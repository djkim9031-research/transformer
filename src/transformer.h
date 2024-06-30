#pragma once

#include <torch/torch.h>
#include <cmath>
#include <memory>

#include "utils.h"

namespace nn_models{

    class SelfAttention : public torch::nn::Module {
        // Single head of self-attention
        public:
            torch::nn::Linear query{nullptr};
            torch::nn::Linear key{nullptr};
            torch::nn::Linear value{nullptr};

            SelfAttention(int embedding_dims, int head_dims, int context_win_size, int seed_num){
                torch::manual_seed(seed_num);
                query = register_module("query", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, head_dims).bias(false)));
                key = register_module("key", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, head_dims).bias(false)));
                value = register_module("value", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, head_dims).bias(false)));
            }

            torch::Tensor forward(torch::Tensor x){
                int B = x.size(0);
                int T = x.size(1);
                int C = x.size(2); // embedding_dims

                auto q = query->forward(x); // B, T, head_dims
                auto k = key->forward(x); // B, T, head_dims
                int H = q.size(2);

                // Compute attention scores ("affinities")
                // w = (q @ k.T)/sqrt(head_dims) from attention paper, to keep uniform variance
                // [B, T, head_dims] x [B, head_dims, T] => [B, T, T]
                auto w = torch::matmul(q, k.transpose(-2, -1)) * std::pow(H, -0.5);
                torch::Tensor trill = torch::tril(torch::ones({T, T}));
                auto mask = trill.unsqueeze(0).expand({B, T, T});
                w = w.masked_fill(mask.eq(0), -std::numeric_limits<float>::infinity());
                w = torch::nn::functional::softmax(w, -1); // [B, T, T]

                // Perform the weighted aggregation of the values.
                auto v = value->forward(x);

                // [B, T, T] x [B, T, H] => [B, T, H]
                return torch::matmul(w, v);
            }
    };

    class MultiHeadSelfAttention : public torch::nn::Module {
        // Multi head self-attention.
        // Running multiple singlehead self-attention in parallel and concatenate the outputs
        // along head dimensions.
        public:
            std::vector<std::shared_ptr<SelfAttention>> attention_heads;
            torch::nn::Linear projection{nullptr};
            torch::nn::Dropout dropout{nullptr};

            MultiHeadSelfAttention(int num_heads, int embedding_dims, int head_dims, int context_win_size, float dropout_prob, int seed_num){
                torch::manual_seed(seed_num);
                for(size_t i=0; i<num_heads; ++i){
                    auto head = std::make_shared<SelfAttention>(embedding_dims, head_dims, context_win_size, seed_num);
                    attention_heads.push_back(register_module("self_attention_" + std::to_string(i), head));
                }
                projection = register_module("projection", torch::nn::Linear(embedding_dims, embedding_dims));
                dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout_prob)));
            }

            torch::Tensor forward(torch::Tensor x){
                std::vector<torch::Tensor> head_outputs;
                for(auto& head : attention_heads){
                    head_outputs.push_back(head->forward(x));
                }
                auto y = torch::cat(head_outputs, /*dims=*/-1);
                y = projection->forward(y);
                return dropout->forward(y);
            }
    };

    class FeedForward : public torch::nn::Module {
        // Feedforward network in the transformer
        // A simple linear layer followed by non-linearity.
        public:
            torch::nn::Sequential net{nullptr};

            FeedForward(int embedding_dims, float dropout_prob, int seed_num){
                torch::manual_seed(seed_num);
                net = register_module("feedforward", torch::nn::Sequential(
                                                            // From the original paper, the inner layer has a multiplier of 4
                                                            torch::nn::Linear(embedding_dims, 4*embedding_dims),
                                                            torch::nn::ReLU(),
                                                            torch::nn::Linear(4*embedding_dims, embedding_dims),
                                                            torch::nn::Dropout(torch::nn::DropoutOptions().p(dropout_prob))
                                    ));
            }

            torch::Tensor forward(torch::Tensor x){
                return net->forward(x);
            }
    };

    class AttentionBlock : public torch::nn::Module {
        // Transformer's attention block - communication followed by computation (multihead attention then feedforward)
        public:
            std::shared_ptr<MultiHeadSelfAttention> at_heads{nullptr};
            std::shared_ptr<FeedForward> feed_forward{nullptr};
            torch::nn::LayerNorm layer_norm1{nullptr};
            torch::nn::LayerNorm layer_norm2{nullptr};

            AttentionBlock(int num_heads, int embedding_dims, int context_win_size, float dropout_prob, int seed_num){
                torch::manual_seed(seed_num);
                at_heads = register_module("attention_heads", std::make_shared<MultiHeadSelfAttention>(num_heads, embedding_dims, embedding_dims/num_heads, context_win_size, dropout_prob, seed_num));
                feed_forward = register_module("feed_forward", std::make_shared<FeedForward>(embedding_dims, dropout_prob, seed_num));
                layer_norm1 = register_module("layer_norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
                layer_norm2 = register_module("layer_norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
            }

            torch::Tensor forward(torch::Tensor x){
                // Layer norms are applied prior to attention and feedforward.
                // This is a deviation from the original paper, but the later research shows
                // that this ensures training stability.
                auto y1 = layer_norm1->forward(x);
                y1 = x + at_heads->forward(y1);

                auto y2 = layer_norm2->forward(y1);
                y2 = y1 + feed_forward->forward(y2);
                return y2;
            }
    };


    class Transformer : public torch::nn::Module {
        public:
            torch::nn::Embedding token_embedding_table{nullptr};
            torch::nn::Embedding position_embedding_table{nullptr};
            torch::nn::Sequential attn_blocks{nullptr};
            torch::nn::Linear lm_head{nullptr};

            Transformer(int vocab_size, int context_win_size, int embedding_dims, int num_attention_heads, int num_attention_blocks, float dropout_prob, int seed_num){
                torch::manual_seed(seed_num);
                token_embedding_table = register_module("token_embedding_table", torch::nn::Embedding(vocab_size, embedding_dims));
                position_embedding_table = register_module("position_embedding_table", torch::nn::Embedding(context_win_size, embedding_dims));
                attn_blocks = register_module("attention_blocks", torch::nn::Sequential());
                for(size_t i=0; i<num_attention_blocks; ++i){
                    attn_blocks->push_back(AttentionBlock(num_attention_heads, embedding_dims, context_win_size, dropout_prob, seed_num));
                }
                attn_blocks->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({embedding_dims})));
                lm_head = register_module("linear_head", torch::nn::Linear(embedding_dims, vocab_size));
            }

            torch::Tensor forward(torch::Tensor &x, torch::Tensor &y, torch::Tensor &nll){
                int B = x.size(0);
                int T = x.size(1);

                auto token_embeddings = token_embedding_table->forward(x); // Shape B, T, C1 (batch, context, embedding_dims)
                auto positions = torch::arange(0, T, torch::kLong); // Shape T (context_win_size)
                auto pos_embeddings = position_embedding_table->forward(positions); // shape T, C1 (context, embedding_dims)
                auto embedding_vectors = token_embeddings + pos_embeddings; // B, T, C1 (batch, context, embedding_dims)

                auto attention_block_output = attn_blocks->forward(embedding_vectors); // B, T, embedding_dims
                auto logits = lm_head->forward(attention_block_output); // Shape B, T, C2 (batch, context, vocab_size)

                if (y.size(0) > 0){
                    int C = logits.size(2);

                    logits = logits.view({B*T, C});
                    auto targets = y.view({-1}); //[B, T] tensor to [B*T] tensor

                    nll = torch::nn::functional::cross_entropy(logits, targets);
                } 
                
                return logits;
            }

            torch::Tensor generate(const torch::Tensor &x, const int max_new_tokens, const int model_context_win_size){
                // Input x is of size [B, T] 
                torch::Tensor generated = x.clone();
                torch::Tensor not_used = torch::Tensor();

                for(size_t i=0; i<max_new_tokens; ++i){

                    torch::Tensor curr_tokens = generated.slice(1, -model_context_win_size);

                    // Get the embedding logits [B, T, C]
                    auto logits = forward(curr_tokens, not_used, not_used);

                    // Get the last token in the context window
                    // [B, 1, C]
                    logits = logits.slice(/*dim=*/1, /*start_idx=*/logits.size(1)-1, /*last_idx=*/logits.size(1));

                    // Get the probabilitiy distribution over the vocab size for all batches
                    auto probs = torch::nn::functional::softmax(logits, /*dim=*/2);

                    // Sample from the distribution
                    // [B, 1, C] => [B, 1]
                    probs = probs.squeeze(1);
                    auto idx_next = torch::multinomial(probs, /*num_samples=*/1, /*replacement=*/true);

                    // Append the sampled indices to the running sequence
                    // [B, curr_T+1]
                    generated = torch::cat({generated, idx_next}, /*dim=*/1);
                }

                return generated;
            }
    };

    // Transformer model pipeline.
    // This includes reading the dataset from the provided data_path,
    // then tokenizing the dataset, splitting it to train/val tokenized data,
    // and train the model
    void transformer_training_pipeline(const std::string &data_path,
                                       const int batch_size,
                                       const int context_win_size,
                                       const int embedding_dims = 384,
                                       const int num_attention_heads = 6,
                                       const int num_attention_blocks = 8,
                                       const float dropout_probs = 0.2,
                                       const int seed_num = 42,
                                       const float train_val_split_ratio = 0.9,
                                       const int max_training_step = 10000,
                                       const int evaluation_interval = 1000,
                                       const int loss_eval_iter = 100,
                                       const int num_tokens_to_generate = 1000);

    // Loss evaluation logic, loss metrics against training and validataion data
    // are evaluated during training to check if a model overfits to the data. 
    float evaluation(Transformer& model,
                     const torch::Tensor& data,
                     const int eval_iter,
                     const int batch_size,
                     const int context_win_size);
}