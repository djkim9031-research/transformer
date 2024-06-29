#pragma once

#include <torch/torch.h>
#include <cmath>
#include <memory>

#include "utils.h"

namespace nn_models{

    class SelfAttentionHead : public torch::nn::Module {
        // Single head of self-attention
        public:
            torch::nn::Linear query{nullptr};
            torch::nn::Linear key{nullptr};
            torch::nn::Linear value{nullptr};
            torch::Tensor trill;

            SelfAttentionHead(int embedding_dims, int head_dims, int context_win_size, int seed_num){
                torch::manual_seed(seed_num);
                query = register_module("query", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, head_dims).bias(false)));
                key = register_module("key", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, head_dims).bias(false)));
                value = register_module("value", torch::nn::Linear(torch::nn::LinearOptions(embedding_dims, head_dims).bias(false)));
                trill = torch::tril(torch::ones({context_win_size, context_win_size}));
            }

            torch::Tensor forward(torch::Tensor& x){
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
                auto mask = trill.unsqueeze(0).expand({B, T, T});
                w = w.masked_fill(mask.eq(0), -std::numeric_limits<float>::infinity());
                w = torch::nn::functional::softmax(w, -1); // [B, T, T]

                // Perform the weighted aggregation of the values.
                auto v = value->forward(x);
                // [B, T, T] x [B, T, H] => [B, T, H]
                auto out = torch::matmul(w, v);

                return out;
            }
    };


    class Transformer : public torch::nn::Module {
        public:
            torch::nn::Embedding token_embedding_table{nullptr};
            torch::nn::Embedding position_embedding_table{nullptr};
            std::unique_ptr<SelfAttentionHead> sa_head;
            torch::nn::Linear lm_head{nullptr};

            Transformer(int vocab_size, int context_win_size, int embedding_dims, int head_dims, int seed_num){
                torch::manual_seed(seed_num);
                token_embedding_table = register_module("token_embedding_table", torch::nn::Embedding(vocab_size, embedding_dims));
                position_embedding_table = register_module("position_embedding_table", torch::nn::Embedding(context_win_size, embedding_dims));
                sa_head = std::make_unique<SelfAttentionHead>(embedding_dims, head_dims, context_win_size, seed_num);
                lm_head = register_module("linear_head", torch::nn::Linear(head_dims, vocab_size));
            }

            torch::Tensor forward(torch::Tensor &x, torch::Tensor &y, torch::Tensor &nll){
                int B = x.size(0);
                int T = x.size(1);

                auto token_embeddings = token_embedding_table->forward(x); // Shape B, T, C1 (batch, context, embedding_dims)
                auto positions = torch::arange(0, T, torch::kLong); // Shape T (context_win_size)
                auto pos_embeddings = position_embedding_table->forward(positions); // shape T, C1 (context, embedding_dims)
                auto embedding_vectors = token_embeddings + pos_embeddings; // B, T, C1 (batch, context, embedding_dims)

                auto attention_output = sa_head->forward(embedding_vectors); // B, T, H (batch, context, head_dims)
                auto logits = lm_head->forward(attention_output);

                if (y.size(0) > 0){
                    int C = logits.size(2);

                    logits = logits.view({B*T, C});
                    auto targets = y.view({-1}); //[B, T] tensor to [B*T] tensor

                    nll = torch::nn::functional::cross_entropy(logits, targets);
                } 
                
                return logits;
            }
    };
}