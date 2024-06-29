#pragma once

#include <torch/torch.h>

#include "utils.h"

namespace nn_models{

    class BigramLanguageModel : public torch::nn::Module{
        public:
            torch::nn::Embedding token_embedding_table{nullptr};
            torch::nn::Linear linear_head{nullptr};

            BigramLanguageModel(int vocab_size, int n_embedding, int seed_num){
                torch::manual_seed(seed_num);
                token_embedding_table = register_module("token_embedding_table", torch::nn::Embedding(vocab_size, n_embedding));
                linear_head = register_module("linear_head", torch::nn::Linear(n_embedding, vocab_size));
            }

            torch::Tensor forward(torch::Tensor &x, torch::Tensor &y, torch::Tensor &nll){
                auto token_embeddings = token_embedding_table->forward(x); // Shape B, T, C (batch, context, n_embdding)
                auto logits = linear_head->forward(token_embeddings); // Shape B, T, C (batch, context, vocab_size)
                if (y.size(0) > 0){
                    int B = logits.size(0);
                    int T = logits.size(1);
                    int C = logits.size(2);

                    logits = logits.view({B*T, C});
                    auto targets = y.view({-1}); //[B, T] tensor to [B*T] tensor

                    nll = torch::nn::functional::cross_entropy(logits, targets);
                } 
        
                return logits;
            }

            torch::Tensor generate(const torch::Tensor &x, const int max_new_tokens){
                // Input x is of size [B, T] 
                torch::Tensor generated = x.clone();
                torch::Tensor not_used = torch::Tensor();
                for(size_t i=0; i<max_new_tokens; ++i){
                    // Get the embedding logits [B, T, C]
                    auto logits = forward(generated, not_used, not_used);

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

    // Bigram model pipeline.
    // This includes reading the dataset from the provided data_path,
    // then tokenizing the dataset, splitting it to train/val tokenized data,
    // and train the model
    void bigram_training_pipeline(const std::string &data_path,
                                  const int batch_size,
                                  const int context_win_size,
                                  const int seed_num = 42,
                                  const float train_val_split_ratio = 0.9,
                                  const int max_training_step = 10000);

    // Loss evaluation logic, loss metrics against training and validataion data
    // are evaluated during training to check if a model overfits to the data. 
    float evaluation(BigramLanguageModel& model,
                     const torch::Tensor& data,
                     const int eval_iter,
                     const int batch_size,
                     const int context_win_size);
}

