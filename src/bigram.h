#pragma once

#include <torch/torch.h>

class BigramLanguageModel : public torch::nn::Module{
    public:
        torch::nn::Embedding token_embedding_table{nullptr};

        BigramLanguageModel(int vocab_size, int seed_num){
            torch::manual_seed(seed_num);
            token_embedding_table = register_module("token_embedding_table", torch::nn::Embedding(vocab_size, vocab_size));
        }

        torch::Tensor forward(torch::Tensor &x, torch::Tensor &y, torch::Tensor &nll){
            auto logits = token_embedding_table->forward(x); // Shape B, T, C (batch, context, vocab)
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