#pragma once

#include <torch/torch.h>
#include <cmath>

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
}