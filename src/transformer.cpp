#include "transformer.h"

namespace nn_models{

    void transformer_training_pipeline(const std::string &data_path,
                                       const int batch_size,
                                       const int context_win_size,
                                       const int embedding_dims,
                                       const int num_attention_heads,
                                       const int num_attention_blocks,
                                       const float dropout_probs,
                                       const int seed_num,
                                       const float train_val_split_ratio,
                                       const int max_training_step,
                                       const int evaluation_interval,
                                       const int loss_eval_iter,
                                       const int num_tokens_to_generate,
                                       const torch::DeviceType device){
        
        // Parse the data
        std::unordered_map<char, int> stoi;
        std::unordered_map<int, char> itos;
        int vocab_size;
        torch::Device run_device(device);
        std::string text = preprocessing::data_parser(data_path, stoi, itos, vocab_size);
        
        // Encode the parsed data
        std::vector<int> encoded_text = tokenizer::encode(text, stoi);
        torch::Tensor data = torch::tensor(encoded_text, torch::dtype(torch::kInt64)).to(run_device);
        
        // Splitting the encoded data to train_data, and val_data
        torch::Tensor train_data, val_data;
        preprocessing::split_dataset(train_val_split_ratio, data, train_data, val_data);
        
        // Construct the transformer model
        Transformer transformer(vocab_size, context_win_size, embedding_dims, num_attention_heads, num_attention_blocks, dropout_probs, seed_num);
        transformer.to(run_device);

        // Training
        torch::optim::AdamW optimizer(transformer.parameters(), torch::optim::AdamWOptions(1e-3));
        for(size_t step = 0; step < max_training_step; ++step){
            torch::Tensor xb, yb;
            preprocessing::create_batch(batch_size, context_win_size, train_data, xb, yb);
            xb.to(run_device);
            yb.to(run_device);
            
            torch::Tensor loss;
            transformer.forward(xb, yb, loss);

            // backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            if(step % evaluation_interval == 0){
                std::cout<<"[Step "<<step<<"] Training mean loss = "
                <<evaluation(transformer, train_data, loss_eval_iter, batch_size, context_win_size)<<std::endl;
                std::cout<<"|__________ Evaluation mean loss = "
                <<evaluation(transformer, val_data, loss_eval_iter, batch_size, context_win_size)<<std::endl;
                std::cout<<"________________________________________________________"<<std::endl;
            }
        }

        // Inference on the trained model.
        std::string inference_text = "First Citizen:\nMy good sir, I must ";
        encoded_text = tokenizer::encode(text, stoi);
        torch::Tensor inference_tensors = torch::tensor(encoded_text, torch::dtype(torch::kInt64)).to(run_device);
        torch::Tensor ignored, init_data;
        preprocessing::create_batch(1, context_win_size, inference_tensors, init_data, ignored);

        auto inference_data = transformer.generate(init_data, num_tokens_to_generate, context_win_size);
        std::vector<int> inference_data_vectorized;
        for(size_t c=0; c<inference_data.size(1); ++c){
            inference_data_vectorized.push_back(inference_data[0][c].item<int>());
        }
        std::cout<<"Given: "<<inference_text<<std::endl;
        std::cout<<"Generated: "<<tokenizer::decode(inference_data_vectorized, itos)<<std::endl;
    }

    float evaluation(Transformer& model,
                     const torch::Tensor& data,
                     const int eval_iter,
                     const int batch_size,
                     const int context_win_size){
        model.eval(); // evaluation mode
        torch::NoGradGuard no_grad; // Disable gradient calculation
        
        torch::Tensor losses = torch::zeros({eval_iter});
        for(size_t k=0; k<eval_iter; ++k){
            torch::Tensor xb, yb, nll;
            preprocessing::create_batch(batch_size, context_win_size, data, xb, yb);
            model.forward(xb, yb, nll);

            losses[k] = nll.item<float>();
        }

        float mean_loss = losses.mean().item<float>();

        model.train(); // back to train mode before returning.

        return mean_loss;
    }
}
