#include "bigram.h"

namespace nn_models{

    void bigram_training_pipeline(const std::string &data_path,
                                  const int batch_size,
                                  const int context_win_size,
                                  const int seed_num,
                                  const float train_val_split_ratio,
                                  const int max_training_step){
        
        // Parse the data
        std::unordered_map<char, int> stoi;
        std::unordered_map<int, char> itos;
        int vocab_size;
        int embedding_dims = 32;
        std::string text = preprocessing::data_parser(data_path, stoi, itos, vocab_size);
        
        // Encode the parsed data
        std::vector<int> encoded_text = tokenizer::encode(text, stoi);
        torch::Tensor data = torch::tensor(encoded_text, torch::dtype(torch::kInt64));
        
        // Splitting the encoded data to train_data, and val_data
        torch::Tensor train_data, val_data;
        preprocessing::split_dataset(train_val_split_ratio, data, train_data, val_data);

        // Construct the bigram language model
        BigramLanguageModel bigram(vocab_size, embedding_dims, seed_num);
        //bigram.train();
        //bigram.eval();

        // Training
        torch::optim::AdamW optimizer(bigram.parameters(), torch::optim::AdamWOptions(1e-3));
        for(size_t step = 0; step < max_training_step; ++step){
            torch::Tensor xb, yb;
            preprocessing::create_batch(batch_size, context_win_size, train_data, xb, yb);
            
            torch::Tensor loss;
            bigram.forward(xb, yb, loss);

            // backward pass
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            
            if(step % 1000 == 0){
                std::cout<<"[Step "<<step<<"] Training mean loss = "
                <<evaluation(bigram, train_data, 100, batch_size, context_win_size)<<std::endl;
                std::cout<<"|__________ Evaluation mean loss = "
                <<evaluation(bigram, val_data, 100, batch_size, context_win_size)<<std::endl;
                std::cout<<"________________________________________________________"<<std::endl;
            }
        }

        // Inference on the trained model.
        torch::Tensor init_data = torch::zeros({1, 1}, torch::kInt64);
        auto inference_data = bigram.generate(init_data, 100);
        std::vector<int> inference_data_vectorized;
        for(size_t c=0; c<inference_data.size(1); ++c){
            inference_data_vectorized.push_back(inference_data[0][c].item<int>());
        }
        std::cout<<"Given: "<<tokenizer::decode({init_data[0][0].item<int>()}, itos)<<std::endl;
        std::cout<<"Generated: "<<tokenizer::decode(inference_data_vectorized, itos)<<std::endl;
    }

    float evaluation(BigramLanguageModel& model,
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