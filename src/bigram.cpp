#include "bigram.h"

namespace nn_models{

    void bigram_training_pipeline(const std::string &data_path,
                                  int batch_size,
                                  int context_win_size,
                                  int seed_num,
                                  float train_val_split_ratio,
                                  int max_training_step){
        
        // Parse the data
        std::unordered_map<char, int> stoi;
        std::unordered_map<int, char> itos;
        int vocab_size;
        std::string text = preprocessing::data_parser(data_path, stoi, itos, vocab_size);
        
        // Encode the parsed data
        std::vector<int> encoded_text = tokenizer::encode(text, stoi);
        torch::Tensor data = torch::tensor(encoded_text, torch::dtype(torch::kInt64));
        
        // Splitting the encoded data to train_data, and val_data
        torch::Tensor train_data, val_data;
        preprocessing::split_dataset(train_val_split_ratio, data, train_data, val_data);

        // Construct the bigram language model
        BigramLanguageModel bigram(vocab_size, seed_num);

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
                std::cout<<"[Step "<<step<<"] Loss = "<<loss.item<float>()<<std::endl;
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
}