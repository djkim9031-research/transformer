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

            std::cout<<"[Step "<<step+1<<"] Loss = "<<loss.item<float>()<<std::endl;
        }
    }
}