#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>

#include <torch/torch.h>


namespace tokenizer {
    // Function to create stoi, itos mappings
    void createMappings(const std::vector<char>& chars,
                        std::unordered_map<char, int>& stoi,
                        std::unordered_map<int, char>& itos){
        for(size_t i=0; i<chars.size(); ++i){
            stoi[chars[i]] = i;
            itos[i] = chars[i];
        }
    }

    // Function to encode a string
    std::vector<int> encode(const std::string& s, 
                            const std::unordered_map<char, int> &stoi){
        std::vector<int> encoded;
        for(char c : s){
            encoded.push_back(stoi.at(c));
        }
        return encoded;
    }

    // Function to decode tokens to corresponding strings
    std::string decode(const std::vector<int>& l,
                       const std::unordered_map<int, char>& itos){
        std::string decoded = "";
        for(int i : l){
            decoded += itos.at(i);
        }
        return decoded;
    }
}

namespace preprocessing {

    // Preprocessing function - parse the data from the given .txt
    // and create the char-to-int, int-to-char mappings (which maps 
    // the unique char in the dataset).
    std::string data_parser(const std::string& data_path,
                            std::unordered_map<char, int>& stoi,
                            std::unordered_map<int, char>& itos){
        
        // Read the content of the file
        std::ifstream file(data_path);
        if (!file.is_open()) {
            std::cerr << "Could not open the file!" << std::endl;
            return "";
        }

        std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        file.close();

        // Extract unique characters
        std::unordered_set<char> unique_chars(text.begin(), text.end());

        // Convert unordered_set to a vector and sort it
        std::vector<char> chars(unique_chars.begin(), unique_chars.end());
        std::sort(chars.begin(), chars.end());

        // Create stoi, itos mappings
        tokenizer::createMappings(chars, stoi, itos);

        return text;
    }

    // Preprocessing function - splitting given data into train/val dataset
    void split_dataset(const float data_split_ratio,
                       const torch::Tensor& data,
                       torch::Tensor& train_data,
                       torch::Tensor& val_data){
        
        int n_train = static_cast<int>(data_split_ratio * data.size(0));
        int n_val = data.size(0) - static_cast<int>(data_split_ratio * data.size(0));

        std::vector<torch::Tensor> splits = torch::split(data, {n_train, n_val}, 0);

        train_data = splits[0];
        val_data = splits[1];

        return;
    }

    // Preprocessing function - creating batches of dataset
    void get_batch(const int batch_size,
                   const int context_win_size,
                   const torch::Tensor& data,
                   const torch::Generator& gen,
                   torch::Tensor& batch_input,
                   torch::Tensor& batch_output){
        
        auto idx = torch::randint(0, data.size(0) - context_win_size, {batch_size}, gen, torch::kInt64);

        // Extract the tensors at the selected index for each batch
        std::vector<torch::Tensor> x_vec, y_vec;
        for(size_t i=0; i<batch_size; ++i){
            int start_idx = idx[i].item<int>();
            x_vec.push_back(data.slice(/*dim=*/0, /*start=*/start_idx, /*end=*/start_idx + context_win_size));
            y_vec.push_back(data.slice(/*dim=*/0, /*start=*/start_idx + 1, /*end=*/start_idx + context_win_size + 1));
        }

        // Torch-ify the tensors to form [batch_size, context_win_size] tensors
        batch_input = torch::stack(x_vec);
        batch_output = torch::stack(y_vec);
    }
}


// Preparing the dataset for nn model.
inline void data_loader(const std::string &data_path){

    // Parse the data
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    std::string text = preprocessing::data_parser(data_path, stoi, itos);

    // Encode the parsed data
    std::vector<int> encoded_text = tokenizer::encode(text, stoi);
    torch::Tensor data = torch::tensor(encoded_text, torch::dtype(torch::kInt64));
    
    // Splitting the encoded data to train_data, and val_data
    torch::Tensor train_data, val_data;
    preprocessing::split_dataset(0.9, data, train_data, val_data);

    // Create batches of training and validation data.
    int batch_size = 4;
    int context_win_size = 8;
    auto gen = at::detail::createCPUGenerator(42);
    torch::Tensor xb, yb;
    preprocessing::get_batch(batch_size, context_win_size, train_data, gen, xb, yb);

    for(int b=0; b<batch_size; ++b){
        for(int c=0; c<context_win_size; ++c){
            torch::Tensor context = xb[b].slice(/*dim=*/0, /*start=*/0, /*end=*/c+1);
            torch::Tensor target = yb[b][c];
            std::cout<<"Input: "<<context<<std::endl;
            std::cout<<"Target: "<<target<<std::endl;
        }
        std::cout<<"___________________________"<<std::endl;
    }

}