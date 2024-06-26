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

namespace preprocessing{

    void split_dataset(const float data_split_ratio,
                       const torch::Tensor& data,
                       torch::Tensor& train_data,
                       torch::Tensor& validation_data){
        
        int n_train = static_cast<int>(data_split_ratio * data.size(0));
        int n_val = data.size(0) - static_cast<int>(data_split_ratio * data.size(0));

        std::vector<torch::Tensor> splits = torch::split(data, {n_train, n_val}, 0);

        train_data = splits[0];
        validation_data = splits[1];

        return;
    }
}

inline void data_parser(const std::string &data_path){

    // Read the content of the file
    std::ifstream file(data_path);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Extract unique characters
    std::unordered_set<char> unique_chars(text.begin(), text.end());

    // Convert unordered_set to a vector and sort it
    std::vector<char> chars(unique_chars.begin(), unique_chars.end());
    std::sort(chars.begin(), chars.end());

    //Determine the vocabulary size
    int vocab_size = chars.size();

    // Output the results
    std::cout << "Unique characters: ";
    for (char c : chars) {
        std::cout << c << " ";
    }
    std::cout << std::endl;

    std::cout << "Vocabulary size: " << vocab_size << std::endl;


    // Encode, decode logic
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;
    tokenizer::createMappings(chars, stoi, itos);
    std::vector<int> encoded_text = tokenizer::encode(text, stoi);
    torch::Tensor data = torch::tensor(encoded_text, torch::dtype(torch::kInt64));
    torch::Tensor train_data, val_data;
    preprocessing::split_dataset(0.9, data, train_data, val_data);

    std::cout<<train_data.sizes()<<std::endl;
    std::cout<<val_data.sizes()<<std::endl;
    std::cout<<val_data<<std::endl;

    // Output the tensor
    std::cout << "Encoded tensor size: " << data.sizes() << std::endl;

    // Example
    std::string example_string = "hii there";
    std::vector<int> encoded = tokenizer::encode(example_string, stoi);
    std::string decoded = tokenizer::decode(encoded, itos);

     // Output the results
    std::cout << "Original string: " << example_string << std::endl;
    std::cout << "Encoded: ";
    for (int i : encoded) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "Decoded string: " << decoded << std::endl;
}