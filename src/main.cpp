#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>

#include "utils.h"
#include "bigram.h"
#include "transformer.h"

void print_help(const std::string& program_name) {
    std::cerr << "Usage: " << program_name << " <model name>\n"
              << " [REQUIRED] <model name>: Name of the model [bigram/transformer]\n"
              << " [REQUIRED] <data path>: Path to the input dataset\n"
              << " [REQUIRED] <batch size>: Number of batches to be used in training\n"
              << " [REQUIRED] <context window size>: Max number of tokens to be used for sequence prediction\n"
              << " ____________________________________________________________________________________________\n"
              << " For transformer model, following parameters can be optionally provided\n"
              << " [OPTIONAL] <embedding dimensions>: Number of embedding dimensions to be used, Default = 384\n"
              << " [OPTIONAL] <num attention heads>: Number of attention heads in the multi self-attention, Default = 6\n"
              << " [OPTIONAL] <num attention blocks>: Number of decoder blocks to be used in the transformer, Default = 8\n"
              << " [OPTIONAL] <dropout probs>: Percetange of dropout in the multi self-attention and feedforward layers, Default = 0.2\n"
              << " [OPTIONAL] <seed num>: Seed number for the reproducibility, Default = 42\n"
              << " [OPTIONAL] <train val split ratio>: Split ratio of train to validation dataset from input data, Default = 0.9\n"
              << " [OPTIONAL] <max train steps>: Total number of training steps, Default = 10000\n"
              << " [OPTIONAL] <evaluation interval>: Interval of training steps at which loss evaluation is triggered for train/val dataset, Default = 1000\n"
              << " [OPTIONAL] <loss eval iteration>: How many iteration to perform during loss evaluation, the mean over the iteration is reported, Default = 100\n"
              << " [OPTIONAL] <num tokens to generate>: Once the training completes, how many tokens to generate (i.e., how many unique character to generate), Default = 1000\n"
              << " ____________________________________________________________________________________________\n";
}
    

int main(int argc, char** argv) {

    if (argc < 5) {
        print_help(argv[0]);
        return 1;
    }

    const std::string model_name = argv[1];
    const std::string data_path = argv[2];
    int batch_size = std::stoi(argv[3]);
    int context_win_size = std::stoi(argv[4]);

    // Default values for optional parameters
    int embedding_dims = 384;
    int num_attention_heads = 6;
    int num_attention_blocks = 8;
    float dropout_probs = 0.2;
    int seed_num = 42;
    float train_val_split_ratio = 0.9;
    int max_train_steps = 10000;
    int evaluation_interval = 1000;
    int loss_eval_iteration = 100;
    int num_tokens_to_generate = 1000;
    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Optional parameters start from argv[5]
    std::map<std::string, std::string> optional_params;
    for (int i = 5; i < argc; ++i) {
        std::string arg(argv[i]);
        auto pos = arg.find('=');
        if (pos != std::string::npos) {
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            optional_params[key] = value;
        }
    }

    // Update the parameters if they are provided
    if (optional_params.find("embedding_dims") != optional_params.end()) {
        embedding_dims = std::stoi(optional_params["embedding_dims"]);
    }
    if (optional_params.find("num_attention_heads") != optional_params.end()) {
        num_attention_heads = std::stoi(optional_params["num_attention_heads"]);
    }
    if (optional_params.find("num_attention_blocks") != optional_params.end()) {
        num_attention_blocks = std::stoi(optional_params["num_attention_blocks"]);
    }
    if (optional_params.find("dropout_probs") != optional_params.end()) {
        dropout_probs = std::stof(optional_params["dropout_probs"]);
    }
    if (optional_params.find("seed_num") != optional_params.end()) {
        seed_num = std::stoi(optional_params["seed_num"]);
    }
    if (optional_params.find("train_val_split_ratio") != optional_params.end()) {
        train_val_split_ratio = std::stof(optional_params["train_val_split_ratio"]);
    }
    if (optional_params.find("max_train_steps") != optional_params.end()) {
        max_train_steps = std::stoi(optional_params["max_train_steps"]);
    }
    if (optional_params.find("evaluation_interval") != optional_params.end()) {
        evaluation_interval = std::stoi(optional_params["evaluation_interval"]);
    }
    if (optional_params.find("loss_eval_iteration") != optional_params.end()) {
        loss_eval_iteration = std::stoi(optional_params["loss_eval_iteration"]);
    }
    if (optional_params.find("num_tokens_to_generate") != optional_params.end()) {
        num_tokens_to_generate = std::stoi(optional_params["num_tokens_to_generate"]);
    }

    try{
        if(model_name=="bigram"){
            nn_models::bigram_training_pipeline(data_path, batch_size, context_win_size);
        } else if(model_name=="transformer"){
            if(embedding_dims%num_attention_heads!=0){
                std::cout << "Embedding dims % Num attention heads should be zero, terminnating..."<<std::endl;
                return 1;
            }
            std::cout <<"Transformer model running on "<<(torch::cuda::is_available() ? "CUDA" : "CPU")<<std::endl;
            nn_models::transformer_training_pipeline(data_path, batch_size, context_win_size, 
                                                     embedding_dims, num_attention_heads, 
                                                     num_attention_blocks, dropout_probs, 
                                                     seed_num, train_val_split_ratio, max_train_steps,
                                                     evaluation_interval, loss_eval_iteration,
                                                     num_tokens_to_generate, device_type);
        } else {
            std::cerr << "Unknown model name: " << model_name << std::endl;
            print_help(argv[0]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

