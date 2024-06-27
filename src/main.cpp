#include <torch/torch.h>

#include <vector>

#include "utils.h"


int main() {
    
    torch::Tensor train_data, val_data;
    data_parser("../data/input.txt", train_data, val_data);
    return 0;
}