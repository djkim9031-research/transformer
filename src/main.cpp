#include <torch/torch.h>
#include <vector>

#include "utils.h"
#include "bigram.h"
#include "transformer.h"


int main() {

    nn_models::transformer_training_pipeline("../data/input.txt",
                                             32, 8, 42, 0.9);

    return 0;
}