#include <torch/torch.h>
#include <vector>

#include "utils.h"
#include "bigram.h"


int main() {

    nn_models::bigram_training_pipeline("../data/input.txt",
                                        32, 8, 42, 0.9);

    return 0;
}