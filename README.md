# Transformer
Recreation of a transformer architecture for sequential linearized inputs. Given the tokenized language inputs, the model infers on the next token.

This repo recreates the "Decoder" block of the transformer architecture, and also provides the baseline bigram model for reference.
As the aim of the current transformer model is to have a "generative" capability based on input dataset, it is an auto regressive model and therefore, do not contain a cross-attention layer in the decoder block.

## How to use

After building the current repo, run the executable with the required arguments.

Example 1: bigram model
```
./transformer bigram ../data/input.txt 32 8
```

Example 2: transformer model (only with required arguments)
```
./transformer transformer ../data/input.txt 32 8
```

To provide the optional arguments, following is the argument list:
```
[REQUIRED] <model name>: Name of the model [bigram/transformer]\n"
[REQUIRED] <data path>: Path to the input dataset\n"
[REQUIRED] <batch size>: Number of batches to be used in training\n"
[REQUIRED] <context window size>: Max number of tokens to be used for sequence prediction\n"

[OPTIONAL] <embedding dimensions>: Number of embedding dimensions to be used, Default = 384\n"
[OPTIONAL] <num attention heads>: Number of attention heads in the multi self-attention, Default = 6\n"
[OPTIONAL] <num attention blocks>: Number of decoder blocks to be used in the transformer, Default = 8\n"
[OPTIONAL] <dropout probs>: Percetange of dropout in the multi self-attention and feedforward layers, Default = 0.2\n"
[OPTIONAL] <seed num>: Seed number for the reproducibility, Default = 42\n"
[OPTIONAL] <train val split ratio>: Split ratio of train to validation dataset from input data, Default = 0.9\n"
[OPTIONAL] <evaluation interval>: Interval of training steps at which loss evaluation is triggered for train/val dataset, Default = 1000\n"
[OPTIONAL] <loss eval iteration>: How many iteration to perform during loss evaluation, the mean over the iteration is reported, Default = 100\n"
[OPTIONAL] <num tokens to generate>: Once the training completes, how many tokens to generate (i.e., how many unique character to generate), Default = 1000\
```

## Disclaimer

Current version is implemented on LibTorch CPU version, and the future implementation will support LibTorch GPU version.
