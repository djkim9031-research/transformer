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

Example 3: transformer model (also with optional arguments)
```
./transformer ../data/input.txt 32 8 embedding_dims=10 num_attention_heads=2 num_attention_blocks=8 dropout_probs=0.2 seed_num=42 train_val_split_ratio=0.9 max_train_steps=10000 evaluation_interval=1000 loss_eval_iteration=100 num_tokens_to_generate=1000
```

To provide the optional arguments, following is the argument list:
```
[REQUIRED] <model name>: Name of the model [bigram/transformer]
[REQUIRED] <data path>: Path to the input dataset
[REQUIRED] <batch size>: Number of batches to be used in training
[REQUIRED] <context window size>: Max number of tokens to be used for sequence prediction

[OPTIONAL] <embedding dimensions>: Number of embedding dimensions to be used, Default = 384
[OPTIONAL] <num attention heads>: Number of attention heads in the multi self-attention, Default = 6
[OPTIONAL] <num attention blocks>: Number of decoder blocks to be used in the transformer, Default = 8
[OPTIONAL] <dropout probs>: Percetange of dropout in the multi self-attention and feedforward layers, Default = 0.2
[OPTIONAL] <seed num>: Seed number for the reproducibility, Default = 42
[OPTIONAL] <train val split ratio>: Split ratio of train to validation dataset from input data, Default = 0.9
[OPTIONAL] <evaluation interval>: Interval of training steps at which loss evaluation is triggered for train/val dataset, Default = 1000
[OPTIONAL] <max train steps>: Total number of training steps, Default = 10000
[OPTIONAL] <loss eval iteration>: How many iteration to perform during loss evaluation, the mean over the iteration is reported, Default = 100
[OPTIONAL] <num tokens to generate>: Once the training completes, how many tokens to generate (i.e., how many unique character to generate), Default = 1000
```

## Disclaimer

Current version is implemented on LibTorch CPU version, and the future implementation will support LibTorch GPU version.
