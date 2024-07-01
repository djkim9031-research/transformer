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
./transformer transformer ../data/input.txt 64 100 embedding_dims=384 num_attention_heads=6 num_attention_blocks=8 dropout_probs=0.2 seed_num=42 train_val_split_ratio=0.9 max_train_steps=10000 evaluation_interval=1000 loss_eval_iteration=100 num_tokens_to_generate=1000
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
[OPTIONAL] <max train steps>: Total number of training steps, Default = 10000
[OPTIONAL] <evaluation interval>: Interval of training steps at which loss evaluation is triggered for train/val dataset, Default = 1000
[OPTIONAL] <loss eval iteration>: How many iteration to perform during loss evaluation, the mean over the iteration is reported, Default = 100
[OPTIONAL] <num tokens to generate>: Once the training completes, how many tokens to generate (i.e., how many unique character to generate), Default = 1000
```

## Example output
This is the example output when the transformer is trained with:
```
batch size: 64
context window length: 30
embedding dimensions: 384
number of attention heads: 6
number of attention blocks (layers): 8
drop out rate: 0.2
seed number: 42
train/val split ratio: 0.9
max training steps: 10000
evaluation interval: 1000
loss eval iteration: 100
num tokens to generate: 1000
```

Prompt:
```
First Citizen:
My good sir, I must
```
Generated:
```
ng bought love with such a blood! without means with his dimpority.

RATCLIFF:
As any thousand, I pray stay, these tender could have discoted
By commands the faction of his duteous is
To sat her bridgetting tongue perceive
To find thee.

PETRUCHIO:
Now, by Saint Clarence Henry Peter.
Marcius was to punish the lords of sorrow
And bath thee, my dear love.

PAULINA:
Give me thy love as long such a true bankrd abulades on my
name they, and goes the tyrant.
And fellows it in his ditetted smiles;
You'll be wenfully but say shame 'tis set an learness so witters I with mire
Than a man in good birthrough in the duke,
How not your presume to come it
In corpost to lay it! I say, and then, holdest-blain,
That will do it so. Do I send it hie to loved upon the common viners' bearing bloods: you
righ'd you hast me leaved with
him or shame.
And folly the king; I fear his bones;
Even of her in majesty, and as thou
shalt have assured tips of that
That was your own colours
He meets with weeds. I had got it not seem.
Well, if in her l
```

Could be better, but for the given compute resource sounds plausible 😅



## Disclaimer

Current version is implemented on LibTorch CPU version, and the future implementation will support LibTorch GPU version.
