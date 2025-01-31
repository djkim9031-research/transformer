# Transformer
Recreation of a transformer architecture for sequential linearized inputs. Given the tokenized language inputs, the model infers on the next token.

This repo recreates the "Decoder" block of the transformer architecture, and also provides the baseline bigram model for reference.
As the aim of the current transformer model is to have a "generative" capability based on input dataset, it is an auto regressive model and therefore, do not contain a cross-attention layer in the decoder block.

## How to build
If you opt to use docker, use ***nvidia-docker***. <br></br>
Ubuntu 20.04 with CUDA 12.0 or above is recommended.

Install dockerfile and run the docker volume
```
nvidia-docker build -t transformer:latest .
nvidia-docker run -v ./:/code -it --name transformer_volume transformer:latest
```

Make bulid directory and navigate there to compile,
```
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=${LIBTORCH_PATH} ..
make -j${nproc}
```

## How to use

After building the current repo, run the executable with the required arguments.
If the CUDA device is available, it will automatically run on CUDA device.

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
context window length: 50
embedding dimensions: 384
number of attention heads: 6
number of attention blocks (layers): 8
drop out rate: 0.4
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
thee,
And all those friends that deign to follow it,
Lest in the nexting husband. Now, let's seek
us a touch'd time and noble increast!
For rove and Mowbray's God's brother, show a man.
Third is thus with thee: but your father, we say,
Though thou departed Harry's brother, Warwick, that all the
deed infant frowness of the other steeds and revokes.

CORIOLANUS:
Avaid it shallow me with me:
Ah, I'll warrant thee. Poor Clarence so
Can holumn and him.

DUKE OF YORK:
What, well, what never sat Lamentor, great:
Now or his 'defermity can, she is known.

AUTOLYCUS:
A mOthacase; I would I do her untirely,
Why happiness of grave author near to be?
Farewell, Clifford.

CAMILLO:
If we be so righteously beheven of you.

Provost:
Marcuteously for this storeking mades,
Were pierce-a-headed blow, will not under this wreck
Was every true buttale; indeed, in this;
From this true here comely chear good?

ISABELLA:
O pain chard-babe,--
As we and gentlemen
As in Crassugatro, as a fearm. Come, few, prithout,
Your child, comfort pretty breaths let's 'gains

```

Could be better, but for the given compute resource sounds plausible 😅



## Disclaimer
Depending on your cuda toolkit version, you may have to change the first line of Dockerfile to your specific cuda version.

