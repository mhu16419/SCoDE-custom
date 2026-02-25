# Steer Your Model: Secure Code Generation with Contrastive Decoding
This is the official repository for ''Steer Your Model: Secure Code Generation with Contrastive Decoding''.

## Setup
First, install Python dependencies:
```console
pip install -r requirements.txt
```
Then, install [GitHub CodeQL](https://codeql.github.com/), which will be used for evaluating the security of LLM-generated code:
```console
./setup_codeql.sh
```
Finally, set up different programming languages studied in this work (`sudo` rights required):
```console
./setup_langs.sh
```

## Training

### functional tuning
Run the following command to fine-tune an pretrained LLM with standard instruction tuning:
```console
cd scripts/

python train.py --pretrain_name starcoderbase-1b --output_name starcoderbase-1b-std --datasets evol
```
Here, `--pretrain_name` specifies the base pretrained LLM, `--output_name` denotes the user-provided name of the standard model, and `--datasets` represents a list of datasets used for training.

### security tuning
Run the following command to fine-tune the standard LLM with security dataset:
#### STCD:
```console
cd steer/

python train_steers.py --pretrain_name starcoderbase-1b-std --output_name starcoderbase-1b-std-steer --datasets sec-desc sec-new-desc --num_steers 2 --dummy_steer 1 --rank 1000 --regularization 1e-6 --epsilon 1e-3
```
Here, `--pretrain_name` specifies the standard pretrained LLM, `--output_name` denotes the user-provided name of the security model, and `--datasets` represents a list of security datasets used for training.

#### SVEN:
```console
cd scripts/

python train.py --pretrain_name starcoderbase-1b-std --output_name starcoderbase-1b-std-sven --datasets sec-desc sec-new-desc --sven --kl_loss_weight 1600
```

#### CoSec:
```console
cd cosec/

bash run_cosec.sh
```

## Evaluation
Our evaluation covers various benchmarks concerning security and utility. You just run the following commands:
```console
bash run_*.sh
```
Here, `*` denotes the methods, like `base`,`stcd`,`sven`, and `cosec` 


## Datasets
The datasets can be found in [`SafeCoder`](https://github.com/eth-sri/SafeCoder).

## Acknowledgement
We are very grateful that the authors of [`SVEN`](https://github.com/eth-sri/sven), [`SafeCoder`](https://github.com/eth-sri/SafeCoder), [`CoSec`](https://github.com/Nero0113/CoSec), and [`LM-Steers`](https://github.com/Glaciohound/LM-Steer) make their code publicly available so that we can build this repository on top of their codes.
