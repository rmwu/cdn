# Identifying biological perturbation targets through causal differential networks

Official implementation of causal differential networks (ICML 2025).

**NOTE** This repository is currently under construction.
Links will be updated as files are cleaned and prepared.

## Overview

Our goal is to identify the root causes that drive differences between
two biological systems.
For example: What transcription factors drive cell differentiation?
What are the direct targets of a drug?
We take a causality-inspired approach: if we could "invert" datasets into
their causal mechanisms, it would be straightforward to read off the
differences.

If you find our work interesting, please check out our paper to learn more:
[Identifying biological perturbation targets through causal differential
networks
](https://arxiv.org/abs/2410.03380).

## Installation

```
conda create -y --name cdn pip python=3.10
conda activate cdn

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm rich pyyaml numpy=1.26.4 pandas matplotlib seaborn
pip install pytorch-lightning==2.4.0 torchmetrics==1.4.1 causal-learn==0.1.3.8 wandb
```

CDN was tested using Python 3.10 with PyTorch 2.4.1.
We trained our models on a single A6000 GPU.

## Quickstart

To run inference using our pretrained models, please modify the data and model paths in
`src/inference.sh`, specify the appropriate config file, and run:
```
./src/inference.sh
```
When benchmarking runtimes, it is assumed that `batch_size=1`.
If you do not need runtimes, you may increase `batch_size` for faster
completion.

To train your own CDN, please modify the data and model paths in
`src/train.sh`, specify the appropriate config file, change the wandb
project to your own, and run:
```
./src/train.sh
```
We recommend at least 10-20 data workers per GPU and a batch size of at least
16 for synthetic, and a batch size of 1 for Perturb-seq.

## Models

You may download our checkpoints [here]().
We provide pretrained weights for 3 versions of CDN:
- Synthetic (all mechanisms)
- Synthetic (exclude scale, for reproducibility)
- Perturb-seq finetuned

## Datasets

You may download our datasets [here](https://figshare.com/articles/dataset/Single_cell_evaluation_datasets/29215766?file=55059587).
The unzipped folder should be placed under `data`, which will be referenced by splits files:
- `data/test_240.csv` Synthetic testing datasets
- `data/perturbseq.csv` Perturb-seq finetuning and testing datasets for *seen* cell line
- `data/perturbseq_{cell_line}.csv` Perturb-seq finetuning and testing datasets for *unseen* cell line splits
- `data/sciplex.csv` Sci-Plex testing datasets (*unseen* cell line and intervention type)

These splits CSVs are formatted as follows.
- `perturbation` is a string that denotes the identifier of the perturbation target
- `name` is a string that denotes the raw perturbation target (e.g. gene name)
- `cluster` (where applicable) is an integer that denotes the k-means cluster of the log-fold
  change in gene expression, used for data splitting purposes
- `split` is a string (either train, val or test)
- `fp_data` is a string that describes the path to the *interventional* dataset.
  The corresponding *observational* dataset can be found by replacing
  `data_interv` by `data` (done dynamically in our codebase).
- `label` is an integer that represents the index of the true target within the
  data.

## Results

TBD

