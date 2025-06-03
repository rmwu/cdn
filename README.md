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

pip install tqdm rich pyyaml numpy==1.26.4 pandas matplotlib seaborn
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install wandb pytorch-lightning==2.4.0 torchmetrics==1.4.1 causal-learn==0.1.3.8
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

You may download our checkpoints
[here](https://figshare.com/articles/software/CDN_checkpoints/29225855).
The unzipped folder should be placed in the root directory of this repository.
We provide pretrained weights for all versions of CDN used in our paper.
- Synthetic (all mechanisms)
  - `cdn_synthetic-all.ckpt` "concatenate" version
    **recommended for benchmarking on synthetic datasets**
  - `cdn_synthetic_diff-all.ckpt` "difference" version
    **recommended for benchmarking on chemical perturbation datasets**
- Synthetic (ablations)
  - `cdn_synthetic_noG-no_scale.ckpt` remove "scale" interventions, remove graph loss
  - `cdn_synthetic-no_scale.ckpt` remove "scale" interventions
  - `cdn_synthetic-no_shift.ckpt` remove "shift" interventions
- Perturb-seq finetuned
  - `cdn_finetuned-seen.ckpt`  trained on all cell lines
    **recommended for benchmarking on genetic perturbation datasets**
  - `cdn_finetuned-unseen-no_hepg2.ckpt` hold out HepG2
  - `cdn_finetuned-unseen-no_jurkat.ckpt` hold out Jurkat
  - `cdn_finetuned-unseen-no_k562.ckpt` hold out K562
  - `cdn_finetuned-unseen-no_rpe1.ckpt` hold out RPE1
- Base SEA weights for training CDN
  - `sea_fci_corr.ckpt`

## Datasets

You may download our datasets [here](https://figshare.com/articles/dataset/Single_cell_evaluation_datasets/29215766?file=55059587).
The unzipped folder should be placed under `data`, which will be referenced by splits files:
- `data/test_240.csv` Synthetic testing datasets
- `data/perturbseq.csv` Perturb-seq finetuning and testing datasets for *seen* cell line
- `data/perturbseq_{cell_line}.csv` Perturb-seq finetuning and testing datasets for *unseen* cell line splits
- `data/sciplex.csv` Sci-Plex testing datasets (*unseen* cell line and intervention type)

For synthetic datasets, the splits CSVs are formatted as follows.
- `fp_data` path to the *interventional* dataset.
  The corresponding *observational* dataset can be found by replacing
  `data_interv` by `data` (done dynamically in our codebase).
- `fp_graph` path to the ground truth synthetic
  graph, which is stored as a numpy array
- `fp_regime` path to the CSV of ground truth interventions (labels)
- `split` is train, val or test

For biological datasets, the splits CSVs are formatted as follows.
- `perturbation` or `pert` string that denotes the ENSG identifier of the perturbation target
- `name` string that denotes the raw perturbation target (e.g. gene name)
- `cluster` (where applicable) integer that denotes the k-means cluster of the log-fold
  change in gene expression, used for data splitting purposes
- `split` is train, val or test
- `fp_data` path to the *interventional* dataset.
  The corresponding *observational* dataset can be found by replacing
  `data_interv` by `data` (done dynamically in our codebase).
- `label` index of the true target within the data.

## Results

TBD

