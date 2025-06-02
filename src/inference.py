"""
Main inference file. Call via inference.sh
"""
import os
import sys
import yaml
import random
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl

from args import parse_args
from data import InferenceDataModule, BaselineDataModule
from model import get_model_cls
from utils import printt, save_pickle, parse_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    printt("Starting...")
    with open("data/goodluck2.txt") as f:
        for line in f:
            print(line, end="")
    np.seterr(invalid="ignore")

    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    #from utils import read_pickle
    #results = read_pickle("tmp.pkl")
    #results = parse_results(results)
    #save_pickle(args.results_file, results)
    #exit(0)

    # data loaders
    if args.model == "baseline":
        data = BaselineDataModule(args)
    else:
        data = InferenceDataModule(args)
    printt("Finished loading raw data.")

    # setup
    set_seed(args.seed)
    if not os.path.exists(args.checkpoint_path):
        raise Exception(args.checkpoint, "does not exist")

    model = get_model_cls(args).load_from_checkpoint(args.checkpoint_path,
                                                     strict=False,
                                                     map_location="cpu")
    model.eval()
    printt("Finished loading model.")

    # inference
    kwargs = {
        "accelerator": "gpu" if args.gpu >= 0 else "cpu"
    }
    if args.gpu >= 0:
        kwargs["devices"] = [args.gpu]
    tester = pl.Trainer(num_nodes=1,
                        enable_checkpointing=False,
                        logger=False,
                        **kwargs)

    results = tester.predict(model, data)

    # post-process results for jupyter analysis
    results = parse_results(results)
    save_pickle(args.results_file, results)
    printt("All done. Exiting.")


if __name__ == "__main__":
    main()

