"""
Finetune per dataset

>>> NOTE: I haven't figured this out quite yet
"""
import os
import sys
import yaml
import time
import random
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr

from tqdm import tqdm
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

from args import parse_args
from data import get_finetune_datasets
from model import get_model_cls
from utils import printt, get_suffix, save_pickle, parse_results


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

    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # setup
    set_seed(args.seed)

    # data loaders
    data = get_finetune_datasets(args)
    printt("Finished loading raw data.")
    printt("Saving to", args.results_file)

    # for every single dataset, finetune separately
    all_results = []
    for train_loader, test_loader in tqdm(data, ncols=40):
        # fml why is this necessary to suppress "stupid user you're missing
        # parameters in your state_dict" "stupid user your model looks like"
        with open("dummy", "w") as f:
            with redirect_stdout(f), redirect_stderr(f):
                model = get_model_cls(args).load_from_checkpoint(
                        args.checkpoint_path,
                        strict=False,
                        map_location="cpu")
            # profile both finetuning and inference time
            start = time.time()
            model = fit(model, train_loader, args)
            results = test(model, test_loader, args)
            end = time.time()

        for item in results:
            item["time"] = end - start
        all_results.extend(results)

    # post-process results for jupyter analysis
    results = parse_results(all_results)

    save_pickle(args.results_file, results)
    printt("All done. Exiting.")


def fit(model, data, args):
    """
    model (pl.LightningModule)
    data (DataLoader)
    """
    # need to explicitly create and disable
    callbacks = [RichProgressBar()]
    callbacks[0].disable()

    trainer_kwargs = {
        "max_epochs": args.finetune_epochs,
        "gradient_clip_val": 1.,
        "callbacks": callbacks,
        "fast_dev_run": args.debug,
        "logger": None,
        "devices": [args.gpu],
        "accelerator": "gpu" if args.gpu >= 0 else "cpu",
    }
    if args.gpu >= 0:
        trainer_kwargs["devices"] = [args.gpu]
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(model, data)
    return model


def test(model, data, args):
    """
    model (pl.LightningModule)
    data (DataLoader)
    """
    # need to explicitly create and disable
    callbacks = [RichProgressBar()]
    callbacks[0].disable()

    kwargs = {
        "callbacks": callbacks,
        "accelerator": "gpu" if args.gpu >= 0 else "cpu",
    }
    if args.gpu >= 0:
        kwargs["devices"] = [args.gpu]
    tester = pl.Trainer(num_nodes=1,
                        enable_checkpointing=False,
                        logger=False,
                        **kwargs)
    model.eval()
    results = tester.predict(model, data)
    return results


if __name__ == "__main__":
    main()

