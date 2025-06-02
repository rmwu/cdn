"""
Main training file. Call via train.sh
"""
import os
import sys
import yaml
import time
import random
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from args import parse_args
from data import DataModule
from model import load_model, get_model_cls
from utils import printt, get_suffix, save_pickle



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    printt("Starting...")
    with open("data/goodluck.txt") as f:
        for line in f:
            print(line, end="")
    np.seterr(invalid="ignore")

    args = parse_args()
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.set_float32_matmul_precision("medium")

    # save args (do not pickle object for readability)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # NOTE: does not capture change to save_path args after wandb init
    with open(args.args_file, "w+") as f:
        yaml.dump(args.__dict__, f)

    # data loaders
    data = DataModule(args)
    printt("Finished loading raw data.")

    # setup
    set_seed(args.seed)
    if os.path.exists(args.checkpoint_path):
        model = get_model_cls(args).load_from_checkpoint(args.checkpoint_path,
                                                         strict=False,
                                                         map_location="cpu")
    else:
        model = load_model(args)
    printt("Finished loading model.")

    # logger
    if args.debug:
        wandb_logger = None
    else:
        name = str(time.time())
        wandb_logger = WandbLogger(project=args.run_name,
                                   name=name)
        wandb_logger.watch(model)  # gradients
        args.save_path = os.path.join(args.save_path, name)

    # train loop
    mode = "max"
    for keyword in ["loss"]:
        if keyword in args.metric:
            mode = "min"

    checkpoint_kwargs = {
        "save_top_k": 1,
        "monitor": args.metric,
        "mode": mode,
        "filename": get_suffix(args.metric),
        "dirpath": args.save_path,
        "save_last": True,
    }
    cb_checkpoint = ModelCheckpoint(**checkpoint_kwargs)

    cb_earlystop = EarlyStopping(
            monitor=args.metric,
            patience=args.patience,
            mode=mode,
    )
    cb_lr = LearningRateMonitor(
            logging_interval="step"
    )
    callbacks=[
            RichProgressBar(),
            cb_checkpoint,
            cb_earlystop,
            #cb_lr
    ]
    if args.no_tqdm:
        callbacks[0].disable()

    device_ids = [args.gpu + i for i in range(args.num_gpu)]

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "min_epochs": args.min_epochs,
        "accumulate_grad_batches": args.accumulate_batches,
        "gradient_clip_val": 1.,
        # evaluate more frequently
        "limit_train_batches": 200,
        "limit_val_batches": 50,
        # logging and saving
        "callbacks": callbacks,
        "log_every_n_steps": args.log_frequency,
        "fast_dev_run": args.debug,
        "logger": wandb_logger,
        # GPU utilization
        "devices": device_ids,
        "accelerator": "gpu",
        #"strategy": "ddp"
        #"precision": "16-mixed",  # doesn't work well with gies?
    }

    trainer = pl.Trainer(**trainer_kwargs)
    printt("Initialized trainer.")

    # if applicable, restore full training
    fit_kwargs = {}
    #if os.path.exists(args.checkpoint_path):
    #    fit_kwargs["ckpt_path"] = args.checkpoint_path
    trainer.fit(model, data, **fit_kwargs)

    print(cb_checkpoint.best_model_path)
    printt("All done. Exiting.")


if __name__ == "__main__":
    main()

