"""
Disable grads for encoder and enable grads only for the downstream MLP.
"""

import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.classification import BinaryAccuracy

from .axial import AxialTransformer, TopLayer
from .utils import get_params_groups


class IndependentModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_vars = args.num_vars

        # Transformer on sequence of predicted graphs
        self.encoder = AxialTransformer(args)
        # we want to toss this layer actually
        # but keeping just in case
        #self.top_layer = TopLayer(
        #    embed_dim=self.args.embed_dim * 2,
        #    output_dim=3
        #)
        # load pretrained checkpoints
        if os.path.exists(args.pretrained_path):
            checkpoint = torch.load(args.pretrained_path, map_location="cpu")
            state_dict = self.state_dict()
            for k, v in checkpoint["state_dict"].items():
                if k in state_dict:
                    state_dict[k] = v
            self.load_state_dict(state_dict)
        else:
            raise Exception(f"Weights not found at {args.pretrained_path}!")
        self.disable_grads(self.encoder)

        # new parameters, specific to target predictor
        self.mlp_out = TopLayer(
            embed_dim=self.args.embed_dim * 2,
            output_dim=self.args.embed_dim
        )
        # this is called after collapsing over incoming edges
        self.linear_out = nn.Linear(self.args.embed_dim, 1)

        self.loss = nn.BCEWithLogitsLoss()

        # validation meters
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.acc = BinaryAccuracy()

        self.save_hyperparameters()

    def disable_grads(self, module):
        """
        Disable gradients for (pretrained) module
        """
        for param in module.parameters():
            param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # cuda oom
        try:
            pred, true = self.encode_batch(batch, reduce=True)
            losses = self.compute_losses(pred, true)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                self.log(f"Train/{k}", v.item(),
                    batch_size=len(batch["label"]), sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # cuda oom
        try:
            pred, _ = self.encode_batch(batch, reduce=False)
            results = self.compute_metrics_per_graph(pred, batch,
                    save_preds=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        # metrics
        for k, v in results.items():
            self.log(f"Val/{k}", np.mean(v),
                batch_size=len(batch["label"]), sync_dist=True)

    def forward(self, batch):
        """
        Used on predict_dataloader
        """
        start = time.time()  # keep track of GPU time

        pred, _ = self.encode_batch(batch, reduce=False)
        results = self.compute_metrics_per_graph(pred, batch,
                save_preds=True)
        # need to save these
        for key in ["key", "dataset_id", "reg_idx"]:
            results[key] = batch[key]
        results["label"] = batch["label"].cpu().tolist()

        end = time.time()  # keep track of GPU time
        # run with batch_size=1 for accurate timing
        results["time"] = batch["time"].item() + (end - start)
        return results

    def encode_batch(self, batch, reduce):
        # NOTE encoder does NOT have grad
        output_obs = self.encoder(batch["input_obs"], batch["index_obs"],
                                  batch["feats_2d_obs"], batch["unique_obs"])
        output_int = self.encoder(batch["input_int"], batch["index_int"],
                                  batch["feats_2d_int"], batch["unique_int"])
        # combine for final prediction
        output = torch.cat([output_obs, output_int], dim=3)
        output, label = self.symmetrize(output, batch["label"], reduce=reduce)
        return output, label

    def symmetrize(self, output_B_N_N_2d, label_B_N=None, reduce=False):
        """
        # two assumptions.
        # 1) padding is already set to 0
        # 2) N_N padding is square

        reduce: bool  True to reduce over B, False to preserve B_(variable N)
                      also controls whether we return true labels
        return: B_N  probability of intervention
        """
        mask_B_N = (output_B_N_N_2d != 0.)[..., 0, 0]
        output_B_N_N_d = self.mlp_out(output_B_N_N_2d)
        # collapse over incoming edges -> dim=2 (columns)
        output_B_N_d = output_B_N_N_d.mean(dim=2)
        # project to output
        output_B_N = self.linear_out(output_B_N_d).squeeze(2)
        if reduce:
            return output_B_N[mask_B_N], label_B_N[mask_B_N]
        else:
            output_B_list = []
            label_B_list = []
            for i, mask in enumerate(mask_B_N):
                output_B_list.append(output_B_N[i][mask])
                label_B_list.append(label_B_N[i][mask])
            return output_B_list, label_B_list

    def compute_losses(self, pred, true):
        losses = {}
        losses["loss"] = self.loss(pred, true.float())
        return losses

    def compute_metrics_per_graph(self, pred, batch, save_preds=False):
        """
        Metrics on individual graphs from batch
        """
        losses, auroc, auprc, acc = [], [], [], []
        if save_preds:
            pred_list, true_list = [], []
        for i, p in enumerate(pred):
            t = batch["label"][i, :len(p)]
            assert p.shape == t.shape  # well this version = trivially true
            ## add metrics / predictions to list
            losses.append(self.loss(p, t.float()).item())
            p = torch.sigmoid(p).cpu()
            t = t.cpu()
            auroc.append(self.auroc(p, t).item())
            acc.append(self.acc(p, t).item())
            if t.sum() > 0:  # non-obs, always predict
                auprc.append(self.auprc(p, t).item())
            elif save_preds:  # we need to make sure these align
                auprc.append(-1)
            # otherwise ok to skip for logging
            if save_preds:
                pred_list.append(p.tolist())
                true_list.append(t.tolist())

        outputs = {}
        outputs["loss"] = losses
        outputs["auroc"] = auroc
        outputs["auprc"] = auprc
        outputs["acc"] = acc
        if save_preds:
            outputs["pred"] = pred_list
            outputs["true"] = true_list
        return outputs

    def configure_optimizers(self):
        param_groups = get_params_groups(self, self.args)
        optimizer = AdamW(param_groups)
        return [optimizer]

    def on_save_checkpoint(self, checkpoint):
        # pop the pretrained weights here
        to_delete = []
        for k in checkpoint["state_dict"]:
            if "encoder" in k:
                to_delete.append(k)
            elif "top_layer" in k:
                to_delete.append(k)
        for k in to_delete:
            del checkpoint["state_dict"][k]

