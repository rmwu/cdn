"""
Main model (h_diff)
"""

import os
import math
import time
import warnings

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

from .axial import AxialTransformer, AxialTransformerLayer, TopLayer
from .utils import get_params_groups


class JointModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_vars = args.num_vars

        # Transformer on sequence of predicted graphs
        self.encoder = AxialTransformer(args)
        self.top_layer = TopLayer(
            embed_dim=self.args.embed_dim * 2,
            output_dim=3
        )
        # load pretrained checkpoints
        if os.path.exists(args.pretrained_path):
            checkpoint = torch.load(args.pretrained_path, map_location="cpu",
                                    weights_only=False)
            state_dict = self.state_dict()
            for k, v in checkpoint["state_dict"].items():
                if k in state_dict:
                    state_dict[k] = v
            self.load_state_dict(state_dict)
        else:
            warnings.warn(f"Weights not found at {args.pretrained_path}!",
                    UserWarning)
        #self.disable_grads(self.encoder)

        # new parameters, for first order statistics
        self.encoder_1d = TopLayer(self.args.embed_dim,
                                   self.args.embed_dim,
                                   input_dim=2)

        # new parameters, specific to target predictor
        axial_kwargs = {
            "embed_dim": self.args.embed_dim,
            "ffn_embed_dim": 256,
            "n_heads": self.args.n_heads,
            "dropout": self.args.dropout,
            "max_tokens": self.args.max_length,
            "rope_cols": False,
            "rope_rows": False,
        }

        # NOTE >1 layers doesn't work as well so we just hardcode 1 layer
        if self.args.target_transformer_num_layers == 1:
            self.axial_N_N_d2 = AxialTransformerLayer(**axial_kwargs)
        else:
            layers = [AxialTransformerLayer(**axial_kwargs) for _ in
                      range(self.args.target_transformer_num_layers)]
            self.axial_N_N_d2 = AxialTransformerStack(layers)

        # this is called after collapsing over incoming edges
        self.linear_out = nn.Linear(self.args.embed_dim, 1)

        self.loss = nn.BCEWithLogitsLoss()

        # validation meters
        self.auroc = BinaryAUROC()
        self.auprc = BinaryAveragePrecision()
        self.acc = BinaryAccuracy()

        self.save_hyperparameters()

    #def disable_grads(self, module):
    #    """
    #    Disable gradients for (pretrained) module
    #    """
    #    for param in module.parameters():
    #        param.requires_grad = False

    def training_step(self, batch, batch_idx):
        # cuda oom
        try:
            results = self.encode_batch(batch, reduce=True)
            losses = self.compute_losses(results, batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        for k, v in losses.items():
            if not torch.is_tensor(v) or v.numel() == 1:
                if type(v) is not float:
                    v = v.item()
                self.log(f"Train/{k}", v,
                    batch_size=len(batch["label"]), sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        # cuda oom
        try:
            results = self.encode_batch(batch, reduce=False)
            results = self.compute_metrics_per_graph(results, batch,
                    save_preds=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return
        # metrics
        for k, v in results.items():
            if type(v) is list:
                v = np.mean(v)
            self.log(f"Val/{k}", v,
                batch_size=len(batch["label"]), sync_dist=True)

    def forward(self, batch):
        """
        Used on predict_dataloader
        """
        start = time.time()  # keep track of GPU time

        results = self.encode_batch(batch, reduce=False)
        results = self.compute_metrics_per_graph(results, batch,
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
        # each is [B, N, N, d]
        output_obs = self.encoder(batch["input_obs"], batch["index_obs"],
                                  batch["feats_2d_obs"], batch["unique_obs"])
        output_int = self.encoder(batch["input_int"], batch["index_int"],
                                  batch["feats_2d_int"], batch["unique_int"])
        output_obs_raw = output_obs  # save for graph prediction
        # encode and combine first order statistics
        output_1d_obs = self.encoder_1d(batch["feats_1d_obs"])#[..., None])
        output_1d_int = self.encoder_1d(batch["feats_1d_int"])#[..., None])
        output_1d_obs = output_1d_obs[:,:,None]  # expand empty dim=2
        output_1d_int = output_1d_int[:,:,None]  # expand empty dim=2
        # >>>
        output_sub = output_int - output_obs
        output_1d_sub = output_1d_int - output_1d_obs
        # <<<
        output = torch.cat([output_sub, output_1d_sub], dim=2)
        # combine for final prediction. fuse has grad.
        output, label = self.fuse(output, batch["label"], reduce=reduce)

        results = {
            "pred": output,
            "true": label,
            "graph": output_obs_raw
        }
        return results

    def fuse(self, out_B_N_N_2d, label_B_N=None, reduce=True):
        """
        assumptions.
        1) padding is already set to 0
        technically it will be [B, N, N+1, 2d] with first order statistics

        reduce: bool  True to reduce over B, False to preserve B_(variable N)
                      also controls whether we return true labels
        return: B_N  probability of intervention
        """
        mask_B_N_N = (out_B_N_N_2d == 0.)[..., 0]  # B, R, C
        inv_mask_B_N_N = (~mask_B_N_N).type_as(out_B_N_N_2d)[..., None]
        # >>> why is this necessary? isn't this circular?
        #out_B_N_N_2d = out_B_N_N_2d * inv_mask_B_N_N
        # B x R x C x D -> R x C x B x D
        out_N_N_B_2d = out_B_N_N_2d.permute(1, 2, 0, 3)
        out_N_N_B_2d = self.axial_N_N_d2(out_N_N_B_2d, mask_B_N_N)
        # R x C x B x D -> B x R x C x D
        out_B_N_N_2d = out_N_N_B_2d.permute(2, 0, 1, 3)
        # collapse over incoming edges -> dim=2 (columns)
        out_B_N_2d = out_B_N_N_2d.mean(dim=2)
        # project to out
        out_B_N = self.linear_out(out_B_N_2d).squeeze(2)
        mask_B_N = ~mask_B_N_N[..., 0]
        if reduce:
            return out_B_N[mask_B_N], label_B_N[mask_B_N]
        else:
            output_B_list = []
            label_B_list = []
            for i, mask in enumerate(mask_B_N):
                output_B_list.append(out_B_N[i][mask])
                label_B_list.append(label_B_N[i][mask])
            return output_B_list, label_B_list

    def symmetrize(self, output, batch, reduce=True):
        """
            P(i->j) = 1 - P(j->i)
            reduce: bool  True to reduce batch, False to preserve graphs
        """
        # symmetrize output
        # select upper and lower triangular, skip diagonal
        # NOTE: tril is NOT same as triu of transpose
        output = output.permute(0, 3, 1, 2)  # move embed_dim to 2 for triu
        #eps = 1e-5  # ... smh
        #print(output.shape)
        forward_edge = torch.triu(output, 1)
        backward_edge = torch.triu(output.transpose(2, 3), 1)
        forward_edge = forward_edge.permute(0, 2, 3, 1)
        backward_edge = backward_edge.permute(0, 2, 3, 1)
        # select away padding as well as wrong direction
        # backward_mask should == forward_mask
        forward_mask = forward_edge[...,0] != 0.  # remove embed_dim dimension
        backward_mask = backward_edge[...,0] != 0.
        # revert after mask computation
        #print(forward_mask.sum(), backward_mask.sum())
        #forward_edge, backward_edge = forward_edge - eps, backward_edge - eps
        #exit(0)
        if reduce:
            forward_edge = forward_edge[forward_mask]
            backward_edge = backward_edge[backward_mask]
            assert len(forward_edge) == len(backward_edge)
            logits = torch.cat([forward_edge, backward_edge], dim=1)
            edge_pred = self.top_layer(logits)  # (B*T, 2*dim) -> (B*T, 3)
        else:
            edge_pred = []
            for i in range(len(forward_edge)):
                forward_i = forward_edge[i][forward_mask[i]]
                backward_i = backward_edge[i][backward_mask[i]]
                logits = torch.cat([forward_i, backward_i], dim=-1)
                edge_pred.append(self.top_layer(logits))

        # get label corresponding to same entries
        label = batch["graph"]
        forward_label = torch.triu(label, 1)
        backward_label = torch.triu(label.transpose(1, 2), 1)
        if reduce:
            forward_label = forward_label[forward_mask]
            backward_label = backward_label[backward_mask] * 2
            # forward/backward should be mutually exclusive
            joint_label = forward_label + backward_label  # {0, 1, 2}
        else:
            joint_label = []
            for i in range(len(forward_edge)):
                forward_i = forward_label[i][forward_mask[i]]
                backward_i = backward_label[i][backward_mask[i]]
                joint_label.append(torch.cat([forward_i, backward_i]))
        return edge_pred, joint_label

    def compute_losses(self, results, batch):
        losses = {}
        # reduce by graph
        reduce = torch.is_tensor(results["pred"])
        if reduce:
            target_loss = self.loss(results["pred"],
                                    results["true"].float())
        # joint version
        else:
            target_loss = [self.loss(p, t.float()).item() for p, t in
                zip(results["pred"], results["true"])]
            target_loss = np.mean(target_loss)
        losses["loss"] = self.args.target_loss_weight * target_loss
        losses["target_loss"] = target_loss.item()

        if self.args.graph_loss_weight > 0:
            pred_graph, true_graph = self.symmetrize(results["graph"], batch,
                                                     reduce=reduce)
            if reduce:
                graph_loss = F.cross_entropy(pred_graph, true_graph)
            else:
                graph_loss = []
                for p, t in zip(pred_graph, true_graph):
                    halfway = len(t) // 2
                    t = t[:halfway] + t[halfway:] * 2
                    graph_loss.append(F.cross_entropy(p, t).item())
                graph_loss = np.mean(graph_loss)
            losses["loss"] += self.args.graph_loss_weight * graph_loss
            losses["graph_loss"] = graph_loss.item()

        return losses

    def compute_metrics_per_graph(self, results, batch, save_preds=False):
        """
        Metrics on individual graphs from batch
        """
        pred = results["pred"]
        auroc, auprc, acc = [], [], []
        if save_preds:
            pred_list, true_list = [], []
        for i, p in enumerate(pred):
            t = batch["label"][i, :len(p)]
            assert p.shape == t.shape, (p.shape, t.shape)  # well this version = trivially true
            ## add metrics / predictions to list
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

        # training
        if "graph" in batch:
            outputs.update(self.compute_losses(results, batch))
            outputs["loss"] = outputs["loss"].item()
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

    #def on_save_checkpoint(self, checkpoint):
    #    # pop the pretrained weights here
    #    to_delete = []
    #    for k in checkpoint["state_dict"]:
    #        if "encoder" in k:
    #            to_delete.append(k)
    #        elif "top_layer" in k:
    #            to_delete.append(k)
    #    for k in to_delete:
    #        del checkpoint["state_dict"][k]


class AxialTransformerStack(nn.Module):
    """
        Stack of layers
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask):
        """
        x, mask  input to AxialTransformerLayer
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x

