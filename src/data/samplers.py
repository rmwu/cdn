"""
    Variety of sampling mechanisms
"""

import time
import math
import itertools
from itertools import accumulate, repeat, chain
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from sklearn.covariance import LedoitWolf, GraphicalLassoCV

from .utils import collate
from .utils import convert_to_graphs, convert_to_item
from model import get_model_cls


class DatasetSampler:
    def __init__(self, args, dataset, visit_counts=None, **kwargs):
        self.args = args
        self.dataset = dataset
        # allow variable sizes since graphs are heterogenous
        self.num_vars = dataset.shape[1]
        self.reset()  # initialize variables
        if visit_counts is not None:
            self.visit_counts = visit_counts

    def compute_scores(self):
        raise Exception("implement me")

    def sample_nodes(self, num_nodes):
        """
            @param (np.ndarray) scores  NxN
        """
        # these scores are not NaN [20250106 10:19]
        scores = self.compute_scores()
        joint_scores = scores * 1 / np.sqrt(1 + self.visit_counts)
        sampled = []
        for i in range(num_nodes):
            sampled_arr = np.array(sampled, dtype=int)
            if len(sampled) == 0:
                p = joint_scores.sum(0)
            else:
                # remove sampled from consideration
                p = joint_scores[sampled_arr]
                p[:,sampled_arr] = 0
                p = p.sum(0)
            p = p / p.sum()  # normalize probabilities to 1
            if np.any(np.isnan(p)):
                p = np.ones_like(p) / len(p)
                print("Here", sampled)
            v = np.random.choice(len(p), (1,), p=p).item()
            sampled.append(v)
        # update visit counts
        update_counts = np.zeros((self.num_vars, self.num_vars))
        sampled = np.array(sampled)
        edges = cartesian_prod([sampled, sampled])
        update_counts[edges[:,0], edges[:,1]] = 1
        self.visit_counts += update_counts
        assert len(sampled) == len(set(sampled)) == num_nodes
        return sampled

    def sample_batches(self):
        raise Exception("implement me")

    def callback(self, *args, **kwargs):
        # optionally called after sampling every batch
        return

    def reset(self):
        self.visit_counts = np.zeros((self.num_vars, self.num_vars))
        # apparently catting to the empty tensor works like a charm
        self.graphs = torch.empty((0,), dtype=torch.long)
        self.orders = torch.empty((0,), dtype=torch.long)


class ObservationalSampler(DatasetSampler):
    """
        Sampler for observational data
    """
    def sample_batches(self, num_batches, batch_size, num_vars_batch):
        """
            Called by Datasets
        """
        batches = []
        batch_size = min(len(self.dataset), batch_size)
        if batch_size == 0:
            print(self.dataset)
            exit(0)
        for i in range(num_batches):
            # sample dataset points
            idxs = np.random.choice(len(self.dataset), (batch_size,),
                    replace=False)
            # sample nodes
            nodes = self.sample_nodes(num_vars_batch)
            # broadcast axis
            batch = self.dataset[idxs[:, np.newaxis], nodes]
            batches.append((batch, nodes))
            self.callback([(batch, nodes)])  # for learned sampler
        # compute global features based on last batch
        # we compute both 1st and 2nd order features
        feats_data = self.dataset[idxs].T
        feats_2d = compute_features(feats_data)
        feats_1d = compute_features(feats_data, order=1)
        feats = (feats_2d, feats_1d)
        # <<<
        return batches, feats


class InterventionalSampler(DatasetSampler):
    """
        Sampler for interventional data
        NOTE we never use this sampler so it's unmodified for the new API
    """
    def sample_batches(self, num_batches, batch_size, num_vars_batch):
        """
            Called by Datasets
        """
        batches = []
        # +1 for observational
        points_per_env = batch_size // (num_vars_batch + 1)
        for i in range(num_batches):
            # sample nodes
            nodes = self.sample_nodes(num_vars_batch)
            # sample regimes that impact those points
            reg_idx = []
            for v in nodes:
                reg_idx.extend(self.dataset.node_to_regime[v])
            # for Sachs
            if len(reg_idx) < num_vars_batch:
                reg_idx = sorted(set(reg_idx))
                for _ in range(num_vars_batch - len(reg_idx)):
                    reg_idx.append(0)  # observational
            else:
                reg_idx = np.random.choice(sorted(set(reg_idx)),
                                           num_vars_batch,
                                           replace=False)
            # sample examples from each regime
            batch = []
            for reg in reg_idx:
                # sample dataset points
                idxs = np.random.choice(self.dataset.regimes[reg],
                                        points_per_env,
                                        replace=False)[:, np.newaxis]
            batch.append(self.dataset.data[idxs, nodes])
            # add observational
            idxs = np.random.choice(self.dataset.regimes[0],
                                    points_per_env,
                                    replace=False)[:, np.newaxis]
            batch.append(self.dataset.data[idxs, nodes])
            batch = np.stack(batch, axis=0)

            # re-number nodes in regimes to local batch idx
            # and remove intervened nodes outside of sampled set
            node_renumber = {node:i for i, node in enumerate(nodes)}
            regimes = [self.dataset.idx_to_regime[reg] for reg in reg_idx]
            regimes = [[node_renumber.get(x) for x in reg] for reg in regimes]
            regimes = [[x for x in reg if x is not None] for reg in regimes]
            regimes.append([])

            batches.append((batch, nodes, regimes))
            self.callback([(batch, nodes, regimes)])  # for learned sampler

        idxs = np.random.choice(len(self.dataset), (batch_size,),
                replace=False)
        feats = compute_features(self.dataset.data[idxs].T)

        return batches, feats


class RandomSampler(DatasetSampler):
    def compute_scores(self):
        scores = np.ones((self.num_vars, self.num_vars))
        scores = scores * (1 - np.eye(len(scores)))  # zero out diagonal
        return scores


class CorrelationSampler(DatasetSampler):
    def compute_scores(self):
        batch_size = min(len(self.dataset), self.args.fci_batch_size)
        idxs = np.random.choice(len(self.dataset),
                batch_size, replace=False)
        batch_score = self.dataset[idxs]
        score = compute_features(batch_score.T)
        score = score * (1 - np.eye(len(score)))  # zero out diagonal
        score = np.abs(score)  # make positive
        return score


def compute_features(x, order=2):
    """
    x: (num_vars, num_samples)
    """
    if order == 2:
        return _second_order_feats(x)
    elif order == 1:
        return _first_order_feats(x)


def _first_order_feats(x):
    """
    x: (num_vars, num_samples)
    """
    feats = [x.mean(axis=1),
             np.var(x, axis=1)]
    feats = np.stack(feats).T
    feats[np.isnan(feats)] = 0
    return feats


def _second_order_feats(x):
    """
    Implementation depends on matrix size
    x: (num_vars, num_samples)
    """
    #return np.cov(x)  # ugh in the service of nice proofs
    # >>>
    score = np.corrcoef(x)
    score[np.isnan(score)] = 0
    return score
    # <<<
    #return np.linalg.pinv(np.cov(x), rcond=1e-10)
    #model = GraphicalLassoCV()
    #model.fit(x.T)
    #invcov = model.precision_
    #return invcov


def cartesian_prod(arrays):
    """
        https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points/49445693#49445693
    """
    la = len(arrays)
    L = *map(len, arrays), la
    dtype = np.result_type(*arrays)
    arr = np.empty(L, dtype=dtype)
    arrs = *accumulate(chain((arr,), repeat(0, la-1)), np.ndarray.__getitem__),
    idx = slice(None), *itertools.repeat(None, la-1)
    for i in range(la-1, 0, -1):
        arrs[i][..., i] = arrays[i][idx[:la-i]]
        arrs[i-1][1:] = arrs[i]
    arr[..., 0] = arrays[0][idx]
    return arr.reshape(-1, la)

