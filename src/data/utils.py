import os
from contextlib import redirect_stdout

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import default_collate
from torch.nn.utils.rnn import pad_sequence

from causallearn.search.ConstraintBased.FCI import fci


edge_map_fci = {
    # 0 reserved for padding
    (2, 2): 1,  # (0, 0) no edge but not padded
    (1, 3): 2,  # (-1, 1)
    (3, 1): 3,  # (1, -1)
    (3, 3): 4,  # (1, 1)
    (3, 4): 5,  # (1, 2)
    (4, 3): 6,  # (2, 1)
    (4, 4): 7   # (2, 2)
}


edge_map_gies = {
    (0, 0): 0,  # reserved for padding
    (1, 1): 1,  # (0, 0) no edge but not padded
    (2, 1): 2,  # (1, 0)
    (1, 2): 3,  # (0, 1)
    (2, 2): 4,  # (1, 1) not DAG but exists (?)
}


def convert_to_item(feats, graphs, orders):
    """
        Convert to batchable item format

        NOTE >>> this time we don't need to pass dataset obj but keeping for
        now until I finalize the API

        NOTE >>> this time feats contains BOTH first order and second order
        as a tuple
    """
    # NOTE inverse is NOT the global ordering, which is given by unique,
    # but it is consistent within this single item.
    # unique is (B*N, 2), inverse is (B*N)
    unique, inverse = torch.unique(orders, dim=0,
            return_inverse=True)

    # get true edges
    item = {
        "input": graphs,
        "feats_2d": torch.from_numpy(feats[0]).float(),  # (N, N)
        "feats_1d": torch.from_numpy(feats[1]).float(),  # (N,)
        "index": inverse.reshape(len(graphs), -1),  # shape is (T, k*k)
        "unique": unique + 1,  # (num unique, 2)  NEED TO 1 INDEX FOR PADDING
    }
    return item


def convert_result_to_lg(g, edge_map):
    edge_attr = []
    for i in range(len(g)):
        for j in range(len(g)):
            # eliminate diagonal from the source
            if i == j:
                continue
            ij = g[i,j]
            ji = g[j,i]
            edge_attr.append(edge_map[(ij, ji)])
    return edge_attr


def convert_to_graphs(results, dataset):
    """
        Convert to PyG line graphs
    """
    graphs = []
    orders = []
    for G, order in results:
        if dataset.algorithm == "fci":
            # G includes {-1, 0, 1, 2} for FCI
            # to include padding, we should map this to {1, 2, 3, 4}
            graphs.append(convert_result_to_lg(G + 2, edge_map_fci))
        elif dataset.algorithm in ["ges", "grasp"]:
            # G includes {-1, 0, 1} for GES and GRaSP
            graphs.append(convert_result_to_lg(G, edge_map_ges))
        else:
            # G includes {0, 1}
            # to include padding, we should map this to {1, 2}
            graphs.append(convert_result_to_lg(G + 1, edge_map_gies))
        orders.append(torch.cartesian_prod(order, order))
    if len(graphs) == 0:
        return None, None
    # (T, k*k)
    graphs = torch.tensor(graphs, dtype=torch.long)
    orders = torch.cat(orders, dim=0)
    # remove diagonal
    orders = orders[orders[:,0] != orders[:,1]]
    return graphs, orders


def run_fci(batch):
    try:
        with open("dummy", "w") as f:
            with redirect_stdout(f):
                G, edges = fci(batch,
                               independence_test_method="fisherz",
                               alpha=0.05,  # default
                               depth=-1,  # no max, fine if only 5-10 vars
                               max_path_length=-1,  # no max
                               verbose=False,
                               show_progress=False)
    # sometimes, very rarely, FCI fails...
    except:
        return
    return G.graph


def collate(args, batch):
    """
        Overwrite default_collate for jagged tensors
    """
    # initialize new batch
    # and skip invalid items haha
    keys = ["key", "dataset_id", "reg_idx", "label", "time"]
    for key in ["input", "index", "feats_2d", "feats_1d", "unique"]:
        keys.append(f"{key}_obs")
        keys.append(f"{key}_int")
    batch = {key:[item[key] for item in batch if key in item] for key in keys}
    new_batch = {}
    for key, val in batch.items():
        # preserve as is
        if key in ["key", "dataset_id", "reg_idx"]:
            new_batch[key] = val
        elif not torch.is_tensor(val[0]) or val[0].ndim == 0:
            new_batch[key] = default_collate(val)
        elif "feats_2d" in key:
            # [N, N] so require 2D padding
            max_nodes = max([len(v) for v in val])
            for i, v in enumerate(val):
                pad = max_nodes - len(v)
                if pad > 0:
                    val[i] = F.pad(v, (0, pad, 0, pad))
            new_batch[key] = torch.stack(val, dim=0)
        else:
            new_batch[f"{key}_len"] = torch.tensor([len(v) for v in val])
            # dimension = 1 is now time
            # each of these should be (length, )
            padded = pad_sequence(val, batch_first=True)
            new_batch[key] = padded
    # >>> ablation
    #new_batch["input_obs"] = torch.zeros_like(new_batch["input_obs"])
    #new_batch["input_int"] = torch.zeros_like(new_batch["input_int"])
    #new_batch["feats_2d_obs"] = torch.ones_like(new_batch["feats_2d_obs"])
    #new_batch["feats_2d_int"] = torch.ones_like(new_batch["feats_2d_int"])
    # <<<
    return new_batch


def get_mask(lens, max_len=None):
    # mask where 0 is padding and 1 is token
    if max_len is None:
        max_len = lens.max()
    mask = torch.arange(max_len)[None, :] < lens[:, None]
    return mask

