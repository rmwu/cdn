"""
Dataset objects

-   InterventionalDataset are individual
    "datasets" with a single graph / set of interventions

-   MetaDataset descendents are datasets of datasets which
    sample individual InterventionalDataset objects
"""

import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from . import samplers
from .utils import run_fci, run_gies
from .utils import convert_to_graphs, convert_to_item
from utils import read_csv

from sklearn.preprocessing import StandardScaler


# ======== Start of individual datasets ========


class PairDataset(Dataset):
    def __init__(self, fp_data, label):
        """
        Single pair of datasets. Typical InterventionalDataset objects contain
        multiple pairs. This is just a single pair.
        """
        super().__init__()
        self.key = fp_data.split("/")[-2]
        # trim .npy and "data" from "data-ENSG00000001497.npy"
        self.dataset_id = fp_data.split(".")[0].split("-")[1]
        # read raw data
        self.data_obs = np.load(fp_data.replace("data_interv-", "data-"))
        self.data_int = np.load(fp_data)
        #print(self.data_obs.shape, fp_data, "obs")
        #print(self.data_int.shape, fp_data, "int")
        # >>> uncertainty quantification
        np.random.seed(4)
        num_pert = min(len(self.data_int), max(50, int(len(self.data_int) * 0.8)))
        idx = np.random.choice(len(self.data_int), num_pert, replace=False)
        self.data_int = self.data_int[idx]
        # <<<
        self.num_vars = self.data_obs.shape[1]
        assert self.num_vars == self.data_int.shape[1]
        self.algorithm = "fci"

        self.time = 0  # placeholder for later

        # single label
        self.label = torch.zeros(self.num_vars, dtype=torch.long)
        self.label[label] = 1

        # single regime
        self.num_regimes = 1

    def __len__(self):
        return 1

    def get_obs(self):
        item = {
            "data": self.data_obs,
        }
        return item

    def __getitem__(self, idx):
        item = {
            "data": self.data_int,
            "label": self.label
        }
        return item


class InterventionalDataset(Dataset):
    def __init__(self, fp_data, fp_graph, fp_regime,
                 min_interv=None, max_interv=None,
                 use_graph=False):
        """
        min_interv (int)  inclusive, minimum number of intervention targets
        max_interv (int)  inclusive, maximum number of intervention targets
        """
        super().__init__()
        self.key = fp_graph.split("/")[-2]
        # trim DAG and .npy
        self.dataset_id = int(fp_graph.split("/")[-1][3:-4])
        # read raw data
        self.data = np.load(fp_data)
        # >>> standardize
        #scaler = StandardScaler()
        #self.data = scaler.fit_transform(self.data)
        # <<<
        self.num_vars = self.data.shape[1]
        self.algorithm = "fci"

        # used for joint finetuning
        self.use_graph = use_graph
        if self.use_graph:
            self.graph = torch.from_numpy(np.load(fp_graph)).long()

        self.time = 0  # placeholder for later

        # read intervention.csv (intervened nodes)
        with open(fp_regime) as f:
            # if >1 node intervened, formatted as a list
            lines = [line.strip() for line in f.readlines()]
        regimes = [tuple(sorted(int(x) for x in line.split(",")))
                if len(line) > 0 else () for line in lines]
        # subset of data based on intervention target count, if applicable
        if min_interv is not None or max_interv is not None:
            if min_interv is None:
                min_interv = 0
            if max_interv is None:
                max_interv = float("inf")
            idx_to_keep = [i for i, interv in enumerate(regimes) if
                           (len(interv) == 0) or  # cannot discard obs
                           (min_interv <= len(interv) <= max_interv)]
            regimes = [regimes[i] for i in idx_to_keep]
            self.data = self.data[idx_to_keep]
        assert len(regimes) == len(self.data)

        # get unique and map to nodes
        unique_regimes = sorted(set(regimes))  # 0 is obs because () first
        self.idx_to_regime = {i: reg for i, reg in enumerate(unique_regimes)}
        self.regime_to_idx = {reg: i for i, reg in enumerate(unique_regimes)}
        self.num_regimes = len(self.idx_to_regime)
        # convert to regime label tensor
        self.idx_to_label = torch.zeros(self.num_regimes, self.num_vars,
                                        dtype=torch.long)
        for idx, regime in self.idx_to_regime.items():
            for node in regime:
                self.idx_to_label[idx, node] = 1

        # map regimes to dataset
        self.regimes = defaultdict(list)
        for i, reg in enumerate(regimes):
            self.regimes[self.regime_to_idx[reg]].append(i)
        self.regimes = {reg: np.array(idx, dtype=int) for reg, idx in
                self.regimes.items()}  # convert to np.ndarray
        self.idx_to_dataset = {}
        # this will be returned as the "Dataset" object
        for i, idxs in self.regimes.items():
            self.idx_to_dataset[i] = self.data[idxs]

    def __len__(self):
        return self.num_regimes

    def get_obs(self):
        return self[0]

    def __getitem__(self, idx):
        data = self.idx_to_dataset[idx]
        label = self.idx_to_label[idx]
        item = {
            "data": data,
            "label": label
        }
        if self.use_graph:
            item["graph"] = self.graph
        return item


# ======== Start of meta datasets ========


class MetaDataset(Dataset):
    """
        Dataset of datasets

        splits_to_load (bool)  only load a subset of splits
        line_to_load (int)  only load a single line in data_to_load

        min_interv (int)  inclusive, minimum number of intervention targets
        max_interv (int)  inclusive, maximum number of intervention targets
    """
    def __init__(self, data_file, args,
                 splits_to_load=None,
                 line_to_load=None,
                 min_interv=None, max_interv=None,
                 use_tqdm=True):
        super().__init__()
        # read raw data
        self.args = args
        self.data_file = data_file
        self.splits = defaultdict(list)
        self.use_graph = args.graph_loss_weight > 0

        # create individual Dataset objects and track how many regimes each has
        data_to_load = []
        for i, item in enumerate(read_csv(self.data_file)):
            split = item["split"]
            # ha ha
            if args.is_singlecell and split == "train":
                if i % 10 == 0:
                    split = "val"
            # >>> for kinase training on Perturb-seq
            #if args.is_singlecell and split == "test":
            #    split = "train"
            # <<<
            if splits_to_load is not None and split not in splits_to_load:
                continue
            self.splits[split].append(len(data_to_load))
            data_to_load.append(item)
            if args.debug and len(data_to_load) > 100:
                break
        # select line (for finetuning application)
        if line_to_load is not None:
            data_to_load = [data_to_load[line_to_load]]

        # load now
        self.data = self._load_raw_data(data_to_load,
                                        args.is_singlecell,
                                        n_cpu=args.num_io_workers)
        # (for length computation)
        self.idx_to_data_regime = []
        for data_idx, dataset in enumerate(self.data):
            for reg_idx in range(dataset.num_regimes):
                self.idx_to_data_regime.append((data_idx, reg_idx))

        # initialize per-class
        self.sampler_classes = None
        self._run_alg = get_run_alg(args.algorithm)

    def _load_raw_data(self, data_to_load, is_real=False,
            min_interv=1, max_interv=3, n_cpu=0):
        """
        Option to multiprocess data loading in case i/o bandwidth > process
        speed (older machines)
        """
        if n_cpu == 0:
            return [_read((item, is_real, min_interv, max_interv,
                    self.use_graph))
                    for item in tqdm(data_to_load, ncols=40)]

        # set up pool and assign
        configs = [(item, is_real, min_interv, max_interv, self.use_graph) for item in data_to_load]
        with Pool(n_cpu) as pool:
            data = pool.map(_read, configs)

        return data

    def _sample_batches(self, dataset, num_batches):
        # this must be initialized per-class
        if self.sampler_classes is None:
            raise Exception("MetaDataset did not initialize sampler_classes")
        # sample batches per sampler
        kwargs = {
            "num_batches": num_batches // len(self.sampler_classes),
            "batch_size": self.args.fci_batch_size,
            "num_vars_batch": self.args.fci_vars,
        }
        for i, create_sampler in enumerate(self.sampler_classes):
            if i == 0:
                sampler = create_sampler(self.args, dataset,
                                         run_alg=self.run_alg)
                batches, feats = sampler.sample_batches(**kwargs)
                # save outputs of traditional algorithms
                if self.args.use_learned_sampler:
                    self.graphs = sampler.graphs
                    self.orders = sampler.orders
            else:
                sampler = create_sampler(self.args, dataset, visit_counts,
                                         run_alg=self.run_alg)
                # no need to replace feats
                batches.extend(sampler.sample_batches(**kwargs)[0])
            # update counts if necessary
            visit_counts = sampler.visit_counts
        return batches, feats

    def __len__(self):
        return len(self.idx_to_data_regime)

    def __getitem__(self, idx):
        raise Exception("Not implemented")

    def get(self, idx, num_batches):
        """
        Each dataset's __getitem__ should wrap this
        """
        start = time.time()  # keep track of CPU time

        data_idx, reg_idx = self.idx_to_data_regime[idx]
        dataset = self.data[data_idx]
        data_obs = dataset.get_obs()["data"]
        data_int = dataset[reg_idx]["data"]

        # sample and run algorithm on observational
        batches_obs, feats_obs = self._sample_batches(data_obs,
                                                      num_batches // 2)
        results_obs = self.run_alg(batches_obs)
        graphs_obs, orders_obs = convert_to_graphs(results_obs, dataset)
        item_obs = convert_to_item(feats_obs, graphs_obs, orders_obs)

        # sample and run algorithm on interventional
        batches_int, feats_int = self._sample_batches(data_int,
                                                      num_batches // 2)
        results_int = self.run_alg(batches_int)
        graphs_int, orders_int = convert_to_graphs(results_int, dataset)
        item_int = convert_to_item(feats_int, graphs_int, orders_int)

        if item_obs is None or item_int is None:
            return {}

        end = time.time()  # keep track of CPU time

        # combine and copy processed keys
        item = {
            "key": dataset.key,
            "reg_idx": reg_idx,
            "dataset_id": dataset.dataset_id,
            "label": dataset[reg_idx]["label"],
            "time": end - start  # CPU time elapsed so far
        }
        if self.use_graph:
            item["graph"] = dataset.get_obs()["graph"]
        for key in ["input", "feats_2d", "feats_1d", "index", "unique"]:
            item[f"{key}_obs"] = item_obs[key]
            item[f"{key}_int"] = item_int[key]
        return item


class TrainDataset(MetaDataset):
    """
        Sample varying # of batches per individual dataset
    """
    def __init__(self, data_file, args, splits_to_load=None, **kwargs):
        super().__init__(data_file, args, splits_to_load, **kwargs)

    def __getitem__(self, idx):
        # number of batches total
        num_batches = np.random.randint(self.args.fci_batches,
                                        self.args.fci_batches * 5, 1).item()
        return self.get(idx, num_batches)


class TestDataset(MetaDataset):
    """
        Sample fixed # of batches per individual dataset
    """
    def __init__(self, data_file, args, splits_to_load=None, **kwargs):
        super().__init__(data_file, args, splits_to_load, **kwargs)

    def __getitem__(self, idx):
        num_batches = self.args.fci_batches_inference
        return self.get(idx, num_batches)


class BaselineDataset(MetaDataset):
    """
        Used for running baseline algorithms only. Samples all variables.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # only use RandomSampler since we sample ALL nodes for baselines
        is_obs = (self.args.algorithm != "gies")
        if is_obs:
            batch_sampler = samplers.ObservationalSampler
        else:
            batch_sampler = samplers.InterventionalSampler
        score_sampler = samplers.RandomSampler
        class Sampler(batch_sampler, score_sampler):
            pass
        self.create_sampler = Sampler

    def __getitem__(self, idx):
        dataset = self.data[idx]
        num_batches = self.args.fci_batches_inference
        start = time.time()  # keep track of CPU time
        batches, feats = self._sample_batches(dataset, num_batches)
        results = self.run_alg(batches)
        graphs, orders = convert_to_graphs(results, dataset)
        end = time.time()  # keep track of CPU time
        dataset.time = end - start
        if graphs is None:
            return {}
        return convert_to_item(feats, graphs, orders)

    def _sample_batches(self, dataset, num_batches):
        # sample all nodes every single time
        sampler = self.create_sampler(self.args, dataset,
                                      run_alg=self.run_alg)
        batches = sampler.sample_batches(
                num_batches=num_batches,
                batch_size=self.args.fci_batch_size,
                # note this line!
                num_vars_batch=dataset.num_vars)
        return batches


class MetaObservationalDataset(MetaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_classes = get_samplers(is_obs=True,
                                    is_learned=self.args.use_learned_sampler)

    def run_alg(self, batches):
        """
        batches: tuples (batch, order) output of sample_batches
        """
        results = []
        for batch, order in batches:
            G = self._run_alg(batch)
            if G is None:
                continue
            order = torch.from_numpy(order).long()
            results.append((G, order))
        return results


class MetaInterventionalDataset(MetaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler_classes = get_samplers(is_obs=False,
                                    is_learned=self.args.use_learned_sampler)

    def run_alg(self, batches):
        """
        batches: tuples (batch, order, regime) output of sample_batches
        """
        results = []
        for batch, order, regime in batches:
            graph = self._run_alg(batch, regime)
            if graph is None:
                continue
            order = torch.from_numpy(order).long()
            results.append((graph, order))
        return results


def _read(config):
    item, is_real, min_interv, max_interv, use_graph = config
    if not is_real:
        dataset = InterventionalDataset(item["fp_data"],
                                        item["fp_graph"],
                                        item["fp_regime"],
                                        min_interv=min_interv,
                                        max_interv=max_interv,
                                        use_graph=use_graph)
    else:
        dataset = PairDataset(item["fp_data"],
                              int(item["label"]))
    return dataset


def get_samplers(is_obs, is_learned):
    # observational vs. interventional determines whether regimes
    # are sampled for each batch
    if is_obs:
        batch_sampler = samplers.ObservationalSampler
    else:
        batch_sampler = samplers.InterventionalSampler
    score_samplers = [samplers.RandomSampler,
                      samplers.CorrelationSampler]
    # combine
    sampler_cls = []
    for score_sampler in score_samplers:
        class Sampler(batch_sampler, score_sampler):
            pass
        sampler_cls.append(Sampler)
    return sampler_cls


def get_run_alg(algorithm):
    if algorithm == "fci":
        return run_fci
    elif algorithm == "ges":
        return run_ges
    elif algorithm == "grasp":
        return run_grasp
    elif algorithm == "gies":
        return run_gies
    else:
        raise Exception("Unsupported algorithm", algorithm)

