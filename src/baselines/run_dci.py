import os
import sys
import csv
import json
import time
from collections import defaultdict

from tqdm import tqdm
from causaldag import dci, dci_stability_selection

import numpy as np


def write_json(fp, data):
    with open(fp, 'w+') as f:
        for item in data:
            json.dump(item, f)
            f.write(os.linesep)


def read_csv(fp, delimiter=','):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for item in reader:
            for k,v in item.items():
                item[k] = v
            data.append(item)
    return data


def load_data(fp_data, fp_interv):
    """
    Copied from my own codebase
    """
    data = np.load(fp_data)

    with open(fp_interv) as f:
        # if >1 node intervened, formatted as a list
        lines = [line.strip() for line in f.readlines()]
    regimes = [tuple(sorted(int(x) for x in line.split(",")))
            if len(line) > 0 else () for line in lines]
    assert len(regimes) == len(data)

    # get unique and map to nodes
    unique_regimes = sorted(set(regimes))  # 0 is obs because () first
    idx_to_regime = {i: reg for i, reg in enumerate(unique_regimes)}
    regime_to_idx = {reg: i for i, reg in enumerate(unique_regimes)}
    # convert to regime label tensor
    num_vars = data.shape[1]
    labels = np.zeros((len(unique_regimes), num_vars)).astype(int)
    for idx, regime in idx_to_regime.items():
        for node in regime:
            labels[idx, node] = 1

    # group data by env
    env_to_idx = defaultdict(list)
    envs = [regime_to_idx[r] for r in regimes]
    for idx, env in enumerate(envs):
        env_to_idx[env].append(idx)

    obs_data = data[env_to_idx[0]]
    int_data = [data[env_to_idx[i]] for i in range(1, len(idx_to_regime))]

    return obs_data, int_data, labels.tolist()


def run_dci_all(obs_data, int_data,
                batch_size=1000, num_cpus=1):
    """
    obs_data  (num_samples, num_vars)
    int_data  [(num_samples, num_vars), ...]
    """
    outputs = []
    for dataset in int_data:
        obs_batch = sample_batch(obs_data, batch_size)
        int_batch = sample_batch(dataset, batch_size)
        ddag = run_dci(obs_batch, int_batch, num_cpus=num_cpus)
        outputs.append(ddag.tolist())
    return outputs


def run_dci(X1, X2, num_cpus=1):
    """
    X1, X2  (num_samples, num_vars)
    """
    # taken from tutorial
    ddag, scores = dci_stability_selection(X1, X2,
                                           alpha_ug_grid=[0.001, 0.01],
                                           alpha_skeleton_grid=[0.1, 0.5],
                                           alpha_orient_grid=[0.001, 0.01],
                                           edge_threshold=0,
                                           max_set_size=3,
                                           n_jobs=num_cpus,
                                           n_bootstrap_iterations=50,
                                           random_state=0,
                                           verbose=0)
    return ddag


def sample_batch(data, batch_size):
    """
    data  (num_samples, num_vars)
    """
    if len(data) <= batch_size:
        return data
    idxs = np.random.choice(len(data), batch_size, replace=False)
    return data[idxs]


def main():
    np.random.seed(0)

    fp = ""
    items_to_load = read_csv(fp)
    # >>> this run is only for missing 60 (uniform)
    #items_to_load = [item for item in items_to_load if "uniform"
    #                 in item["fp_data"]]
    #items_to_load = items_to_load[0:15]
    #items_to_load = items_to_load[15:30]
    #items_to_load = items_to_load[30:45]
    #items_to_load = items_to_load[45:]
    # <<<

    # save results here
    exp_root = ""
    # >>> uncomment desired setting
    #items_to_load = [item for item in items_to_load if "uniform"
    #                 not in item["fp_data"]]
    #items_to_load = items_to_load[0:50]
    #items_to_load = items_to_load[50:100]
    #items_to_load = items_to_load[100:150]
    #items_to_load = items_to_load[150:]
    #items_to_load = items_to_load[::-1]
    # <<<
    # iterate through our test set
    for item in tqdm(items_to_load):
        if item["split"] != "test":
            continue
        # load data and filter
        fp_data = item["fp_data"]
        key = fp_data.split("/")[-2]
        # use all of Sachs
        if "sachs" not in item["fp_data"]:
            nodes = int(key.split("_")[0][1:])
            edges = int(key.split("_")[1][1:])
        else:
            key = "sachs"
        # check if output exists
        dataset_id = item["fp_data"].split("data_interv")[1].split(".")[0]

        # remove .npy filename
        data_path = item["fp_data"].rsplit("/", 1)[0]

        fp_out = f"{exp_root}/{key}_{dataset_id}.json"
        if os.path.exists(fp_out):
            continue

        try:
            print(f"working on {fp_out}")
            obs_data, int_data, labels = load_data(item["fp_data"],
                                                   item["fp_regime"])
            start = time.time()
            # while I appreciate that Chandler wrote nice multi-cpu code, for
            # fairness of comparison, I'm going to use 1 CPU, since everyone
            # else only gets 1 CPU, including my own model.
            outputs = run_dci_all(obs_data, int_data,
                                  num_cpus=1)
            end = time.time()
            results = {
                "true": labels,
                "outputs": outputs,
                "time": end - start
            }
            write_json(fp_out, [results])
        except Exception as e:
            print(item["fp_data"], "CRASHED:", e)
            raise
            continue


if __name__ == '__main__':
    main()

