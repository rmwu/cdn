import os
import sys
import csv
import json
import time
from collections import defaultdict
from multiprocessing import Pool

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


def load_data(fp_data, batch_size=1000):
    """
    Copied from my own codebase
    """
    data = np.load(fp_data)
    data_interv = np.load(fp_data.replace("data-", "data_interv-"))

    np.random.seed(0)
    obs_data = sample_batch(data, batch_size)
    int_data = sample_batch(data_interv, batch_size)

    return obs_data, int_data


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
                                           max_set_size=2,
                                           n_jobs=num_cpus,
                                           n_bootstrap_iterations=10,
                                           random_state=0,
                                           verbose=0)
    return ddag.tolist()


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

    # save results here
    exp_root = ""
    configs = []
    # run on everything, pairwise
    for item in items_to_load:
        fp_data = item["fp_data"]
        key = fp_data.split("/")[-2]
        dataset_id = fp_data.split(".")[0].split("-")[1]

        fp_out = f"{exp_root}/{key}_{dataset_id}.json"
        if os.path.exists(fp_out):
            continue
        configs.append((fp_data, fp_out))

    save_batch = 100
    results = []
    for config in tqdm(configs):
        results.append(worker(config))
        # save
        if len(results) % 100 == 0:
            for fp_out, preds in results:
                write_json(fp_out, [preds])
            results = []
    # last batch
    for fp_out, preds in results:
        write_json(fp_out, [preds])


def worker(config):
    fp_data, fp_out = config
    obs_data, int_data = load_data(fp_data)
    print(fp_data, obs_data.shape, int_data.shape)
    start = time.time()
    # sigh I don't know how to turn off many cpus for UT-IGSP
    outputs = run_dci(obs_data, int_data,
                      num_cpus=1)
    end = time.time()
    results = {
        "outputs": outputs,
        "time": end - start
    }
    return fp_out, results


if __name__ == '__main__':
    main()

