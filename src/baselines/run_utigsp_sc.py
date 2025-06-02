import os
import sys
import csv
import json
import time
from multiprocessing import Pool
from collections import defaultdict

from tqdm import tqdm

from causaldag import unknown_target_igsp
from utils.ci_tests import gauss_ci_suffstat, gauss_ci_test, MemoizedCI_Tester
from utils.invariance_tests import gauss_invariance_suffstat, gauss_invariance_test, MemoizedInvarianceTester


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
    data_interv = np.load(fp_data)
    data = np.load(fp_data.replace("data_interv-", "data-"))

    np.random.seed(0)
    obs_data = sample_batch(data, batch_size)
    int_data = sample_batch(data_interv, batch_size)

    return obs_data, int_data


def run_utigsp(X_obs, X_int):
    """
    targets  is a list of (list of) targets
    X_obs    should be n_samples, n_nodes
    X_ints   is a list of datasets, each the same format as X_obs
    """
    nodes = set(range(X_obs.shape[1]))
    if len(nodes) > 200:
        return [None]

    # Form sufficient statistics
    obs_suffstat = gauss_ci_suffstat(X_obs)
    invariance_suffstat = gauss_invariance_suffstat(X_obs, [X_int])

    # Create conditional independence tester and invariance tester
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test,
            invariance_suffstat, alpha=alpha_inv)

    # Run UT-IGSP
    setting_list = [dict(known_interventions=[])]
    try:
        dag, targets = unknown_target_igsp(
            setting_list,
            nodes,
            ci_tester,
            invariance_tester,
            depth=3,
            nruns=5,
            verbose=False
        )
    # numerical issues
    except:
        return [None]
    targets = [list(t) for t in targets]
    return targets


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

    fp = f""
    items_to_load = read_csv(fp)

    # save results here
    exp_root = ""
    # iterate through our test set
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

    #n_cpu = 1
    #with Pool(n_cpu) as pool:
    #    for item in pool.map(worker, configs):
    # >>>
    #configs = configs[3950:]
    # <<<
    save_batch = 10
    results = []
    for config in tqdm(configs, ncols=40):
        results.append(worker(config))
        # save
        if len(results) % save_batch == 0:
            for fp_out, preds in results:
                write_json(fp_out, [preds])
            results = []
    # last batch
    for fp_out, preds in results:
        write_json(fp_out, [preds])


def worker(config):
    fp_data, fp_out = config
    obs_data, int_data = load_data(fp_data)
    start = time.time()
    outputs = run_utigsp(obs_data, int_data)
    end = time.time()
    results = {
        "outputs": outputs,
        "time": end - start
    }
    return fp_out, results


if __name__ == '__main__':
    main()


