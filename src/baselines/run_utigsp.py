import os
import sys
import csv
import json
import time
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


def load_data(fp_data, fp_interv, batch_size=1000):
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

    np.random.seed(0)
    obs_data = sample_batch(data[env_to_idx[0]], batch_size)
    int_data = [data[env_to_idx[i]] for i in range(1, len(idx_to_regime))]

    return obs_data, int_data, labels.tolist()[1:]


def run_utigsp(X_obs, X_ints, targets):
    """
    targets  is a list of (list of) targets
    X_obs    should be n_samples, n_nodes
    X_ints   is a list of datasets, each the same format as X_obs
    """
    assert len(X_ints) == len(targets)
    # Form sufficient statistics
    obs_suffstat = gauss_ci_suffstat(X_obs)
    invariance_suffstat = gauss_invariance_suffstat(X_obs, X_ints)

    # Create conditional independence tester and invariance tester
    alpha = 1e-3
    alpha_inv = 1e-3
    ci_tester = MemoizedCI_Tester(gauss_ci_test, obs_suffstat, alpha=alpha)
    invariance_tester = MemoizedInvarianceTester(gauss_invariance_test,
            invariance_suffstat, alpha=alpha_inv)

    # Run UT-IGSP
    setting_list = [dict(known_interventions=[]) for _ in targets]
    nodes = set(range(X_obs.shape[1]))
    dag, targets = unknown_target_igsp(
        setting_list,
        nodes,
        ci_tester,
        invariance_tester
    )
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
        if "shift" in item["fp_data"]:
            continue
        if "normal" in item["fp_data"]:
            continue
        # load data and filter
        fp_data = item["fp_data"]
        key = fp_data.split("/")[-2]
        if "sachs" in fp_data:
            key = "sachs"
        else:
            nodes = int(key.split("_")[0][1:])
            edges = int(key.split("_")[1][1:])
            #if (nodes, edges) not in [(10, 10), (20, 40)]:
            #    continue
            if (nodes, edges) not in [(10, 20), (20, 20)]:
                continue
        # check if output exists
        dataset_id = item["fp_data"].split("data_interv")[1].split(".")[0]

        # remove .npy filename
        data_path = item["fp_data"].rsplit("/", 1)[0]

        fp_out = f"{exp_root}/{key}_{dataset_id}.json"
        if os.path.exists(fp_out):
            pass
            #continue

        try:
            #print(f"working on {fp_out}")
            obs_data, int_data, labels = load_data(item["fp_data"],
                                                   item["fp_regime"])
            start = time.time()
            outputs = run_utigsp(obs_data, int_data, labels)
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


