import os
import sys
import csv
import pickle
from collections import defaultdict

from datetime import datetime


# -------- evaluation

def parse_results(results_list):
    """
    Post-process list of outputs from pl.Trainer.predict.
    You should insert your own custom logic here
    """
    # flatten batches
    results_dict = group_by_key(results_list)

    # organize by individual dataset and regime
    dataset_to_metrics = defaultdict(lambda: defaultdict(dict))
    # key contains both setting and dataset_id
    for i, key in enumerate(zip(results_dict["key"],
                                results_dict["dataset_id"])):
        # we should only have run each regime through once
        reg_idx = results_dict["reg_idx"][i]
        assert reg_idx not in dataset_to_metrics[key]

        # gather each metric
        dataset_to_metrics[key][reg_idx] = {
            "time": results_dict["time"][i],
            "true": results_dict["true"][i],
            "pred": results_dict["pred"][i],
            "auroc": results_dict["auroc"][i],
            "auprc": results_dict["auprc"][i],
        }
    dataset_to_metrics = dict(dataset_to_metrics)
    return dataset_to_metrics


def group_by_key(results_list):
    """
    Aggregate list of batched outputs from pl.Trainer.predict:
        list[dict k: list] -> dict k: list
    """
    results_dict = defaultdict(list)
    # you should be using batch_size=1 for eval but this is for generality
    for batch in results_list:
        for k, v in batch.items():
            if type(v) is list:
                results_dict[k].extend(v)
            else:
                results_dict[k].append(v)
    return results_dict

# -------- general

def save_pickle(fp, data):
    with open(fp, "wb+") as f:
        pickle.dump(data, f)


def read_pickle(fp):
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data


def read_csv(fp, fieldnames=None, delimiter=',', str_keys=[]):
    data = []
    with open(fp) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        # iterate and append
        for item in reader:
            data.append(item)
    return data

# -------- general

def get_timestamp():
    return datetime.now().strftime('%H:%M:%S')


def printt(*args, **kwargs):
    print(get_timestamp(), *args, **kwargs)


def get_suffix(metric):
    suffix = "model_best_"
    suffix = suffix + "{global_step}_{epoch}_{"
    suffix = suffix + metric + ":.3f}_{Val/loss:.3f}"
    return suffix

