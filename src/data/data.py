"""
DataModule is entrypoint into all data-related objects.

We create different datasets via diamond inheritence
which is arguably horrible and clever hehe.

"""

import os
from functools import partial
from contextlib import redirect_stdout

import numpy as np
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from .dataset import MetaObservationalDataset, MetaInterventionalDataset
from .dataset import TrainDataset, TestDataset, BaselineDataset
from .utils import collate
from utils import read_csv


def get_base_dataset(algorithm):
    if algorithm in ["fci", "ges", "grasp"]:
        return MetaObservationalDataset
    elif algorithm in ["gies"]:
        return MetaInterventionalDataset
    else:
        raise Exception("Unsupported algorithm", algorithm)


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_file = args.data_file

        BaseDataset = get_base_dataset(args.algorithm)
        class BaseTrainDataset(BaseDataset, TrainDataset):
            pass
        class BaseTestDataset(BaseDataset, TestDataset):
            pass

        self.subset_train = BaseTrainDataset(self.data_file, args,
                splits_to_load=["train"])
        self.subset_val = BaseTestDataset(self.data_file, args,
                splits_to_load=["val"])

    def train_dataloader(self):
        train_loader = DataLoader(self.subset_train,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  persistent_workers=(not self.args.debug),
                                  collate_fn=partial(collate, self.args))
        return train_loader

    def val_dataloader(self):
        # batch_size smaller since we sample more batches on average
        val_loader = DataLoader(self.subset_val,
                                batch_size=max(self.batch_size // 4, 1),
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True,
                                persistent_workers=(not self.args.debug),
                                collate_fn=partial(collate, self.args))
        return val_loader


class InferenceDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        # for proper timing, ALWAYS set batch_size to 1
        self.batch_size = 1
        self.num_workers = args.num_workers
        self.data_file = args.data_file

        BaseDataset = get_base_dataset(args.algorithm)
        class BaseTestDataset(BaseDataset, TestDataset):
            pass

        self.subset_test = BaseTestDataset(self.data_file, args,
                splits_to_load=["test"])

    def predict_dataloader(self):
        test_loader = DataLoader(self.subset_test,
                                 batch_size=self.batch_size,
                                 num_workers=10,#self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return test_loader

    def test_dataloader(self):
        return self.predict_dataloader()


class BaselineDataModule(pl.LightningDataModule):
    """
        Used for running baseline algorithms only. Samples all variables.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.data_file = args.data_file

        BaseDataset = get_base_dataset(args.algorithm)
        class BaseTestDataset(BaseDataset, BaselineDataset):
            pass

        self.subset_test = BaseTestDataset(self.data_file, args,
                splits_to_load=["test"])

    def predict_dataloader(self):
        test_loader = DataLoader(self.subset_test,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=False,
                                 pin_memory=False,
                                 collate_fn=partial(collate, self.args))
        return test_loader

    def test_dataloader(self):
        return self.predict_dataloader()


def get_finetune_datasets(args):
    """
    Ok. So I think the easiest workflow is to loop through MetaDataset
    objects, where each only contains a single real dataset + multiple
    interventions. We can split the interventions into train/test
    (manually for now since brain ded)

    Obv we only finetune on the test weirdos
    """
    BaseDataset = get_base_dataset(args.algorithm)
    class BaseTrainDataset(BaseDataset, TrainDataset):
        pass
    class BaseTestDataset(BaseDataset, TestDataset):
        pass

    data_to_load = read_csv(args.data_file)
    # in my test file, technically this should == range(len(data_to_load))
    lines_to_load = [line for line, item in enumerate(data_to_load)
                     if item["split"] == "test"]
    # iterate through each individual dataset
    data_loaders = []
    for line in lines_to_load:
        train_dataset = BaseTrainDataset(args.data_file, args,
                                         line_to_load=line,
                                         max_interv=0, use_tqdm=False)
        test_dataset = BaseTestDataset(args.data_file, args,
                                       line_to_load=line,
                                       min_interv=1, use_tqdm=False)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=1,
                                  shuffle=True,
                                  collate_fn=partial(collate, args))

        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=1,
                                 shuffle=False,
                                 collate_fn=partial(collate, args))
        data_loaders.append((train_loader, test_loader))
    return data_loaders

