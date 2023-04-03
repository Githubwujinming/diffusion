import torch
from functools import partial
from itertools import cycle
from packaging import version
import numpy as np
import pytorch_lightning as pl
from lightning.pytorch.utilities import CombinedLoader
from src.data.base import Txt2ImgIterableBaseDataset
from src.util import instantiate_from_config
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from src.data.SCDDataloader import SCDDataLoader
def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
# 用于将多个dataloader组合成一个dataloader
class SequentialLoader:
    def __init__(self, sup_data_loader, unsup_data_loader):
        self.sup_data_loader = sup_data_loader
        self.unsup_data_loader = unsup_data_loader
        self.len = max(len(sup_data_loader), len(unsup_data_loader))
    def __len__(self):
        return self.len

    def __iter__(self):
        for sup_data, unsup_data in zip(cycle(self.sup_data_loader), self.unsup_data_loader):
            yield sup_data, unsup_data
            
class SCDDatasetFromConfig(pl.LightningDataModule):
    def __init__(self, train_supervised=None, train_unsupervised=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=4, use_worker_init_fn=False,
                 sup_train_size=4, unsup_train_size=8, val_size=8,
                 test_size=1):
        super().__init__()
        self.dataset_configs = dict()
        self.sup_train_size = sup_train_size
        self.unsup_train_size = unsup_train_size
        self.val_size = val_size
        self.test_size = test_size
        self.num_workers = num_workers 
        self.use_worker_init_fn = use_worker_init_fn
        if train_supervised is not None:
            if train_unsupervised is not None:
                self.dataset_configs["unsup_train"] = train_unsupervised
            self.dataset_configs["sup_train"] = train_supervised
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=False)

        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, False)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['sup_train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        sup_loader = SCDDataLoader(self.datasets["sup_train"], batch_size=self.sup_train_size,
                        num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                        worker_init_fn=init_fn)
        
        if "unsup_train" in self.datasets.keys():
            unsup_loader = SCDDataLoader(self.datasets["unsup_train"], batch_size=self.unsup_train_size,
                        num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                        worker_init_fn=init_fn)
            if version.parse(pl.__version__) >= version.parse('1.9.0'):
                return CombinedLoader([sup_loader, unsup_loader], mode='max_size_cycle')
            else:
                return SequentialLoader(sup_loader, unsup_loader)
        else:
            return sup_loader

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return SCDDataLoader(self.datasets["validation"],
                          batch_size=self.val_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['test'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return SCDDataLoader(self.datasets["test"], batch_size=self.test_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return SCDDataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)

