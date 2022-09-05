import logging
import os
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn
import re

import h5py
import numpy as np
import pandas as pd
import requests
import torch
import yaml
from torchvision import transforms


def fetch_dir(
    key: str, data_config_file: Union[str, Path, os.PathLike] = "pet_dirs.yaml"
) -> Path:
    """
    Data directory fetcher.
    This is a brute-force simple way to configure data directories for a
    project.
    Args:
        key: key to retrieve path from data_config_file.
        data_config_file: Optional; Default path config file to fetch path
            from.

    Returns:
        The path to the specified directory.
    """
    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "data_path": "/path/to/data",
            "log_path": ".",
        }
        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class CombinedSliceDataset(torch.utils.data.Dataset):
    """
    A container for combining slice datasets.
    """

    def __init__(
        self,
        roots: Sequence[Path],
        drfs: Sequence[int],
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
    ):
        """
        Args:
            roots: Paths to the datasets.
            drfs: dose reduced factor.
            sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the slices should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            volume_sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the volumes should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
        """
        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError(
                "either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both"
            )

        if sample_rates is None:
            sample_rates = [None] * len(roots)
        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)
        if not (
            len(roots)
            == len(drfs)
            == len(sample_rates)
            == len(volume_sample_rates)
        ):
            raise ValueError(
                "Lengths of roots, transforms, challenges, sample_rates do not match"
            )

        self.datasets = []
        self.examples: List[Tuple[Path, int, Dict[str, object]]] = []
        for i in range(len(roots)):
            self.datasets.append(
                SliceDataset(
                    root=roots[i],
                    drf=drfs[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file
                )
            )

            self.examples = self.examples + self.datasets[-1].examples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to PET image slices.
    """
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        drf: int = 0,
        patch_size: int = 320,
        data_partition: str ='train',
        use_dataset_cache: bool = True,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
    ):
        """
        Args:
            root: Path to the dataset.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets .
        """
        if drf not in (0, 2, 4, 10, 20, 50, 100):
            raise ValueError('drf should be one of values 0, 2, 4, 10, 20, 50, 100')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.examples = []  # [(h5_path, #slice, meta), (..)]

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())  # iterate all h5 files
            for fname in sorted(files):
                volum, _ = self._read_hdf5(fname)
                num_slices = volum.shape[0]
                drf = int(re.split('(\d+)', str(fname))[-4])

                self.examples += [
                    (fname, slice_ind, drf) for slice_ind in range(num_slices)
                ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]  # num_examples
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        self.drf = drf

        if data_partition == 'train':
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop(patch_size)
            ])
        elif data_partition == 'val' or data_partition == 'test':
            if patch_size > 0:
                self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.CenterCrop(patch_size)
                ])
        else:
            raise NotImplementedError

    def _read_hdf5(self, file_h5: str):
        file = h5py.File(file_h5, 'r')
        # list(file.keys())
        image_low = file['lr']
        image_target = file['hr']
        return image_low, image_target

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, drf = self.examples[i]
        
        with h5py.File(fname, "r") as hf:
            image_low = hf["lr"][dataslice]
            image_target = hf["hr"][dataslice]
            drf = hf["drf"][0]  # int
            lr_mean = hf["lr_mean"][0]
            ori_w = hf["ori_w"][0]
            ori_h = hf["ori_h"][0]

        img_lr = self.transforms(np.float32(image_low))
        img_hr = self.transforms(np.float32(image_target))  # tensor(1,320,320)
        sample = {'lr': img_lr, 'hr': img_hr, 'fname': fname.stem, 'slice_num': dataslice, 'drf': drf,
                  'lr_mean': lr_mean, 'ori_w': ori_w, 'ori_h': ori_h}

        return sample