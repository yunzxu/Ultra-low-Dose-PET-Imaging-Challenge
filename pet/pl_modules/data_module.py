from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Union
import re
import time

import pytorch_lightning as pl
import torch
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import pickle
import logging
import h5py
from torchvision import transforms
import functools
from torchvision.transforms import functional
from scipy.ndimage import gaussian_filter
import numpy as np
import sys##引用同级文件夹下的
sys.path.append("/raid/yunzhi_raid/Low_dose_PET_5/")###
from pet.data import SliceDataset, CombinedSliceDataset
from pet.data import Patch3D_Data,Patch3D_SegData


class PETDataModule(pl.LightningDataModule):
    """
    Data module class for PET data sets.
    """

    def __init__(
        self,
        data_path: Path,
        drf: int = 0,
        combine_train_val: bool = False,
        patch_size: int = 320,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ):
        """
        Args:
            data_path: Path to root data directory.
            use_dataset_cache_file: Whether to cache dataset metadata.
            batch_size: Batch size.
            num_workers: Number of workers for PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This
                should be set to True if training with ddp.
        """
        super().__init__()

        self.data_path = data_path
        self.drf = drf
        self.patch_size = patch_size
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate

    def _create_data_loader(
        self,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:

        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
        else:
            is_train = False
            # sample_rate = 1.0
            # volume_sample_rate = None  # default case, no subsampling

        # if desired, combine train and val together for the train split
        dataset: Union[SliceDataset, CombinedSliceDataset]
        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
            ]

            drfs = [self.drf, self.drf]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            dataset = CombinedSliceDataset(
                roots=data_paths,
                drfs=drfs,
                sample_rates=sample_rates,
                volume_sample_rates=volume_sample_rates,
                use_dataset_cache=self.use_dataset_cache_file,
            )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{data_partition}"  # val

            dataset = SliceDataset(
                root=data_path,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                # patch_size=self.patch_size,####
                drf=self.drf,
                use_dataset_cache=self.use_dataset_cache_file,
            )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=is_train,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        # dataset.pkl = dict{'root_train':train data, 'root_val':val data, 'root_test':test data}
        tic = time.perf_counter()
        if self.use_dataset_cache_file:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
                self.data_path / "test",
            ]
            for i, data_path in enumerate(data_paths):
                sample_rate = self.sample_rate if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate if i == 0 else None
                _ = SliceDataset(  # preload data into dataset.pkl
                    root=data_path,
                    drf=self.drf,
                    patch_size=self.patch_size,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    use_dataset_cache=self.use_dataset_cache_file,
                )
        toc = time.perf_counter()
        print(f"Prepare data in {toc - tic:0.4f} seconds")

    def train_dataloader(self)-> torch.utils.data.DataLoader:
        tic = time.perf_counter()
        train_data_loader = self._create_data_loader(data_partition="train")
        toc = time.perf_counter()
        print(f"Train data loader in {toc - tic:0.4f} seconds")
        return train_data_loader

    def val_dataloader(self)-> torch.utils.data.DataLoader:
        tic = time.perf_counter()
        val_data_loader = self._create_data_loader(data_partition="val", sample_rate=1.0)
        toc = time.perf_counter()
        print(f"Val data loader in {toc - tic:0.4f} seconds")
        return val_data_loader


    def test_dataloader(self):
        test_data_loader = self._create_data_loader(data_partition=self.test_split, sample_rate=1.0)
        return test_data_loader


    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to data root",
        )

        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )

        parser.add_argument(
            "--drf",
            choices=(0, 2, 4, 10, 20, 50, 100),
            default=0,
            type=int,
            help="Which low dose to preprocess for, dose=0 means all doses",
        )
        parser.add_argument(
            "--test_split",
            choices=("test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help="Fraction of slices in the dataset to use (train split only). If not given all will be used. Cannot set together with volume_sample_rate.",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes of the dataset to use (train split only). If not given all will be used. Cannot set together with sample_rate.",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=True,
            type=bool,
            help="Whether to cache dataset meta info in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size",
            default=1,
            type=int,
            help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=16,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser




class PETData3DModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        drf: int = 0,
        combine_train_val: bool = False,
        patch_size: int = 96,
        use_dataset_cache_file: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ):##use_dataset_cache_file设置为FALSE，防止读取2D数据

        super().__init__()

        self.data_path = data_path
        self.drf = drf
        self.patch_size = patch_size
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
    def _create_data_loader(
        self,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        patch_training: bool = True,
    ) -> torch.utils.data.DataLoader:

        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
        else:
            is_train = False
            # sample_rate = 1.0
            # volume_sample_rate = None  # default case, no subsampling

        # if desired, combine train and val together for the train split
        # dataset: Union[SliceDataset, CombinedSliceDataset]


        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
            ]

            drfs = [self.drf, self.drf]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            # dataset = CombinedSliceDataset(
            #     roots=data_paths,
            #     drfs=drfs,
            #     sample_rates=sample_rates,
            #     volume_sample_rates=volume_sample_rates,
            #     use_dataset_cache=self.use_dataset_cache_file,
            # )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{data_partition}"  # val
         
            dataset = Patch3D_Data(
                root=data_path,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                patch_size=self.patch_size,####
                drf=self.drf,
                use_dataset_cache=self.use_dataset_cache_file,
                patch_training=patch_training,####对于val,使用完整图像测试，即patch_training=False
            )


        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        # dataset.pkl = dict{'root_train':train data, 'root_val':val data, 'root_test':test data}
        tic = time.perf_counter()
        if self.use_dataset_cache_file:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
                self.data_path / "test",
            ]
            for i, data_path in enumerate(data_paths):
                sample_rate = self.sample_rate if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate if i == 0 else None
                _ = Patch3D_Data(  # preload data into dataset.pkl
                    root=data_path,
                    drf=self.drf,
                    patch_size=self.patch_size,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    use_dataset_cache=self.use_dataset_cache_file,
                )
        toc = time.perf_counter()
        print(f"Prepare data in {toc - tic:0.4f} seconds")

    def train_dataloader(self)-> torch.utils.data.DataLoader:
        tic = time.perf_counter()
        train_data_loader = self._create_data_loader(data_partition="train")
        toc = time.perf_counter()
        print(f"Train data loader in {toc - tic:0.4f} seconds")
        return train_data_loader

    def val_dataloader(self)-> torch.utils.data.DataLoader:
        tic = time.perf_counter()
        val_data_loader = self._create_data_loader(data_partition="val", sample_rate=1.0,patch_training=False)##Fasle输出完整图像
        toc = time.perf_counter()
        print(f"Val data loader in {toc - tic:0.4f} seconds")
        return val_data_loader


    def test_dataloader(self):
        test_data_loader = self._create_data_loader(data_partition=self.test_split, sample_rate=1.0)
        return test_data_loader


    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to data root",
        )

        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )

        parser.add_argument(
            "--drf",
            choices=(0, 2, 4, 10, 20, 50, 100),
            default=0,
            type=int,
            help="Which low dose to preprocess for, dose=0 means all doses",
        )
        parser.add_argument(
            "--test_split",
            choices=("test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help="Fraction of slices in the dataset to use (train split only). If not given all will be used. Cannot set together with volume_sample_rate.",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes of the dataset to use (train split only). If not given all will be used. Cannot set together with sample_rate.",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=False,
            type=bool,
            help="Whether to cache dataset meta info in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size",
            default=1,
            type=int,
            help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=16,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser




class PETDataSegModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Path,
        drf: int = 0,
        combine_train_val: bool = False,
        patch_size: int = 96,
        use_dataset_cache_file: bool = False,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        patch_training: bool = True,
    ):##use_dataset_cache_file设置为FALSE，防止读取2D数据

        super().__init__()

        self.data_path = data_path
        self.patch_size = patch_size
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.patch_training=patch_training
    def _create_data_loader(
        self,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        patch_training: bool = False,
        batch_size:Optional[float] = 4,
    ) -> torch.utils.data.DataLoader:

        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = self.volume_sample_rate if volume_sample_rate is None else volume_sample_rate
        else:
            is_train = False
            # sample_rate = 1.0
            # volume_sample_rate = None  # default case, no subsampling

        # if desired, combine train and val together for the train split
        # dataset: Union[SliceDataset, CombinedSliceDataset]


        if is_train and self.combine_train_val:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
            ]

            drfs = [self.drf, self.drf]
            sample_rates, volume_sample_rates = None, None  # default: no subsampling
            if sample_rate is not None:
                sample_rates = [sample_rate, sample_rate]
            if volume_sample_rate is not None:
                volume_sample_rates = [volume_sample_rate, volume_sample_rate]
            # dataset = CombinedSliceDataset(
            #     roots=data_paths,
            #     drfs=drfs,
            #     sample_rates=sample_rates,
            #     volume_sample_rates=volume_sample_rates,
            #     use_dataset_cache=self.use_dataset_cache_file,
            # )
        else:
            if data_partition in ("test", "challenge") and self.test_path is not None:
                data_path = self.test_path
            else:
                data_path = self.data_path / f"{data_partition}"  # val
         
            dataset = Patch3D_SegData(
                root=data_path,
                sample_rate=sample_rate,
                volume_sample_rate=volume_sample_rate,
                patch_size=self.patch_size,####
                use_dataset_cache=self.use_dataset_cache_file,
                patch_training=patch_training,####对于val,使用完整图像测试，即patch_training=False
            )


        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        return dataloader

    def prepare_data(self):
        # call dataset for each split one time to make sure the cache is set up on the
        # rank 0 ddp process. if not using cache, don't do this
        # dataset.pkl = dict{'root_train':train data, 'root_val':val data, 'root_test':test data}
        tic = time.perf_counter()
        if self.use_dataset_cache_file:
            data_paths = [
                self.data_path / "train",
                self.data_path / "val",
                self.data_path / "test",
            ]
            for i, data_path in enumerate(data_paths):
                sample_rate = self.sample_rate if i == 0 else 1.0
                volume_sample_rate = self.volume_sample_rate if i == 0 else None
                _ = Patch3D_SegData(  # preload data into dataset.pkl
                    root=data_path,
                    patch_size=self.patch_size,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    use_dataset_cache=self.use_dataset_cache_file,
                )
        toc = time.perf_counter()
        print(f"Prepare data in {toc - tic:0.4f} seconds")

    def train_dataloader(self)-> torch.utils.data.DataLoader:
        tic = time.perf_counter()
        train_data_loader = self._create_data_loader(data_partition="train",patch_training=True,batch_size=self.batch_size)
        toc = time.perf_counter()
        print(f"Train data loader in {toc - tic:0.4f} seconds")
        return train_data_loader

    def val_dataloader(self)-> torch.utils.data.DataLoader:
        tic = time.perf_counter()
        val_data_loader = self._create_data_loader(data_partition="val", sample_rate=1.0,patch_training=False,batch_size=1)##Fasle输出完整图像
        toc = time.perf_counter()
        print(f"Val data loader in {toc - tic:0.4f} seconds")
        return val_data_loader


    def test_dataloader(self):
        test_data_loader = self._create_data_loader(data_partition=self.test_split, sample_rate=1.0)
        return test_data_loader


    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # dataset arguments
        parser.add_argument(
            "--data_path",
            default=None,
            type=Path,
            help="Path to data root",
        )

        parser.add_argument(
            "--test_path",
            default=None,
            type=Path,
            help="Path to data for test mode. This overwrites data_path and test_split",
        )
        parser.add_argument(
            "--test_split",
            choices=("test", "challenge"),
            default="test",
            type=str,
            help="Which data split to use as test split",
        )
        parser.add_argument(
            "--sample_rate",
            default=None,
            type=float,
            help="Fraction of slices in the dataset to use (train split only). If not given all will be used. Cannot set together with volume_sample_rate.",
        )
        parser.add_argument(
            "--volume_sample_rate",
            default=None,
            type=float,
            help="Fraction of volumes of the dataset to use (train split only). If not given all will be used. Cannot set together with sample_rate.",
        )
        parser.add_argument(
            "--use_dataset_cache_file",
            default=False,
            type=bool,
            help="Whether to cache dataset meta info in a pkl file",
        )
        parser.add_argument(
            "--combine_train_val",
            default=False,
            type=bool,
            help="Whether to combine train and val splits for training",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size",
            default=1,
            type=int,
            help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=16,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parser




if __name__ == "__main__":
    
    data_path= Path('/raid/yunzhi_raid/data/hecktor2022/data_all_patch/')

    # dataset = PETDataSegModule(
    #             root=data_path,
    #             sample_rate=None,
    #             volume_sample_rate=None,
    #             patch_size=96,####
    #             use_dataset_cache=False,
    #             patch_training=True,####对于val,使用完整图像测试，即patch_training=False
    #         )


    data_module = PETDataSegModule(
        data_path=data_path,
        # test_split=args.test_split,
        # test_path=args.test_path,
        sample_rate=None,
        volume_sample_rate=None,
        batch_size=4,
        patch_size=96,
        use_dataset_cache_file=False,
        num_workers=0,
    )


    dataset=data_module.train_dataloader()

    batch=next(iter(dataset))
    # {'image': X_patch, 'label': Y_patch, 'fname': fname.stem}

    img=batch['image'].numpy()
    label=batch['label'].numpy()
    fname=batch['fname']
    print(img.shape,label.shape,fname)

    # label=label.transpose(1,2,3,0)
    # print(label.shape)
    # label=restore_labels(label,range(8))
    # save_nifit(img[0],'/raid/yunzhi_raid/Low_dose_PET_5/pet/data/imageCT.nii.gz')
    # save_nifit(img[1],'/raid/yunzhi_raid/Low_dose_PET_5/pet/data/imagePT.nii.gz')
    # save_nifit(label,'/raid/yunzhi_raid/Low_dose_PET_5/pet/data/label.nii.gz')