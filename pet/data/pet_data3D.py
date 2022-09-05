import numpy as np
import scipy.ndimage as ndimage
import time
import torch

import pickle
import torch.nn as nn
from torch.utils import data
import h5py
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from pathlib import Path
import copy
import os
import re
from warnings import warn
import pandas as pd
import requests
import yaml
from torchvision import transforms
from scipy.ndimage import zoom



class Patch3D_Data(data.IterableDataset):##数据量未知，使用IterableDataset，取patch这种合适
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
        augmentation: bool =False,
        patch_training: bool =True,
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
        self.patch_training=patch_training ##在val时设置为否，验证集使用滑动窗口计算
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

                self.examples += [(fname, num_slices, drf)]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        # if sample_rate < 1.0:  # sample by slice
        #     random.shuffle(self.examples)
        #     num_examples = round(len(self.examples) * sample_rate)
        #     self.examples = self.examples[:num_examples]  # num_examples
        # elif volume_sample_rate < 1.0:  # sample by volume
        #     vol_names = sorted(list(set([f[0].stem for f in self.examples])))
        #     random.shuffle(vol_names)
        #     num_volumes = round(len(vol_names) * volume_sample_rate)
        #     sampled_vols = vol_names[:num_volumes]
        #     self.examples = [
        #         example for example in self.examples if example[0].stem in sampled_vols
        #     ]##这段有点看不懂，先去掉。

        self.drf = drf

        self.image_size=volum.shape[0:3]##
        self.patch_size = [patch_size,patch_size,patch_size]
        self.n_pixels = np.prod([patch_size,patch_size,patch_size])
        self.aug = augmentation
        self.patch_edge = self.__calc_edge__()

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

    def __calc_edge__(self):
            # index = []
            # start_pos = []
            
            n0 = self.image_size[0] - self.patch_size[0] + 1
            n1 = self.image_size[1] - self.patch_size[1] + 1
            n2 = self.image_size[2] - self.patch_size[2] + 1
            # index = np.array(np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2))).T.reshape(-1,3).astype(np.int8)
            # for n in range(self.n_subject):
            #     for pos in start_pos:
            #         index.append([n, pos])
            return (n0,n1,n2)

    def _edge(self,image_size):
        n0 = image_size[0] -  self.patch_size[0] + 1
        n1 = image_size[1] -  self.patch_size[1] + 1
        n2 = image_size[2] -  self.patch_size[2] + 1
        return (n0,n1,n2)

    
    def __get_patch(self, X, start_pos, patch_size):
        Y = X[start_pos[0]:start_pos[0] + patch_size[0],
            start_pos[1]:start_pos[1] + patch_size[1],
            start_pos[2]:start_pos[2] + patch_size[2]]
        return Y
    

    
    
    def __iter__(self):
        # np.random.shuffle(self.patch_index)
        while True:

            n_subject=len(self.examples)
            # print('n_subject',n_subject)
            selected_subject = np.random.randint(n_subject)
            # print('self.patch_edge',self.patch_edge)
            
            fname, dataslice, drf = self.examples[selected_subject]
            with h5py.File(fname, "r") as hf:
                image_low = hf["lr"][()]
                image_target = hf["hr"][()]
                drf = hf["drf"][0]  # int
                lr_mean = hf["lr_mean"][0]
                ori_w = hf["ori_w"][0]
                ori_h = hf["ori_h"][0]  
           
            ###使用zoom计算：
            # image_low=zoom(np.array(image_low),zoom=[96/image_low.shape[0],96/image_low.shape[1],96/image_low.shape[2]],order=1)
            # image_target=zoom(np.array(image_target),zoom=[96/image_target.shape[0],96/image_target.shape[1],96/image_target.shape[2]],order=1)

            ##进行标准化操作：
            # image_low=image_low-np.mean(image_low)
            # image_target=image_target-np.mean(image_target)
            #print("image",image_low.shape,image_target.shape)
            image_size=image_low.shape[0:3]

            index = np.random.randint(self._edge(image_size))
            #print('index',index)

         
            if self.patch_training :###对于训练的使用patch,对于val使用滑动窗口
                # print('index',index)
                # print(type(image_target))
                Y_patch = self.__get_patch(image_target, index, self.patch_size)
                patch_probability = np.sum((Y_patch>0).astype(np.float))/(self.n_pixels).astype(np.float)/0.1
                ##(Y_patch>0)可以认为是取全脑区域
                #print(np.sum((Y_patch>0).astype(np.float)),(patch_probability>5))
                if patch_probability>np.random.uniform():#np.random.uniform() < patch_probability:##起码patch_probability必须大于0或者说不能太小，才能使用patch
                    #print("subject",selected_subject)
                    X_patch = self.__get_patch(image_low, index, self.patch_size)
                    X_patch = X_patch[np.newaxis,:,:,:]
                    #X_patch = np.expand_dims(normlize_mean_std(X_patch), -1)
                    #Y_patch = to_hot_label(Y_patch,np.arange(self.n_labels))
                    #Y_patch = multi_class_labels(Y_patch,np.arange(self.n_labels))
                    ##########使用tf.onehot进行编码

                    ##
                    Y_patch = Y_patch[np.newaxis,:,:,:]
                    Y_patch=torch.tensor(Y_patch).type(torch.float32)
                    #print(Y_patch.shape)
                    #Y_patch=torch.nn.functional.one_hot(Y_patch, num_classes=self.n_labels)
                    a=time.time()
                    #print(Y_patch.shape)
                    if self.aug:
                        Y_patch=Y_patch.numpy()
            
                        X_patch,Y_patch=flip_lr(X_patch,Y_patch)
                        X_patch=torch.tensor(X_patch)
                        Y_patch=torch.tensor(Y_patch)
                        X_patch,Y_patch=shift_3d_int(X_patch,Y_patch)#####使用shift_3d_int里面的torch.roll运行速度极快
                    
                    X_patch=torch.tensor(X_patch).type(torch.FloatTensor)
                    Y_patch=Y_patch.type(torch.FloatTensor)
                    # print('X_patch',X_patch.shape)
                    sample = {'lr': X_patch, 'hr': Y_patch, 'fname': fname.stem, 'slice_num': dataslice, 'drf': drf,
                    'lr_mean': lr_mean, 'ori_w': ori_w, 'ori_h': ori_h}
                    # print('一次迭代')          
                    yield sample
            else:
                X_patch=crop_pad3D(image_low,[644,256,256])
                Y_patch=crop_pad3D(image_target,[644,256,256])
                X_patch = X_patch[np.newaxis,:,:,:]
                Y_patch = Y_patch[np.newaxis,:,:,:]
                X_patch=torch.tensor(X_patch).type(torch.FloatTensor)
                Y_patch=torch.tensor(Y_patch).type(torch.FloatTensor)
                #print('X_patch',X_patch.shape)




                sample = {'lr': X_patch, 'hr': Y_patch, 'fname': fname.stem, 'slice_num': dataslice, 'drf': drf,
                    'lr_mean': lr_mean, 'ori_w': ori_w, 'ori_h': ori_h}
                # print('一次迭代')          
                yield sample
    
###用于aug3D的函数：


def rot_3d_torch(X, Y):
    axis = ((0,1),(0,2),(1,2))
    #t=time.time()
    for n in range(len(axis)):
        X=torch.rot90(X,1,axis[n])
        Y=torch.rot90(Y,1,axis[n])
    return X,Y
    
    
    

def flip_lr(X, Y):
    if np.random.uniform() > 0.5:
        X = X[::-1,:,:,:].copy()
        Y = Y[::-1,:,:,:].copy()####加copy()防止报错
    return X, Y

def rot_3d(X, Y, max_angle=15):
    axis = ((0,1),(0,2),(1,2))
    #t=time.time()
    for n in range(len(axis)):
        theta = np.random.uniform(-max_angle, max_angle)
        X = ndimage.rotate(np.squeeze(X), theta, axes=axis[n], reshape=False, mode='reflect')
        Y = ndimage.rotate(np.squeeze(Y), theta, axes=axis[n], reshape=False, mode='reflect', order=1)
    #return X[:,:,:,np.newaxis], (Y>0.5).astype(int)[:,:,:,np.newaxis]####对于多标签的情况Y>0.5肯定是有大问题的
    ##报错出在这，因为已经把onehot改在了前面，那么这里就是多了维度的不用np.newaxis
    #print(time.time()-t,'s')
    return X[:,:,:,np.newaxis], (Y>0.5).astype(int)##X[:,:,:,np.newaxis]还是要加上，因为有一个np.squeeze(X)，会把维度1给变没


def tf_random_rotate_image(image, label):
    im_shape = image.shape
    l_shape = label.shape
    [image,label] = tf.py_function(rot_3d, [image, label], [tf.float16, tf.float16])
    image.set_shape(im_shape)
    label.set_shape(l_shape)

    print("image,label",image.shape,label.shape)
    return image, label

def shift_3d(X ,Y, max_shift=10):
    t=time.time()
    #print("shift_3d",X.shape,Y.shape)
    random_shift = np.random.uniform(-max_shift,max_shift,4)
    random_shift[3]=0 # channel demension 
    print(time.time()-t,'s')
    X = ndimage.shift(X, shift=random_shift, mode='reflect')
    print(time.time()-t,'s')
    Y = ndimage.shift(Y, shift=random_shift, mode='reflect', order=1)
    print(time.time()-t,'s')
    #print("shift_3d输出",X.shape,Y.shape)
    return X, (Y>0.5).astype(int)

def tf_random_shift_image(image, label):

    # label=tf.cast(label[:,:,:,0],dtype=tf.int32)
    # label=tf.one_hot(indices=label, depth=8, on_value=1.0, off_value=0.0, axis=-1)
    # print("label",label.shape)
    im_shape = image.shape
    l_shape = label.shape

    [image,label] = tf.py_function(shift_3d, [image, label], [tf.float32, tf.float32])
    image.set_shape(im_shape)
    label.set_shape(l_shape)
    print("image,label",image.shape,label.shape)
    return image, label

# shift by int is about 40% faster
def shift_3d_int(X ,Y, max_shift=10):
    random_shift = np.random.randint(-max_shift, max_shift, size=3)
    X = torch.roll(X, list(random_shift), dims=(0,1,2))
    Y = torch.roll(Y, list(random_shift), dims=(0,1,2))
    return X, Y

def crop_pad3D(x, target_size, shift=[0, 0, 0]):
    'crop or zero-pad the 3D volume to the target size'
    x = np.asarray(x)
    small = 0
    y = np.ones(target_size, dtype=np.float32) * small
    current_size = x.shape
    pad_size = [0, 0, 0]
    # print('current_size:',current_size)
    # print('pad_size:',target_size)
    for dim in range(3):
        if current_size[dim] > target_size[dim]:
            pad_size[dim] = 0
        else:
            pad_size[dim] = int(np.ceil((target_size[dim] - current_size[dim])/2.0))#np.ceil向上取整
    # pad first
    x1 = np.pad(x, [[pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]]], 'constant', constant_values=small)
    # crop on x1
    start_pos = np.ceil((np.asarray(x1.shape) - np.asarray(target_size))/2.0)##当目标更大时，填充情况因为向上取值X1有可能会比目标大小大1(0.5向丄取整)，就向前填补坐标
    start_pos = start_pos.astype(int)##若为裁剪，相差基数，向上取整，相当于前面多裁剪1 如17/2=9  9:xx 0到8被裁剪，即前面裁剪9
    y = x1[(shift[0]+start_pos[0]):(shift[0]+start_pos[0]+target_size[0]),
           (shift[1]+start_pos[1]):(shift[1]+start_pos[1]+target_size[1]),
           (shift[2]+start_pos[2]):(shift[2]+start_pos[2]+target_size[2])]
    
    return y 

# def tf_random_shift_image_int(image, label):
#     im_shape = image.shape
#     l_shape = label.shape
#     [image,label] = tf.py_function(shift_3d, [image, label], [tf.float32, tf.float32])##报错区域
#     image.set_shape(im_shape)
#     label.set_shape(l_shape)
#     return image, label


# def tf_random_patch(image, label, patch_size=(64,64,64), stride=(1,1,1)):
#     im_shape = patch_size + (1,)
#     [image,label] = tf.py_function(get_patch, [image, label, patch_size, stride], [tf.float32, tf.float32])
#     image.set_shape(im_shape)
#     label.set_shape(im_shape)
#     return image, label


class MyData(data.Dataset):
 
    def __init__(self, X ,Y, patch_size, stride=[1,1,1],augmentation=False):
        self.n_subject = X.shape[0]
        self.image_size = X.shape[1:4]
        self.patch_size = patch_size
        self.n_pixels = np.prod(patch_size)
        self.X = X
        self.Y = Y
        self.n_labels = int(np.max(Y)+1)###注意加int

        self.patch_edge = self.__calc_edge__()
        self.aug=augmentation
    def __len__(self):
        return len(self.X)
    def __calc_edge__(self):
        # index = []
        # start_pos = []

        n0 = self.image_size[0] - self.patch_size[0] + 1
        n1 = self.image_size[1] - self.patch_size[1] + 1
        n2 = self.image_size[2] - self.patch_size[2] + 1
        # index = np.array(np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2))).T.reshape(-1,3).astype(np.int8)
        # for n in range(self.n_subject):
        #     for pos in start_pos:
        #         index.append([n, pos])
        return (n0,n1,n2)
    
    def __get_patch(self, X, start_pos, patch_size):
        Y = X[start_pos[0]:start_pos[0] + patch_size[0],
            start_pos[1]:start_pos[1] + patch_size[1],
            start_pos[2]:start_pos[2] + patch_size[2]]
        return Y
    

    def __getitem__(self, idx):
        while True:
            selected_subject = np.random.randint(self.n_subject)
            index = np.random.randint(self.patch_edge)
            Y_patch = self.__get_patch(self.Y[selected_subject], index, self.patch_size)
            patch_probability = np.sum((Y_patch>0).astype(np.float))/(self.n_pixels).astype(np.float)/0.1
            ##(Y_patch>0)可以认为是取全脑区域
            #print(np.sum((Y_patch>0).astype(np.float)),(patch_probability>5))
            if patch_probability>0:#np.random.uniform() < patch_probability:##起码patch_probability必须大于0或者说不能太小，才能使用patch
                X_patch = self.__get_patch(self.X[selected_subject], index, self.patch_size)
                X_patch = X_patch[:,:,:,np.newaxis]
            
                Y_patch=torch.tensor(Y_patch).type(torch.int64)
                #print(Y_patch.shape)
                Y_patch=torch.nn.functional.one_hot(Y_patch, num_classes=self.n_labels)
                a=time.time()
                #print(Y_patch.shape)
                if self.aug:
                    Y_patch=Y_patch.numpy()

                    X_patch,Y_patch=flip_lr(X_patch,Y_patch)
                    X_patch,Y_patch=shift_3d(X_patch,Y_patch)
#                     X_patch,Y_patch=rot_3d(X_patch,Y_patch)
                print(time.time()-a,'s')


        return x_image, y_image




class Patch3DAndSeg_Data(data.IterableDataset):##数据量未知，使用IterableDataset，取patch这种合适
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
        augmentation: bool =False,
        patch_training: bool =True,
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
        self.patch_training=patch_training ##在val时设置为否，验证集使用滑动窗口计算
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

                self.examples += [(fname, num_slices, drf)]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]
        self.drf = drf

        self.image_size=volum.shape[0:3]##
        self.patch_size = [patch_size,patch_size,patch_size]
        self.n_pixels = np.prod([patch_size,patch_size,patch_size])
        self.aug = augmentation
        self.patch_edge = self.__calc_edge__()

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

    def __calc_edge__(self):
            # index = []
            # start_pos = []
            
            n0 = self.image_size[0] - self.patch_size[0] + 1
            n1 = self.image_size[1] - self.patch_size[1] + 1
            n2 = self.image_size[2] - self.patch_size[2] + 1
            # index = np.array(np.meshgrid(np.arange(n0), np.arange(n1), np.arange(n2))).T.reshape(-1,3).astype(np.int8)
            # for n in range(self.n_subject):
            #     for pos in start_pos:
            #         index.append([n, pos])
            return (n0,n1,n2)

    def _edge(self,image_size):
        n0 = image_size[0] -  self.patch_size[0] + 1
        n1 = image_size[1] -  self.patch_size[1] + 1
        n2 = image_size[2] -  self.patch_size[2] + 1
        return (n0,n1,n2)

    
    def __get_patch(self, X, start_pos, patch_size):
        Y = X[start_pos[0]:start_pos[0] + patch_size[0],
            start_pos[1]:start_pos[1] + patch_size[1],
            start_pos[2]:start_pos[2] + patch_size[2]]
        return Y
    

    
    
    def __iter__(self):
        # np.random.shuffle(self.patch_index)
        while True:

            n_subject=len(self.examples)
            # print('n_subject',n_subject)
            selected_subject = np.random.randint(n_subject)
            # print('self.patch_edge',self.patch_edge)
            
            fname, dataslice, drf = self.examples[selected_subject]

            with h5py.File(fname, "r") as hf:
                image_low = hf["lr"][()]
                image_target = hf["hr"][()]
                drf = hf["drf"][0]  # int
                lr_mean = hf["lr_mean"][0]
                ori_w = hf["ori_w"][0]
                ori_h = hf["ori_h"][0]  
                mask  =  hf['mask'][()]
                
            X_patch=image_low
            Y_patch=image_target
            # mask_name=str(fname).split('/')[-1].split('_')[2]       
            # Array=sitk.GetArrayFromImage(sitk.ReadImage('/data/xuyunzhi/data/Amax/suv_seg_mask/'+str(mask_name)+'.nii.gz'))
            # Array=zoom(Array,zoom=[1,256/Array.shape[1],256/Array.shape[2]],order=1)
            # Array=np.array(Array,dtype=np.int16)    
            
            # ##通过crop获取含有mask的patch:
            # if np.sum(Array)>1000:
            # X_patch,Y_patch,mask=crop_edge(image_low,image_target,Array,patch=[96,96,96])
            X_patch = X_patch[np.newaxis,:,:,:]
            Y_patch = Y_patch[np.newaxis,:,:,:]
            X_patch=torch.tensor(X_patch).type(torch.FloatTensor)
            Y_patch=torch.tensor(Y_patch).type(torch.FloatTensor)
            
            sample = {'lr': X_patch, 'hr': Y_patch, 'fname': fname.stem, 'slice_num': dataslice, 'drf': drf,
                'lr_mean': lr_mean, 'ori_w': ori_w, 'ori_h': ori_h,'mask':mask}
            # print('一次迭代')          
            yield sample