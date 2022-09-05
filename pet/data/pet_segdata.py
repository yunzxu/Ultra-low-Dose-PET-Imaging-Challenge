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
import nibabel as nib

def restore_labels(x, labels):
    tmp = np.squeeze(np.argmax(x, -1)).astype(np.int8)
    y = np.zeros(tmp.shape, np.int8)
    n_labels = len(labels)
    for label_index in range(n_labels):
        y[tmp == label_index] = labels[label_index]
    return y



def save_nifit(data, filename):
    # print(data.dtype)
    img = nib.Nifti1Image(data, np.eye(4))
    nib.save(img, filename)





class Patch3D_SegData(data.IterableDataset):##数据量未知，使用IterableDataset，取patch这种合适
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        patch_size: int = 320,
        data_partition: str ='train',
        use_dataset_cache: bool = True,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        augmentation: bool =True,
        patch_training: bool =True,
        n_label=3,
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
        self.n_labels=n_label
        self.patch_training=patch_training ##在val时设置为否，验证集使用滑动窗口计算


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

                self.examples += [(fname, num_slices)]

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
        image= file['CT']
        label = file['label']
        return image, label

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
            
            fname, dataslice = self.examples[selected_subject]
            with h5py.File(fname, "r") as hf:
                image_ct = hf["CT"][()]
                image_target = hf["label"][()]
                image_pt = hf["Pet"][()]

            ##对CT，PT分别归一化操作：
            image_ct=image_ct/1024
            image_ct[image_ct<-1]=-1
            image_ct[image_ct>1]=1


            
            image_pt=image_pt/12
            image_pt[image_pt<0]=0
            image_pt[image_pt>1]=1

            ##不支持uint16,将label转为int6
            image_target=np.array(image_target,dtype=np.int16)
            #print(image_target.shape)

    

    
        
            image_size=image_ct.shape[0:3]
            h,d,l=image_size[0],image_size[1],image_size[2]
            if (image_size[0]<96) or  (image_size[1]<96)  or (image_size[2]<96):
                #print('wrong image',image_size)
                if h<96:
                    h=96
                if d<96:
                    d=96
                if l<96:
                    l=96
                image_size=[h,d,l]
                image_pt=crop_pad3D(image_pt,[h,d,l])
                image_ct=crop_pad3D(image_ct,[h,d,l])
                image_target=crop_pad3D(image_target,[h,d,l])

        #    #index = np.random.randint(self._edge(image_size))
        #     print('index',index)
            #print(self.patch_training)

         
            if self.patch_training :###对于训练的使用patch,对于val使用滑动窗口
                # print('index',index)
                # print(type(image_target))
                patch_probability=0
                while patch_probability<=0:
                    index = np.random.randint(self._edge(image_size))
                    #print('index',index)
                    Y_patch = self.__get_patch(image_target, index, self.patch_size)
                    patch_probability = np.sum((Y_patch>0).astype(np.float))/(self.n_pixels).astype(np.float)/0.1
                
                    #print(patch_probability)
                # if patch_probability>0 :#np.random.uniform() < patch_probability:##起码patch_probability必须大于0或者说不能太小，才能使用patch
                    #print("subject",selected_subject)
                    #X_ct = self.__get_patch(image_ct, index, self.patch_size)
                X_pt = self.__get_patch(image_pt, index, self.patch_size)

                X_patch =X_pt[:,:,:,np.newaxis] #np.stack((X_ct,X_pt),axis=0)考虑aug代码，必须先是通道在后
    

                Y_patch=torch.tensor(Y_patch).type(torch.int64)
                #print(Y_patch.shape)
                Y_patch=torch.nn.functional.one_hot(Y_patch, num_classes=self.n_labels)
                Y_patch=Y_patch.permute(3,0,1,2)#transpose(3,2).transpose(2,1).transpose(1,0) ##tensor的transpose只能2个维度交换
                if self.aug:
                    Y_patch=Y_patch.numpy()
        
                    X_patch,Y_patch=flip_lr(X_patch,Y_patch)
                    X_patch=torch.tensor(X_patch)
                    Y_patch=torch.tensor(Y_patch)
                    X_patch,Y_patch=shift_3d_int(X_patch,Y_patch)#####使用shift_3d_int里面的torch.roll运行速度极快
                
                X_patch=torch.tensor(X_patch).type(torch.FloatTensor)
                X_patch=X_patch.permute(3,0,1,2)
                Y_patch=Y_patch.type(torch.FloatTensor)
                # print('X_patch',X_patch.shape)
                #print(X_patch.shape,Y_patch.shape)
                sample = {'image': X_patch, 'label': Y_patch, 'fname': fname.stem}
                # print('一次迭代')          
                yield sample
            else:
                X_ct =image_ct
                X_pt =image_pt
                label=image_target
                ##存在某些label维度差一点的情况，因此设置一下：
                if (X_pt.shape[0]!=label.shape[0]) or (X_pt.shape[1]!=label.shape[1])  or (X_pt.shape[2]!=label.shape[2]):
                    target_shape=X_pt.shape[0:3]
                    label=crop_pad3D(label,target_shape)

                #print(image_ct.shape)
                # if X_ct.shape[0]==96 and X_ct.shape[1]==96 and X_ct.shape[2]==96:
                X_patch=X_pt[np.newaxis,:,:,:]
                #X_patch = np.stack((X_ct,X_pt),axis=0)
                Y_patch=torch.tensor(label).type(torch.int64)
                Y_patch=torch.nn.functional.one_hot(Y_patch, num_classes=self.n_labels)
                Y_patch=Y_patch.transpose(3,2).transpose(2,1).transpose(1,0)
                X_patch=torch.tensor(X_patch).type(torch.FloatTensor)
                Y_patch=torch.tensor(Y_patch).type(torch.FloatTensor)
                #print('X_patch',X_patch.shape)



                #print('val',X_patch.shape,Y_patch.shape)
                sample = {'image': X_patch, 'label': Y_patch, 'fname': fname.stem}
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
    if np.random.uniform() > 0.2:
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
    if np.random.uniform() > 0.5:
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


 
if __name__ == "__main__":

    data_path= '/raid/yunzhi_raid/data/hecktor2022/hdf5_space3/train/'

    dataset = Patch3D_SegData(
                root=data_path,
                sample_rate=None,
                volume_sample_rate=None,
                patch_size=96,####
                use_dataset_cache=False,
                patch_training=False,####对于val,使用完整图像测试，即patch_training=False
            )


    batch=next(iter(dataset))
    # {'image': X_patch, 'label': Y_patch, 'fname': fname.stem}

    img=batch['image'].numpy()
    label=batch['label'].numpy()
    fname=batch['fname']
    print(img.shape,label.shape,fname)

    label=label.transpose(1,2,3,0)
    print(label.shape)
    label=restore_labels(label,range(3))
    save_nifit(img[0],'/raid/yunzhi_raid/Low_dose_PET_5/pet/data/tempCT.nii.gz')
    # save_nifit(img[1],'/raid/yunzhi_raid/Low_dose_PET_5/pet/data/imagePT.nii.gz')
    save_nifit(label,'/raid/yunzhi_raid/Low_dose_PET_5/pet/data/templabel.nii.gz')