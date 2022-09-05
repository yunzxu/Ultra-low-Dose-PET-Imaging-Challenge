from pandas import date_range
import torch
import os
import time
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import glob
from pet.pytorch_ssim import ssim3D
from pet import SSIM
import torch.nn as nn
from pet import SSIM
import torch.nn as nn
from pet.models import SRResNet,EDSR,UNetPlusPlus_v1,UNETR
import argparse
import warnings

import pytorch_lightning as pl
import pathlib
import os
import pet
import h5py
from pet.data.pet_data import fetch_dir
from pet.pl_modules import PETDataModule, SRGANModule,EDSRModule,UNetPlusPlusModule,UNETRModule,ConvNext3DModule,UNETppModule,UNET3PlusModule,ConvNext3DStemModule,RRDBNet3DModule
import random
from monai.inferers import sliding_window_inference
from scipy.ndimage import zoom
from test_dicom_save import dcm_reader_, dcm_reader_,pre_zoom#array_to_dicom_diff,load_dicom_array
from utils_dicom import array_to_dicom_diff
from pathlib import Path

from suv_calculator import suv_calculator
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import SimpleITK as sitk
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"]="2"



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')


ssim=SSIM(window_size=11)
l1loss=nn.L1Loss()

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)



def dicom_sitkread(file_path):
    time0=time.time()
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path)
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(series_file_names)
    image3D = series_reader.Execute()
    #print('read',time.time()-time0,'s')
    array=sitk.GetArrayFromImage(image3D)
    return array

def read_dicom(dicom_path):
    reader = sitk.ImageSeriesReader()
    seriesIDs = reader.GetGDCMSeriesIDs(dicom_path)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_path, seriesIDs[0])
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    data = sitk.GetArrayFromImage(image) # z, y, x
    return data



def ssim_tensor(X,Y,axis3d=True,data_range=1):
    ssim_test3d=pet.SSIM3D(window_size=11)
    ssim_test2d=pet.SSIM(window_size=11)
    x0=X[np.newaxis,np.newaxis,...]
    y0=Y[np.newaxis,np.newaxis,...]
    x0=torch.tensor(x0).type(torch.float32).cuda()
    y0=torch.tensor(y0).type(torch.float32).cuda()
    if axis3d:
        ssim_result=ssim_test3d(x0,y0,data_range)

    else:
        ssim_result=ssim_test2d(x0,y0)
    return ssim_result

def psnr_tensor(X,Y):
    psnr_test=pet.PSNR()
    x0=torch.tensor(X).type(torch.float32)#.cuda()
    y0=torch.tensor(Y).type(torch.float32)#.cuda()
    psnr_result=psnr_test(x0,y0)
    return psnr_result


def inference_one(X,model):
    model=model.eval()
    ##test_aug
    # X_flip1=X[::-1,:,:]
    # X_flip2=X[:,::-1,:]
    # X_flip3=X[:,:,::-1]
    # X_flip1=torch.tensor(X_flip1[np.newaxis,np.newaxis,:,:,:].copy()).type(torch.FloatTensor).cuda()
    # X_flip2=torch.tensor(X_flip2[np.newaxis,np.newaxis,:,:,:].copy()).type(torch.FloatTensor).cuda()
    # X_flip3=torch.tensor(X_flip3[np.newaxis,np.newaxis,:,:,:].copy()).type(torch.FloatTensor).cuda()

    with torch.no_grad():
        x0=X[np.newaxis,np.newaxis,:,:,:]
        x0=torch.tensor(x0).type(torch.FloatTensor).to(device)
        y0=sliding_window_inference(x0, (96, 96, 96), 8, model,overlap=0.5,mode="gaussian",sigma_scale=0.05)
        y0=y0.cpu().numpy()

        # y_flip1=np.squeeze(sliding_window_inference(X_flip1, (96, 96, 96), 64, model).cpu().numpy())[::-1,:,:]
        # y_flip2=np.squeeze(sliding_window_inference(X_flip2, (96, 96, 96), 64, model).cpu().numpy())[:,::-1,:]
        # y_flip3=np.squeeze(sliding_window_inference(X_flip3, (96, 96, 96), 64, model).cpu().numpy())[:,:,::-1]


        result=np.squeeze(y0)
        #result=(result+y_flip1+y_flip2+y_flip3)/4
    return result

def psnr_my(predictions, targets):
    max_val = np.max(targets)
    mse = compare_mse(predictions,targets)#np.sum((predictions-targets)**2)/np.size(targets)
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr

def pre_process(X,Y,h=644,d=256,w=256):
    X,xh,xd,xw=pre_zoom(X,h,d,w)
    Y,yh,yd,yw=pre_zoom(Y,h,d,w)
    mean_lr=np.mean(X)
    mean_hr=np.mean(Y)
    X=X/mean_lr
    Y=Y/mean_hr
    return X,Y,mean_lr,mean_hr


def post_process(X,mean_lr,h=644,d=440,w=440):
    X,xh,xd,xw=pre_zoom(X,h,d,w)
    # Y,yh,yd,yw=pre_zoom(Y,644,440,440)
    X=X*mean_lr
    # Y=Y*mean_hr
    return X #,Y

def mse_tensor(X,Y):
    x0=X[np.newaxis,np.newaxis,...]
    y0=Y[np.newaxis,np.newaxis,...]
    x0=torch.tensor(x0).type(torch.float32)
    y0=torch.tensor(y0).type(torch.float32)
    result=F.mse_loss(x0,y0)
    return result















def run_inference_challenge(img_path,model=None,name='test',out_put_dir=None):
    
    time0=time.time()
    image3D=sitk.ReadImage(img_path)
    img=sitk.GetArrayFromImage(image3D)
    #target=np.array(dcm_reader_(target_path),dtype=np.float32)

    ##为了节约时间，只预测不算，所以target直接设置为0
    img_h,img_d,img_l=img.shape[0:3]
    target=np.ones([img_h,img_d,img_l])
    print(img_h,img_d,img_l)
    print('read data',time.time()-time0,'s')


    input=zoom(img,zoom=[644/img.shape[0],256/img.shape[1],256/img.shape[2]],order=3)

    input[input<=0]=0

    print('zoom',input.dtype,np.min(input),np.max(input))
    mean_lr=np.mean(input)
    input=input/mean_lr
    mean_hr=np.mean(target)
    print(mean_lr,mean_hr)
    print('pre zoom',time.time()-time0,'s')
    time0=time.time()

    pred=inference_one(input,model)
    print('inference time',time.time()-time0,'s')

    time0=time.time()
    # pred_final=post_process(pred,mean_lr,img_h,img_d,img_l)
    # pred_final=np.array(pred_final,dtype=img.dtype)

    pred_final=zoom(pred,zoom=[img_h/pred.shape[0],img_d/pred.shape[1],img_l/pred.shape[1]],order=3)

    pred_final[pred_final<=0]=0

    pred_final=pred_final#*mean_lr
    pred_final=np.array(pred_final,dtype=img.dtype)

    print('final',pred_final.dtype,np.min(pred_final),np.max(pred_final))
    pred_final[pred_final<=0]=0

    print('post',time.time()-time0,'s')


    time0=time.time()
    print(pred_final.dtype,pred_final.shape)

    save_name=str(name)
    print(save_name)

    pred_final=sitk.GetImageFromArray(pred_final)
    pred_final.SetDirection(image3D.GetDirection())##加载元信息，一般是这个三个，.GetDirection，.GetSpacing，.GetOrigin()
    pred_final.SetSpacing(image3D.GetSpacing())
    pred_final.SetOrigin(image3D.GetOrigin())
    sitk.WriteImage(pred_final, out_put_dir+'/'+str(save_name))
    #array_to_dicom_diff(img_path,pred_final,out_put_dir+'/'+str(save_name),'_pred')
    print('save',time.time()-time0,'s')



def challenge_test():
    model =UNET3PlusModule(
    summary_step=100,
    patch_size=96,
    ).to(device)
    dose=[10,100,2,20,4,50]

    checkpoint=torch.load('./model/UNET3plus_epoch-val_loss=0.18.ckpt',map_location=device)
  
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    dicom_path='/raid/data/lei_data/PET/Ultra-low-dose_PET_MICCAI_challenge/data_all_BQML/test_NII'
   # 'Q:\raid\data\lei_data\PET\Ultra-low-dose_PET_MICCAI_challenge\data_all_BQML\test_NII'


    output_dir='/raid/yunzhi_raid/data/Dicom_challenge/challenge_case/unet3p_new/' #'/raid/yunzhi_raid/data/Dicom_challenge/test_case/nii_20_case/RRDBNet/'
    files=sorted(list(Path(dicom_path).iterdir()))


    #files=['/raid/yunzhi_raid/data/Dicom_challenge/challenge_dicom_lr/30122021_2_20211230_170722','/raid/yunzhi_raid/data/Dicom_challenge/challenge_dicom_lr/Anonymous_ANO_20220225_1933412_110551']
    for file in files:
        fname=str(file).split('/')[-1]
        print('fname',fname)
        filedir=output_dir+str(fname)
        os.system('mkdir %s' %(str(filedir)))
        dicom_files=sorted(list(Path(file).iterdir()))
        index=0
        for dicom in dicom_files:
            name=str(dicom).split('/')[-1]
            index+=1
            # print(filedir)
            print(name)
            # print(dicom)
            # print(target_path)
            img_path=str(dicom)
            time0=time.time()
            run_inference_challenge(img_path,model=model,name=name,out_put_dir=filedir)
            print(time.time()-time0,'s')  


    

if __name__ == '__main__':

    time0=time.time()
   
    challenge_test()
   
    print(time.time()-time0,'s')

    

        


            









