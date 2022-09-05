import argparse
import torch
from torch.autograd import Variable
from scipy.ndimage import zoom
from imageio import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
import scipy.io as sio#sio.loadmat
import h5py
# from dataset import DatasetFromHdf5
import SimpleITK as sitk
import os
# from helper.utils import read_nii, read_h5
from torchvision.transforms import ToTensor
# from skimage.measure import compare_ssim as ssim
import copy
import pydicom
from pydicom.encaps import encapsulate
from pydicom.uid import JPEG2000
from imagecodecs import jpeg2k_encode


parser = argparse.ArgumentParser(description="PyTorch Dense_Unet Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--slice_num", type=int, default=1, help="slice number")
parser.add_argument("--chan_in", type=int, default=1, help="channel number into the net")
parser.add_argument("--chan_out", type=int, default=1, help="channel number out of the net")
parser.add_argument("--th_mask", type=int, default=30, help="threshold value for getting the mask")
parser.add_argument("--image_size", type=int, default=256, help="image size when zoom")
parser.add_argument("--net", default="UNet", type=str, help="network")

parser.add_argument("--test_path", type=str, default="/raid/lei_raid/projects/FastMRI_UW/experiments/train/sub_33_in.nii.gz", help="test_path")
parser.add_argument("--LR_T1_path", type=str, default="C:\DeepLearning\PET_CT_v2_3_1_demo_lei\data\lr", help="LR_T1_path")
parser.add_argument("--LR_T2_path", type=str, default="C:\DeepLearning\PET_CT_v2_3_1_demo_lei\data\lr", help="LR_T2_path")
parser.add_argument("--HR_T1_path", type=str, default="C:\DeepLearning\PET_CT_v2_3_1_demo_lei\data\Full_dose", help="HR_T1_path")
parser.add_argument("--HR_T2_path", type=str, default="C:\DeepLearning\PET_CT_v2_3_1_demo_lei\data\Full_dose", help="HR_T2_path")
parser.add_argument("--SR_T1_path", type=str, default="C:\DeepLearning\PET_CT_v2_3_1_demo_lei\data\sr", help="SR_T1_path")
parser.add_argument("--SR_T2_path", type=str, default="C:\DeepLearning\PET_CT_v2_3_1_demo_lei\data\sr", help="SR_T2_path")


parser.add_argument('--dataset', default="SR_T1_registered ", type=str, help="which dataset")
parser.add_argument("--long_skip", action='store_true', help="Use long skip connection")
parser.add_argument("--zoom", action='store_true', help="apply zoom when inference")
parser.add_argument("--save_hi_re", action='store_true', help="save dicom with high resolution")
args = parser.parse_args(args=[])

def register_im(im_fixed, im_moving, param_map=None, verbose=True, im_fixed_spacing=None, im_moving_spacing=None, max_iter=200):
    '''
    Image registration using SimpleElastix.
    Register im_moving to im_fixed
    '''

    default_transform = 'translation'

    sim0 = sitk.GetImageFromArray(im_fixed)
    sim1 = sitk.GetImageFromArray(im_moving)

    if im_fixed_spacing is not None:
        sim0.SetSpacing(im_fixed_spacing)

    if im_moving_spacing is not None:
        sim1.SetSpacing(im_moving_spacing)

    if param_map is None:
        if verbose:
            print("using default '{}' parameter map".format(default_transform))
        param_map = sitk.GetDefaultParameterMap(default_transform)

    param_map['MaximumNumberOfIterations'] = [str(max_iter)]

    ef = sitk.ElastixImageFilter()
    ef.SetFixedImage(sim0)
    ef.SetMovingImage(sim1)
    ef.SetParameterMap(param_map)

    if verbose:
        print('image registration')
        tic = time.time()

    # TODO: Set mask for registration by using ef.SetFixedMask(brain_mask)
    ef.Execute()

    if verbose:
        toc = time.time()
        print('registration done, {:.3} s'.format(toc - tic))

    sim_out = ef.GetResultImage()
    param_map_out = ef.GetTransformParameterMap()

    im_out = sitk.GetArrayFromImage(sim_out)
    im_out = np.clip(im_out, 0, im_out.max())

    return im_out, param_map_out



def load_dicom_array(dicom_dir: str):
    
    """"

    load volume data from DICOM directory int a int16 array

    """

    # num of slices in one volume
    num_slices = len(os.listdir(dicom_dir))
    # list the names of slices
    files_slices = os.listdir(dicom_dir)
    # read out the data and axis location, then sorted
    data = []
    location = []
    RIntercepts = []
    RSlopes = []
    ds = pydicom.dcmread(dicom_dir + '/' + files_slices[0])
    for i in range(num_slices):
        ds = pydicom.dcmread(dicom_dir + '/' + files_slices[i])
        RIntercept = ds.RescaleIntercept
        RSlope = ds.RescaleSlope
        data.append(ds.pixel_array * RSlope + RIntercept)
        location.append(get_projected_z_pos(ds))
        RIntercepts.append(ds.RescaleIntercept)
        RSlopes.append(ds.RescaleSlope)
    Img_loc, Img, RIntercepts, RSlopes = zip(*sorted(zip(location, data, RIntercepts, RSlopes), reverse=False))
    Img = np.asarray(Img)
    return Img

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)
    

def PSNR_3D(pred, gt, img_3D_T1):
    pred_mask = pred[img_3D_T1>=0]
    gt_mask = gt[img_3D_T1>=0]
    m_pred = np.max(abs(pred_mask[:]))-np.min(abs(pred_mask[:]))
    m_gt = np.max(abs(gt_mask[:]))-np.min(abs(gt_mask[:]))
    L = max(m_pred,m_gt)  
#    print('max L is ' + str(L))
    imdff = pred - gt 
    rmse = math.sqrt(np.mean(imdff**2))
    if rmse == 0:
        return 100
    psnr = 10* math.log10(L**2/np.mean(imdff**2))
    return psnr,rmse

def dcm_reader(PathDicom):
    # get dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    
    # load images
    Img = []
    Img_loc = []
    RIntercepts = []
    RSlopes = []
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)
        ax_dir = ds.PatientPosition[0]
#         if ax_dir is 'F':
#             ax_reverse = False
#         else:
#             ax_reverse = True

        RIntercept = ds.RescaleIntercept
        RSlope = ds.RescaleSlope
#         store the raw image data
        Img.append(ds.pixel_array*RSlope+RIntercept)
        Img_loc.append(get_projected_z_pos(ds))
        RIntercepts.append(ds.RescaleIntercept)
        RSlopes.append(ds.RescaleSlope)
    Img_loc, Img, RIntercepts, RSlopes = zip(*sorted(zip(Img_loc, Img, RIntercepts, RSlopes), reverse=False))
    Img = np.asarray(Img)
    return Img, RIntercepts, RSlopes 

def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 





def dcm_reader_(PathDicom):
    # get dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".ima" or '.dcm' in filename.lower():  # check whether the file's DICOM  '.dcm' or '.ima'
                lstFilesDCM.append(os.path.join(dirName,filename))
    # load images
    Img = []
    Img_ = []
    Img_loc = []

#    RIntercepts = []
#    RSlopes = []
    for filenameDCM in lstFilesDCM:
        # read the file
        ds = pydicom.read_file(filenameDCM)
#        ax_dir = ds.PatientPosition[0]
#         if ax_dir is 'F':
#             ax_reverse = False
#         else:
#             ax_reverse = True

#        RIntercept = ds.RescaleIntercept
#        RSlope = ds.RescaleSlope
#         store the raw image data
        Img.append(ds.pixel_array)
        Img_loc.append(get_projected_z_pos(ds))
        tmp = ds.pixel_array
#        RIntercepts.append(ds.RescaleIntercept)
#        RSlopes.append(ds.RescaleSlope)

    try:
        Img_loc, Img = zip(*sorted(zip(Img_loc, Img), reverse=False))
    except:
        filenameDCM_sorted = sort_list(lstFilesDCM, Img_loc)#sort_index)
        for filenameDCM in filenameDCM_sorted:
            # read the file
            ds = pydicom.read_file(filenameDCM)
            # store the raw image data
            Img_.append(ds.pixel_array) 
            Img = Img_
    Img = np.asarray(Img)
    return Img


def array_to_dicom_diff(in_dicom, data_file, out_dicom, series_desc_suffix):
    print("print template files")
    in_dicom_files = [os.path.join(in_dicom, f) for f in os.listdir(in_dicom) ]##并非所有都是dcm,if f.endswith("dcm")
    list_dcm_ds = []
    for f in in_dicom_files:
        try:
            list_dcm_ds.append(pydicom.read_file(f))
        except pydicom.errors.InvalidDicomError:
            pass

    series_number_offset = int(
        pydicom.read_file(in_dicom_files[0]).SeriesNumber) + 100
    list_sorted_dcm_ds = sorted(list_dcm_ds,
                                key=lambda dcm: get_projected_z_pos(dcm),
                                reverse=False)
    print('reading pixel data...')
    os.makedirs(out_dicom, exist_ok=True)
    print('writing output DICOM...')
    save_series_with_template(data_file, list_sorted_dcm_ds, out_dicom,
                              series_number_offset, series_desc_suffix)

def save_series_with_template(pixel_data, templates, out_dir,
                              series_number_offset,
                              series_desc_suffix):
    """save pixel data to DICOM according to a template"""
    print (pixel_data.shape, len(templates))
    assert len(pixel_data.shape) == 3 and pixel_data.shape[0] == len(templates)
    series_uid = pydicom.uid.generate_uid()
    uid_pool = set()
    uid_pool.add(series_uid)
    for i, data_set in enumerate(templates):
        uid_pool = save_slice_with_template(pixel_data[i], data_set, out_dir, i,
                                            series_number_offset,
                                            series_desc_suffix,
                                            series_uid, uid_pool)


def save_slice_with_template(pixel_data, template_dataset, out_dir, i_slice,
                             series_number_offset, series_desc_suffix,
                             series_uid, uid_pool):
    assert len(pixel_data.shape) == 2
    out_data_set = copy.deepcopy(template_dataset)
    sop_uid = pydicom.uid.generate_uid()
    while sop_uid in uid_pool:
        sop_uid = pydicom.uid.generate_uid()
    uid_pool.add(sop_uid)
    data_type = template_dataset.pixel_array.dtype
    # resolve the bits storation issue
    bits_stored = template_dataset.get("BitsStored", 16)
    if template_dataset.get("PixelRepresentation", 0) != 0:
        # signed
        t_min, t_max = (-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1)
    else:
        # unsigned
        t_min, t_max = 0, (1 << bits_stored) - 1
    pixel_data = np.array(pixel_data).copy()


    
    # # jpeg2k_encode to perform JPEG2000 compression
    # arr_jpeg2k = jpeg2k_encode(pixel_data)
    # # convert from bytearray to bytes before saving to PixelData
    # arr_jpeg2k = bytes(arr_jpeg2k)

    # # ds.PixelData = encapsulate([arr_jpeg2k])




    pixel_data[pixel_data < t_min] = t_min
    pixel_data[pixel_data > t_max] = t_max
    out_data_set.PixelData = pixel_data.astype(data_type).tostring()
    out_data_set.SeriesInstanceUID = series_uid
    out_data_set.SOPInstanceUID = sop_uid
    out_data_set.SeriesNumber += series_number_offset
    out_data_set.SeriesDescription += series_desc_suffix
    out_path = os.path.join(out_dir, 'IMG_{:06d}.dcm'.format(i_slice))
    out_data_set.save_as(out_path)
    return uid_pool

def get_projected_z_pos(dataset: pydicom.Dataset):
    """Calculate the projected vertical location
    Assumes ImagePositionPatient and ImageOrientationPatient exist in the
    dataset object
    :param dataset: the pydicom.Dataset object containing info
    :return: the projected z position
    """
    ipp = np.array([float(v) for v in dataset.ImagePositionPatient])
    iop_v1, iop_v2 =\
        np.array([float(v) for v in dataset.ImageOrientationPatient[:3]]),\
        np.array([float(v) for v in dataset.ImageOrientationPatient[3:]])
    norm_vec = np.cross(iop_v1, iop_v2)
    return np.dot(ipp, norm_vec)
            
def pre_process(args, img):
    """
    resize image to [*, 256, 256] and normalize by mean
    """
    shape_zoom = [args.image_size, args.image_size]
    mean_img = np.mean(img)
    img = img/mean_img
    if args.zoom:
        img_reshape = np.zeros([img.shape[0], shape_zoom[0], shape_zoom[1]])
        zoom_factor = [shape_zoom[0]/(img.shape[1]+0.0), shape_zoom[1]/(img.shape[2]+0.0)]
        for i in range(img.shape[0]):
            img_reshape[i,:,:] = zoom(
                np.squeeze(img[i,:,:]), 
                zoom_factor,
                order=3)
    else:
        img_reshape = img
    ori_1, ori_w, ori_h = img.shape
    return img_reshape, mean_img, ori_w, ori_h

def pre_zoom(img_move, tar_1, tar_w, tar_h):
    """
    3D zoom image
    """
    shape_zoom = [tar_1, tar_w, tar_h]
    zoom_factor = [shape_zoom[0]/(img_move.shape[0]+0.0), shape_zoom[1]/(img_move.shape[1]+0.0), shape_zoom[2]/(img_move.shape[2]+0.0)]
    img_reshape = zoom(
            img_move, 
            zoom_factor,
            order=3)
    ori_1, ori_w, ori_h = img_move.shape
    return img_reshape, ori_1, ori_w, ori_h
            
def post_process(args, img, mean_img, target_h, target_w):
    """
    rescale back to original shape, and rescale back by mean
    """
    if args.zoom:
        shape_zoom = [target_h, target_w]
        img_reshape = np.zeros([img.shape[0], shape_zoom[0], shape_zoom[1]])
        zoom_factor = [shape_zoom[0]/(img.shape[1]+0.0), shape_zoom[1]/(img.shape[2]+0.0)]
        for i in range(img.shape[0]):
            img_reshape[i,:,:] = zoom(
                np.squeeze(img[i,:,:]), 
                zoom_factor,
                order=3)
    else:
        img_reshape = img
    img_result = img_reshape * mean_img
    print('==> After reshapeed back, shape is {}'.format(img_result.shape))
    return img_result    
    
def inference(args, LR_T1, LR_T2):
    args.eval = True
    args.cuda = True
    args.saveimg = True #True
    iters_all = [100]
    height, width = args.image_size, args.image_size

#    low_img = read_nii(low_path)
#    full_img = read_nii(full_path)
    print('cude is '+str(args.cuda))    
    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    img_3D_low_T1 = np.array(np.squeeze(LR_T1)).astype(np.float32)  # squeeze out dimension which is 1
    img_3D_low_T2 = np.array(np.squeeze(LR_T2)).astype(np.float32)
    [img_height,img_width,img_deepth] = img_3D_low_T1.shape
    for iters in iters_all:#[50,60,80,90,95]:
        img_3D_Pre_T1 = np.zeros(img_3D_low_T1.shape).astype(np.float32)
        img_3D_Pre_T2 = np.zeros(img_3D_low_T2.shape).astype(np.float32)
        
        mask_global = np.zeros(img_3D_low_T1.shape).astype(np.float32)
        if not args.long_skip:
            model_path = './experiments/model/'+str(args.dataset)+'_'+str(args.net)+'/model_epoch_' +str(iters)+'.pth'
        elif args.long_skip:
            model_path = './experiments/model/'+str(args.dataset)+'_'+str(args.net)+'_long_skip/model_epoch_' +str(iters)+'.pth'
        model = torch.load(model_path)['model']
        if args.eval==True:
            model.eval()
            model.half()
        print('==> model is loaded')
        slice_height = args.slice_num
        slice_height_half = int((slice_height - 1)/2)
        start_time = time.time()
        
        for i in range(slice_height - slice_height_half-1,(img_3D_low_T1.shape[0]-slice_height_half)):
            img_slice_low_T1 = img_3D_low_T1[(i-slice_height_half):(i+slice_height_half+1),:,:]
            img_slice_low_T2 = img_3D_low_T2[(i-slice_height_half):(i+slice_height_half+1),:,:]
            [im_height, im_w, im_h] = img_slice_low_T1.shape
            img_slice_3D_add_T1 = np.zeros([slice_height, height, width],dtype='float32')
            img_slice_3D_add_T2 = np.zeros([slice_height, height, width],dtype='float32')
            img_slice_3D_add_T1[0:slice_height,0:im_w,0:im_h] = img_slice_low_T1
            img_slice_3D_add_T2[0:slice_height,0:im_w,0:im_h] = img_slice_low_T2
            im_input = np.array(np.concatenate((img_slice_3D_add_T1, img_slice_3D_add_T2), 0))

            im_input = Variable(torch.from_numpy(im_input).contiguous().float()).view(1, -1, im_input.shape[1], im_input.shape[2])
            if args.cuda:
                model = model.cuda()
                im_input = im_input.half().cuda()
        
            start_time_slice = time.time()
            out = model(im_input)
            elapsed_time_slice = time.time() - start_time_slice           
            OutputImage = out.cpu().data[0].numpy()#.astype(np.float64)#.astype(np.float64)#maybe 1 channel 
            OutputImage = OutputImage.clip(0, 1000.0)

            # mask for average
            mask_local = np.ones([slice_height,img_width,img_deepth])
            mask_global[(i - slice_height_half):(i + slice_height_half+1),:,:] = \
                mask_global[(i - slice_height_half):(i + slice_height_half+1),:,:] + mask_local 
            # for recon
            img_3D_Pre_T1[(i - slice_height_half) : (i + slice_height_half+1),:,:] = \
                img_3D_Pre_T1[(i - slice_height_half) : (i + slice_height_half+1),:,:] + OutputImage[0,0:im_w,0:im_h]
            img_3D_Pre_T2[(i - slice_height_half) : (i + slice_height_half+1),:,:] = \
                img_3D_Pre_T2[(i - slice_height_half) : (i + slice_height_half+1),:,:] + OutputImage[1,0:im_w,0:im_h]
            if i %50 ==0:
                print('===> slice ' +str(i) +' iter ' +str(iters) +' is Done !!! ')
        # average
        img_3D_Pre_T1 = img_3D_Pre_T1 / mask_global
        img_3D_Pre_T2 = img_3D_Pre_T2 / mask_global
        img_3D_Pre_T1 = np.array(img_3D_Pre_T1).astype(np.float32)
        img_3D_Pre_T2 = np.array(img_3D_Pre_T2).astype(np.float32)
        elapsed_time = time.time() - start_time
        print('Time for one subjefull: ' +str(elapsed_time) + '  for one slice: ' +str(elapsed_time_slice))
#        psnr,rmse = PSNR_3D(img_3D_Pre_full,img_3D_full,img_3D_full)
#        ssim_value = ssim(img_3D_Pre_full,img_3D_full)
#        print("PSNR {0:0.2f},  RMSE {1:0.4f}, SSIM {2:0.4f}".format(psnr, rmse, ssim_value))
        ##save nii
        saveFolder = './experiments/result/'+str(args.dataset)+'_'+str(args.net)+'/'
        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)
        if args.saveimg:
            name = 'low_T1'
            img1_nii = sitk.GetImageFromArray(img_3D_low_T1)              
            sitk.WriteImage(img1_nii,os.path.join(saveFolder,name+'.nii.gz')) 
            
            name = 'pre_T1'
            img1_nii = sitk.GetImageFromArray(np.float32(img_3D_Pre_T1))      
            sitk.WriteImage(img1_nii,os.path.join(saveFolder,name+'.nii.gz')) 
            
            name = 'low_T2'
            img1_nii = sitk.GetImageFromArray(img_3D_low_T2)              
            sitk.WriteImage(img1_nii,os.path.join(saveFolder,name+'.nii.gz')) 
            
            name = 'pre_T2'
            img1_nii = sitk.GetImageFromArray(np.float32(img_3D_Pre_T2))      
            sitk.WriteImage(img1_nii,os.path.join(saveFolder,name+'.nii.gz'))             
            
        return img_3D_Pre_T1, img_3D_Pre_T2
         
def demo_inference(args):
    start_time = time.time()
    
    # case_path_T1 = '/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T1_BRAVO_SUBTLE_SRE_8'
    # better_case_path_T1 = '/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T1_BRAVO_SUBTLE_SRE_8'
    # dcm_out_folder_T1 = './experiments/result/15280_preT1_T1toT2+'
    
    # case_path_T2 = '/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T2_FLAIR_3D_SUBTLE_SRE_9'
    # better_case_path_T2 = '/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T2_FLAIR_3D_4'
    # dcm_out_folder_T2 = './experiments/result/15280_preFLAIR_T1toT2+'

    case_path_T1 = args.LR_T1_path #'/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T1_BRAVO_SUBTLE_SRE_8'
    better_case_path_T1 = args.HR_T1_path#'/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T1_BRAVO_SUBTLE_SRE_8'
    dcm_out_folder_T1 = args.SR_T1_path #'./experiments/result/15280_preT1_T1toT2+'
    
    case_path_T2 = args.LR_T2_path #'/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T2_FLAIR_3D_SUBTLE_SRE_9'
    better_case_path_T2 = args.HR_T2_path #'/raid/SubtleMR/UW_VWI/Multi_sequences/data/Radnet/15280/Mr_Brain_Wo - 15280/SAG_T2_FLAIR_3D_4'
    dcm_out_folder_T2 = args.SR_T2_path #'./experiments/result/15280_preFLAIR_T1toT2+'

    
    LR_T1 = dcm_reader_(case_path_T1)
    print('LR T1 image shape is {}'.format(LR_T1.shape))
    LR_T2 = dcm_reader_(case_path_T2)
    print('LR T2 image shape is {}'.format(LR_T2.shape))
    HR_T1 = dcm_reader_(better_case_path_T1)
    print('HR T1 image shape is {}'.format(HR_T1.shape))   
    HR_T2 = dcm_reader_(better_case_path_T2)
    print('HR T2 image shape is {}'.format(HR_T2.shape))    

    # rescale T1 to T2
    LR_T1,  oriT1_l, oriT1_w, oriT1_h = pre_zoom(LR_T1, LR_T2.shape[0], LR_T2.shape[1],  LR_T2.shape[2])# LR_T1 zoomed, LR_T2 as reference
    # zoom image to [x, 512, 512]
#    Img_norm, Img_mean, ori_w, ori_h = pre_process(args, Img) # 'zoom to [x,512,512] and norm by mean'
#    print('original image shape: {0}, after process shape: {1}'.format(Img.shape, Img_norm.shape))
#    Img_norm = np.float32(Img_norm)
#    Img_norm[Img_norm<0] = 0
    
    # LR_T1_img, _ = register_im(LR_T2, LR_T1)# LR_T2 fixed, LR_T1 moved
    LR_T1_img = LR_T1  #
    LR_T2_img = LR_T2  # (0, 32767)/29.73
    LR_T1_img_mean, LR_T2_img_mean = np.mean(LR_T1_img), np.mean(LR_T2_img)
    breakpoint()
    LR_T1_img, LR_T2_img = LR_T1_img/LR_T1_img_mean, LR_T2_img/LR_T2_img_mean  # (0, 1101.8)

    # inference
    rec_img_T1, rec_img_T2 = inference(args, LR_T1_img, LR_T2_img)
    
    # rescale back,T2
    Img_o_T2 = post_process(args, rec_img_T2, LR_T2_img_mean, LR_T2_img.shape[1], LR_T2_img.shape[2])
    array_to_dicom_diff(case_path_T2, Img_o_T2, dcm_out_folder_T2, "_inf_mul_simel+_EDSR_v4_T1toT2_dicom_test_v2")

    # rescale back,T1
    Img_o_T1 = post_process(args, rec_img_T1, LR_T1_img_mean, LR_T2_img.shape[1], LR_T2_img.shape[2])
    Img_o_T1,  _, _, _= pre_zoom(Img_o_T1,   oriT1_l, oriT1_w, oriT1_h )
    array_to_dicom_diff(case_path_T1, Img_o_T1, dcm_out_folder_T1, "_inf_mul_simel+_EDSR_v4_T1toT2_dicom_test_v2")    
    # HR_T2
#    array_to_dicom_diff(case_path_T2, T2_img_reg, dcm_out_folder_T2_HR, "_T2_reg")
    # LR_T2
#    array_to_dicom_diff(case_path_T2, Img_o_T2_LR, dcm_out_folder_T2_LR, "_T2_LR_reg")

    elapsed_time = time.time() - start_time
    print('===> Inference is Done!!! Cost time: {}'.format(elapsed_time))      

            
if __name__ == '__main__':
#    test_demo(args)
    breakpoint()
    demo_inference(args)
