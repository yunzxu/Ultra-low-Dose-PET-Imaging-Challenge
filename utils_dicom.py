import h5py
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib.style as style
import argparse
import h5py
from skimage import exposure
from pathlib import Path
import copy

def load_dicom_hu(dicom_dir: str):
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

    # get the tracer info
    ds = pydicom.dcmread(dicom_dir + '/' + files_slices[0])

    tracer = ds[0x00540016].value[0].Radiopharmaceutical  # in ['Fluorodeoxyglucose', 'FDG']
    # print(tracer.value[0].to_json_dict()['00180031']['Value'][0])

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

    return Img, RIntercepts, RSlopes, tracer


def load_dicom(dicom_dir: str):
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
    # get the tracer info
    ds = pydicom.dcmread(dicom_dir + '/' + files_slices[0])
    tracer = ds[0x00540016].value[0].Radiopharmaceutical  # in ['Fluorodeoxyglucose', 'FDG']
    # print(tracer.value[0].to_json_dict()['00180031']['Value'][0])

    for i in range(num_slices):
        ds = pydicom.dcmread(dicom_dir + '/' + files_slices[i])

        data.append(ds.pixel_array)
        location.append(get_projected_z_pos(ds))
    Img_loc, Img = zip(*sorted(zip(location, data), reverse=False))
    Img = np.asarray(Img)

    return Img, tracer


def dcm_reader(PathDicom, check_file=True):
    # get dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if check_file:
                if ".dcm" in filename.lower():  # check whether the file's DICOM
                    lstFilesDCM.append(os.path.join(dirName, filename))
            else:
                lstFilesDCM.append(os.path.join(dirName, filename))

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
        # store the raw image data
        Img.append(ds.pixel_array * RSlope + RIntercept)
        Img_loc.append(get_projected_z_pos(ds))
        RIntercepts.append(ds.RescaleIntercept)
        RSlopes.append(ds.RescaleSlope)
    Img_loc, Img, RIntercepts, RSlopes = zip(*sorted(zip(Img_loc, Img, RIntercepts, RSlopes), reverse=False))
    Img = np.asarray(Img)
    return Img, RIntercepts, RSlopes

def array_to_dicom_diff(in_dicom, data_file, out_dicom, series_desc_suffix):
    """

    :param in_dicom: templete dicom folder
    :param data_file:
    :param out_dicom:
    :param series_desc_suffix:
    :return:
    """
    print("print template files")
    # in_dicom_files = [os.path.join(in_dicom, f) for f in os.listdir(in_dicom) if f.endswith("IMA")]
    in_dicom_files = [os.path.join(in_dicom, f) for f in os.listdir(in_dicom) if f.endswith("IMA") or f.endswith('dcm')]
    list_dcm_ds = []  # dicom dataset len(list_dcm_ds)
    for f in in_dicom_files:
        try:
            list_dcm_ds.append(pydicom.read_file(f))  # each slice has its meta data
        except pydicom.errors.InvalidDicomError:
            pass

    series_number_offset = int(
        pydicom.read_file(in_dicom_files[0]).SeriesNumber) + 100  # same patient has same SeriesNumber for all slices
    list_sorted_dcm_ds = sorted(list_dcm_ds,
                                key=lambda dcm: get_projected_z_pos(dcm),
                                reverse=False)  # sort according to the z position
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
    pixel_data = np.array(pixel_data)
    pixel_data[pixel_data < t_min] = t_min
    pixel_data[pixel_data > t_max] = t_max
    out_data_set.PixelData = pixel_data.astype(data_type).tobytes()
    out_data_set.SeriesInstanceUID = series_uid
    out_data_set.SOPInstanceUID = sop_uid
    out_data_set.SeriesNumber += series_number_offset
    out_data_set.SeriesDescription += series_desc_suffix
    # for compressed data in PixelData
    from pydicom.uid import ExplicitVRLittleEndian
    out_data_set.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

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
    ipp = np.array([float(v) for v in dataset.ImagePositionPatient])  # x, y, and z coordinates of the upper left hand corner of the image; it is the center of the first voxel transmitted.
    iop_v1, iop_v2 =\
        np.array([float(v) for v in dataset.ImageOrientationPatient[:3]]),\
        np.array([float(v) for v in dataset.ImageOrientationPatient[3:]])
    norm_vec = np.cross(iop_v1, iop_v2)
    return np.dot(ipp, norm_vec)

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z


def dcm_reader_(PathDicom):
    # get dicom files
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".ima" in filename.lower():  # check whether the file's DICOM  '.dcm' or '.ima'
                lstFilesDCM.append(os.path.join(dirName, filename))

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
        filenameDCM_sorted = sort_list(lstFilesDCM, Img_loc)  # sort_index)
        for filenameDCM in filenameDCM_sorted:
            # read the file
            ds = pydicom.read_file(filenameDCM)
            # store the raw image data
            Img_.append(ds.pixel_array)
            Img = Img_
    Img = np.asarray(Img)
    return Img