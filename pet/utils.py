import numpy as np
import imageio
import torch
import matplotlib.pyplot as plt
import matplotlib.style as style
import argparse
import h5py
from skimage import exposure
import skimage.util

def img_show(img, cmap='gray'):
    p_lo, p_hi = np.percentile(img, (0, 99.5))
    img_rescale_1 = exposure.rescale_intensity(img, in_range=(p_lo, p_hi))

    plt.imshow(img_rescale_1, cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    return


def plot_scans(original, recon):
    plt.figure(figsize=(6, 6), dpi=100)
    plt.subplot(1, 2, 1)
    plt.title('HR', fontsize=14)
    img_show(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('LR', fontsize=14)
    img_show(recon, cmap='gray')
    plt.axis('off')

    plt.show()
    plt.savefig('result' + '.png')

def img_save(img, title, idx, save_dir):

    plt.figure(figsize=(6, 6), dpi=100)

    p_lo, p_hi = np.percentile(img, (0, 99.5))
    # print(title + "p_lo = ", p_lo)
    # print(title + "p_hi = ", p_hi)
    # .cpu().detach().numpy()
    img_rescale_1 = exposure.rescale_intensity(img[0].numpy(), in_range=(p_lo, p_hi))

    plt.imshow(img_rescale_1, cmap=plt.cm.gray)

    plt.xticks([])
    plt.yticks([])
    plt.title(title, fontsize=14)
    plt.axis('off')
    # save_name = save_dir / ('{}.png'.format(batch_size * index + i))
    # imageio.imwrite(str(save_name),img_to_save[0, :, :].astype(np.uint8))
    plt.savefig(str(save_dir) + '/train_recon_' + title + '_' + str(idx) + '.png')
    return

def save_recon(img_lr, img_sr, img_hr, index, save_dir, error_scale=1,do_save=True):
    """ Save the reconstruction result and compare with reference

    Parameters:
    ----------

    index: int
        index of the current batch
    save_dir: pathlib.Path
        location to save the image
    do_save: bool
        to write the image to save_dir or not (use during inference)
    error_scale: float
        how much to magnify the error map

    Outputs:
    ----------
    None. The magnitude image of the slice are saved at save_dir
    """
    batch_size = img_lr.shape[0]
    img_lr_np = img_lr.detach().cpu()
    img_sr_np = img_sr.detach().cpu()
    img_hr_np = img_hr.detach().cpu()

    for i in range(batch_size):
        img_lr_mag = img_lr_np[i] / img_lr_np.max()
        img_sr_mag = img_sr_np[i] / img_sr_np.max()
        img_hr_mag = img_hr_np[i] / img_hr_np.max()

        diff = error_scale*np.abs(img_hr_mag - img_sr_mag)
        # p_lo, p_hi = np.percentile(img_lr_np[i], (0, 99.5))
        # print("lr p_lo = ", p_lo)
        # print("lr p_hi = ", p_hi)
        #
        # p_lo, p_hi = np.percentile(img_hr_np[i], (0, 99.5))
        # print("hr p_lo = ", p_lo)
        # print("hr p_hi = ", p_hi)
        #
        # p_lo, p_hi = np.percentile(img_sr_np[i], (0, 99.5))
        # print("sr p_lo = ", p_lo)
        # print("sr p_hi = ", p_hi)
        #
        # p_lo, p_hi = np.percentile(diff, (0, 99.5))
        # print("diff p_lo = ", p_lo)
        # print("diff p_hi = ", p_hi)
        diff = diff / diff.max()

        img_to_save = np.concatenate((img_lr_mag,img_sr_mag,img_hr_mag,diff), axis=2)

        if do_save:
            idx = batch_size*index + i
            img_save(img_lr_mag, 'lr', idx, save_dir)
            img_save(img_sr_mag, 'sr', idx, save_dir)
            img_save(img_hr_mag, 'hr', idx, save_dir)
            img_save(diff, 'diff', idx, save_dir)

            # save_name = save_dir / ('{}.png'.format(batch_size * index + i))
            # imageio.imwrite(str(save_name),img_to_save[0, :, :].astype(np.uint8))

    return img_to_save