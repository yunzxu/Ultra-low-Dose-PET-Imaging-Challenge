from argparse import ArgumentParser
from math import sqrt, ceil
import time
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms
from pathlib import Path

import pytorch_lightning as pl
import matplotlib.pyplot as plt

from pet.models import UNetPlusPlus_v1
import pet
import numpy as np


class UNetPlusPlusModule(pl.LightningModule):
    """
    UNet++ training module.
    This can be used to train UNet++ networks from the paper:
    UNet++: A Nested U-Net Architecture for Medical Image Segmentation
    code refer to https://towardsdatascience.com/biomedical-image-segmentation-unet-991d075a3a4b
    """
    def __init__(
        self,
        summary_step: int = 1,
        patch_size: int = 4,
        n1: int = 64,
        lr: float = 1e-4,
        lr_step_size: float = 2e+5,
        lr_gamma: float = 0.5,
        **kwargs,
    ):
        """
        Args:
            summary_step: print out summary
            patch_size: center crop in image
            f_chans: number of channels of features
            n_blocks: number of residual cnn blocks
            res_scale: scale the output of each residual block
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.summary_step = summary_step
        self.patch_size = patch_size

        self.n1 = n1

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        # networks
        self.net = UNetPlusPlus_v1(
            in_channels=1,
            out_channels=1,
            n1=self.n1,
        )

        # training criterions
        self.criterion_L1 = nn.L1Loss()

        # validation metrics
        self.criterion_PSNR = pet.PSNR()
        self.criterion_SSIM = pet.SSIM(window_size=11)  # pytorch-ssim

    def forward(self, input):
        return self.net(input)

    def training_step(self, batch, batch_nb):
        img_lr = batch['lr']
        img_hr = batch['hr']

        self.img_sr = self.forward(img_lr)

        l1_loss = self.criterion_L1(self.img_sr, img_hr)
        ssim_loss = self.criterion_SSIM(self.img_sr, img_hr)
        content_loss = l1_loss + (1 - ssim_loss)

        self.log("train_loss", content_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_l1_loss", l1_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_ssim_loss", ssim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        # train_accum_logs = {"train_content_loss": content_loss, 'train_l1_loss': l1_loss, "train_ssim_loss": ssim_loss}

        return {'loss': content_loss,
                'img_lr': img_lr,
                'img_hr': img_hr,
                # 'prog': {'tng/l1_loss': l1_loss,
                #          'tng/ssim_loss': ssim_loss,},
                # 'log': train_tensorboard_logs,
                }

    def training_step_end(self, step_output):

        if self.global_step % self.summary_step == 0:
            # nrow = ceil(sqrt(self.summary_step))  img_lr=tensor(b,c,h,w)cuda
            # sample_img = pet.save_recon(img_lr, self.img_sr, img_hr, batch_nb, Path("train_result"), 1, False)
            # self.logger.experiment.add_image('sample_recon', sample_img, self.global_step)
            img_lr = step_output['img_lr']
            img_hr = step_output['img_hr']
            nrow = ceil(sqrt(img_lr.shape[0]))
            self.logger.experiment.add_image(
                tag='train/lr_img',
                img_tensor=make_grid(img_lr/img_lr.max(), nrow=nrow, padding=0),
                global_step=self.global_step
            )
            self.logger.experiment.add_image(
                tag='train/hr_img',
                img_tensor=make_grid(img_hr/img_hr.max(), nrow=nrow, padding=0),
                global_step=self.global_step
            )
            self.logger.experiment.add_image(
                tag='train/sr_img',
                img_tensor=make_grid(self.img_sr/self.img_sr.max(), nrow=nrow, padding=0),
                global_step=self.global_step
            )
            img_sr_np = self.img_sr.detach().cpu()
            img_hr_np = img_hr.detach().cpu()
            diff = np.abs(img_sr_np - img_hr_np)
            self.logger.experiment.add_image(
                tag='train/diff_img',
                img_tensor=make_grid(diff/diff.max(), nrow=nrow, padding=0),
                global_step=self.global_step
            )

    def validation_step(self, batch, batch_nb):
        with torch.no_grad():
            img_lr = batch['lr']  # b,c,patch_size, patch_size
            img_hr = batch['hr']
            self.img_sr = self.forward(img_lr)

            l1_loss = self.criterion_L1(self.img_sr, img_hr)
            ssim_loss = self.criterion_SSIM(self.img_sr, img_hr)
            psnr = self.criterion_PSNR(self.img_sr, img_hr)
            content_loss = l1_loss + (1 - ssim_loss)
            self.log("val_loss", content_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_l1_loss", l1_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_ssim_loss", ssim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_psnr", psnr, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            val_accum_logs = {"val_content_loss": content_loss, 'val_l1_loss': l1_loss,
                                  "val_ssim_loss": ssim_loss}

        return {
            # "batch_idx": batch_nb,
            # "fname": batch['fname'],
            # "slice_num": batch['slice_num'],
            # "img_lr": img_lr,
            # "img_hr": img_hr,
            # "img_sr": self.img_sr,
            'val_loss': content_loss,
            'ssim': ssim_loss,
            'psnr':psnr,
            # 'log': val_accum_logs,
        }

    def validation_epoch_end(self, outputs):
        print(f'len of outputs at validation_epoch_end: {len(outputs)}')
        val_psnr_mean = 0
        val_ssim_mean = 0
        for output in outputs:
            val_psnr_mean += output['psnr']
            val_ssim_mean += output['ssim']
        val_psnr_mean /= len(outputs)
        val_ssim_mean /= len(outputs)
        self.log("val_psnr_mean", val_psnr_mean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_ssim_mean", val_ssim_mean, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'val/psnr': val_psnr_mean.item(),
                'val/ssim': val_ssim_mean.item()}

    def test_step(self, batch, batch_idx):
        img_lr = batch['lr']  # (b, 1, patchsize, patchsize)
        img_hr = batch['hr']
        img_sr = self.forward(img_lr)
        mid_slice = 125

        psnr = self.criterion_PSNR(img_sr, img_hr).item()
        ssim = self.criterion_SSIM(img_sr, img_hr).item()
        slices = batch['slice_num']
        if batch['fname'][0] == 'P17_dcm' and mid_slice in slices.tolist():

            sample_img = pet.save_recon(img_lr, img_sr, img_hr, batch_idx, Path("test_result"), 1, True)

            print(f'[{batch_idx}] PSNR: {psnr:.4}, SSIM: {ssim:.4}')

        return {'psnr': psnr,
                'ssim': ssim
                }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.lr_step_size, self.lr_gamma)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # param overwrites

        # network params
        parser.add_argument('--n1', type=int, default=64, help="feature numbers in first layer")

        # training params (opt)
        parser.add_argument("--lr", default=1e-4, type=float, help="Adam learning rate")
        parser.add_argument(
            "--lr_step_size",
            default=2e+5,
            type=float,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.5,
            type=float,
            help="Extent to which step size should be decreased",
        )

        return parser
