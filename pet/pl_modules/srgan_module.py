from argparse import ArgumentParser
from math import sqrt, ceil

import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision import transforms
from pathlib import Path

import pytorch_lightning as pl
import matplotlib.pyplot as plt


from pet.models import SRResNet, Discriminator

import pet



import numpy as np

# from .mri_module import MriModule


class SRGANModule(pl.LightningModule):
    """
    SRGAN training module.
    This can be used to train SRGAN networks from the paper:
    https://arxiv.org/pdf/1609.04802.
    """
    def __init__(
        self,
        summary_step: int = 1,
        patch_size: int = 4,
        g_chans: int = 64,
        n_blocks: int = 16,
        d_chans: int = 64,
        lr_G: float = 1e-4,
        lr_D: float = 1e-4,
        lr_step_size: float = 1e+3,
        lr_gamma: float = 0.5,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            summary_step: print out summary
            patch_size: center crop in image
            g_chans: number of channels in first conv layer for G
            n_blocks: number of residual cnn blocks in G
            d_chans: number of channels in first conv layer for D
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.summary_step = summary_step
        self.patch_size = patch_size

        self.g_chans = g_chans
        self.n_blocks = n_blocks
        self.d_chans = d_chans

        self.lr_G = lr_G
        self.lr_D = lr_D
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        # networks
        self.net_G = SRResNet(
            in_channels=1,
            out_channels=1,
            ngf=self.g_chans,
            n_blocks=self.n_blocks,
        )

        self.net_D = Discriminator(
            in_channels=1,
            out_channels=1,
            ndf=self.d_chans,
        )

        # training criterions
        self.criterion_L1 = nn.L1Loss()
        self.criterion_GAN = pet.GANLoss(gan_mode='vanilla')
        self.criterion_SSIM = pet.SSIM(window_size=11)  # old SSIMLoss, new SSIM
        # validation metrics
        self.criterion_PSNR = pet.PSNR()


    def forward(self, input):
        return self.net_G(input)

    def training_step(self, batch, batch_nb, optimizer_idx):
        img_lr = batch['lr']
        img_hr = batch['hr']

        if optimizer_idx == 0:  # train discriminator correspond to index of the output from function configure_optimizers
            self.img_sr = self.forward(img_lr)  # \in [0, 1] predicted full-dose

            # for real image
            d_out_real = self.net_D(img_hr)
            d_loss_real = self.criterion_GAN(d_out_real, True)
            # for fake image
            d_out_fake = self.net_D(self.img_sr.detach())
            d_loss_fake = self.criterion_GAN(d_out_fake, False)

            # combined discriminator loss
            # d_loss = 1 + d_loss_real + d_loss_fake
            d_loss = (d_loss_real + d_loss_fake)/2

            self.log("train_d_loss", d_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_d_real_loss", d_loss_real, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_d_fake_loss", d_loss_fake, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            d_accum_logs = {'train_d_loss': d_loss}

            return {'loss': d_loss,
                    # 'prog': {'tng/d_loss': d_loss},
                    # 'log': d_accum_logs,
                    }

        elif optimizer_idx == 1:  # train generator
            self.img_sr = self.forward(img_lr)
            # content loss
            l1_loss = self.criterion_L1(self.img_sr, img_hr)
            ssim_loss = self.criterion_SSIM(self.img_sr, img_hr)
            content_loss = l1_loss + (1 - ssim_loss)
            # adversarial loss
            adv_loss = self.criterion_GAN(self.net_D(self.img_sr), True)

            # combined generator loss
            g_loss = content_loss + 1e-3 * adv_loss

            self.log("train_g_loss", g_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_g_l1_loss", l1_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_g_ssim_loss", ssim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("train_g_adv_loss", adv_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)

            g_accum_logs = {'train_g_loss': g_loss, 'train_g_l1_loss': l1_loss, "train_g_adv_loss": adv_loss}

            return {'loss': g_loss,
                    'img_lr': img_lr,
                    'img_hr': img_hr,
                    # 'prog': {'tng/g_loss': g_loss,
                    #          'tng/l1_loss': l1_loss,
                    #          'tng/adv_loss': adv_loss},
                    # 'log': g_tensorboard_logs,
                    }

    def training_step_end(self, step_output):

        if self.global_step % self.summary_step == 1:
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

            psnr = self.criterion_PSNR(self.img_sr, img_hr)
            # ssim = 1 - self.criterion_SSIM(img_sr, img_hr, data_range=img_sr.max() - img_sr.min())  # 1-ssim

            l1_loss = self.criterion_L1(self.img_sr, img_hr)
            ssim_loss = self.criterion_SSIM(self.img_sr, img_hr)
            content_loss = l1_loss + (1 - ssim_loss)
            self.log("val_loss", content_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_l1_loss", l1_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_ssim", ssim_loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_psnr", psnr, on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return {
            # "batch_idx": batch_nb,
            # "fname": batch['fname'],
            # "slice_num": batch['slice_num'],
            # "img_lr": img_lr,
            # "img_hr": img_hr,
            # "img_sr": img_sr,
            'val_loss': content_loss,
            'psnr': psnr,
            'ssim': ssim_loss}

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
        ssim = 1 - self.criterion_SSIM(img_sr, img_hr, data_range=img_sr.max() - img_sr.min()).item()
        slices = batch['slice_num']
        if batch['fname'][0] == 'P17_dcm' and mid_slice in slices.tolist():

            sample_img = pet.save_recon(img_lr, img_sr, img_hr, batch_idx, Path("test_result"), 1, True)

            print(f'[{batch_idx}] PSNR: {psnr:.4}, SSIM: {ssim:.4}')

        return {'psnr': psnr,
                'ssim': ssim
                }

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.lr_G, weight_decay=self.weight_decay)
        optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.lr_D, weight_decay=self.weight_decay)
        scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, self.lr_step_size, self.lr_gamma)  # decay by gamma after each lr_step_size
        scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, self.lr_step_size, self.lr_gamma)

        return [optimizer_D, optimizer_G], [scheduler_D, scheduler_G]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # param overwrites

        # network params
        parser.add_argument('--g_chans', type=int, default=64)
        parser.add_argument('--n_blocks', type=int, default=16)
        parser.add_argument('--d_chans', type=int, default=64)

        # training params (opt)
        parser.add_argument("--lr_G", default=1e-4, type=float, help="Adam learning rate")
        parser.add_argument("--lr_D", default=1e-4, type=float, help="Adam learning rate")
        parser.add_argument(
            "--lr_step_size",
            default=1e+4,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma",
            default=0.5,
            type=float,
            help="Extent to which step size should be decreased",
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
