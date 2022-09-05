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

from pet.models import UNETR,EDSR3D,UNetPlusPlus_nest3_3d,UNETR_NoBN,UNet_nest3_3d
import pet
import numpy as np
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from pet import Diceceloss,Diceloss




class Dice_weight_loss(nn.Module):
    def __init__(self,weight_loss=[1,1,1],softmax=True):
        super().__init__()
        """
        multi label dice loss with weighted
        WDL=1-2*(sum(w*sum(r&p))/sum((w*sum(r+p)))),w=array of shape (C,)
        :param Y_pred: [None, self.image_depth, self.image_height, self.image_width,
                                                        self.numclass],Y_pred is softmax result
        :param Y_gt:[None, self.image_depth, self.image_height, self.image_width,
                                                        self.numclass],Y_gt is one hot result
        :param weight_loss: numpy array of shape (C,) where C is the number of classes
        :return:
        """
        self.weight_loss=weight_loss
        self.softmax=softmax

    def __dice(self,Y_pred, Y_gt):
        weight_loss = np.array(self.weight_loss)
        smooth =1.e-5
        smooth_tf = smooth#torch.Tensor(smooth).type(torch.float32)
        if self.softmax:
            Y_pred=torch.softmax(Y_pred,dim=1)
        Y_pred =Y_pred.to(torch.float32)#torch.Tensor(Y_pred).type(torch.float32)
        Y_gt =Y_gt.to(torch.float32)#torch.Tensor(Y_gt).type(torch.float32)
        # Compute gen dice coef:
        numerator = Y_gt * Y_pred
        numerator = torch.sum(numerator, dim=(2, 3, 4))
        denominator = Y_gt + Y_pred
        denominator = torch.sum(denominator, dim=(2, 3, 4))
        gen_dice_coef =torch.mean(2. * (numerator + smooth_tf) / (denominator + smooth_tf), dim=0)
        #weight_loss=list(np.array(weight_loss))
        weight_loss=torch.tensor(weight_loss).cuda()#####先转成tensor
        dice = torch.mean(weight_loss * gen_dice_coef)
        return dice,gen_dice_coef
        


    def forward(self,Y_pred, Y_gt):
        dice,gendice=self.__dice(Y_pred, Y_gt)
        return gendice


class UNETSEGModule(pl.LightningModule):
    def __init__(
        self,
        summary_step: int = 1,
        patch_size: int = 4,
        in_channels=2,
        out_channels=3,
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

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma

        # networks
        # self.net = UNETR(
        #     input_dim=1, 
        #     output_dim=1,
        # )

        # self.net = EDSR3D()

        # self.net = EDSR3D(
        #     in_channels=2,
        #     out_channels=3,
        #     ngf=16,
        #     n_blocks=8,
        #     res_scale=0.1)
        self.net=UNet_nest3_3d(
                in_channels=in_channels,
                out_channels=out_channels,
                n1=16,)


        # training criterions
        self.loss=Diceceloss()

        # validation metrics

        self.dicemetric=Diceloss()#Dice_weight_loss
       
    def forward(self, input):
        return self.net(input)

    def training_step(self, batch, batch_nb):
        # print('test')
        image = batch['image']
        label = batch['label']
        # print("image",image.shape)

        self.train_pred = self.forward(image)
        dice_weight=Dice_weight_loss()
        gendice=dice_weight(self.train_pred,label)




        diceceloss=self.loss(self.train_pred,label)
        dicemetriec=self.dicemetric(self.train_pred,label)

        self.log("train_loss", diceceloss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_dice", dicemetriec, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_dice_1", gendice[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_dice_2", gendice[2], on_step=True, on_epoch=True, prog_bar=False, logger=True)

        return {'loss':  diceceloss,
                'image': image,
                'label': label,
                'dice': dicemetriec,
                # 'prog': {'tng/l1_loss': l1_loss,
                #          'tng/ssim_loss': ssim_loss,},
                # 'log': train_tensorboard_logs,
                }

    # def training_step_end(self, output):
    #     ##对于3D图像理论也是要改这一部分的
    #     if self.global_step % self.summary_step == 0:
    #         avg_loss = torch.stack([x["loss"] for x in output]).mean()
    #         avg_dice = torch.stack([x["dice"] for x in output]).mean()
    #     self.log("train_loss_mean", avg_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
    #     self.log("train_dice_mean", avg_dice, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        


    def validation_step(self, batch, batch_nb):

        model=self.net
        model=model.eval()
        #torch.save(model.state_dict(), '/raid/yunzhi_raid/Low_dose_PET_5/examples/unetr_net/test_unetpp_weight.pth')
        #torch.save(model, '/raid/yunzhi_raid/Low_dose_PET_5/examples/unetr_net/test_unetpp.pth')
        #print(batch['fname'])
        model=model.cuda()
        with torch.no_grad():
            image = batch['image'].cuda()
            label = batch['label'].cuda()
            #print('val',image.shape,label.shape)
            drf=batch['fname']
            #print('drf',drf)
            self.val_pred=sliding_window_inference(image,(96, 96, 96), 4,model)

            #self.img_sr = self.forward(img_lr)

            # l1_loss = self.criterion_L1(self.img_sr, img_hr)
            # ssim_loss = self.criterion_SSIM(self.img_sr, img_hr)
            # mse_loss=self.mse_loss(self.img_sr, img_hr)
            # print('ssim',ssim_loss)
            # psnr = self.criterion_PSNR(self.img_sr, img_hr)
            # content_loss = l1_loss + (1 - ssim_loss)
            print('self.val',self.val_pred.shape,label.shape)
            diceceloss=self.loss(self.val_pred,label)
            dicemetriec=self.dicemetric(self.val_pred,label)


            dice_weight=Dice_weight_loss()
            gendice=dice_weight(self.val_pred,label)
            print(gendice)



            self.log("val_loss", diceceloss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_dice", dicemetriec, on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_dice_1", gendice[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log("val_dice_1", gendice[2], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {
            # "batch_idx": batch_nb,
            # "fname": batch['fname'],
            # "slice_num": batch['slice_num'],
            # "img_lr": img_lr,
            # "img_hr": img_hr,
            # "img_sr": self.img_sr,
            'val_loss': diceceloss,
            'val_dice': dicemetriec,
            # 'log': val_accum_logs,
        }

    def validation_epoch_end(self, outputs):
        print(f'len of outputs at validation_epoch_end: {len(outputs)}')
        val_loss=0
        val_dice = 0
        for output in outputs:
            val_loss += output['val_loss']
            val_dice += output['val_dice']
           
        val_loss /= len(outputs)
        val_dice /= len(outputs)


        
        self.log("val_loss_mean", val_loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_dice_mean", val_dice, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'val/loss': val_loss.item(),
                'val/dice': val_dice.item()}

    def test_step(self, batch, batch_idx):
        image = batch['image']
        label = batch['label']

        self.train_pred = self.forward(image)




        diceceloss=self.loss(self.train_pred,label)
        dicemetriec=self.dicemetric(self.train_pred,label)
        # if batch['fname'][0] == 'P17_dcm' and mid_slice in slices.tolist():

        #     sample_img = pet.save_recon(img_lr, img_sr, img_hr, batch_idx, Path("test_result"), 1, True)

        #     print(f'[{batch_idx}] PSNR: {psnr:.4}, SSIM: {ssim:.4}')

        return {'dicece': diceceloss,
                'dicemetriec': dicemetriec
                }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr,weight_decay=1e-5)
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


