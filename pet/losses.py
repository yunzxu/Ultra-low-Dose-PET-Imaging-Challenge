from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg


class GANLoss(nn.Module):
    """
    PyTorch module for GAN loss.
    This code is inspired by https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
    """
    def __init__(self,
                 gan_mode='wgangp',
                 target_real_label=1.0,
                 target_fake_label=0.0):

        super(GANLoss, self).__init__()

        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).detach()

    def forward(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = - prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class VGGLoss(nn.Module):
    """
    PyTorch module for VGG loss.

    Parameter
    ---------
    net_type : str
        type of vgg network, i.e. `vgg16` or `vgg19`.
    layer : str
        layer where the mean squared error is calculated.
    rescale : float
        rescale factor for VGG Loss
    """
    def __init__(self, net_type='vgg19', layer='relu2_2', rescale=0.006):
        super(VGGLoss, self).__init__()

        if net_type == 'vgg16':
            assert layer in ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']
            self.__vgg_net = VGG16()
            self.__layer = layer
        elif net_type == 'vgg19':
            assert layer in ['relu1_2', 'relu2_2', 'relu3_4',
                             'relu4_4', 'relu5_4']
            self.__vgg_net = VGG19()
            self.__layer = layer

        self.register_buffer(
            name='vgg_mean',
            tensor=torch.tensor([[[0.485]], [[0.456]], [[0.406]]],
                                requires_grad=False)
        )
        self.register_buffer(
            name='vgg_std',
            tensor=torch.tensor([[[0.229]], [[0.224]], [[0.225]]],
                                requires_grad=False)
        )
        self.register_buffer(   # to balance VGG loss with other losses.
            name='rescale',
            tensor=torch.tensor(rescale, requires_grad=False)
        )

    def __normalize(self, img):
        img = img.sub(self.vgg_mean.detach())
        img = img.div(self.vgg_std.detach())
        return img

    def forward(self, x, y):
        """
        Paramenters
        ---
        x, y : torch.Tensor
            input or output tensor. they must be normalized to [0, 1].

        Returns
        ---
        out : torch.Tensor
            mean squared error between the inputs.
        """
        norm_x = self.__normalize(x)
        norm_y = self.__normalize(y)
        feat_x = getattr(self.__vgg_net(norm_x), self.__layer)
        feat_y = getattr(self.__vgg_net(norm_y), self.__layer)
        out = F.mse_loss(feat_x, feat_y) * self.rescale
        return out


class VGG16(nn.Module):
    """
    Blockwise pickable VGG16.

    This code is inspired by https://gist.github.com/crcrpar/a5d46738ffff08fc12138a5f270db426 
    """
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)

        return out


class VGG19(nn.Module):
    """
    Blockwise pickable VGG19.

    This code is inspired by https://gist.github.com/crcrpar/a5d46738ffff08fc12138a5f270db426
    """
    def __init__(self, requires_grad=False):
        super(VGG19, self).__init__()
        vgg_pretrained_features = vgg.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_4 = h
        h = self.slice4(h)
        h_relu4_4 = h
        h = self.slice5(h)
        h_relu5_4 = h

        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2',
                           'relu3_4', 'relu4_4', 'relu5_4'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_4, h_relu4_4, h_relu5_4)

        return out


class TVLoss(nn.Module):
    """
    Total Variation Loss.

    This code is copied from https://github.com/leftthomas/SRGAN/blob/master/loss.py
    """
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class PSNR(nn.Module):
    """
    Peak Signal/Noise Ratio.
    """
    def __init__(self, max_val=1.):
        super(PSNR, self).__init__()
        self.max_val = max_val

    def forward(self, predictions, targets):
        self.max_val = torch.max(targets)
        mse = F.mse_loss(predictions, targets)
        psnr = 10 * torch.log10(self.max_val ** 2 / mse)
        return psnr


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)


        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return S.mean()



class Diceloss(nn.Module):
    def __init__(self, softmax=True,is_3D=True,squared_pred=False,smooth=1e-5,jaccard=False):
        super().__init__()
        self.softmax=softmax
        self.iS_3D= is_3D
        self.squared_pred=squared_pred
        self.smooth_nr=smooth
        self.smooth_dr=smooth
        self.jaccard=jaccard
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.softmax:
            input = torch.softmax(input, 1)##?????????axis=1
        

        if self.iS_3D:
            reduce_axis=[2,3,4]
        else:
            reduce_axis=[2,3]

        if self.squared_pred:
            target = torch.pow(target, 2)
            input = torch.pow(input, 2)

        intersection = torch.sum(target * input, dim=reduce_axis)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

        denominator = ground_o + pred_o

        if self.jaccard:
            denominator = 2.0 * (denominator - intersection)

        f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / (denominator + self.smooth_dr)

        f = torch.mean(f)  # the batch and channel average
       
        return f


class Diceceloss(nn.Module):
    def __init__(self, lambda1=1.0,lambda2=1.0,softmax=True,is_3D=True,squared_pred=False,smooth=1e-5,ce_weight=None,reduction="mean"):
        super().__init__()
        self.dice=Diceloss(softmax,is_3D,squared_pred,smooth)
        
        self.ce=nn.CrossEntropyLoss(weight=ce_weight, reduction=reduction)
        self.lambda_dice=lambda1
        self.lambda_ce=lambda2

        


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice(input, target)
        target = torch.argmax(target, dim=1)
        ce_loss = self.ce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_ce * ce_loss

        return total_loss
