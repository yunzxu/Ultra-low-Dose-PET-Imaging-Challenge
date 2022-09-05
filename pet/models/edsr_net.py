from math import log2

import torch
from torch import nn

from torchsummary import summary


class EDSR(nn.Module):
    """
    PyTorch Module for EDSR, https://arxiv.org/pdf/1707.02921.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=256, n_blocks=32, res_scale=0.1):
        super(EDSR, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, ngf, kernel_size=3),
        )
        self.body = nn.Sequential(
            *[EDSRBlock(ngf, res_scale) for _ in range(n_blocks)],
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3),
        )
        self.tail = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, out_channels, kernel_size=3)
        )

        # mean value of DIV2K
        self.register_buffer(
            name='mean',
            tensor=torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]],
                                requires_grad=False)
        )

    def __normalize(self, x):
        x.sub_(self.mean.detach())

    def __denormalize(self, x):
        x.add_(self.mean.detach())

    def forward(self, x):
        # self.__normalize(x)

        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)

        # self.__denormalize(x)

        return x


#################
# Building Blocks
#################

class EDSRBlock(nn.Module):
    """
    Building block of EDSR.
    """

    def __init__(self, dim, res_scale=0.1):
        super(EDSRBlock, self).__init__()
        self.res_scale = res_scale
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
        )

    def forward(self, x):
        return x + self.net(x) * self.res_scale


class UpscaleBlock(nn.Sequential):
    """
    Upscale block using sub-pixel convolutions.
    `scale_factor` can be selected from {2, 3, 4, 8}.
    """

    def __init__(self, scale_factor, dim, act=None):
        assert scale_factor in [2, 3, 4, 8]

        layers = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            layers += [
                nn.ReflectionPad2d(1),
                nn.Conv2d(dim, dim * r * r, kernel_size=3),
                nn.PixelShuffle(r),
            ]

            if act == 'relu':
                layers += [nn.ReLU(True)]
            elif act == 'prelu':
                layers += [nn.PReLU()]

        super(UpscaleBlock, self).__init__(*layers)





class EDSR3D(nn.Module):
    """
    PyTorch Module for EDSR, https://arxiv.org/pdf/1707.02921.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=256, n_blocks=32, res_scale=0.1):
        super(EDSR3D, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels, ngf, kernel_size=3),
        )
        self.body = nn.Sequential(
            *[EDSR3DBlock(ngf, res_scale) for _ in range(n_blocks)],
            nn.ReflectionPad3d(1),
            nn.Conv3d(ngf, ngf, kernel_size=3),
        )
        self.tail = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(ngf, out_channels, kernel_size=3)
        )

        # mean value of DIV2K
        self.register_buffer(
            name='mean',
            tensor=torch.tensor([[[0.4488]], [[0.4371]], [[0.4040]]],
                                requires_grad=False)
        )

    def __normalize(self, x):
        x.sub_(self.mean.detach())

    def __denormalize(self, x):
        x.add_(self.mean.detach())

    def forward(self, x):
        # self.__normalize(x)

        x = self.head(x)
        x = self.body(x) + x
        x = self.tail(x)

        # self.__denormalize(x)

        return x


#################
# Building Blocks
#################

class EDSR3DBlock(nn.Module):
    """
    Building block of EDSR.
    """

    def __init__(self, dim, res_scale=0.1):
        super(EDSR3DBlock, self).__init__()
        self.res_scale = res_scale
        self.net = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(dim, dim, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad3d(1),
            nn.Conv3d(dim, dim, kernel_size=3),
        )

    def forward(self, x):
        return x + self.net(x) * self.res_scale


class UpscaleBlock(nn.Sequential):
    """
    Upscale block using sub-pixel convolutions.
    `scale_factor` can be selected from {2, 3, 4, 8}.
    """

    def __init__(self, scale_factor, dim, act=None):
        assert scale_factor in [2, 3, 4, 8]

        layers = []
        for _ in range(int(log2(scale_factor))):
            r = 2 if scale_factor % 2 == 0 else 3
            layers += [
                nn.ReflectionPad3d(1),
                nn.Conv3d(dim, dim * r * r, kernel_size=3),
                nn.PixelShuffle(r),
            ]

            if act == 'relu':
                layers += [nn.ReLU(True)]
            elif act == 'prelu':
                layers += [nn.PReLU()]

        super(UpscaleBlock, self).__init__(*layers)
        
        
        
if __name__ == '__main__':
    data = torch.randn((1,3,256,256,256))
    model =EDSR3D(ngf=32, n_blocks=8).cuda()
    # with torch.no_grad():
    #     out = model(data)
    # print(out)



    print(summary(model,[1,96,96,96]))

    data = torch.randn(4,1,96,96,96).cuda()
    y=model(data)

