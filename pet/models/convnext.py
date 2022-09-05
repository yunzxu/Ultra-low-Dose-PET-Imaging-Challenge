# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

###代码地址：

from functools import partial
import sys
#sys.path.append('.')
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from torchsummary import summary

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


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


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        #print('input',x.shape)
        x = self.dwconv(x) #卷积   卷积之后大小通道不变   [4, 96, 64, 64]
        #print('dwconv',x.shape)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C) [4, 96, 64, 64])
        #print('dwconv',x.shape)
        x = self.norm(x) 
        #print('norm',x.shape)         #Layernorm 
        x = self.pwconv1(x)       #1*1卷积   [4, 64, 64, 384]   384 = dim(96) * 4    (把(4,64,64)看做整体输入线性层)
        #print('pwconv1',x.shape)      #1*1卷积   [4, 64, 64, 384]   384 = dim(96) * 4    (把(4,64,64)看做整体输入线性层)

        x = self.act(x)           #GELU
        x = self.pwconv2(x)       #1*1卷积    [4, 64, 64, 384]
        #print('pwconv2',x.shape)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2,3) # (N, H, W, C) -> (N, C, H, W)  [4, 96, 64, 64]

        x = input + self.drop_path(x)   #残差
        return x


# class Block(nn.Module):
#     r""" ConvNeXt Block. There are two equivalent implementations:
#     (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
#     (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
#     We use (2) as we find it slightly faster in PyTorch
    
#     Args:
#         dim (int): Number of input channels.
#         drop_path (float): Stochastic depth rate. Default: 0.0
#         layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
#     """
#     def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
#         super().__init__()
#         self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
#         self.norm = LayerNorm(dim, eps=1e-6)
#         self.pwconv1 = nn.Conv3d(dim, 4*dim, kernel_size=1,padding=0)#nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
#         self.act = nn.GELU()
#         self.pwconv2 = nn.Conv3d(4*dim, dim, kernel_size=1,padding=0)#nn.Linear(4 * dim, dim)
#         self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
#                                     requires_grad=True) if layer_scale_init_value > 0 else None
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x) #卷积   卷积之后大小通道不变   [4, 96, 64, 64]
#         x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C) [4, 96, 64, 64])
#         x = self.norm(x)
#         x = x.permute(0, 4, 1, 2,3)          #Layernorm 
#         x = self.pwconv1(x)       #1*1卷积   [4, 64, 64, 384]   384 = dim(96) * 4    (把(4,64,64)看做整体输入线性层)
#         x = self.act(x)           #GELU
#         x = self.pwconv2(x)       #1*1卷积    [4, 64, 64, 384]
#         x = x.permute(0, 2, 3, 4, 1)
#         if self.gamma is not None:
#             x = self.gamma * x
#         x = x.permute(0, 4, 1, 2,3) # (N, H, W, C) -> (N, C, H, W)  [4, 96, 64, 64]

#         x = input + self.drop_path(x)   #残差
#         return x




class ConvNeXt(nn.Module):

    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() 
        
        #stem层为transformer中的，4*4卷积，stride=4 替换resnet的池化
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        ###stage2-stage4的3个downsample
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        
        ##这里才用到block
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        
        ##分类的区别，分类在卷积层输出后拉平(N, C, H, W) -> (N, C)
        # 而分割直接接卷积的输出，里面的结构模块都是一样的
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x) #[4,3,256,256]-->[4, 96, 64, 64]--->[4,192,32,32]--->[4, 384, 16, 16]--.[4,768,8,8]
            x = self.stages[i](x) #为什么不加上这一部分
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x

class LayerNorm(nn.Module):
    #看数据输入是[n,c,w,d]还是[n,w,d,c]来决定参数channels_last or channels_first
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None,None] * x + self.bias[:, None, None,None]
            return x


class Convnext3D_isotropic(nn.Module):
    """
    PyTorch Module for EDSR, https://arxiv.org/pdf/1707.02921.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=96, n_blocks=32, res_scale=0.1, layer_scale_init_value=1e-6,
                depths=[2,2,2,2,2,2],dims=[96,96,96,96,96,96]):
        super(Convnext3D_isotropic, self).__init__()

        self.head = nn.Sequential(
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_channels, ngf, kernel_size=3),
        )
        self.body=nn.ModuleList()
        for i in range(len(depths)):
            body = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=0.1, 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])],
                    nn.ReflectionPad3d(1),
                    nn.Conv3d(ngf, ngf, kernel_size=3),)
            self.body.append(body)




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
        x0=x
        for i in range(len(self.body)):
            x=self.body[i](x)+x
            print(i,x.shape)
        x=x+x0
        x = self.tail(x)

        # self.__denormalize(x)

        return x






class Convnext3D(nn.Module):
    """
    PyTorch Module for EDSR, https://arxiv.org/pdf/1707.02921.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=96,layer_scale_init_value=1e-6,
                depths=[6,6,6,6,6,6]):
        super(Convnext3D, self).__init__()

        self.head = nn.Conv3d(in_channels, ngf, 3,1,1)
      
        self.body=nn.ModuleList()
        for i in range(len(depths)):
            body = nn.Sequential(
                    *[Block(dim=ngf, drop_path=0.1, 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])],
                    nn.Conv3d(ngf, ngf, 3,1,1),)
            self.body.append(body)

        self.conv_after_body = nn.Conv3d(ngf,ngf, 3, 1, 1)

        self.tail =nn.Conv3d(ngf, out_channels, 3,1,1)


        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def __denormalize(self, x):
        x.add_(self.mean.detach())

    def forward(self, x):
        # self.__normalize(x)
        input=x
        x = self.head(x)
        x0=x
        for body in self.body:
            x=body(x)+x
            #print(x.shape)
        x=self.conv_after_body(x)+x0
        x = self.tail(x)+input

        # self.__denormalize(x)

        return x



class Convnext3D_stem(nn.Module):
    """
    PyTorch Module for EDSR, https://arxiv.org/pdf/1707.02921.
    """

    def __init__(self, in_channels=1, out_channels=1, ngf=96,layer_scale_init_value=1e-6,
                depths=[6,6,6,6,6,6]):
        super(Convnext3D_stem, self).__init__()

        self.head = nn.Conv3d(ngf, ngf, 3,1,1)##使用stem时这里改为in_channels->ngf
      
        self.body=nn.ModuleList()
        for i in range(len(depths)):
            body = nn.Sequential(
                    *[Block(dim=ngf, drop_path=0.1, 
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])],
                    nn.Conv3d(ngf, ngf, 3,1,1),)
            self.body.append(body)

        self.conv_after_body = nn.Conv3d(ngf,ngf, 3, 1, 1)
        self.stem=nn.Conv3d(in_channels,ngf,2,2,0)
        self.tail =nn.Conv3d(ngf, out_channels, 3,1,1)
        self.up=nn.ConvTranspose3d(ngf,ngf,2,2,0)


        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def __denormalize(self, x):
        x.add_(self.mean.detach())

    def forward(self, x):
        # self.__normalize(x)
        input=x
        x = self.stem(x)
        #print('stem',x.shape)
        x=self.head(x)
        x0=x
        for body in self.body:
            x=body(x)+x
            #print(x.shape)
        x=self.conv_after_body(x)+x0

        x=self.up(x)
        x = self.tail(x)+input

        # self.__denormalize(x)

        return x







from torchstat import stat
        
if __name__ == '__main__':
    #data = torch.randn((1,3,256,256,256))
    #model = ConvNeXt(in_chans=1, depths=[1, 1, 3, 1], dims=[16, 32, 64, 128]).cuda()
    model =Convnext3D_stem(ngf=48,depths=[3,3,9,3]).cuda()
    #     out = model(data)
    # print(out)

   
    # 导入模型，输入一张输入图片的尺寸
    #stat(model, input_size=[1,96,96,96])



    print(summary(model,[1,96,96,96]))
    data = torch.randn(2,1,96,96,96).cuda()
    y=model(data)
    # block = Block(1)
    # out = block(data)
    # print(out)
    # depths=[3, 3, 9, 3]
    # dp_rates=[x.item() for x in torch.linspace(0, 1, sum(depths))] 
    # print(dp_rates)