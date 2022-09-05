from matplotlib import use
import torch
import torch.nn as nn
"""refer to  https://towardsdatascience.com/biomedical-image-segmentation-unet-991d075a3a4b"""






class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        output = self.activation(x)

        return output
class UNetPlusPlus_v1(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, n1=64):
        super(UNetPlusPlus_v1, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class UNetPlusPlus_nest4(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, n1=64, deep_supervision=False):
        super(UNetPlusPlus_nest4, self).__init__()
        print('nest4 used')
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            return output


class UNetPlusPlus_nest3(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, n1=64, deep_supervision=False):
        super(UNetPlusPlus_nest3, self).__init__()
        print('nest3 used')

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):

        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]

        else:
            output = self.final(x0_3)
            return output


class conv_block_nested_3D(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch,bn=False,use_res=False):
        super(conv_block_nested_3D, self).__init__()
        self.use_res=use_res
        self.bn=bn
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(out_ch)
        self.res= nn.Conv3d(in_ch, out_ch, kernel_size=1,bias=True)
        self.bn3 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        input=x
        #input=self.conv3(x)###残差连接
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = self.activation(x)
    
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        if self.use_res:
            res=self.res(input)
            res=self.bn3(res)
            x=x+res
        output = self.activation(x)#+input



        return output


class UNetPlusPlus_nest3_3d(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=1, n1=16, deep_supervision=False):
        super(UNetPlusPlus_nest3_3d, self).__init__()
        print('nest3 used')

        filters = [n1, n1 * 2, n1 * 4, n1 * 8]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv1_1 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv1_2 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_1 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)





        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0])
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3])

        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2])

        self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1])

        self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0])

        if self.deep_supervision:
            self.final1 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    # def forward(self, x):
    #     print(x.shape)
    #     x0_0 = self.conv0_0(x)
    #     x1_0 = self.conv1_0(self.pool(x0_0))
    #     x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

    #     x2_0 = self.conv2_0(self.pool(x1_0))
    #     x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
    #     x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

    #     x3_0 = self.conv3_0(self.pool(x2_0))
    #     x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
    #     x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
    #     x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

    #     if self.deep_supervision:
    #         output1 = self.final1(x0_1)
    #         output2 = self.final2(x0_2)
    #         output3 = self.final3(x0_3)
    #         return [output1, output2, output3]

    #     else:
    #         output = self.final(x0_3)
    #         return output

    def forward(self, x):
        # print(x.shape)
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Deconv1_0(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Deconv2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Deconv1_1(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Deconv3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Deconv2_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Deconv1_2(x1_2)], 1))

        # print('x1_0',x1_0.shape)
        # print('x2_0',x2_0.shape)
        # print('x3_0',x3_0.shape)
        # print('x2_1',x2_1.shape)
        # print('x1_2',x1_2.shape)
        # print('x1_1',x1_1.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            return [output1, output2, output3]

        else:
            output = self.final(x0_3)
            return output

class UNet_nest3_3d(nn.Module):###在unet++基础上的unet
        
    def __init__(self, in_channels=1, out_channels=1, n1=16, deep_supervision=False,bn=True,use_res=True):
        super(UNet_nest3_3d, self).__init__()
        print('nest3 used')

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 *16]
        self.deep_supervision = deep_supervision
     

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv4_0 =nn.ConvTranspose3d(filters[4], filters[4], kernel_size=2, stride=2, padding=0, output_padding=0)

        # self.Deconv1_1 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.Deconv1_2 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        # self.Deconv2_1 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)





        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0],bn,use_res)
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1],bn,use_res)
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2],bn,use_res)
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3],bn,use_res)
        self.conv4_0 = conv_block_nested_3D(filters[3], filters[4], filters[4],bn,use_res)
        



        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0],bn,use_res)
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1],bn,use_res)
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2],bn,use_res)
        self.conv3_1 = conv_block_nested_3D(filters[3] + filters[4], filters[3], filters[3],bn,use_res)

        self.dropout=nn.Dropout(0.5)


        # self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0],bn)
        # self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1],bn)

        # self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0],bn)

        if self.deep_supervision:
            self.final1 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv3d(filters[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv3d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # print(x.shape)
    
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))
        #print('x1-4',x0_0.shape,x1_0.shape,x2_0.shape,x3_0.shape,x4_0.shape)
        # print('x1_0',x1_0.shape)
        # print('x2_0',x2_0.shape)
        #print('self.Deconv3_0(x3_0)',self.Deconv3_0(x3_0).shape)


        x4_0=self.dropout(x4_0)
        x3_1 =self.conv3_1(torch.cat([x3_0, self.Deconv4_0(x4_0)], 1))
        #print('x3_1',x3_1.shape)


        x2_1 = self.conv2_1(torch.cat([x2_0, self.Deconv3_0(x3_1)], 1))
        # print('X2_1',x2_1.shape)
        # print('self.Deconv2_1(x2_1)',self.Deconv2_1(x2_1).shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Deconv2_0(x2_1)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Deconv1_0(x1_1)], 1))

        # if self.deep_supervision:
        #     output1 = self.final1(x0_1)
        #     output2 = self.final2(x0_2)
        #     output3 = self.final3(x0_3)
        #     return [output1, output2, output3]

        # else:
        output = self.final(x0_1)
        return output




class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bath_normal=False):
        super(DoubleConv, self).__init__()
        channels = out_channels 
        # if in_channels > out_channels:
        #     channels = in_channels // 2

        layers = [
            # in_channels：输入通道数
            # channels：输出通道数
            # kernel_size：卷积核大小
            # stride：步长
            # padding：边缘填充
            nn.Conv3d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            nn.Conv3d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True)
        ]
        if bath_normal: # 如果要添加BN层
            layers.insert(1, nn.BatchNorm3d(channels))
            layers.insert(len(layers) - 1, nn.BatchNorm3d(out_channels))

        # 构造序列器
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, batch_normal)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels, batch_normal=False, bilinear=False):
        super(UpSampling, self).__init__()
        if bilinear:
            # 采用双线性插值的方法进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 采用反卷积进行上采样
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, batch_normal)

    # inputs1：上采样的数据（对应图中黄色箭头传来的数据）
    # inputs2：特征融合的数据（对应图中绿色箭头传来的数据）
    def forward(self, inputs1, inputs2):
        # 进行一次up操作
        inputs1 = self.up(inputs1)

        # 进行特征融合
        outputs = torch.cat([inputs1, inputs2], dim=1)
        print('outputs',outputs.shape)
        outputs = self.conv(outputs)
        return outputs

class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels ):
        super(LastConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1 )

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels, num_classes=2, batch_normal=False, bilinear=False,ngf=16):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.batch_normal = batch_normal
        self.bilinear = bilinear


        self.inputs = DoubleConv(in_channels, ngf, self.batch_normal)
        self.down_1 = DownSampling(ngf, ngf*2, self.batch_normal)
        self.down_2 = DownSampling(ngf*2, ngf*4, self.batch_normal)
        self.down_3 = DownSampling(ngf*4, ngf*8, self.batch_normal)

        self.up_1 = UpSampling(ngf*8, ngf*4, self.batch_normal, self.bilinear)
        self.up_2 = UpSampling(ngf*4, ngf*2, self.batch_normal, self.bilinear)
        self.up_3 = UpSampling(ngf*2, ngf, self.batch_normal, self.bilinear)
        self.outputs = LastConv(ngf, num_classes)

    def forward(self, x):
        # down 部分
        x1 = self.inputs(x)
        #print('x1',x1.shape)
        x2 = self.down_1(x1)
        #print('x2',x2.shape)
        x3 = self.down_2(x2)
        #print('x3',x3.shape)
        x4 = self.down_3(x3)
        #print('x4',x4.shape)

        # up部分
        print('x4,x3',x4.shape,x3.shape)
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x = self.outputs(x7)

        return x



    
from torchsummary import summary


if __name__ == "__main__":
    model=UNet_nest3_3d(
                in_channels=1,
                out_channels=1,
                n1=48,
                use_res=True,
                bn=True
            )#.cuda()
    #model=UNet3D(in_channels=1).cuda()
    summary(model,[1,96,96,96])
    # x=torch.randn([2,1,96,96,96]).cuda()
    # model(x)