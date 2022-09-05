from matplotlib import use
import torch
import torch.nn as nn

# from unetplusplus_v1 import conv_block_nested_3D
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




class conv_block_3D(nn.Module):###
    
    def __init__(self, in_ch, out_ch,bn=True,use_res=False):
        super(conv_block_3D, self).__init__()
        self.use_res=use_res
        self.bn=bn
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(out_ch)
    def forward(self, x):
        input=x
        #input=self.conv3(x)###残差连接
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
    
        output = self.activation(x)#+input
        return output



class UNet3plus_nest3_3d(nn.Module):###在unet++基础上的unet
        
    def __init__(self, in_channels=1, out_channels=1, n1=16, deep_supervision=False,bn=True,use_res=False):
        super(UNet3plus_nest3_3d, self).__init__()
        print('nest3 used')
        filters_cat=16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 *16]
        filters_cat_total=filters_cat*5
        self.deep_supervision = deep_supervision
     

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv4_0 =nn.ConvTranspose3d(filters[4], filters[4], kernel_size=2, stride=2, padding=0, output_padding=0)



        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0],bn,use_res)
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1],bn,use_res)
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2],bn,use_res)
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3],bn,use_res)
        self.conv4_0 = conv_block_nested_3D(filters[3], filters[4], filters[4],bn,use_res)
        



        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0],bn,use_res)
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1],bn,use_res)
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2],bn,use_res)
        self.conv3_1 = conv_block_nested_3D(filters[3] + filters[4], filters[3], filters[3],bn,use_res)


        # self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0],bn)
        # self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1],bn)

        # self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0],bn)


        ##hd3每个编码端都使用5个map,5x64=320
        self.h0_hd3_PT = nn.Sequential(nn.MaxPool3d(8, 8, ceil_mode=True),conv_block_3D(filters[0],filters_cat))##hd代表decoder的意思
        self.h1_hd3_PT = nn.Sequential(nn.MaxPool3d(4, 4, ceil_mode=True),conv_block_3D(filters[1],filters_cat))
        self.h2_hd3_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[2],filters_cat))

        self.h3_hd3_PT = conv_block_3D(filters[3],filters_cat)

        self.h4_hd3_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],2,2,0,0),conv_block_3D(filters[4],filters_cat))###上采样原始网络使用的nn.Upsample(scale_factor=2, mode='bilinear') ，也可以使用：nn.ConvTranspose3d(filters[4],filters[4],2,2,0,0)


        ##hd2
        self.h0_hd2_PT = nn.Sequential(nn.MaxPool3d(4, 4, ceil_mode=True),conv_block_3D(filters[0],filters_cat))
        self.h1_hd2_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[1],filters_cat))

        self.h2_hd2_PT = conv_block_3D(filters[2],filters_cat)

        self.h3_hd2_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,2,2,0,0),conv_block_3D(filters_cat_total,filters_cat))

        self.h4_hd2_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],4,4,0,0),conv_block_3D(filters[4],filters_cat))



        ##hd1
        self.h0_hd1_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[0],filters_cat))

        self.h1_hd1_PT = conv_block_3D(filters[1],filters_cat)

        self.h2_hd1_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,2,2,0,0),conv_block_3D(filters_cat_total,filters_cat))

        self.h3_hd1_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,4,4,0,0),conv_block_3D(filters_cat_total,filters_cat))

        self.h4_hd1_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],8,8,0,0),conv_block_3D(filters[4],filters_cat))


        ##hd0:
        self.h0_hd0_PT = conv_block_3D(filters[0],filters_cat)
        self.h1_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,2,2,0,0),conv_block_3D(filters_cat_total,filters_cat))
        self.h2_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,4,4,0,0),conv_block_3D(filters_cat_total,filters_cat))
        self.h3_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,8,8,0,0),conv_block_3D(filters_cat_total,filters_cat))
        self.h4_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],16,16,0,0),conv_block_3D(filters[4],filters_cat))



        ##deconder_conv:320-320
        self.conv_hd3_hd2=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd2_hd1=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd1_hd0=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd0_hd0=conv_block_3D(filters_cat_total,filters_cat_total)



        self.out= nn.Conv3d(filters_cat_total ,out_channels, kernel_size=1)
 

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

        h0_3=self.h0_hd3_PT(x0_0)
        h1_3=self.h1_hd3_PT(x1_0)
        h2_3=self.h2_hd3_PT(x2_0)
        h3_3=self.h3_hd3_PT(x3_0)
        h4_3=self.h4_hd3_PT(x4_0)

        h3_0= self.conv_hd3_hd2(torch.cat([h0_3,h1_3,h2_3,h3_3,h4_3], 1))

        h0_2=self.h0_hd2_PT(x0_0)
        h1_2=self.h1_hd2_PT(x1_0)
        h2_2=self.h2_hd2_PT(x2_0)
        h3_2=self.h3_hd2_PT(h3_0)
        h4_2=self.h4_hd2_PT(x4_0)
        print('h_2',h0_2.shape,h1_2.shape,h2_2.shape,h3_2.shape,h4_2.shape)
        h2_0= self.conv_hd2_hd1(torch.cat([h0_2,h1_2,h2_2,h3_2,h4_2], 1))


        h0_1=self.h0_hd1_PT(x0_0)
        h1_1=self.h1_hd1_PT(x1_0)
        h2_1=self.h2_hd1_PT(h2_0)
        h3_1=self.h3_hd1_PT(h3_0)
        h4_1=self.h4_hd1_PT(x4_0)


        h1_0= self.conv_hd1_hd0(torch.cat([h0_1,h1_1,h2_1,h3_1,h4_1], 1))


        h0_0=self.h0_hd0_PT(x0_0)
        h1_0=self.h1_hd0_PT(h1_0)
        h2_0=self.h2_hd0_PT(h2_0)
        h3_0=self.h3_hd0_PT(h3_0)
        h4_0=self.h4_hd0_PT(x4_0)

        h0_0= self.conv_hd0_hd0(torch.cat([h0_0,h1_0,h2_0,h3_0,h4_0], 1))

        output = self.out(h0_0)
        return output



class UNet3plus_nest3_3d_depth3(nn.Module):###在unet++基础上的unet
        
    def __init__(self, in_channels=1, out_channels=1, n1=16, filters_cat=16, deep_supervision=False,bn=True,use_res=False):
        super(UNet3plus_nest3_3d_depth3, self).__init__()
        print('nest3 used')
        filters_cat=16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 *16]
        filters_cat_total=filters_cat*4
        self.deep_supervision = deep_supervision
     

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv4_0 =nn.ConvTranspose3d(filters[4], filters[4], kernel_size=2, stride=2, padding=0, output_padding=0)



        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0],bn,use_res)
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1],bn,use_res)
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2],bn,use_res)
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3],bn,use_res)
        self.conv4_0 = conv_block_nested_3D(filters[3], filters[4], filters[4],bn,use_res)
        



        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0],bn,use_res)
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1],bn,use_res)
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2],bn,use_res)
        self.conv3_1 = conv_block_nested_3D(filters[3] + filters[4], filters[3], filters[3],bn,use_res)


        # self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0],bn)
        # self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1],bn)

        # self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0],bn)


        ##hd3每个编码端都使用5个map,5x64=320
        # self.h0_hd3_PT = nn.Sequential(nn.MaxPool3d(8, 8, ceil_mode=True),conv_block_3D(filters[0],filters_cat))##hd代表decoder的意思
        # self.h1_hd3_PT = nn.Sequential(nn.MaxPool3d(4, 4, ceil_mode=True),conv_block_3D(filters[1],filters_cat))
        # self.h2_hd3_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[2],filters_cat))

        # self.h3_hd3_PT = nn.Sequential(nn.ConvTranspose3d(filters[3],filters[4],2,2,0,0),conv_block_3D(filters[4],filters_cat)

        # self.h4_hd3_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],2,2,0,0),conv_block_3D(filters[4],filters_cat))###上采样原始网络使用的nn.Upsample(scale_factor=2, mode='bilinear') ，也可以使用：nn.ConvTranspose3d(filters[4],filters[4],2,2,0,0)


        ##hd2
        self.h0_hd2_PT = nn.Sequential(nn.MaxPool3d(4, 4, ceil_mode=True),conv_block_3D(filters[0],filters_cat))
        self.h1_hd2_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[1],filters_cat))

        self.h2_hd2_PT = conv_block_3D(filters[2],filters_cat)

        self.h3_hd2_PT = nn.Sequential(nn.ConvTranspose3d(filters[3],filters[3],2,2,0,0),conv_block_3D(filters[3],filters_cat))

        self.h4_hd2_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],4,4,0,0),conv_block_3D(filters[4],filters_cat))



        ##hd1
        self.h0_hd1_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[0],filters_cat))

        self.h1_hd1_PT = conv_block_3D(filters[1],filters_cat)

        self.h2_hd1_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,2,2,0,0),conv_block_3D(filters_cat_total,filters_cat))

        self.h3_hd1_PT = nn.Sequential(nn.ConvTranspose3d(filters[3],filters[3],4,4,0,0),conv_block_3D(filters[3],filters_cat))

        self.h4_hd1_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],8,8,0,0),conv_block_3D(filters[4],filters_cat))


        ##hd0:
        self.h0_hd0_PT = conv_block_3D(filters[0],filters_cat)
        self.h1_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,2,2,0,0),conv_block_3D(filters_cat_total,filters_cat))
        self.h2_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters_cat_total,filters_cat_total,4,4,0,0),conv_block_3D(filters_cat_total,filters_cat))
        self.h3_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters[3],filters[3],8,8,0,0),conv_block_3D(filters[3],filters[3],filters_cat))
        self.h4_hd0_PT = nn.Sequential(nn.ConvTranspose3d(filters[4],filters[4],16,16,0,0),conv_block_3D(filters[4],filters_cat))



        ##deconder_conv:320-320
        self.conv_hd3_hd2=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd2_hd1=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd1_hd0=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd0_hd0=conv_block_3D(filters_cat_total,filters_cat_total)



        self.out= nn.Conv3d(filters_cat_total ,out_channels, kernel_size=1)
 

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

        h0_2=self.h0_hd2_PT(x0_0)
        h1_2=self.h1_hd2_PT(x1_0)
        h2_2=self.h2_hd2_PT(x2_0)
        h3_2=self.h3_hd2_PT(x3_0)
       
        #print('h_2',h0_2.shape,h1_2.shape,h2_2.shape,h3_2.shape)
        h2_0= self.conv_hd2_hd1(torch.cat([h0_2,h1_2,h2_2,h3_2], 1))


        h0_1=self.h0_hd1_PT(x0_0)
        h1_1=self.h1_hd1_PT(x1_0)
        h2_1=self.h2_hd1_PT(h2_0)
        h3_1=self.h3_hd1_PT(x3_0)


        h1_0= self.conv_hd1_hd0(torch.cat([h0_1,h1_1,h2_1,h3_1], 1))


        h0_0=self.h0_hd0_PT(x0_0)
        h1_0=self.h1_hd0_PT(h1_0)
        h2_0=self.h2_hd0_PT(h2_0)
        h3_0=self.h3_hd0_PT(x3_0)

        h0_0= self.conv_hd0_hd0(torch.cat([h0_0,h1_0,h2_0,h3_0], 1))

        output = self.out(h0_0)
        return output


class UNet3plus_nest3_3d_depth3_linear(nn.Module):###在unet++基础上的unet,将转置卷积改为upsmaple
        
    def __init__(self, in_channels=1, out_channels=1, n1=16, filters_cat=16, deep_supervision=False,bn=True,use_res=False):
        super(UNet3plus_nest3_3d_depth3_linear, self).__init__()
        print('nest3 used')
        filters_cat=filters_cat
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 *16]
        filters_cat_total=filters_cat*4
        self.deep_supervision = deep_supervision
     

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Deconv1_0 =nn.ConvTranspose3d(filters[1], filters[1], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv2_0 =nn.ConvTranspose3d(filters[2], filters[2], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv3_0 =nn.ConvTranspose3d(filters[3], filters[3], kernel_size=2, stride=2, padding=0, output_padding=0)
        self.Deconv4_0 =nn.ConvTranspose3d(filters[4], filters[4], kernel_size=2, stride=2, padding=0, output_padding=0)



        self.conv0_0 = conv_block_nested_3D(in_channels, filters[0], filters[0],bn,use_res)
        self.conv1_0 = conv_block_nested_3D(filters[0], filters[1], filters[1],bn,use_res)
        self.conv2_0 = conv_block_nested_3D(filters[1], filters[2], filters[2],bn,use_res)
        self.conv3_0 = conv_block_nested_3D(filters[2], filters[3], filters[3],bn,use_res)
        self.conv4_0 = conv_block_nested_3D(filters[3], filters[4], filters[4],bn,use_res)
        



        self.conv0_1 = conv_block_nested_3D(filters[0] + filters[1], filters[0], filters[0],bn,use_res)
        self.conv1_1 = conv_block_nested_3D(filters[1] + filters[2], filters[1], filters[1],bn,use_res)
        self.conv2_1 = conv_block_nested_3D(filters[2] + filters[3], filters[2], filters[2],bn,use_res)
        self.conv3_1 = conv_block_nested_3D(filters[3] + filters[4], filters[3], filters[3],bn,use_res)


        # self.conv0_2 = conv_block_nested_3D(filters[0]*2 + filters[1], filters[0], filters[0],bn)
        # self.conv1_2 = conv_block_nested_3D(filters[1]*2 + filters[2], filters[1], filters[1],bn)

        # self.conv0_3 = conv_block_nested_3D(filters[0]*3 + filters[1], filters[0], filters[0],bn)

        ##hd2
        self.h0_hd2_PT = nn.Sequential(nn.MaxPool3d(4, 4, ceil_mode=True),conv_block_3D(filters[0],filters_cat))
        self.h1_hd2_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[1],filters_cat))

        self.h2_hd2_PT = conv_block_3D(filters[2],filters_cat)

        self.h3_hd2_PT = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),conv_block_3D(filters[3],filters_cat))

        self.h4_hd2_PT = nn.Sequential(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),conv_block_3D(filters[4],filters_cat))



        ##hd1
        self.h0_hd1_PT = nn.Sequential(nn.MaxPool3d(2, 2, ceil_mode=True),conv_block_3D(filters[0],filters_cat))

        self.h1_hd1_PT = conv_block_3D(filters[1],filters_cat)

        self.h2_hd1_PT = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),conv_block_3D(filters_cat_total,filters_cat))

        self.h3_hd1_PT = nn.Sequential(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),conv_block_3D(filters[3],filters_cat))

        self.h4_hd1_PT = nn.Sequential(nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),conv_block_3D(filters[4],filters_cat))


        ##hd0:
        self.h0_hd0_PT = conv_block_3D(filters[0],filters_cat)
        self.h1_hd0_PT = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),conv_block_3D(filters_cat_total,filters_cat))
        self.h2_hd0_PT = nn.Sequential(nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True),conv_block_3D(filters_cat_total,filters_cat))
        self.h3_hd0_PT = nn.Sequential(nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True),conv_block_3D(filters[3],filters_cat))
        self.h4_hd0_PT = nn.Sequential(nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True),conv_block_3D(filters[4],filters_cat))



        ##deconder_conv:320-320
        self.conv_hd3_hd2=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd2_hd1=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd1_hd0=conv_block_3D(filters_cat_total,filters_cat_total)
        self.conv_hd0_hd0=conv_block_3D(filters_cat_total,filters_cat_total)



        self.out= nn.Conv3d(filters_cat_total ,out_channels, kernel_size=1)
 

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

        #print('x0',x0_0.shape,x1_0.shape,x2_0.shape,x3_0.shape)
        h0_2=self.h0_hd2_PT(x0_0)
        h1_2=self.h1_hd2_PT(x1_0)
        h2_2=self.h2_hd2_PT(x2_0)
        h3_2=self.h3_hd2_PT(x3_0)
       
        #print('h_2',h0_2.shape,h1_2.shape,h2_2.shape,h3_2.shape)
        h2_0= self.conv_hd2_hd1(torch.cat([h0_2,h1_2,h2_2,h3_2], 1))
        #print('h2_0',h2_0.shape)


        h0_1=self.h0_hd1_PT(x0_0)
        h1_1=self.h1_hd1_PT(x1_0)
        h2_1=self.h2_hd1_PT(h2_0)
        h3_1=self.h3_hd1_PT(x3_0)

        #print('h_2',h0_1.shape,h1_1.shape,h2_1.shape,h3_1.shape)

        h1_0= self.conv_hd1_hd0(torch.cat([h0_1,h1_1,h2_1,h3_1], 1))
        #print('h1_0',h1_0.shape)


        h0_0=self.h0_hd0_PT(x0_0)
        h1_0=self.h1_hd0_PT(h1_0)
        h2_0=self.h2_hd0_PT(h2_0)
        h3_0=self.h3_hd0_PT(x3_0)
        #print('h_0',h0_0.shape,h1_0.shape,h2_0.shape,h3_0.shape)
        h0_0= self.conv_hd0_hd0(torch.cat([h0_0,h1_0,h2_0,h3_0], 1))

        output = self.out(h0_0)
        return output




from torchsummary import summary


if __name__ == "__main__":
    model=UNet3plus_nest3_3d_depth3(
                in_channels=1,
                out_channels=1,
                n1=16)
    #model=UNet3D(in_channels=1).cuda()
    summary(model,[1,96,96,96])



