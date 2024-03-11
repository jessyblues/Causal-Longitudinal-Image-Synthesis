from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, Sequential, Module
from torch.nn import Conv3d, BatchNorm3d, MaxPool3d, ConvTranspose3d, Upsample
from GAN_mri_package.dataset import MriFileFolderDataset
import pdb
import torch.nn as nn
from GAN_mri_package.model_3D import StyledGenerator
import sys
#sys.path.append('/home1/yujiali/cf_mri_2/Encoder_GAN/encoder4editing')
#from models.psp import pSp

class DoubleConv(Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = Sequential(
            Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm3d(out_channels),
            ReLU(inplace=True),
            Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm3d(out_channels),
            ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)

class Down(Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = Sequential(
            MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = Upsample(scale_factor=2)#align_corners=True)
        else:
            self.up = ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)# //为整数除法
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])
 
        x1 = F.pad(x1, [torch.div(diffX, 2, rounding_mode='floor'), diffX - torch.div(diffX, 2, rounding_mode='floor'),
                        torch.div(diffY, 2, rounding_mode='floor'), diffY - torch.div(diffY, 2, rounding_mode='floor'),
                        torch.div(diffZ, 2, rounding_mode='floor'), diffZ - torch.div(diffZ, 2, rounding_mode='floor')])
 
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)

class OutConv(Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = Conv3d(in_channels, out_channels, kernel_size=1)
        self.actv = Sigmoid()
 
    def forward(self, x):
        
        return self.conv(x)

class noise_encoder(Module):
    def __init__(self, n_channels, n_classes=1, bilinear=True):
        super(noise_encoder, self).__init__()
        self.n_channels = n_channels # 输入通道数
        self.n_classes = n_classes # 输出类别数
        self.bilinear = bilinear # 上采样方式
 
        self.inc = DoubleConv(n_channels, 4) # 输入层
        self.down1 = Down(4, 8)
        self.down2 = Down(8, 16)
        self.down3 = Down(16, 32)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 64)


        self.up1 = Up(128, 32, bilinear)
        self.up2 = Up(64, 16, bilinear)
        self.up3 = Up(32, 8, bilinear)
        self.up4 = Up(16, 8, bilinear)
        #self.up4 = Up(16, 4, bilinear)
        #self.outc = OutConv(8, n_classes) # 输出层

        self.out0 = OutConv(64, 1)
        self.out1 = OutConv(32, 1)
        self.out2 = OutConv(16, 1)
        self.out3 = OutConv(8, 1)
        self.out4 = OutConv(8, 1)
 
    def forward(self, x):
        x1 = self.inc(x) # 一开始输入
        x2 = self.down1(x1) # 四层左部分
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        out0 = self.out0(x6)
        x = self.up1(x6, x5)
        out1 = self.out1(x)# 四层右部分
        x = self.up2(x, x4)
        out2 = self.out2(x)
        x = self.up3(x, x3)
        out3 = self.out3(x)
        x = self.up4(x, x2)
        out4 = self.out4(x)


        return out0, out1, out2, out3, out4
