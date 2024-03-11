import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from GAN_face.model.core.wing import FAN

from collections import namedtuple
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm3d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter
import pdb

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv3d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm3d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm3d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv3d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool3d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool3d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm3d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        #pdb.set_trace()
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        pdb.set_trace()
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, short_cut=True, 
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()

        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.short_cut = short_cut
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv3d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv3d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv3d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.short_cut:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out
    

class style_transformer(nn.Module):
    def __init__(self, style_dim=64):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(style_dim,style_dim)
                                    ,nn.ReLU()
                                    ,nn.Linear(style_dim,style_dim))

    def forward(self, s):
        return self.layers(s)



class Generator(nn.Module):
    def __init__(self, img_size=(64, 64, 64), style_dim=64, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size[0]
        self.img_size = img_size

        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4

        for _ in range(repeat_num): #resolution from 8*8*8 to 64*64*64
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))



    def forward(self, x, s_new, s_old, masks=None, layer_new_s = [16, 32]):
        x = self.from_rgb(x)
        
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)

        for block in self.decode:
            #pdb.set_trace()
            if x.size(2) in layer_new_s:
                x = block(x, s_new)
            else:
                x = block(x, s_old)
            
            
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])
        return self.to_rgb(x)