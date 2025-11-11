from enum import Enum
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module, Conv3d, BatchNorm3d, LeakyReLU

from models.e4e.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE, _upsample_add, bottleneck_IR_SE_3D, bottleneck_IR_3D, _upsample_add_3d
from models.e4e.stylegan2.model import EqualLinear


class ProgressiveStage(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Delta12Training = 12
    Delta13Training = 13
    Delta14Training = 14
    Delta15Training = 15
    Delta16Training = 16
    Delta17Training = 17
    Inference = 18

class ProgressiveStage_3D(Enum):
    WTraining = 0
    Delta1Training = 1
    Delta2Training = 2
    Delta3Training = 3
    Delta4Training = 4
    Delta5Training = 5
    Delta6Training = 6
    Delta7Training = 7
    Delta8Training = 8
    Delta9Training = 9
    Delta10Training = 10
    Delta11Training = 11
    Inference = 12

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x
    
class GradualStyleBlock3D(Module):
    def __init__(self, in_c, out_c, spatial):
        """
        spatial: int or tuple
            If int: cubic input (S, S, S)
            If tuple: (D, H, W)
        """
        super().__init__()

        # --- parse spatial shape ---
        if isinstance(spatial, int):
            D = H = W = spatial
        else:
            D, H, W = spatial

        self.out_c = out_c

        # --- how many 2x downsamplings until reaching 1x1x1 ---
        # Each downsample divides D,H,W by 2
        num_pools = int(np.log2(min(D, H, W)))

        modules = []

        # First conv
        modules += [
            Conv3d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            LeakyReLU(0.2)
        ]

        # Additional convs
        for _ in range(num_pools - 1):
            modules += [
                Conv3d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                LeakyReLU(0.2)
            ]

        self.convs = nn.Sequential(*modules)

        # Linear: input dim = out_c * 1 * 1 * 1 (after downsampling)
        # But safer to compute dynamically (in case D,H,W don't reduce to 1)
        test_in = torch.zeros(1, in_c, D, H, W)
        with torch.no_grad():
            test_out = self.convs(test_in)
            flat_dim = test_out.view(1, -1).shape[1]

        self.linear = EqualLinear(flat_dim, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)        # (N, out_c, D', H', W')
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 4
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out


class Encoder4Editing(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(Encoder4Editing, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        ''' Get a list of the initial dimension of every delta from which it is applied '''
        return list(range(self.style_count))  # Each dimension has a delta applied to it

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print('Changed progressive stage to: ', new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer main W and duplicate it
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        stage = self.progressive_stage.value
        features = c3
        for i in range(1, min(stage + 1, self.style_count)):  # Infer additional deltas
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))  # FPN's middle features
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))  # FPN's fine features
                features = p1
            delta_i = self.styles[i](features)
            w[:, i] += delta_i
        return w

class Encoder4Editing3D(nn.Module):
    def __init__(self, num_layers, mode='ir_3d', opts=None):
        super().__init__()

        assert num_layers in [25]
        assert mode in ['ir_3d', 'ir_se_3d']

        # ===== Backbone blocks =====
        blocks = get_blocks(num_layers)
        if mode == 'ir_3d':
            unit_module = bottleneck_IR_3D
        else:
            unit_module = bottleneck_IR_SE_3D

        # ===== Input conv (3D) =====
        self.input_layer = nn.Sequential(
            Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm3d(16),
            PReLU(16)
        )

        # ===== Construct full body =====
        modules = []
        for block in blocks:
            for b in block:
                modules.append(
                    unit_module(b.in_channel, b.depth, b.stride)
                )
        self.body = nn.Sequential(*modules)

        # ===== Style blocks (3D) =====
        self.styles = nn.ModuleList()
        log_size = int(math.log(opts.stylegan_size, 2))  # e.g., 256 → 8
        self.style_count = 2 * log_size - 2

        self.coarse_ind = 3
        self.middle_ind = 7

        # --- YOUR REQUIREMENT: smallest cube = (6,7,6) ---
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock3D(512, 512, (6, 7, 6))
            elif i < self.middle_ind:
                style = GradualStyleBlock3D(512, 512, (12, 14, 12))
            else:
                style = GradualStyleBlock3D(512, 512, (24, 28, 24))
            self.styles.append(style)

        # ===== FPN Lateral layers (3D) =====
        self.latlayer1 = Conv3d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = Conv3d(128, 512, kernel_size=1, stride=1, padding=0)

        self.progressive_stage = ProgressiveStage.Inference

    def get_deltas_starting_dimensions(self):
        return list(range(self.style_count))

    def set_progressive_stage(self, new_stage: ProgressiveStage):
        self.progressive_stage = new_stage
        print("Changed progressive stage to:", new_stage)

    def forward(self, x):
        x = self.input_layer(x)

        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x     # highest resolution
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x     # smallest (deepest)

        # ===== Main W =====
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)

        # ===== Progressive deltas =====
        stage = self.progressive_stage.value
        features = c3

        for i in range(1, min(stage + 1, self.style_count)):

            if i == self.coarse_ind:
                p2 = _upsample_add_3d(c3, self.latlayer1(c2))
                features = p2

            elif i == self.middle_ind:
                p1 = _upsample_add_3d(p2, self.latlayer2(c1))
                features = p1

            delta_i = self.styles[i](features)
            w[:, i] += delta_i

        return w

class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(opts.stylegan_size, 2))
        self.style_count = 2 * log_size - 2

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2)

from monai.networks.nets import UNet

class NoiseEncoder3D(nn.Module):
    """
    A 3D noise encoder that predicts per-layer noise maps for StyleGAN3D.
    Uses a MONAI UNet backbone and outputs noise for multiple generator layers.
    """
    def __init__(self, 
                 in_channels=1,
                 base_channels=32,
                 noise_scales=(4, 8, 16, 32, 64)):
        """
        Args:
            in_channels: MRI channels (default 1)
            base_channels: UNet initial feature channels
            noise_scales: resolutions for generator noise layers
        """
        super().__init__()

        self.noise_scales = noise_scales

        # MONAI 3D UNet backbone
        self.unet = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=base_channels,
            channels=(base_channels, base_channels*2, base_channels*4, base_channels*8),
            strides=(2, 2, 2)
        )

        # For each noise scale, output a separate conv to generate noise
        self.heads = nn.ModuleDict()
        for s in noise_scales:
            self.heads[f"noise_{s}"] = nn.Conv3d(base_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: MRI volume, shape (N, 1, D, H, W)
        Returns:
            dict: { "noise_4": tensor, "noise_8": tensor, ... }
        """
        feat = self.unet(x)

        noises = {}
        N, C, D, H, W = feat.shape
        device = feat.device

        # Predict noise for each scale
        for s in self.noise_scales:
            # target shape for noise layer
            target = (N, 1, s, s, s)

            # Downsample or upsample UNet feature map to match noise size
            resized = torch.nn.functional.interpolate(
                feat, size=(s, s, s), mode="trilinear", align_corners=False
            )

            # pass through 1x1 conv
            noise_map = self.heads[f"noise_{s}"](resized)
            noises[f"noise_{s}"] = noise_map

        return noises