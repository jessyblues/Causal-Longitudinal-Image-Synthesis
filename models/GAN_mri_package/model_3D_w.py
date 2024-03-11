import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random
import pdb
import numpy as np

def init_linear(linear):
    init.xavier_normal_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class EqualLR: ## set equal learning rate
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        #pdb.set_trace()
        fan_in = torch.nn.init._calculate_correct_fan(weight, mode='fan_in')

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:, 1:]
            + weight[:, :, :-1, 1:, 1:]
            + weight[:, :, 1:, :-1, 1:]
            + weight[:, :, :-1, :-1, 1:]
            + weight[:, :, 1:, 1:, :-1]
            + weight[:, :, :-1, 1:, :-1]
            + weight[:, :, 1:, :-1, :-1]
            + weight[:, :, :-1, :-1, :-1]
        ) / 8

        out = F.conv_transpose3d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:, 1:]
            + weight[:, :, :-1, 1:, 1:]
            + weight[:, :, 1:, :-1, 1:]
            + weight[:, :, :-1, :-1, 1:]
            + weight[:, :, 1:, 1:, :-1]
            + weight[:, :, :-1, 1:, :-1]
            + weight[:, :, 1:, :-1, :-1]
            + weight[:, :, :-1, :-1, :-1]
        ) / 8

        out = F.conv3d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv3d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv3d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv3d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
                               [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
                               [[1, 2, 1], [2, 4, 2], [1, 2, 1]], ], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv3d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv3d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv3d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel, kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv3d(out_channel, out_channel, kernel2, padding=pad2),
                    nn.AvgPool3d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv3d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm3d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        #pdb.set_trace()
        style = self.style(style).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        gamma, beta = style.chunk(2, 1)
        #pdb.set_trace()
        ## not modified yet
        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=(7, 7, 8)):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size[0], size[1], size[2]))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=64,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)

        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv3d(
                            in_channel, out_channel, kernel_size, padding=padding
                        ),
                        Blur(out_channel),
                    )

            else:
                self.conv1 = EqualConv3d(
                    in_channel, out_channel, kernel_size, padding=padding
                )

        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv3d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        
        if type(style) in [tuple, list]:
            pass
        else:
            style = [style, style]

        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style[0])

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style[1])

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(128, 128, 3, 1, initial=True, style_dim=code_dim),  # (7, 7, 8)
                StyledConvBlock(128, 128, 3, 1, upsample=True, style_dim=code_dim),  # (14, 14, 16)
                StyledConvBlock(128, 64, 3, 1, upsample=True, style_dim=code_dim),  # (28, 28, 32)
                StyledConvBlock(64, 32, 3, 1, upsample=True, style_dim=code_dim),  # (56, 56, 64)
                StyledConvBlock(32, 16,  3, 1, upsample=True, style_dim=code_dim),  # (112, 112, 128)
                StyledConvBlock(16, 8, 3, 1, upsample=True, fused=fused, style_dim=code_dim),  # (224, 224, 256)
                #StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 256
                #StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
                #StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
            ]
        )

        self.to_mri = nn.ModuleList(
            [
                EqualConv3d(128, 1, 1),
                EqualConv3d(128, 1, 1),
                EqualConv3d(64, 1, 1),
                #EqualConv2d(512, 3, 1),
                #EqualConv2d(512, 3, 1),
                #EqualConv3d(256, 3, 1),
                EqualConv3d(32, 1, 1),
                EqualConv3d(16, 1, 1),
                EqualConv3d(8, 1, 1),
                #EqualConv3d(16, 1, 1),
            ]
        )
        #self.relu = nn.ReLU

        # self.blur = Blur()
        self.n_latent = len(self.progression)*2

    def forward(self, styles, noise, step=0, alpha=-1, mixing_range=(-1, -1), inject_index=None):
        
        out = noise[0]
        
        if len(styles) < 2:
            inject_index = [len(self.progression) + 1]

        else:
            inject_index = sorted(random.sample(list(range(step)), len(styles) - 1))

        crossover = 0

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if mixing_range == (-1, -1):
                if crossover < len(inject_index) and i > inject_index[crossover]:
                    crossover = min(crossover + 1, len(styles))

                style_step = styles[crossover]

            else:
                if mixing_range[0] <= i <= mixing_range[1]:
                    style_step = styles[1]

                else:
                    style_step = styles[0]

            if i > 0 and step > 0:
                out_prev = out
                
            out = conv(out, style_step, noise[i])

            if i == step:
                #pdb.set_trace()
                out = to_rgb(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_mri = self.to_mri[i - 1](out_prev)
                    skip_mri = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')(skip_mri)
                    out = (1 - alpha) * skip_mri + alpha * out

                break
        

        return out


class StyledGenerator(nn.Module):
    def __init__(self, code_dim=64, n_mlp=8):
        super().__init__()

        self.generator = Generator(code_dim)
        self.n_latent = self.generator.n_latent

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)
        self.style_dim = code_dim

    def forward(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
        input_is_latent=True,
        return_latents=False
    ):
        styles = []

        if type(input) not in (list, tuple):
            input = [input]
        #pdb.set_trace()
        for i in input:
            if not input_is_latent: ## input is z 
                styles.append(self.style(i))
            else:
                styles.append(i)


        batch = input[0].shape[0]

        if noise is None:
            noise = []

            for i in range(step + 1):
                size_x = 7 * 2 ** i
                size_y = 8 * 2 ** i 
                noise.append(torch.randn((batch, 1, size_x, size_x, size_y), device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        out, latent = self.generator(styles, noise, step, alpha, mixing_range=mixing_range)
       #
       # pdb.set_trace()
        if return_latents:
            return out, latent
        else: 
            return out

    def mean_style(self, input):
        style = self.style(input).mean(0, keepdim=True)

        return style
    
    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent
    
    def get_latent(self, input):
        return self.style(input)


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_mri_activate=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(8, 16, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 32
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 16
                #ConvBlock(256, 512, 3, 1, downsample=True),  # 
                #ConvBlock(512, 512, 3, 1, downsample=True),  # 
                #ConvBlock(512, 512, 3, 1, downsample=True),  # 
                ConvBlock(128, 128, 3, 1, downsample=True),  # 8
                ConvBlock(129, 128, 3, 1, 7, 0),
            ]
        )
        

        def make_from_mri(out_channel):
            if from_mri_activate:
                return nn.Sequential(EqualConv3d(1, out_channel, 1), nn.LeakyReLU(0.2))

            else:
                return EqualConv3d(1, out_channel, 1)

        self.from_mri = nn.ModuleList(
            [
                make_from_mri(8),
                make_from_mri(16),
                make_from_mri(32),
                make_from_mri(64),
                make_from_mri(128),
                make_from_mri(128),
                #make_from_mri(128),
                #make_from_rgb(512),
                #make_from_rgb(512),
            ]
        )

        # self.blur = Blur()

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(256, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):

            #pdb.set_trace()
            
            index = self.n_layer - i - 1

            if i == step: ## 
                #pdb.set_trace()
                out = self.from_mri[index](input)

            if i == 0:
                #pdb.set_trace()
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 7, 7, 8)
                #pdb.set_trace()
                out = torch.cat([out, mean_std], 1)

            #pdb.set_trace()
            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_mri = F.avg_pool3d(input, 2)
                    skip_mri = self.from_mri[index + 1](skip_mri)

                    out = (1 - alpha) * skip_mri + alpha * out
        #pdb.set_trace()
        out = out.squeeze(2).squeeze(2)
        out = out.view(out.shape[0], -1)
        #pdb.set_trace()
        out = self.linear(out)

        return out