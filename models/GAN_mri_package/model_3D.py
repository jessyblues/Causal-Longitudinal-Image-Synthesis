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
    def __init__(self, channel, size=(6, 7, 6)):
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
        init_size=(6, 7, 6)
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel, init_size)

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
    def __init__(self, code_dim, fused=True, init_size=(6, 7, 6)):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                StyledConvBlock(128, 128, 3, 1, initial=True, style_dim=code_dim, init_size=init_size),  # (7, 7, 8)
                StyledConvBlock(128, 128, 3, 1, upsample=True, style_dim=code_dim, init_size=init_size),  # (14, 14, 16)
                StyledConvBlock(128, 64, 3, 1, upsample=True, style_dim=code_dim, init_size=init_size),  # (28, 28, 32)
                StyledConvBlock(64, 32, 3, 1, upsample=True, style_dim=code_dim, init_size=init_size),  # (56, 56, 64)
                StyledConvBlock(32, 16,  3, 1, upsample=True, style_dim=code_dim, init_size=init_size),  # (112, 112, 128)
                StyledConvBlock(16, 8, 3, 1, upsample=True, fused=fused, style_dim=code_dim, init_size=init_size),  # (224, 224, 256)

            ]
        )

        self.to_mri = nn.ModuleList(
            [
                EqualConv3d(128, 1, 1),
                EqualConv3d(128, 1, 1),
                EqualConv3d(64, 1, 1),
                EqualConv3d(32, 1, 1),
                EqualConv3d(16, 1, 1),
                EqualConv3d(8, 1, 1)
            ]
        )

        self.n_latent = len(self.progression)*2

    def forward(self, styles, noise, step=0, alpha=-1, mixing_range=(-1, -1), inject_index=None):
        out = noise[0]

        #pdb.set_trace()
        if len(styles) < 2:
            
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)
            #pdb.set_trace()
            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        #crossover = 0

        #pdb.set_trace()
        for i, (conv, to_mri) in enumerate(zip(self.progression, self.to_mri)):

            #pdb.set_trace()
            if i > 0 and step > 0:
                out_prev = out
                
            out = conv(out, (latent[:, 2*i], latent[:, 2*i+1]), noise[i])

            if i == step:
                #pdb.set_trace()
                out = to_mri(out)

                if i > 0 and 0 <= alpha < 1:
                    skip_mri = self.to_mri[i - 1](out_prev)
                    #pdb.set_trace()
                    skip_mri = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')(skip_mri)
                    #pdb.set_trace()
                    out = (1 - alpha) * skip_mri + alpha * out

                break
        

        return out, latent


class StyledGenerator(nn.Module):
    
    def __init__(self, code_dim=128, n_mlp=8, init_size=(6, 7, 6)):
        
        super().__init__()

        self.generator = Generator(code_dim, init_size=init_size)
        self.n_latent = self.generator.n_latent

        layers = [PixelNorm()]
        for i in range(n_mlp):
            layers.append(EqualLinear(code_dim, code_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.style = nn.Sequential(*layers)
        self.style_dim = code_dim
        self.init_size = init_size

    def forward(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
        mean_style=None,
        style_weight=0,
        mixing_range=(-1, -1),
        input_is_latent=False,
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
                size_x = self.init_size[0] * 2 ** i
                size_y = self.init_size[1] * 2 ** i 
                size_z = self.init_size[2] * 2 ** i
                noise.append(torch.randn((batch, 1, size_x, size_y, size_z), device=input[0].device))

        if mean_style is not None:
            styles_norm = []

            for style in styles:
                styles_norm.append(mean_style + style_weight * (style - mean_style))

            styles = styles_norm

        out, latent = self.generator(styles, noise, step, alpha, mixing_range=mixing_range)

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
        ).cuda()
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent
    
    def get_latent(self, input):
        return self.style(input)


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_mri_activate=False, init_size=(6, 7, 6), set_mini_batch_discriminator=False):
        super().__init__()

        self.progression = nn.ModuleList(
            [
                ConvBlock(8, 16, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 32
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 16
                ConvBlock(128, 128, 3, 1, downsample=True),  # 8
                ConvBlock(129, 128, 3, 1, 6, 0)
            ]
        ) if (init_size == (6, 7, 6) or init_size == (6, 6, 6)) else nn.ModuleList(
            [
                ConvBlock(8, 16, 3, 1, downsample=True, fused=fused),  # 128
                ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 64
                ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 32
                ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 16
                ConvBlock(128, 128, 3, 1, downsample=True),  # 8
                ConvBlock(129, 128, 3, 1, 7, 0)
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
                make_from_mri(128)

            ]
        )

        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)
        self.init_size = init_size
        self.set_mini_batch_discriminator = set_mini_batch_discriminator
        

        self.linear = EqualLinear(256, 1)


    def forward(self, input, step=0, alpha=-1, return_feature=False):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step: 
                out = self.from_mri[index](input)

            if i == 0:

                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand((out.size(0), 1) + self.init_size)
                #pdb.set_trace()
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha < 1:
                    skip_mri = F.avg_pool3d(input, 2)
                    skip_mri = self.from_mri[index + 1](skip_mri)

                    out = (1 - alpha) * skip_mri + alpha * out

        out = out.squeeze(2).squeeze(2)
        feature = out.view(out.shape[0], -1)

        result = self.linear(feature)

        return result, feature


    


class MinibatchDiscrimination(nn.Module):
    
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims))
        init.normal_(self.T, 0, 1)
        

        self.norm_layer1 = nn.BatchNorm1d(num_features=256)
        #self.norm_layer2 = nn.BatchNorm2d()
        self.linear1 = EqualLinear(out_features, 1)
        #self.init_size = init_size
        


    def forward(self, feature):
        # x is NxA
        # T is AxBxC

        #x = self.compute_feature(input=image, step=step, alpha=alpha)

        x = feature.view(feature.shape[0], -1)
        x = self.norm_layer1(x)
        #pdb.set_trace()

        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        #expnorm = torch.exp(-norm)
        #o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        #pdb.set_trace()
        o_b = norm.sum(0) 
        out = self.linear1(o_b)
        #pdb.set_trace()
        #x = torch.cat([x, o_b], 1)
        
        return out
    

class delta_w_encoder(nn.Module):

    def __init__(self, in_dim, num_layers, dims_per_layer=None, same_dim_per_layer=1000, out_dim=128*12):
        super().__init__()

        self.mlp = []
        self.actv = nn.LeakyReLU(0.2)

        if dims_per_layer is None:

            for idx in range(num_layers):
                if idx == 0:
                    self.mlp += [nn.Linear(in_dim, same_dim_per_layer), self.actv]
                else:
                    self.mlp += [nn.Linear(same_dim_per_layer, same_dim_per_layer), self.actv]
            self.mlp += [nn.Linear(same_dim_per_layer, out_dim), self.actv]
        
        else:
            for idx, dim in enumerate(dims_per_layer):
                if idx == 0:
                    self.mlp += [nn.Linear(in_dim, dim), self.actv]
                else:
                    self.mlp += [nn.Linear(dims_per_layer[idx-1], dim), self.actv]
            self.mlp += [nn.Linear(dim, out_dim), self.actv]

        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):

        delta_w = self.mlp(x)

        return delta_w
    

class regression_volume(nn.Module):

    def __init__(self, w_dim, volume_dim, hidden_dims=[]):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)

        if hidden_dims == []:
            self.fcs = [nn.Linear(w_dim, volume_dim)]
        else:
            for idx, hidden_dim in enumerate(hidden_dims):
                if idx == 0:
                    self.fcs = [nn.Linear(w_dim, hidden_dim), self.actv]
                else:
                    self.fcs += [nn.Linear(hidden_dims[idx-1], hidden_dim), self.actv]
            self.fcs += [nn.Linear(hidden_dim, volume_dim), self.actv]
    
        self.fcs = nn.Sequential(*self.fcs)
    
    def forward(self, w):
        predict_volume = self.fcs(w)
        return predict_volume

class regression_volume2(nn.Module):

    def __init__(self, w_dim, volume_dim, hidden_dims=[]):
        super().__init__()
        self.actv = nn.LeakyReLU(0.2)

        
        self.direction_model = nn.Linear(w_dim, volume_dim)
        
        if hidden_dims == []:
            self.attitude_fcs = [nn.Linear(w_dim, volume_dim)]
        else:
            for idx, hidden_dim in enumerate(hidden_dims):
                if idx == 0:
                    self.attitude_fcs = [nn.Linear(w_dim, hidden_dim), self.actv]
                else:
                    self.attitude_fcs += [nn.Linear(hidden_dims[idx-1], hidden_dim), self.actv]
            self.attitude_fcs += [nn.Linear(hidden_dim, volume_dim), self.actv]
    
        self.attitude_fcs = nn.Sequential(*self.attitude_fcs)
    
    
    def forward(self, bs_w, delta_w):
        predict_volume = self.attitude_fcs(bs_w) * self.direction_model(delta_w)
        return predict_volume







