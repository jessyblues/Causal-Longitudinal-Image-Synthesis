import matplotlib


matplotlib.use('Agg')
import torch
from torch import nn
from models.encoders import psp_encoders
from GAN_mri_package.model_3D import StyledGenerator
from GAN_mri_package.utils import get_generator_from_ckpt
from configs.paths_config import model_paths
import numpy as np
import pdb
from models.encoders.my_modules import noise_encoder

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        self.encoder = torch.nn.DataParallel(self.encoder.cuda(), device_ids=np.arange(torch.cuda.device_count()).tolist())
        self.decoder = StyledGenerator(code_dim=128, init_size=(6, 7, 6))
        self.decoder = torch.nn.DataParallel(self.decoder, device_ids=np.arange(torch.cuda.device_count()).tolist())
        if opts.noise_encoder:
            self.noise_encoder = noise_encoder(n_channels=1, n_classes=1)
            self.noise_encoder = torch.nn.DataParallel(self.noise_encoder, device_ids=np.arange(torch.cuda.device_count()).tolist())

        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'E4E_3d':
            encoder = psp_encoders.E4E_3d(32, 'ir_se', self.opts)

        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cuda')
            #pdb.set_trace()
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
            if self.opts.encoder and 'noise_encoder' in ckpt.keys():
                self.noise_encoder.load_state_dict(get_keys(ckpt, 'noise_encoder'), strict=True)
            self.__load_latent_avg(ckpt)
        
        else:
            #print('Loading encoders weights from sc!')
            #encoder_ckpt = torch.load(model_paths['ir_se50'])
            #self.encoder.load_state_dict(encoder_ckpt, strict=False)
            print('Loading decoder weights from pretrained!')
            ckpt = torch.load(self.opts.stylegan_weights)
            self.decoder.load_state_dict(ckpt['generator'])
            self.__load_latent_avg(ckpt, repeat=self.encoder.module.style_count)

    def forward(self, x, input_code=False, 
                return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)


        input_is_latent = not input_code
        if self.opts.noise_encoder:
            
            noise_list = list(self.noise_encoder(x))
            noise_list.append(torch.randn((x.shape[0], 1, 192, 224, 192)))

            images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             return_latents=return_latents,
                                             step=5, noise=noise_list)
        
        else:
            images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             return_latents=return_latents,
                                             step=5)
                                            

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
        elif self.opts.start_from_latent_avg:
            # Compute mean code based on a large number of latents (10,000 here)
            with torch.no_grad():
                self.latent_avg = self.decoder.mean_latent(10000).to(self.opts.device)
        else:
            self.latent_avg = None
        if repeat is not None and self.latent_avg is not None:
            self.latent_avg = self.latent_avg.repeat(repeat, 1)
