"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import os.path as osp
os.environ['HTTP_PROXY'] = 'http://fp:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp:3210/'
import copy
import math

import whisper
from munch import Munch
import numpy as np
import torch
import torch.nn as nn
from argparse import Namespace
import torch.nn.functional as F
from parallel_wavegan.utils import load_model
from Utils.EMORECOG.model_alternate import build_model as bm
from Utils.EMO_ENCODER.model import build_model as full_emo_bm

class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class UpSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.interpolate(x, scale_factor=(2, 1), mode='nearest')
        elif self.layer_type == 'half':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        else:
            raise RuntimeError('Got unexpected upsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)


class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample(x)
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
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample='none'):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = UpSample(upsample)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        x = self.upsample(x)
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.filter = torch.tensor([[-1, -1, -1],
                                    [-1, 8., -1],
                                    [-1, -1, -1]]).to(device) / w_hpf

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=48*8, w_hpf=1, F0_channel=0, F0_model = None):
        super().__init__()

        self.stem = nn.Conv2d(1, dim_in, 3, 1, 1)
        #self.conv11_x = nn.Conv2d(512, 256, 1, 1)
        self.proj_x_l1 = nn.Conv2d(512, 512, 1, 1)
        self.proj_x_l2 = nn.Conv2d(512, 512, 1, 1)
        self.proj_x_l3 = nn.Conv2d(512, 512, 1, 1)
        self.x_norm = nn.InstanceNorm2d(512, affine=True)
        self.F0_model = F0_model
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_out = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, 1, 1, 1, 0))
        self.F0_channel = F0_channel
        # down/up-sampling blocks
        repeat_num = 4 #int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1

        for lid in range(repeat_num):
            if lid in [1, 3]:
                _downtype = 'timepreserve'
            else:
                _downtype = 'half'

            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=_downtype))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=_downtype))  # stack-like
            dim_in = dim_out

        # bottleneck blocks (encoder)
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))

        # F0 blocks
        if F0_channel != 0:
            self.decode.insert(
                0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out, style_dim, w_hpf=w_hpf))

        # bottleneck blocks (decoder)
        for _ in range(2):
            self.decode.insert(
                    0, AdainResBlk(dim_out + int(F0_channel / 2), dim_out + int(F0_channel / 2), style_dim, w_hpf=w_hpf))

        if F0_channel != 0:
            self.F0_conv = nn.Sequential(
                ResBlk(F0_channel, int(F0_channel / 2), normalize=True, downsample="half"),
            )
            self.F0_GRU_level1 = nn.GRU(int(F0_channel / 2)*5, int(F0_channel / 4)*5, 1, batch_first=True, bidirectional=True)
            self.F0_GRU_level2 = nn.GRU(int(F0_channel / 2) * 5, int(F0_channel / 4) * 5, 2, batch_first=True,
                                        bidirectional=True)
            self.F0_GRU_level3 = nn.GRU(int(F0_channel/2) * 5, int(F0_channel / 4) * 5, 3, batch_first=True,
                                        bidirectional=True)
            self.F0_Norm = nn.InstanceNorm2d(int(F0_channel/2), affine=True)

        # Added transformer layer
        self.attention_layers_l1 = nn.ModuleList()
        for i in range(1):
            self.attention_layers_l1.append(
                nn.TransformerEncoderLayer(d_model=2560, nhead=5, activation="relu", batch_first=True, dim_feedforward=2560))

        self.attention_layers_l2 = nn.ModuleList()
        for i in range(1):
            self.attention_layers_l2.append(
                nn.TransformerEncoderLayer(d_model=2560, nhead=5, activation="relu", batch_first=True,dim_feedforward=2560))

        self.attention_layers_l3 = nn.ModuleList()
        for i in range(1):
            self.attention_layers_l3.append(
                nn.TransformerEncoderLayer(d_model=2560, nhead=5, activation="relu", batch_first=True, dim_feedforward=2560))


        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None, F0=None, training=False):
        """if self.F0_model is not None:
            F0 = self.F0_model.get_feature_GAN(x)"""

        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)

        if F0 is not None:
            F0 = self.F0_conv(F0)
            F0 = F.adaptive_avg_pool2d(F0, [x.shape[-2], x.shape[-1]])

            x = torch.cat([x, F0], axis=1)


        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        return self.to_out(x)

    def get_self_attended_spectrogram(self, x):
        frame = x.size(-1)
        channel = 1
        bins = x.size(-2)
        x = x.view(x.size(0), frame, bins)
        for attn_layer in self.attention_layers:
            x = attn_layer(x)
        x = x.view(x.size(0), channel, bins, frame)

        return x

    def get_content_representation(self, x, masks=None):
        x = self.stem(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)

        return x



class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=48, num_domains=2, hidden_dim=384):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, hidden_dim)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, num_domains=2, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]

    def forward(self, x, y):
        h = self.shared(x)

        h = h.view(h.size(0), -1)
        out = []

        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s

    def get_shared_feature(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        return h


class Discriminator(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()

        # real/fake discriminator
        self.dis = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # adversarial classifier
        self.cls = Discriminator2d(dim_in=dim_in, num_domains=num_domains,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num)
        # auxiliary classifier while training on ESD
        self.aux_cls = Discriminator2d(dim_in= dim_in, num_domains= 5,
                                  max_conv_dim=max_conv_dim, repeat_num=repeat_num )
        self.num_domains = num_domains

    def forward(self, x, y):
        return self.dis(x, y)

    def classifier(self, x):
        return self.cls.get_feature(x)

    def aux_classifier(self, x):
        return self.aux_cls.get_feature(x)


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Discriminator2d(nn.Module):
    def __init__(self, dim_in=48, num_domains=2, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [nn.Conv2d(1, dim_in, 3, 1, 1)]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 5, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def get_feature(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out

    def forward(self, x, y):
        out = self.get_feature(x)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
       # print('-------',idx, out.shape, y)
        out = out[idx, y]  # (batch)
        return out


def build_model(args, F0_model, ASR_model):
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel, F0_model=copy.deepcopy(F0_model))
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    discriminator = Discriminator(args.dim_in, args.num_domains, args.max_conv_dim, args.n_repeat)
    vocoder = load_model(args.vocoder_path)
    vocoder.remove_weight_norm()

    _, emotion_encoder = bm(F0_model=F0_model, ASR_model=ASR_model)
    model_params = torch.load(args.Emotion_encoder_path, map_location='cpu')['model_ema']['classifier']
    emotion_encoder.classifier.load_state_dict(model_params)

    """_, emotion_encoder = full_emo_bm()
    model_params = torch.load(args.Emotion_encoder_path, map_location='cpu')['model_ema']['coder']
    emotion_encoder.coder.load_state_dict(model_params)"""

    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator,
                 f0_model=F0_model,
                 asr_model=ASR_model,
                 vocoder= vocoder,
                 emotion_encoder=emotion_encoder.classifier)

    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema