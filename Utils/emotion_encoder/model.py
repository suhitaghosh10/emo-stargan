"""
EmoStarGAN
Copyright (c) 2023-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import torch
import torch.nn as nn
from munch import Munch
from Models.style_module import StyleEncoder

class EmoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = StyleEncoder(64, 64, 5, 512)
        # first stage pre-trained emotion style encoder
        model_params = torch.load('emotion_style_encoder_pretrained_first_stage.pth', map_location='cpu')['model']['style_encoder']
        self.encoder.load_state_dict(model_params)


        self.fc_o = nn.Sequential(nn.Linear(512, 256),
                                  nn.LeakyReLU(0.1),
                                  nn.Dropout(0.2),
                                  nn.Linear(256, 64),
                                  nn.LeakyReLU(0.1),
                                  nn.Dropout(0.1),
                                  nn.Linear(64, 5),
                                  nn.Softmax(dim=-1))


    def forward(self, mel):
        latent = self.encoder.get_shared_feature(mel)
        final = self.encoder.get_all_pojections(mel)
        weights = self.fc_o(latent).unsqueeze(-1)
        weights = weights ** 2
        code = torch.sum(weights*final, dim=1)

        return  code

    def get_distribution(self, mel):
        latent = self.encoder.get_shared_feature(mel).detach()
        weights = self.fc_o(latent).unsqueeze(-1)

        return weights



def build_model():
    coder = EmoEncoder()
    coder_ema = copy.deepcopy(coder)

    nets = Munch(coder=coder)
    nets_ema = Munch(coder=coder_ema)

    return nets, nets_ema

