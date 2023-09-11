"""
EmoStarGAN
Copyright (c) 2023-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import torch
from munch import Munch


def compute_coding_loss(nets, args, mel, label, gt_model):
    args = Munch(args)
    with torch.no_grad():
        true_label = gt_model(mel, label).detach()
    _ = [nets[key].to(mel.device) for key in nets.keys()]
    code = nets.coder(mel)

    code_loss = torch.nn.functional.smooth_l1_loss(code, true_label)
    #code_angle = torch.arccos((code * true_label).sum(dim=-1) / (torch.linalg.norm(code, dim= -1) * torch.linalg.norm(true_label, dim=-1)))
    code_loss = args.lambda_coding_loss * code_loss #+ 0.1 * code_angle.sum()

    return code_loss, Munch(coding_loss=code_loss.item())
