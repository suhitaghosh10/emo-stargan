import torch
from munch import Munch
import torch.nn.functional as F


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
