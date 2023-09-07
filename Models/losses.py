#coding:utf-8
"""
EmoStarGAN
Copyright (c) 2020-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import librosa
import torch
import torchaudio
from munch import Munch
from Data.transforms import build_transforms
from Utils.constants import *
from Models.features import get_spectral_centroid, get_loudness

import torch.nn.functional as F

def compute_d_loss(nets, args, x_real, y_org, sp_org, y_trg, z_trg=None, x_ref=None, use_r1_reg=True, use_adv_cls=False, use_con_reg=False, use_aux_cls=False):
    args = Munch(args)

    assert (z_trg is None) != (x_ref is None)
    # with real audios
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    
    # R1 regularizaition (https://arxiv.org/abs/1801.04406v4)
    if use_r1_reg:
        loss_reg = r1_reg(out, x_real)
    else:
        loss_reg = torch.FloatTensor([0]).to(x_real.device)
    
    # consistency regularization (bCR-GAN: https://arxiv.org/abs/2002.04724)
    loss_con_reg = torch.FloatTensor([0]).to(x_real.device)
    if use_con_reg:
        t = build_transforms()
        out_aug = nets.discriminator(t(x_real).detach(), y_org)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)

    
    # with fake audios
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
        #s_emo = nets.emotion_encoder.get_shared_feature(x_real)
            
        F0 = nets.f0_model.get_feature_GAN(x_real)
        x_fake = nets.generator(x_real, s_trg, masks=None, F0=F0)
    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)
    if use_con_reg:
        out_aug = nets.discriminator(t(x_fake).detach(), y_trg)
        loss_con_reg += F.smooth_l1_loss(out, out_aug)
    
    # adversarial classifier losses
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_real_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_org[y_org != y_trg])
        
        if use_con_reg:
            out_de_aug = nets.discriminator.classifier(t(x_fake).detach())
            loss_con_reg += F.smooth_l1_loss(out_de, out_de_aug)
    else:
        loss_real_adv_cls = torch.zeros(1).mean()

    # Aux speaker classifier
    if args.use_aux_cls and use_aux_cls:
        out_aux = nets.discriminator.aux_classifier(x_real)
        loss_index = (sp_org != -1)
        loss_aux_cls = F.cross_entropy(out_aux[loss_index], sp_org[loss_index])
    else:
        loss_aux_cls = torch.zeros(1).mean()

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg + \
            args.lambda_adv_cls * loss_real_adv_cls + \
            args.lambda_con_reg * loss_con_reg + \
            args.lambda_aux_cls * loss_aux_cls

    return loss, Munch(real=loss_real.item(),
                       fake=loss_fake.item(),
                       reg=loss_reg.item(),
                       real_adv_cls=loss_real_adv_cls.item(),
                       con_reg=loss_con_reg.item(),
                       real_aux_cls=loss_aux_cls.item())

def compute_g_loss(nets, args, x_real, y_org, sp_org, y_trg, z_trgs=None, x_refs=None, use_adv_cls=False, use_feature_loss=False, use_aux_cls=False):
    args = Munch(args)
    feature_loss_param = Munch(args.feature_loss)
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs
        
    # compute style vectors
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    
    # compute ASR/F0 features (real)
    with torch.no_grad():
        F0_real, GAN_F0_real, cyc_F0_real = nets.f0_model(x_real)
        ASR_real = nets.asr_model.get_feature(x_real)


    # adversarial losses
    x_fake = nets.generator(x_real, s_trg, masks=None, F0=GAN_F0_real)
    out = nets.discriminator(x_fake, y_trg) 
    loss_adv = adv_loss(out, 1)

    # compute ASR/F0 features (fake)
    F0_fake, GAN_F0_fake, _ = nets.f0_model(x_fake)
    ASR_fake = nets.asr_model.get_feature(x_fake)

    # diversity sensitive losses
    if z_trgs is not None:
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
    else:
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
    x_fake2 = nets.generator(x_real, s_trg2, masks=None, F0=GAN_F0_real)
    x_fake2 = x_fake2.detach()
    _, GAN_F0_fake2, _ = nets.f0_model(x_fake2)
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    loss_ds += F.smooth_l1_loss(GAN_F0_fake, GAN_F0_fake2.detach())

    # handcrafted feature losses
    if use_feature_loss:
        mel_fake = x_fake.transpose(-1, -2).squeeze()
        mel_real = x_real.transpose(-1, -2).squeeze()
        batch_size = mel_fake.size(0)
        with torch.no_grad():
            wav_real = [nets.vocoder.inference(mel_real[idx]) for idx in range(batch_size)]
            wav_real = torch.stack(wav_real, dim=0).squeeze()
        wav_fake = [nets.vocoder.inference(mel_fake[idx]) for idx in range(batch_size)]
        wav_fake = torch.stack(wav_fake, dim=0).squeeze()

        if feature_loss_param.use_loudness_loss:
            # loudness losses
            loudness_real = get_loudness(wav_real.squeeze())
            loudness_fake = get_loudness(wav_fake.squeeze())
            if feature_loss_param.use_log_loudness:
                loudness_real = torch.log10(loudness_real)
                loudness_fake  =torch.log10(loudness_fake)
            if feature_loss_param.use_norm_loudness:
                loudness_real = loudness_real/torch.mean(loudness_real, dim=-1, keepdim=True)
                loudness_fake = loudness_fake/torch.mean(loudness_fake, dim=-1, keepdim=True)
            if feature_loss_param.use_delta_loudness:
                loudness_fake_change = torchaudio.functional.compute_deltas(loudness_fake.unsqueeze(0)).squeeze()
                with torch.no_grad():
                    loudness_real_change = torchaudio.functional.compute_deltas(loudness_real.unsqueeze(0)).squeeze()
                loudness_loss = F.smooth_l1_loss(loudness_real_change, loudness_fake_change)
            else:
                loudness_loss = F.smooth_l1_loss(loudness_real, loudness_fake)
        else:
            loudness_loss = torch.zeros(1).mean()

        if feature_loss_param.use_spectral_centroid_loss \
                or feature_loss_param.use_spectral_bandwidth_loss:
            to_spectral_centroid = get_spectral_centroid(feature_loss_param) \
                .to(wav_fake.device)
            sc_fake = to_spectral_centroid(wav_fake)
            sc_real = to_spectral_centroid(wav_real)

            if feature_loss_param.use_spectral_centroid_loss:
                sc_fake = sc_fake / (torch.mean(sc_fake, dim=-1, keepdim=True) + 1e-6)
                sc_real = sc_real / (torch.mean(sc_real, dim=-1, keepdim=True) + 1e-6)
                sc_loss = F.smooth_l1_loss(sc_real, sc_fake)
                sb_loss = torch.zeros(1).mean()
            else:
                to_spectrogram = torchaudio.transforms.Spectrogram(n_fft=MEL_PARAMS["n_fft"], win_length=512, hop_length=256,
                                                                   normalized=True, power=1., return_complex=True).to(
                    wav_fake.device)
                freqs = torch.from_numpy(librosa.fft_frequencies(sr=feature_loss_param.sr, n_fft=MEL_PARAMS["n_fft"]))\
                    .type(sc_real.dtype)\
                    .to(sc_real.device).unsqueeze(0).unsqueeze(-1)
                real_spectrogram = torch.abs(to_spectrogram(wav_real))
                fake_spectrogram = torch.abs(to_spectrogram(wav_fake))
                real_deviation = torch.abs(freqs - sc_real.unsqueeze(1))
                fake_deviation = torch.abs(freqs - sc_fake.unsqueeze(1))
                sb_fake = torch.sum(fake_spectrogram * fake_deviation ** feature_loss_param.order, axis=-2, keepdims=True) **\
                         (1.0 / feature_loss_param.order)
                sb_real = torch.sum(real_spectrogram * real_deviation ** feature_loss_param.order, axis=-2, keepdims=True) **\
                         (1.0 / feature_loss_param.order)
                sb_fake = sb_fake / (torch.mean(sb_fake, dim=-1, keepdim=True) + 1e-6)
                sb_real = sb_real / (torch.mean(sb_real, dim=-1, keepdim=True) + 1e-6)
                sb_loss = F.smooth_l1_loss(sb_real, sb_fake)
                sc_loss = torch.zeros(1).mean()
        else:
            sc_loss = torch.zeros(1).mean()
            sb_loss = torch.zeros(1).mean()


        if feature_loss_param.use_spectral_kurtosis_loss:
            to_spectrogram = torchaudio.transforms.Spectrogram(n_fft=MEL_PARAMS["n_fft"], win_length=512,
                                                               hop_length=256,
                                                               normalized=False, power=2., return_complex=True).to(
                wav_fake.device)
            to_spectral_centroid = get_spectral_centroid(feature_loss_param) \
                .to(wav_fake.device)
            sc_fake = to_spectral_centroid(wav_fake).unsqueeze(-1)
            sc_real = to_spectral_centroid(wav_real).unsqueeze(-1)
            real_spectrogram = torch.abs(to_spectrogram(wav_real)).transpose(-2, -1)
            fake_spectrogram = torch.abs(to_spectrogram(wav_fake)).transpose(-2, -1)

            freqs = torch.from_numpy(librosa.fft_frequencies(sr=feature_loss_param.sr, n_fft=MEL_PARAMS["n_fft"])) \
                .type(real_spectrogram.dtype) \
                .to(real_spectrogram.device).unsqueeze(0).unsqueeze(0)

            fake_spread = torch.sqrt(torch.sum((freqs - sc_fake)**2 * fake_spectrogram, dim=-1, keepdim=True) / torch.sum(fake_spectrogram, dim=-1, keepdim=True))
            real_spread = torch.sqrt(torch.sum((freqs - sc_real)**2 * real_spectrogram, dim=-1, keepdim=True) / torch.sum(real_spectrogram, dim=-1, keepdim=True))

            fake_kurtosis = torch.sqrt(torch.sum((freqs - sc_fake)**4 * fake_spectrogram, dim=-1, keepdim=True) / (fake_spread**4 * torch.sum(fake_spectrogram, dim=-1, keepdim=True)))
            real_kurtosis = torch.sqrt(torch.sum((freqs - sc_real)**4 * real_spectrogram, dim=-1, keepdim=True) / (real_spread**4 * torch.sum(real_spectrogram, dim=-1, keepdim=True)))


            kurtosis_loss = F.smooth_l1_loss(real_kurtosis, fake_kurtosis)

        else:
            kurtosis_loss = torch.zeros(1).mean()

        if feature_loss_param.use_deep_emotion_feature_loss:
            with torch.no_grad():
                deep_emo_feature_real = nets.emotion_encoder.encoder.get_shared_feature(x_real)
                for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
                    deep_emo_feature_real = l(deep_emo_feature_real)
            deep_emo_feature_fake = nets.emotion_encoder.encoder.get_shared_feature(x_fake)
            for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
                deep_emo_feature_fake = l(deep_emo_feature_fake)
            deep_emo_feature_fake2 = nets.emotion_encoder.encoder.get_shared_feature(x_fake2)
            for l in list(nets.emotion_encoder.fc_o.modules())[1:-1]:
                deep_emo_feature_fake2 = l(deep_emo_feature_fake2)
            deep_emo_feature_loss = F.smooth_l1_loss(deep_emo_feature_fake, deep_emo_feature_real)
            deep_emo_feature_loss += F.smooth_l1_loss(deep_emo_feature_fake2, deep_emo_feature_real)
        else:
            deep_emo_feature_loss = torch.zeros(1).mean()

        if feature_loss_param.use_f0_related_loss:
            f0_related_loss = 0

            # F0 Slope duration and relative amplitude matching
            delta_f0_real = F0_real[...,1:] - F0_real[...,:-1]
            delta_f0_fake = F0_fake[..., 1:] - F0_fake[..., :-1]

            real_f0_local_mean = (F0_real[..., 1:] + F0_real[..., :-1])/2.
            fake_f0_local_mean = (F0_fake[..., 1:] + F0_fake[..., :-1]) / 2.

            relative_delta_f0_real = torch.nan_to_num(delta_f0_real / real_f0_local_mean)
            relative_delta_f0_fake = torch.nan_to_num(delta_f0_fake / fake_f0_local_mean)

            f0_related_loss += F.smooth_l1_loss(relative_delta_f0_real, relative_delta_f0_fake)

            # Binarize
            delta_f0_real_up_slope = torch.clamp(F.relu(delta_f0_real), min=0., max=1.)
            delta_f0_real_down_slope = torch.clamp(F.relu(delta_f0_real * -1), min=0., max=1.)

            delta_f0_fake_up_slope = torch.clamp(F.relu(delta_f0_fake), min=0., max=1.)
            delta_f0_fake_down_slope = torch.clamp(F.relu(delta_f0_fake * -1), min=0., max=1.)

            f0_related_loss += F.smooth_l1_loss(delta_f0_real_up_slope, delta_f0_fake_up_slope)
            f0_related_loss += F.smooth_l1_loss(delta_f0_real_down_slope, delta_f0_fake_down_slope)

        else:
            f0_related_loss = torch.zeros(1).mean()


    else:
        loudness_loss = torch.zeros(1).mean()
        sc_loss = torch.zeros(1).mean()
        sb_loss = torch.zeros(1).mean()
        kurtosis_loss = torch.zeros(1).mean()
        deep_emo_feature_loss = torch.zeros(1).mean()
        f0_related_loss = torch.zeros(1).mean()
    
    # norm consistency losses
    x_fake_norm = log_norm(x_fake)
    x_real_norm = log_norm(x_real)
    loss_norm = ((torch.nn.ReLU()(torch.abs(x_fake_norm - x_real_norm) - args.norm_bias))**2).mean()
    
    # F0 losses
    loss_f0 = f0_loss(F0_fake, F0_real)
    
    # style F0 losses (style initialization)
    if x_refs is not None and args.lambda_f0_sty > 0 and not use_adv_cls:
        F0_sty, _, _ = nets.f0_model(x_ref)
        loss_f0_sty = F.l1_loss(compute_mean_f0(F0_fake), compute_mean_f0(F0_sty))
    else:
        loss_f0_sty = torch.zeros(1).mean()
    
    # ASR losses
    loss_asr = F.smooth_l1_loss(ASR_fake, ASR_real)
    
    # style reconstruction losses
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))
    
    # cycle-consistency losses
    s_org = nets.style_encoder(x_real, y_org)
    x_rec = nets.generator(x_fake, s_org, masks=None, F0=GAN_F0_fake)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    # content preservation losses
    loss_cyc += F.smooth_l1_loss(nets.generator.get_content_representation(x_fake.detach()),
                                 nets.generator.get_content_representation(x_real.detach()).detach())
    loss_cyc += F.smooth_l1_loss(nets.generator.get_content_representation(x_fake2.detach()),
                                 nets.generator.get_content_representation(x_real.detach()).detach())

    # F0 losses in cycle-consistency losses
    if args.lambda_f0 > 0:
        _, _, cyc_F0_rec = nets.f0_model(x_rec)
        loss_cyc += F.smooth_l1_loss(cyc_F0_rec, cyc_F0_real)
    if args.lambda_asr > 0:
        ASR_recon = nets.asr_model.get_feature(x_rec)

        loss_cyc += F.smooth_l1_loss(ASR_recon, ASR_real)
    
    # adversarial classifier losses
    if use_adv_cls:
        out_de = nets.discriminator.classifier(x_fake)
        loss_adv_cls = F.cross_entropy(out_de[y_org != y_trg], y_trg[y_org != y_trg])
    else:
        loss_adv_cls = torch.zeros(1).mean()

    if args.use_aux_cls and use_aux_cls:
        out_aux = nets.discriminator.aux_classifier(x_fake)
        loss_aux_cls = F.cross_entropy(out_aux[sp_org != -1], sp_org[sp_org != -1])
    else:
        loss_aux_cls = torch.zeros(1).mean()
    
    loss = args.lambda_adv * loss_adv + args.lambda_sty * loss_sty \
           - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc\
           + args.lambda_norm * loss_norm \
           + args.lambda_asr * loss_asr \
           + args.lambda_f0 * loss_f0 \
           + args.lambda_f0_sty * loss_f0_sty \
           + args.lambda_adv_cls * loss_adv_cls \
           + args.lambda_aux_cls * loss_aux_cls \
           + feature_loss_param.lambda_loudness * loudness_loss \
           + feature_loss_param.lambda_spectral_centroid * sc_loss \
           + feature_loss_param.lambda_spectral_kurtosis * kurtosis_loss \
           + feature_loss_param.lambda_deep_emotion_feature * deep_emo_feature_loss \
           + feature_loss_param.lambda_f0_related_loss * f0_related_loss

    return loss, Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item(),
                       norm=loss_norm.item(),
                       asr=loss_asr.item(),
                       f0=loss_f0.item(),
                       adv_cls=loss_adv_cls.item(),
                       aux_cls = loss_aux_cls.item(),
                       loudness_loss= loudness_loss.item(),
                       sc_loss=sc_loss.item(),
                       sb_loss=sb_loss.item(),
                       kurtosis_loss=kurtosis_loss.item(),
                       deep_emo_feature_loss=deep_emo_feature_loss.item(),
                       f0_related_loss=f0_related_loss.item())


# for norm consistency losses
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x

# for adversarial losses
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization losses
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for F0 consistency losses
def compute_mean_f0(f0):
    f0_mean = f0.mean(-1)
    f0_mean = f0_mean.expand(f0.shape[-1], f0_mean.shape[0]).transpose(0, 1) # (B, M)
    return f0_mean


def f0_loss(x_f0, y_f0):
    """
    x.shape = (B, 1, M, L): predict
    y.shape = (B, 1, M, L): target
    """
    # compute the mean
    x_mean = compute_mean_f0(x_f0)
    y_mean = compute_mean_f0(y_f0)
    loss = F.l1_loss(x_f0 / x_mean, y_f0 / y_mean)
    return loss

def max_min_norm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x

