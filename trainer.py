# -*- coding: utf-8 -*-

import os
import os.path as osp
import sys
import time
from collections import defaultdict

import librosa
import numpy as np
import torch
from torch import nn
from PIL import Image
from tqdm import tqdm
import torchaudio
import soundfile as sf
from losses import compute_d_loss, compute_g_loss

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Trainer(object):
    def __init__(self,
                 args,
                 model=None,
                 model_ema=None,
                 optimizer=None,
                 scheduler=None,
                 config={},
                 device=torch.device("cpu"),
                 logger=logger,
                 train_dataloader=None,
                 val_dataloader=None,
                 initial_steps=0,
                 initial_epochs=0,
                 fp16_run=False,
                 save_samples= False,
                 gen_dataloader = None
    ):
        self.args = args
        self.steps = initial_steps
        self.epochs = initial_epochs
        self.model = model
        self.model_ema = model_ema
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.finish_train = False
        self.logger = logger
        self.fp16_run = fp16_run
        self.save_samples = save_samples
        self.gen_dataloader = gen_dataloader

    def _train_epoch(self):
        """Train model one epoch."""
        raise NotImplementedError

    @torch.no_grad()
    def _eval_epoch(self):
        """Evaluate model one epoch."""
        pass

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.
        Args:
            checkpoint_path (str): Checkpoint path to be saved.
        """
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs,
            "model": {key: self.model[key].state_dict() for key in self.model}
        }
        if self.model_ema is not None:
            state_dict['model_ema'] = {key: self.model_ema[key].state_dict() for key in self.model_ema}

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for key in self.model:
            self._load(state_dict["model"][key], self.model[key])

        if self.model_ema is not None:
            for key in self.model_ema:
                self._load(state_dict["model_ema"][key], self.model_ema[key])
        
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer.load_state_dict(state_dict["optimizer"])


    def _load(self, states, model, force_load=True):
        model_states = model.state_dict()
        for key, val in states.items():
            try:
                if key not in model_states:
                    continue
                if isinstance(val, nn.Parameter):
                    val = val.data

                if val.shape != model_states[key].shape:
                    self.logger.info("%s does not have same shape" % key)
                    print(val.shape, model_states[key].shape)
                    if not force_load:
                        continue

                    min_shape = np.minimum(np.array(val.shape), np.array(model_states[key].shape))
                    slices = [slice(0, min_index) for min_index in min_shape]
                    model_states[key][slices].copy_(val[slices])
                else:
                    model_states[key].copy_(val)
            except:
                self.logger.info("not exist :%s" % key)
                print("not exist ", key)

    @staticmethod
    def get_gradient_norm(model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

        total_norm = np.sqrt(total_norm)
        return total_norm

    @staticmethod
    def length_to_mask(lengths):
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask+1, lengths.unsqueeze(1))
        return mask

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            lr = param_group['lr']
            break
        return lr

    @staticmethod
    def moving_average(model, model_test, beta=0.999):
        for param, param_test in zip(model.parameters(), model_test.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def _train_epoch(self):
        self.epochs += 1
        
        train_losses = defaultdict(list)
        _ = [self.model[k].train() for k in self.model]
        scaler = torch.cuda.amp.GradScaler() if (('cuda' in str(self.device)) and self.fp16_run) else None

        use_con_reg = (self.epochs >= self.args.con_reg_epoch)
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)
        use_aux_cls = (self.epochs > self.args.aux_cls_epoch)
        use_feature_loss = (self.epochs >= self.args.g_loss['feature_loss']['feature_loss_epoch'])
        
        for train_steps_per_epoch, batch in enumerate(tqdm(self.train_dataloader, desc="[train]"), 1):

            ### load data
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, sp_org, x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch
            
            # train the discriminator (by random reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_losses_latent = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, sp_org, y_trg, z_trg=z_trg, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg,
                                                             use_aux_cls= use_aux_cls)
                scaler.scale(d_loss).backward()
            else:
                d_loss, d_losses_latent = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, sp_org, y_trg, z_trg=z_trg, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg,
                                                         use_aux_cls= use_aux_cls)
                d_loss.backward()
            self.optimizer.step('discriminator', scaler=scaler)

            # train the discriminator (by target reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    d_loss, d_losses_ref = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, sp_org, y_trg, x_ref=x_ref, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg,
                                                          use_aux_cls= use_aux_cls)
                scaler.scale(d_loss).backward()
            else:
                d_loss, d_losses_ref = compute_d_loss(self.model, self.args.d_loss, x_real, y_org, sp_org, y_trg, x_ref=x_ref, use_adv_cls=use_adv_cls, use_con_reg=use_con_reg,
                                                      use_aux_cls= use_aux_cls)
                d_loss.backward()

            self.optimizer.step('discriminator', scaler=scaler)

            # train the generator (by random reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses_latent = compute_g_loss(
                        self.model, self.args.g_loss, x_real, y_org, sp_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls,
                        use_aux_cls= use_aux_cls)
                scaler.scale(g_loss).backward()
            else:
                g_loss, g_losses_latent = compute_g_loss(
                    self.model, self.args.g_loss, x_real, y_org, sp_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls,
                use_aux_cls= use_aux_cls)
                g_loss.backward()

            self.optimizer.step('generator', scaler=scaler)
            self.optimizer.step('mapping_network', scaler=scaler)
            self.optimizer.step('style_encoder', scaler=scaler)

            # train the generator (by target reference)
            self.optimizer.zero_grad()
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    g_loss, g_losses_ref = compute_g_loss(
                        self.model, self.args.g_loss, x_real, y_org, sp_org, y_trg, x_refs=[x_ref, x_ref2], use_adv_cls=use_adv_cls, use_feature_loss=use_feature_loss,
                    use_aux_cls= use_aux_cls)
                scaler.scale(g_loss).backward()
            else:
                g_loss, g_losses_ref = compute_g_loss(
                    self.model, self.args.g_loss, x_real, y_org, sp_org, y_trg, x_refs=[x_ref, x_ref2], use_adv_cls=use_adv_cls, use_feature_loss=use_feature_loss,
                use_aux_cls= use_aux_cls)
                g_loss.backward()
            self.optimizer.step('generator', scaler=scaler)
            # compute moving average of network parameters
            self.moving_average(self.model.generator, self.model_ema.generator, beta=0.999)
            self.moving_average(self.model.mapping_network, self.model_ema.mapping_network, beta=0.999)
            self.moving_average(self.model.style_encoder, self.model_ema.style_encoder, beta=0.999)
            self.optimizer.scheduler()

            for key in d_losses_latent:
                train_losses["train/%s" % key].append(d_losses_latent[key])
            for key in g_losses_latent:
                train_losses["train/%s" % key].append(g_losses_latent[key])
            for key in g_losses_ref:
                train_losses["train_ref/%s" % key].append(g_losses_ref[key])


        train_losses = {key: np.mean(value) for key, value in train_losses.items()}
        return train_losses

    @torch.no_grad()
    def _eval_epoch(self):
        use_adv_cls = (self.epochs >= self.args.adv_cls_epoch)
        use_aux_cls = (self.epochs > self.args.aux_cls_epoch)
        
        eval_losses = defaultdict(list)
        eval_images = defaultdict(list)
        _ = [self.model[k].eval() for k in self.model]
        for eval_steps_per_epoch, batch in enumerate(tqdm(self.val_dataloader, desc="[eval]"), 1):

            ### load data
            batch = [b.to(self.device) for b in batch]
            x_real, y_org, sp_org,x_ref, x_ref2, y_trg, z_trg, z_trg2 = batch

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                self.model, self.args.d_loss, x_real, y_org, sp_org, y_trg, z_trg=z_trg, use_r1_reg=False, use_adv_cls=use_adv_cls, use_aux_cls= use_aux_cls)
            d_loss, d_losses_ref = compute_d_loss(
                self.model, self.args.d_loss, x_real, y_org, sp_org, y_trg, x_ref=x_ref, use_r1_reg=False, use_adv_cls=use_adv_cls, use_aux_cls= use_aux_cls)

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                self.model, self.args.g_loss, x_real, y_org, sp_org, y_trg, z_trgs=[z_trg, z_trg2], use_adv_cls=use_adv_cls, use_aux_cls= use_aux_cls)
            g_loss, g_losses_ref = compute_g_loss(
                self.model, self.args.g_loss, x_real, y_org, sp_org, y_trg, x_refs=[x_ref, x_ref2], use_adv_cls=use_adv_cls, use_aux_cls= use_aux_cls)

            for key in d_losses_latent:
                eval_losses["eval/%s" % key].append(d_losses_latent[key])
            for key in g_losses_latent:
                eval_losses["eval/%s" % key].append(g_losses_latent[key])

        eval_losses = {key: np.mean(value) for key, value in eval_losses.items()}
        eval_losses.update(eval_images)
        return eval_losses

    @torch.no_grad()
    def _sample_write_epoch(self, sample_write_params, device='cpu'):
        if self.save_samples and self.gen_dataloader is not None and self.epochs%5 == 0:
            _ = [self.model[k].eval() for k in self.model]
            if "Emotional Speech Dataset (ESD)" in sample_write_params['real_sample_path']:
                base_str = "_Neutral_0012_000014_target_"
            elif "EmoDB" in sample_write_params['real_sample_path']:
                base_str = "_Angry_13_b02_target_"
            else:
                base_str = "_p258_101_target_p"
            for gen_steps_per_epoch, batch in enumerate(tqdm(self.gen_dataloader, desc="[generate sample]"), 1):
                ### load data
                batch = [b.to(self.device) for b in batch]
                x_ref, y_trg, _, _, _, _, _, _ = batch
                source_path = sample_write_params['real_sample_path']
                save_path = sample_write_params['sample_save_path']
                os.makedirs(save_path, exist_ok=True)
                audio, source_sr = librosa.load(source_path)
                if source_sr != 24000:
                    audio = librosa.resample(audio, source_sr, 24000)
                audio = audio / np.max(np.abs(audio))
                audio.dtype = np.float32
                batch_size = x_ref.size(0)
                real = self.preprocess(audio).to(device).unsqueeze(1).repeat(batch_size, 1, 1, 1)
                s_trg = self.model_ema.style_encoder(x_ref, y_trg)
                F0 = self.model.f0_model.get_feature_GAN(real)
                x_fake = self.model_ema.generator(real, s_trg, masks=None, F0=F0)
                x_fake = x_fake.transpose(-1, -2).squeeze()
                target_list= y_trg.cpu().numpy().tolist()
                speaker_target_map = {}
                for speakers in list(sample_write_params['selected_speakers']):
                    p, t = speakers.split("_")
                    speaker_target_map[int(t)] = p
                for idx, target in enumerate(target_list):
                    y_out = self.model.vocoder.inference(x_fake[idx].squeeze())
                    y_out = y_out.view(-1).cpu()
                    sf.write(save_path+'epoch_'+str(self.epochs)+"_"+str(idx)+"_"+ base_str+speaker_target_map[int(target)]+'.wav', y_out.numpy().squeeze(), 24000, 'PCM_24')
        return None

    @torch.no_grad()
    def preprocess(self, wave):
        to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
        mean, std = -4, 4
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor
