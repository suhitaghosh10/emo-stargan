#coding: utf-8

import random
import torch
import torchaudio

import librosa
import numpy as np
import soundfile as sf

from torch.utils.data import DataLoader
from Utils.constants import MEL_PARAMS, EMO_DICT, EMO_DB_DICT

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

np.random.seed(1)
random.seed(1)

class MelDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_list,
                 sr=24000,
                 validation=False,
                 domain=None
                 ):

        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(path, int(label.replace('p',''))) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target] \
            for target in list(set([label for _, label in self.data_list]))}

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4
        self.validation = validation
        self.max_mel_length = 192
        self.domain = domain

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        mel_tensor, label = self._load_data(data)
        if self.domain == "emotions":
            sp_label = int(data[0].split("/")[-1].split("_")[0]) - 11
        elif "ESD" in data[0]:
            sp_label = EMO_DICT[data[0].split("/")[-3]]
        elif "EmoDB" in data[0]:
            sp_label = EMO_DB_DICT[data[0].split("/")[-1][5]]
        else:
            sp_label = -1
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label = self._load_data(ref_data)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _ = self._load_data(ref2_data)
        return mel_tensor, label, sp_label, ref_mel_tensor, ref2_mel_tensor, ref_label
    
    def _load_data(self, path):
        wave_tensor, label = self._load_tensor(path)
        
        if not self.validation: # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor

        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label

    def _preprocess(self, wave_tensor, ):
        mel_tensor = self.to_melspec(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor) - self.mean) / self.std
        return mel_tensor

    def _load_tensor(self, data):
        wave_path, label = data
        label = int(label)
        wave, sr = sf.read(wave_path)
        if sr != self.sr:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=self.sr)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor, label

class Collater(object):
    """
    Args:
      adaptive_batch_size (bool): if true, decrease batch size when long data comes.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        sp_labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()

        for bid, (mel, label, sp_label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel
            
            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel
            
            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel
            
            labels[bid] = label
            sp_labels[bid] = sp_label
            ref_labels[bid] = ref_label

        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)
        
        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, sp_labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2

def build_dataloader(path_list,
                     validation=False,
                     batch_size=4,
                     num_workers=1,
                     device='cpu',
                     collate_config={},
                     domain=None):

    dataset = MelDataset(path_list, validation=validation, domain =domain)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader
