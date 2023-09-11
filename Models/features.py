"""
EmoStarGAN
Copyright (c) 2023-present NAVER Corp.
This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import torch
import librosa
import numpy as np
import torchaudio.functional as F
import torchaudio
from Utils.constants import MEL_PARAMS

def get_loudness(batched_waveform : torch.Tensor, n_fft: int = 124, hop_length: int = 64, sampling_rate :int = 16000 ) -> torch.Tensor:
    """
        This function calculates loudness from batched waveform. It converts the waveform to a A weighted spectrogram
        and then calculates the loudness based on librosa.feature.rms function

        batched_waveform : batched tensor. Must be a 2D tenssor, having shape b*T, where,
                           b is batch size and T is time domain sample length
    """
    window_length = n_fft
    spectrogram = torch.abs(torch.stft(batched_waveform, n_fft=n_fft, hop_length=hop_length,
                                       return_complex=True, window=torch.hann_window(window_length=window_length).to(batched_waveform.device),
                                       pad_mode='constant'))
    spectogram_db = F.amplitude_to_DB(spectrogram, multiplier=20, db_multiplier=1.0, amin=1e-5, top_db=80.0)

    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    a_weights = librosa.A_weighting(freqs)
    a_weights = torch.from_numpy(np.expand_dims(a_weights, axis=(0, -1))).to(batched_waveform.device)

    spectogram_dba = spectogram_db + a_weights
    spectrogram_mag_a = F.DB_to_amplitude(spectogram_dba, power=0.5 , ref=1)

    spectrogram_mag_a = torch.square(spectrogram_mag_a)
    spectrogram_mag_a[:, 0, :] *= 0.5
    spectrogram_mag_a[:, -1, :] *= 0.5
    # loudness = torch.sqrt(torch.mean(torch.square(spectrogram_mag_a), dim=1))
    loudness = 2 * torch.sum(spectrogram_mag_a, dim=1) / (n_fft ** 2)
    loudness = torch.sqrt(loudness)

    return loudness

def frame(signal, frame_length, frame_step, pad_end=False, pad_value=0, axis=-1):
    """
    equivalent of tf.signal.frame
    """
    signal_length = signal.shape[axis]
    if pad_end:
        frames_overlap = frame_length - frame_step
        rest_samples = abs(signal_length - frames_overlap) % abs(frame_length - frames_overlap)
        pad_size = int(frame_length - rest_samples)
        if pad_size != 0 and rest_samples != 0:
            pad_axis = [0] * signal.ndim
            pad_axis[axis] = pad_size
            signal = torch.nn.functional.pad(signal, pad_axis, "constant", pad_value)
    frames=signal.unfold(axis, frame_length, frame_step)
    return frames

def torch_like_frame(signal, frame_length, frame_step, pad_mode="reflect" ):
    signal_dim = signal.dim()
    extended_shape = [1] * (3 - signal_dim) + list(signal.size())
    pad = int(frame_length // 2)
    input = torch.nn.functional.pad(signal.view(extended_shape), [pad, pad-1], pad_mode)
    input = input.view(input.shape[-signal_dim:])
    return input.unfold(-1, frame_length, frame_step)

def get_spectral_centroid(feature_loss_param):
    return torchaudio.transforms.SpectralCentroid(sample_rate=feature_loss_param.sr,
                                                  n_fft=MEL_PARAMS["n_fft"],
                                                  win_length=512, hop_length=256)
