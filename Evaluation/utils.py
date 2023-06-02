import os
import shutil
import scipy
import yaml
import torch
import copy
import string
import librosa
import numpy as np
import torchaudio
from munch import Munch
from Utils.JDC.model import JDCNet
from model_augmented import Generator, StyleEncoder, MappingNetwork
from parallel_wavegan.utils import load_model
import joblib
from Evaluation.constants import ESD, VCTK
import torch.nn.functional as F

CONFIG_PATH = "../Configs/config.yml"
CONFIG_PATH = "/project/ardas/StarGAN_v2/Configs/config.yml"
MEL_PARAMS = {
            "n_mels": 80,
            "n_fft": 2048,
            "win_length": 1200,
            "hop_length": 300
        }
to_mel = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
mean, std = -4, 4

def preprocess(wave, truncate=False):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    mel_length = mel_tensor.size(-1)
    if mel_length > 192 and truncate:
        random_start = np.random.randint(0, mel_length - 192)
        mel_tensor = mel_tensor[..., random_start:random_start + 192]
    return mel_tensor

def build_model(model_params={}, F0_model=None):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel,
                              F0_model=copy.deepcopy(F0_model))
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

    nets_ema = Munch(generator=generator,
                         mapping_network=mapping_network,
                         style_encoder=style_encoder)

    return nets_ema

def get_svm(path='/scratch/sghosh/exp/IS2023/emotion-classifier/SVM/model/svm_tess+ravdess/ravdess_tess_esd_5emotion_24k.pkl'):
    with open(path, 'rb') as fid:
        svm_loaded = joblib.load(fid)
    return svm_loaded

def mfcc(
    *, y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0, **kwargs
):
    if S is None:
        # multichannel behavior may be different due to relative noise floor differences between channels
        S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, **kwargs))

    _ = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]
    M = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]
    return M

def extract_feature(file_name, mfcc_):
        X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast', sr=24000)
        if mfcc_:
            mfccs = np.mean(mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((mfccs))
            return result
        else:
            return None

def get_vocoder(path="/scratch/ardas/StarGAN_v2/checkpoint-1790000steps.pkl"):
    vocoder = load_model(path)
    vocoder.remove_weight_norm()
    vocoder = vocoder.to("cuda")
    return vocoder

def get_F0_model():
    # load pretrained F0 model
    config_path = CONFIG_PATH
    config = yaml.safe_load(open(config_path))
    F0_path = config.get('F0_path', False)
    F0_model = JDCNet(num_class=1, seq_len=192)
    params = torch.load(F0_path, map_location='cpu')['net']
    # params = torch.load(F0_path, map_location='cpu')['model']
    F0_model.load_state_dict(params)
    F0_model = F0_model.to('cuda')
    _ = F0_model.eval()
    return F0_model

def get_model(F0_model, model_path):
    config_path = CONFIG_PATH
    config = yaml.safe_load(open(config_path))
    starganv2 = build_model(Munch(config['model_params']), F0_model=F0_model)
    params = torch.load(model_path, map_location='cpu')
    params = params['model_ema']

    _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
    _ = [starganv2[key].eval() for key in starganv2]
    starganv2.style_encoder = starganv2.style_encoder.to('cuda')
    starganv2.mapping_network = starganv2.mapping_network.to('cuda')
    starganv2.generator = starganv2.generator.to('cuda')
    return starganv2


def get_ref(ref_wav_path, fs=24000):
    _ref, _sr = librosa.load(ref_wav_path, sr=fs)
    if _sr != fs:
        _ref = librosa.resample(_ref, orig_sr=_sr, target_sr=fs)
    ref_wav = preprocess(_ref, truncate=True)
    return ref_wav.unsqueeze(0).to('cuda')


def get_converted_wav(starganv2, source_wav, ref_wav, F0_model, target_label, dataset=ESD):
    target_labels = torch.tensor([_get_target_spkr_idx(target_label, dataset)])
    target_labels = torch.LongTensor(target_labels).to("cuda")
    ref = starganv2.style_encoder(ref_wav, target_labels)
    source_wav = source_wav.unsqueeze(0).to("cuda")
    f0_feat = F0_model.get_feature_GAN(source_wav)
    converted = starganv2.generator(source_wav, ref, masks=None, F0=f0_feat)
    converted = converted.transpose(-1, -2).squeeze(0).to('cuda')
    vocoder = get_vocoder()
    wav_out = vocoder.inference(converted.squeeze())
    wav_out = wav_out.view(-1).cpu()
    return wav_out.detach().numpy().squeeze()

def _get_target_spkr_idx(spk_id, dataset=ESD):
    if dataset==ESD:
        lbl_dct = {'13': 12, '14': 15, '16': 14, '17': 16, '18': 17, '20': 19}
    elif dataset==VCTK:
        #lbl_dct = {'p273': 0, 'p228':5, 'p254':10, 'p244':11, 'p240':11}
        lbl_dct = {'p273': 0, 'p228': 5, 'p254': 3, 'p244': 9, 'p240': 8}
    return lbl_dct[spk_id]

def get_src_wave(source_speech, fs=24000, truncate= False):
    source_wav, sr = librosa.load(source_speech, sr=fs)
    if sr != fs:
        source_wav = librosa.resample(source_wav, orig_sr=sr, target_sr=fs)
    _ = preprocess(source_wav)
    return preprocess(source_wav, truncate=truncate)

def split_source_wave(source_spec):
    speech_len = source_spec.shape[-1]
    if (speech_len % 192) > 0 :
        padding  = (0, 192 - (speech_len % 192))
        source_spec = F.pad(source_spec, padding, "constant", 0)
    parts = torch.stack(torch.split(source_spec, 192, dim= -1), dim=0)
    return parts
