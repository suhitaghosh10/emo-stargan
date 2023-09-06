import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['HTTP_PROXY'] = 'http://fp:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp:3210/'
import torch
import librosa
import torchaudio
import numpy as np
import soundfile as sf
from model_augmented import StyleEncoder
from Utils.emotion_encoder.model import build_model
import matplotlib.pyplot as plt


device = 'cuda'
target_sr = 24000
MEL_PARAMS = {
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}
to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)
mean, std = -4, 4
max_mel_length = 192
wave_path_orig = "/scratch/ardas/Dataset/ESD/Emotional Speech Dataset (ESD)/0015/Surprise/train/0015_001502.wav"
wave_path_conv = "/scratch/ardas/StarGAN_v2/output/samples/VCTK10_ESD10_baseline_withPitch/epoch_75_3__Neutral_0012_000014_target_273.wav"


def return_mel(wave_path):
    wave, sr = sf.read(wave_path)
    if sr != target_sr:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=target_sr)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_melspec(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor) - mean) / std
    mel_length = mel_tensor.size(1)
    print("Wave length", mel_length)
    if mel_length > max_mel_length:
        random_start = np.random.randint(0, mel_length - max_mel_length)
    print("Random start frame", random_start)
    mel_tensor = mel_tensor[:, random_start:random_start + max_mel_length]
    mel = mel_tensor.unsqueeze(0).unsqueeze(0).to(device)

    return mel
model, model_ema = build_model()
model_params = torch.load("/project/ardas/experiments/stargan-v2/esdall_emotion_self_coding_stopLeakage_round3_epochs200/epoch_00154.pth",
                              map_location='cpu')['model']['coder']
model.coder.load_state_dict(model_params)
_ = [model[key].to(device) for key in model]

GT_Model = StyleEncoder(64, 64, 5, 512)
GT_Model_params = torch.load("/project/ardas/experiments/stargan-v2/esdall_emotion_conversion_aux_classifier_epochs200/epoch_00198.pth",
                              map_location='cpu')['model']['style_encoder']
GT_Model.load_state_dict(GT_Model_params)
GT_Model.to(device)

code =  model.coder(return_mel(wave_path_conv)).detach().cpu().squeeze().numpy()
gt = GT_Model(return_mel(wave_path_orig), torch.asarray([0])).detach().cpu().squeeze().numpy()
distribution = model.coder.get_distribution(return_mel(wave_path_conv)).detach().cpu().squeeze().numpy()

print("Emo distribution",  distribution)
print("Emo class", distribution.argmax())
plt.plot(code, label= "pred")
plt.plot(gt, label= "gt")
plt.legend()
plt.savefig("plot.png")