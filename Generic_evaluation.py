import os
import yaml
import copy
import json
import torch
import librosa
import whisper
import torchaudio
import numpy as np
import soundfile as sf
from munch import Munch
from model_augmented import Generator
from Utils.JDC.model import JDCNet
from parallel_wavegan.utils import load_model
from model_augmented import MappingNetwork, StyleEncoder
from Utils.EMORECOG.model_alternate import build_model as bm
from Utils.evaluation_metrics import levenshtein, text_normalizer, stoi, mos, pitchCorr_f


os.environ['CUDA_VISIBLE_DEVICES'] = "1"

lang = 'en'
fs = 24000


class Greedy_CTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

model_w2v = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H.get_model().eval().to("cuda")
decoder_w2v = Greedy_CTCDecoder(labels=torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H.get_labels()).eval()

def get_transcription(waveform):
    emission_w2v = (model_w2v(waveform))
    transcript_w2v = decoder_w2v(emission_w2v[0].squeeze().cpu()).replace('|', ' ').lower()
    return transcript_w2v

with open('Configs/config.yml') as f:
    starganv2_config = yaml.safe_load(f)

# load pretrained F0 model
F0_path = starganv2_config.get('F0_path', False)
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load(F0_path, map_location='cpu')['net']
F0_model.load_state_dict(params)
F0_model = F0_model.to('cuda')
_ = F0_model.eval()



def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    emotion_encoder = None

    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema, emotion_encoder

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




all_speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]
target_speaker = [259, 228, 233, 273]

vocoder = load_model("/voice-conversion/experiments/hifigan2/out_train_cremaD_ESD/checkpoint-1790000steps.pkl").to('cuda')
vocoder.remove_weight_norm()
_ = vocoder.eval()

# load asr
whisper_model = whisper.load_model("medium.en")

source_wav_path = "/StarGAN_v2/Data/voxceleb2.txt"
with open(source_wav_path, 'r') as f:
    data_list = f.readlines()
data_list = [l[:-1].split('|') for l in data_list]
data_list = [(path, int(label.replace('p',''))) for path, label in data_list]
sources_path = [pair[0] for pair in data_list]
sources_speaker_index = [121 for pair in data_list]
source_text = [whisper_model.transcribe(wav_path, language="en")["text"] for wav_path in sources_path]

model_dir = "/experiments/stargan-v2/vctkall_baseline_epochs1000/"
WEIGHTS_STR = "epoch_00150"
target_dir  = "/StarGAN_v2/output/validation/"
target_dir = os.path.join(target_dir, "voxceleb2","vctkall_baseline_epochs1000")
os.makedirs(target_dir, exist_ok=True)
sources = [librosa.load(src, sr=24000)[0] for src in sources_path]
_ = preprocess(sources[0])
sources = [preprocess(src) for src in sources]

refs = ["/datasets/vctk-original/p259/p259_021_mic1.flac",
        "/datasets/vctk-original/p228/p228_003_mic1.flac",
        "/datasets/vctk-original/p233/p233_004_mic1.flac",
        "/datasets/vctk-original/p273/p273_035_mic1.flac"]

refs = [librosa.load(ref, sr=24000)[0] for ref in refs]
refs = [preprocess(ref, truncate=True) for ref in refs]
target_labels = [all_speakers.index(speaker) for speaker in target_speaker]
target_speakers = copy.deepcopy(target_labels)
target_labels = torch.tensor(target_labels)
target_labels = torch.LongTensor(target_labels).to("cuda")
ref_wavs = torch.stack(refs, 0).to("cuda")


for weights in os.listdir(model_dir):
    if WEIGHTS_STR in weights:
        complete_model_path = os.path.join(model_dir, weights)
        write_dir = target_dir
        os.makedirs(write_dir, exist_ok=True)

        # loading model
        starganv2,emotion_encoder = build_model(model_params=starganv2_config["model_params"])
        params = torch.load(complete_model_path, map_location='cpu')
        params = params['model_ema']
        _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
        _ = [starganv2[key].eval() for key in starganv2]
        starganv2.style_encoder = starganv2.style_encoder.to('cuda')
        starganv2.mapping_network = starganv2.mapping_network.to('cuda')
        starganv2.generator = starganv2.generator.to('cuda')

        result = {}
        for idx, source in enumerate(sources):
            if not sources_speaker_index[idx] in target_speakers:
                with torch.no_grad():
                    source = source.unsqueeze(0).repeat(len(target_speaker), 1, 1, 1).to("cuda")
                    ref = starganv2.style_encoder(ref_wavs, target_labels)
                    f0_feat = F0_model.get_feature_GAN(source)
                    converted = starganv2.generator(source, ref, masks=None, F0=f0_feat)
                    converted = converted.transpose(-1, -2).squeeze()
                    batch_size = converted.shape[0]


                    for iter in range(batch_size):
                        result[sources_path[idx]+ "_to_" + str(all_speakers[target_labels[iter].squeeze().item()])] = {}
                        wav_out = vocoder.inference(converted[iter].squeeze())
                        wav_out = wav_out.reshape(-1)

                        sf.write(os.path.join(write_dir,
                                 str(idx+1) + "_to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + ".wav"),
                                 wav_out.cpu().numpy().squeeze(), fs, 'PCM_24')

                        conv_text = whisper_model.transcribe(os.path.join(write_dir,
                                 str(idx+1) + "_to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + ".wav"), language="en")["text"]
                        msd, _, _, _ = (levenshtein(text_normalizer(conv_text), text_normalizer(source_text[idx])))
                        cer = msd / max(len(conv_text), len(source_text)) * 100
                        result[sources_path[idx]+ "_to_" + str(all_speakers[target_labels[iter].squeeze().item()])]["cer"] = cer

                        pred_8000 = torch.tensor(librosa.load(os.path.join(write_dir,
                                 str(idx+1) + "_to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + ".wav"), sr=8000)[0]).squeeze()
                        pred_16000 = torch.tensor(librosa.load(os.path.join(write_dir,
                                 str(idx+1) + "_to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + ".wav"), sr=16000)[0]).squeeze()
                        target_8000 = torch.tensor(librosa.load(sources_path[idx], sr=8000)[0]).squeeze()
                        target_16000 = torch.tensor(librosa.load(sources_path[idx], sr=16000)[0]).squeeze()
                        min_length_8000 = min(len(pred_8000), len(target_8000))
                        min_length_16000 = min(len(pred_16000), len(target_16000))


                        mos_score = mos(os.path.join(write_dir, str(idx+1) + "_to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + ".wav"))
                        result[sources_path[idx]+ "_to_" + str(all_speakers[target_labels[iter].squeeze().item()])]["mos"] = mos_score

                        stoi_score = stoi(pred_16000[:min_length_16000], target_16000[:min_length_16000]).cpu().item()
                        result[sources_path[idx]+ "_to_" + str(all_speakers[target_labels[iter].squeeze().item()])]["stoi"] = stoi_score

                        _ = pitchCorr_f(sources_path[idx], os.path.join(write_dir, str(idx + 1) + "_to_" + str(
                            all_speakers[target_labels[iter].squeeze().item()]) + ".wav"))

                        pcm_score = pitchCorr_f(sources_path[idx], os.path.join(write_dir, str(idx + 1) + "_to_" + str(
                            all_speakers[target_labels[iter].squeeze().item()]) + ".wav"))
                        result[sources_path[idx]+ "_to_" + str(all_speakers[target_labels[iter].squeeze().item()])]["pitch corr"] = pcm_score

        with open(os.path.join(write_dir,"result.json"), "w") as outfile:
            json.dump(result, outfile)



