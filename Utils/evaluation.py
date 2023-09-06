import os
import yaml
import torch
import string
import librosa
import numpy as np
import torchaudio
import soundfile as sf
from munch import Munch
from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from Models.models import Generator, MappingNetwork, StyleEncoder
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from parallel_wavegan.utils import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['HTTP_PROXY'] = 'http://fp:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp:3210/'

#@title Choose English ASR model { run: "auto" }
lang = 'en'
fs = 24000
tag = 'Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave'
d = ModelDownloader()
speech2text = Speech2Text(
    **d.download_and_unpack(tag),
    device="cpu",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=0,
    nbest=1
)

def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1]), size_x, size_y, matrix

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)

    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

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


# generate transcript

class GreedyCTCDecoder(torch.nn.Module):
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

model_w2v = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H.get_model()
decoder_w2v = GreedyCTCDecoder(labels=torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H.get_labels())

def _get_transcription(waveform):
    emission_w2v = (model_w2v(waveform))
    transcript_w2v = decoder_w2v(emission_w2v[0].squeeze()).replace('|', ' ').lower()
    return transcript_w2v

output_string = "Evaluation output: \n"
test_data_dir = "/datasets/vctk-original/"
all_speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]
selected_speakers = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
source_speech = "/datasets/vctk-original/p273/p273_022_mic1.flac"
source_text = "the actual primary rainbow observed is said to be the effect of super imposition of a number of bows"
source_speaker = 273
output_string += "Source speaker: " + str(source_speaker) + "\n"
output_string += "Source speech: " + str(source_speech) + "\n"
output_string += "Source text: " + str(source_text) + "\n"
destination_dir = "/StarGAN_v2/output/evaluation/"
os.makedirs(destination_dir, exist_ok=True)
model_dir = "/experiments/stargan-v2/"
allowed_experiment_list = ["vctkall_vox20_epochs1000",
                           "vctkall_mfcc_delta_delta_epochs1000",
                           "vctkall_deep_emo_feature_epochs1000",
                           "vctkall_spectral_kurtosis_epochs1000"
                           ]
WEIGHTS_STR = "epoch_00750"
with open('Configs/config.yml') as f:
    starganv2_config = yaml.safe_load(f)

# load pretrained ASR model
ASR_config = starganv2_config.get('ASR_config', False)
ASR_path = starganv2_config.get('ASR_path', False)
with open(ASR_config) as f:
        ASR_config = yaml.safe_load(f)
ASR_model_config = ASR_config['model_params']
ASR_model = ASRCNN(**ASR_model_config)
params = torch.load(ASR_path, map_location='cpu')['model']
ASR_model.load_state_dict(params)
ASR_model = ASR_model.to("cuda")
_ = ASR_model.eval()

# load vocoder
vocoder = load_model("/StarGAN_v2/vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()

# load pretrained F0 model
F0_path = starganv2_config.get('F0_path', False)
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load(F0_path, map_location='cpu')['net']
F0_model.load_state_dict(params)
F0_model = F0_model.to('cuda')
_ = F0_model.eval()

source_wav, sr = librosa.load(source_speech, sr=fs)
if sr != fs:
    source_wav = librosa.resample(source_wav, orig_sr=sr, target_sr=fs)
_ = preprocess(source_wav)
source_wav = preprocess(source_wav)

ref_wav_paths = [os.path.join(test_data_dir, "p"+ str(speaker), "p"+ str(speaker) +"_002_mic1.flac" ) for speaker in selected_speakers]
ref_wavs = []
for reference in ref_wav_paths:
    _ref, _sr = librosa.load(reference, sr=fs)
    if _sr != fs:
        _ref = librosa.resample(_ref, orig_sr=_sr, target_sr=fs)
    ref_wavs.append(_ref)
ref_wavs = [preprocess(ref_wav, truncate=True) for ref_wav in ref_wavs]
target_labels = [all_speakers.index(speaker) for speaker in selected_speakers]
target_labels = torch.tensor(target_labels)
target_labels = torch.LongTensor(target_labels).to("cuda")
ref_wavs = torch.stack(ref_wavs, 0).to("cuda")
source_wav = source_wav.unsqueeze(0).repeat(len(selected_speakers), 1, 1, 1).to("cuda")

for experiments in os.listdir(model_dir):
    if experiments in allowed_experiment_list:
        for weights in os.listdir(os.path.join(model_dir, experiments)):
            if WEIGHTS_STR in weights:
                file_write_dir = os.path.join(destination_dir, experiments+"_p"+str(source_speaker)+"_weights_"+WEIGHTS_STR)
                os.makedirs(file_write_dir, exist_ok=True)
                output_string += "Experiment Name " + str(experiments) + "\n"
                output_string += "Model weight " + str(weights) + "\n"
                complete_model_path = os.path.join(model_dir, experiments, weights)
                print(complete_model_path)

                #loading model
                starganv2 = build_model(model_params=starganv2_config["model_params"])
                params = torch.load(complete_model_path, map_location='cpu')
                params = params['model_ema']
                _ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
                _ = [starganv2[key].eval() for key in starganv2]
                starganv2.style_encoder = starganv2.style_encoder.to('cuda')
                starganv2.mapping_network = starganv2.mapping_network.to('cuda')
                starganv2.generator = starganv2.generator.to('cuda')

                with torch.no_grad():
                    ref = starganv2.style_encoder(ref_wavs, target_labels)
                    f0_feat = F0_model.get_feature_GAN(source_wav)
                    converted = starganv2.generator(source_wav, ref, masks=None, F0=f0_feat)
                    converted = converted.transpose(-1, -2).squeeze().to('cuda')
                    batch_size = converted.shape[0]
                    cer_list = []
                    for iter in range(batch_size):
                        wav_out = vocoder.inference(converted[iter].squeeze())
                        wav_out = wav_out.view(-1).cpu()
                        conv_text = _get_transcription(wav_out.unsqueeze(0))
                        output_string += "to_" +str(all_speakers[target_labels[iter].squeeze().item()]) + ": "+  str(conv_text) + "\n"
                        msd, _, _, _ = (levenshtein(conv_text, source_text))
                        cer = msd / max(len(conv_text), len(source_text)) * 100
                        cer_list.append(cer)
                        output_string += "to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + " cer: " + str(
                            round(cer, 4)) + "%\n"
                        sf.write(os.path.join(file_write_dir, "to_" + str(all_speakers[target_labels[iter].squeeze().item()]) + ".wav"),
                                 wav_out.numpy().squeeze(), 24000, 'PCM_24')
                    output_string += "cer mean: " + str(round(np.array(cer_list).mean(), 4)) +  "%\n"
                    output_string += "cer std: " + str(round(np.array(cer_list).std(), 4)) + "%\n"
                    output_string += "-"*300 + "\n"

#writing summary file
f = open(os.path.join(destination_dir, 'evaluation_summary.txt'), "w")
f.write(output_string)
f.close()
