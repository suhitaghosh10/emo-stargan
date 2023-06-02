import torch
import string
import warnings
import torchaudio
import numpy as np
import parselmouth
import pandas as pd
from wvmos import get_wvmos
from scipy.interpolate import interp1d
from torchmetrics import SignalNoiseRatio
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


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

def pesq_nb(predicted, target, sampling_frequency=8000):
    g = torch.manual_seed(1)
    nb_pesq = PerceptualEvaluationSpeechQuality(sampling_frequency, 'nb')
    return nb_pesq(predicted, target)

def pesq_wb(predicted, target, sampling_frequency=16000):
    g = torch.manual_seed(1)
    wb_pesq = PerceptualEvaluationSpeechQuality(sampling_frequency, 'wb')
    return wb_pesq(predicted, target)

def mos(converted_wav_path, model= None):
    warnings.filterwarnings('ignore')
    if model is None:
        model = get_wvmos(cuda=True)
    mos = model.calculate_one(converted_wav_path)
    return mos

def snr(predicted, target):
    snr = SignalNoiseRatio()
    return snr(predicted, target)

def stoi(predicted, target, sampling_frequency=16000):
    g = torch.manual_seed(1)
    stoi = ShortTimeObjectiveIntelligibility(sampling_frequency, False)
    return stoi(predicted, target)




def pitchCorr_f(wav_path_orig,wav_path_anon, pitch_ceiling=500):

    # get waveforms
    snd_orig = parselmouth.Sound(wav_path_orig)
    snd_anon = parselmouth.Sound(wav_path_anon)

    # extract pitches
    pitch_orig = snd_orig.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=pitch_ceiling)
    pitch_anon = snd_anon.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=pitch_ceiling)
    a = pitch_orig.selected_array['frequency']
    b = pitch_anon.selected_array['frequency']

    # linear interpolation wrt longest pitch signal
    if len(a)>len(b):
        b = interp1d(np.linspace(1, len(b), num=len(b)),b)(np.linspace(1, len(b), num=len(a)))
    elif len(a)<len(b):
        a = interp1d(np.linspace(1, len(a), num=len(a)),a)(np.linspace(1, len(a), num=len(b)))

    # keep pitch values between 75 and 500 Hz after the interpolation process
    a[a < 75] = 0
    a[a > pitch_ceiling] = 0
    b[b < 75] = 0
    b[b > pitch_ceiling] = 0

    # align the original and anonymized pitches
    cab = np.correlate(a, b, 'full')
    lags = np.linspace(-len(a) + 1, len(a) - 1, 2 * len(a) - 1)
    I = np.argmax(cab)
    T = lags[I].astype(int)
    b = np.roll(b, T)

    # get correlation coefficients between the 2 pitches excluding unvoiced parts
    df = pd.DataFrame(np.stack((a, b)).T.squeeze(), columns=list('ab'))
    c = df.corr().fillna(0)

    return c.iloc[1]['a']

if __name__ == '__main__':
    src = "/exp/Evaluation/Subjective/results_v2/44_Angry_SRC_19-17_F-F_000667.wav"
    trg = "/Evaluation/Objective/based_on_subjective/Angry/ESD_000667_0019-17_F0_related_feature_noPitch.wav"
    print(pitchCorr_f(src, trg))
    print(pitchCorr_f(trg, src))

