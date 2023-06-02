ESD = 'ESD'
RAVDESS = 'RAVDESS'
VCTK = 'VCTK'
VOXCELEB = 'VOXCELEB'
DATASETS = {ESD, RAVDESS, VCTK, VOXCELEB}
F = 'Female'
M = 'Male'
EMOTIONS = ['Happy', 'Sad', 'Surprise', 'Angry', 'Neutral']
NEUTRAL = 'Neutral'

ESD_TGT_SPKRS = [13, 14, 16, 17, 18, 20]
fs = 24000


REF_ESD_MAP = {
    'Angry_16': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0016/Neutral/evaluation/0016_000019.wav',
    'Neutral_16': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0016/Neutral/evaluation/0016_000019.wav',
    'Sad_16': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0016/Angry/train/0016_000699.wav',
    'Surprise_16': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0016/Sad/train/0016_001394.wav',
    'Happy_16': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0016/Sad/train/0016_001394.wav',
    'Angry_13': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0013/Neutral/train/0013_000064.wav',
    'Neutral_13': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0013/Neutral/train/0013_000064.wav',
    'Sad_13': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0013/Angry/train/0013_000699.wav',
    'Surprise_13': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0013/Sad/train/0013_001394.wav',
    'Happy_13': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0013/Sad/train/0013_001394.wav',
    'Angry_14': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0014/Neutral/evaluation/0014_000019.wav',
    'Neutral_14': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0014/Neutral/evaluation/0014_000019.wav',
    'Sad_14': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0014/Angry/train/0014_000699.wav',
    'Surprise_14': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0014/Sad/train/0014_001394.wav',
    'Happy_14': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0014/Sad/train/0014_001394.wav',
    'Angry_17': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0017/Neutral/train/0017_000348.wav',
    'Neutral_17': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0017/Neutral/train/0017_000348.wav',
    'Sad_17': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0017/Angry/train/0017_000676.wav',
    'Surprise_17': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0017/Sad/train/0017_001383.wav',
    'Happy_17': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0017/Neutral/train/0017_000348.wav',
    'Angry_18': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0018/Neutral/train/0018_000348.wav',
    'Neutral_18': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0018/Neutral/train/0018_000348.wav',
    'Sad_18': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0018/Angry/train/0018_000676.wav',
    'Surprise_18': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0018/Sad/train/0018_001383.wav',
    'Happy_18': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0018/Neutral/train/0018_000348.wav',
    'Angry_20': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0020/Neutral/train/0020_000348.wav',
    'Neutral_20': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0020/Neutral/train/0020_000348.wav',
    'Sad_20': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0020/Happy/train/0020_001032.wav',
    'Surprise_20': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0020/Sad/train/0020_001383.wav',
    'Happy_20': '/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0020/Neutral/train/0020_000348.wav'
}

REF_VCTK_MAP = {'p273':'/scratch/sghosh/datasets/vctk-original/p273/p273_004_mic1.flac',
'p228':'/scratch/sghosh/datasets/vctk-original/p228/p228_010_mic1.flac',
'p254':'/scratch/sghosh/datasets/vctk-original/p254/p254_020_mic2.flac',
'p244':'/scratch/sghosh/datasets/vctk-original/p244/p244_007_mic2.flac',
'p240':'/scratch/sghosh/datasets/vctk-original/p240/p240_012_mic2.flac'}

DATASET_MAP = {
    'ESD': '/project/sughosh/dataset/ESD/EmotionalSpeechDataset/',
    'RAVDESS': '/project/sughosh/dataset/ravdess/raw/',
    'VCTK': '/project/sughosh/dataset/VCTK-Original/',
    'VOXCELEB': '/project/sughosh/dataset/voxceleb/voxceleb1_test/'
}

ESD_GENDER_MAP = {'0011': M,
                  '0012': M,
                  '0013': M,
                  '0014': M,
                  '0015': F,
                  '0016': F,
                  '0017': F,
                  '0018': F,
                  '0019': F,
                  '0020': M}

ESD_SPEAKERS = ['0015', '0020', '0012', '0018', '0016', '0011','0019', '0013', '0014', '0017']

DATASET_CONV_MAP = {ESD: [VCTK, ESD],
                    RAVDESS: [VCTK, ESD],
                    VOXCELEB: [VCTK, ESD],
                    VCTK: [VCTK]
                    }
complete_model_path = "/scratch/ardas/experiments/stargan-v2/VCTK10_ESD10_kur_demo_SS_withPitch/epoch_00064.pth"
complete_modelb_path = "/scratch/ardas/experiments/stargan-v2/VCTK10_ESD10_baseline_withPitch/epoch_00076.pth"

all_speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]
VCTK_SPKRS = ['p258', 'p228', 'p229', 'p273', 'p294', 'p297', 'p345', 'p311']
VCTK_GENDER_MAP = {'p258':M, 'p228': F, 'p229': F, 'p273': M,
                   'p294': F, 'p297': F, 'p345': M, 'p311': M, 'p254': M, 'p244':F, 'p240':F}
VCTK_REF_SPEAKERS = ['p273', 'p228', 'p254', 'p244', 'p240']
VCTK_ACCENT_ENGLISH = ['p228', 'p229', 'p273', 'p258']
VCTK_ACCENT_AMERICAN = ['p294', 'p297', 'p345', 'p311']


RAVDESS_SPKRS = ['Actor_24', 'Actor_23', 'Actor_22', 'Actor_21', 'Actor_20', 'Actor_19', 'Actor_18', 'Actor_17',
                 'Actor_16', 'Actor_15']
RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprised'
}

RAVDESS_GENDER_MAP = {
'Actor_24': F, 'Actor_23': M, 'Actor_22': F, 'Actor_21': M, 'Actor_20':F, 'Actor_19': M, 'Actor_18': F, 'Actor_17': M,
                 'Actor_16': F, 'Actor_15':M
}
