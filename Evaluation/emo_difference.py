import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import torch
import pandas as pd
from Utils.evaluation_metrics import pitchCorr_f
from Utils.EMO_ENCODER.model import build_model as full_emo_bm
from Evaluation.utils import get_src_wave, split_source_wave

emotion_encoder_path = "/scratch/ardas/StarGAN_v2/epoch_00194.pth"
_, emotion_encoder = full_emo_bm()
model_params = torch.load(emotion_encoder_path, map_location='cpu')['model_ema']['coder']
emotion_encoder.coder.load_state_dict(model_params)


file_name = "/scratch/sghosh/exp/IS2023/Evaluation/Objective/test/final/objective_ravdess_to_esd_5emotions.csv"
target_file_name = "/scratch/ardas/Evaluation/Objective/objective_ravdess_to_esd_5emotions.csv"
df = pd.read_csv(file_name)

base_path = "/scratch/sghosh".split("/")

for pos in range(df.shape[0]):
    print(pos)
    source_path = "/".join(base_path + df.loc[pos, "source wav"].split("/")[3:])
    try:
        emo_diff = emotion_encoder.coder(split_source_wave(get_src_wave(source_path))) -\
               emotion_encoder.coder(split_source_wave(get_src_wave(df.loc[pos, "converted wav"])))
    except:
        print("Error", pos)
        continue

    df.loc[pos,'emo_code_diff'] = torch.mean(torch.abs(emo_diff)).detach().item()

df.to_csv(target_file_name, index=False)
print("End")