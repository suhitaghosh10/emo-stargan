import os
import pandas as pd
from tqdm import tqdm
from speechbrain.pretrained import SpeakerRecognition
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['HTTP_PROXY'] = 'http://fp:3210/'
os.environ['HTTPS_PROXY'] = 'http://fp:3210/'

csv_path = "/scratch/ardas/Evaluation/Objective/objective_Demo_alt_style_con_objective_update.csv"
df = pd.read_csv(csv_path)
#df = df.loc[(df["source dataset"]=="ESD")]
df = df.reset_index()
#src_path = "/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)/0015/Surprise/train/0015_001502.wav"
#trg_path = "/scratch/ardas/StarGAN_v2/output/samples/VCTK10_ESD10_kur_demo_SS_withPitch/epoch_65_2__Neutral_0012_000014_target_273.wav"
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

for pos in tqdm(range(df.shape[0])):
    src = df.loc[pos, "source wav"]
    #src = src.replace("/project/sughosh/dataset/VCTK-Original", "/scratch/sghosh/datasets/vctk-original")
    waveform_x = verification.load_audio(src)
    waveform_y = verification.load_audio(df.loc[pos, "converted wav"])
    batch_x = waveform_x.unsqueeze(0)
    batch_y = waveform_y.unsqueeze(0)
    # Verify:
    score, prediction = verification.verify_batch(batch_x, batch_y, threshold=0.3)
    df.loc[pos, "speaker similarity score"] = score.item()
    df.loc[pos, "speaker similarity class"] = prediction.item()

df.to_csv(csv_path, index=False)
acc_gt = ((df["speaker similarity class"] == True).sum() / df.shape[0]) * 100
print("SSS mean", df["speaker similarity score"].mean())
print("SSS Std", df["speaker similarity score"].std())
print(acc_gt)
print("End")