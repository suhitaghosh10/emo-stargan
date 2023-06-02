import csv
import os
import whisper
import soundfile as sf
from transformers import logging
import sys
sys.path.append("/project/ardas/StarGAN_v2/")
import pandas as pd
import torch
from wvmos import get_wvmos

logging.set_verbosity_error()
from Evaluation.constants import *
from Evaluation.utils import get_model, get_svm, extract_feature, get_F0_model, get_vocoder, get_ref, get_src_wave, \
    get_converted_wav, split_source_wave
from Utils.evaluation_metrics import pitchCorr_f, mos
from not_to_checkin.cer import cer_whisper
from Utils.EMO_ENCODER.model import build_model as full_emo_bm
import shutil
import glob


def generate_sample(src_spch, source_wav, tgt_spkr, emo, sv_path, whisper_model, ds, mos_model = None):
    ref_wav = get_ref(REF_VCTK_MAP[tgt_spkr]) if ds == VCTK else \
        get_ref(REF_ESD_MAP[emo + '_' + str(tgt_spkr).replace('00','')])
    conv = get_converted_wav(starganv2=starganv2, source_wav=source_wav, F0_model=F0_model,
                               ref_wav=ref_wav, target_label=str(tgt_spkr), dataset=ds)
    sf.write(sv_path, conv, 24000, 'PCM_24')

    p = pitchCorr_f(src_spch, sv_path)
    src_emo = svm_loaded.predict([extract_feature(src_spch, True)])[0].capitalize()
    m_emo = svm_loaded.predict([extract_feature(sv_path, True)])[0].capitalize()
    mosm = mos(sv_path, mos_model)
    cer, trans_s, trans_o = cer_whisper(src_spch, sv_path, whisper_model)
    print(src_emo, m_emo, p, mosm, cer, trans_s, trans_o)
    return src_emo, m_emo, p, mosm, cer, trans_s, trans_o


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    os.environ['HTTP_PROXY'] = 'http://fp:3210/'
    os.environ['HTTPS_PROXY'] = 'http://fp:3210/'
    fs = 24000
    F0_model = get_F0_model()
    vocoder = get_vocoder()
    svm_loaded = get_svm()
    pretrained_mos_model = get_wvmos(cuda=True)

    emotion_encoder_path = "/scratch/ardas/StarGAN_v2/epoch_00194.pth"
    _, emotion_encoder = full_emo_bm()
    model_params = torch.load(emotion_encoder_path, map_location='cpu')['model_ema']['coder']
    emotion_encoder.coder.load_state_dict(model_params)
    emotion_encoder.coder = emotion_encoder.coder.to("cuda")


    sv_root_path = '/scratch/ardas/Evaluation/Objective/Eval_results/'
    sv_path = os.path.join(sv_root_path)
    for emo in EMOTIONS:
        os.makedirs(os.path.join(sv_root_path, emo), exist_ok=True)
    ##chnge here
    complete_model_path = "/scratch/ardas/experiments/stargan-v2/VCTK10_ESD10_demo_alt_with_pitch/epoch_00060.pth"
    model = 'demo_alt_withPitch_epoch60_objective'
    ### end change here

    whisper_model = whisper.load_model("medium.en")
    starganv2 = get_model(F0_model, complete_model_path)
    vctk_csv = "/scratch/sghosh/exp/IS2023/Evaluation/Objective/test/final/objective_meta_vctk_vctk_5emo_svm.csv"
    esd_csv = "/scratch/sghosh/exp/IS2023/Evaluation/Objective/test/final/objective_esd_to_esd_5emo_svm.csv"
    df_esd = pd.read_csv(esd_csv)
    df_vctk = pd.read_csv(vctk_csv)
    df_esd_vctk = df_vctk.append(df_esd, ignore_index=True)
    #df_esd_vctk = df_esd

    header = ['Model', 'source wav', 'source dataset', 'source speaker', 'source gender', 'source accent', 'emotion GT',
              'converted wav', 'target dataset', 'target speaker', 'target gender',
              'target accent', 'source emotion SVM', 'converted emotion SVM', 'Pitch Corr', 'PMOS', 'CER',
              'source trans', 'conv trans', 'emo_code_diff']
    with open('/scratch/ardas/Evaluation/Objective/objective_' + model + '.csv', mode='w') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(header)

        #
        for pos in range(df_esd_vctk.shape[0]):
            if df_esd_vctk.loc[pos, "Model"] == "baseline":
                continue
            src_spch = df_esd_vctk.loc[pos, "source wav"]
            ds = VCTK if VCTK in src_spch else ESD
            if ESD in src_spch:
                src_spch = src_spch.replace("/project/sughosh/dataset/ESD/EmotionalSpeechDataset", "/scratch/ardas/dataset/ESD/Emotional Speech Dataset (ESD)")
            else:
                src_spch = src_spch.replace("/project/sughosh/dataset/VCTK-Original", "/scratch/sghosh/datasets/vctk-original")

            idx = str(pos)
            emotion_gt = df_esd_vctk.loc[pos, "emotion GT"]
            src_spkr = df_esd_vctk.loc[pos, "source speaker"]
            tgt_spkr = df_esd_vctk.loc[pos, "target speaker"]
            src_g = df_esd_vctk.loc[pos, "source gender"]
            tgt_g = df_esd_vctk.loc[pos, "target gender"]
            src_accent = "NA" if ds == ESD else df_esd_vctk.loc[pos, "source accent"]
            fid = "_".join(src_spch.split("/")[-1].split(".")[0].split("_")[1:])

            source_wav = get_src_wave(src_spch)
            fname_sv = ds + '_' + fid + '_' + str(src_spkr) + '-' + str(tgt_spkr)
            model_sv_path = os.path.join(sv_path, emotion_gt, fname_sv + '_'+model+'.wav')
            src_emo, m_emo, p, mosm, cer, trans_s, trans_o = generate_sample(
                src_spch, source_wav, tgt_spkr, emotion_gt, model_sv_path, whisper_model, ds, pretrained_mos_model)
            print(idx, src_spkr, tgt_spkr)
            try:
                emo_diff = emotion_encoder.coder(split_source_wave(get_src_wave(src_spch)).to("cuda")) - \
                           emotion_encoder.coder(split_source_wave(get_src_wave(model_sv_path)).to("cuda"))
                emo_diff = torch.mean(torch.abs(emo_diff)).detach().cpu().item()
            except:
                print("Error", pos, src_spch)
                writer.writerow([model, src_spch, ds, src_spkr, src_g, src_accent, emotion_gt,
                                 model_sv_path,
                                 ds, tgt_spkr, tgt_g, 'NA', src_emo, m_emo, p, mosm, cer, trans_s, trans_o, 0])
                continue
            writer.writerow([model, src_spch, ds, src_spkr, src_g, src_accent, emotion_gt,
                             model_sv_path,
                             ds, tgt_spkr, tgt_g, 'NA', src_emo, m_emo, p, mosm, cer, trans_s, trans_o, emo_diff])
