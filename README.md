# Emo-StarGAN 

This repository contains the source code of the paper *[Emo-StarGAN: A Semi-Supervised Any-to-Many Non-Parallel Emotion-Preserving Voice Conversion](https://www.researchgate.net/publication/373161292_Emo-StarGAN_A_Semi-Supervised_Any-to-Many_Non-Parallel_Emotion-Preserving_Voice_Conversion), accepted in Interspeech 2023*. An overview of the method and the results can be found [here](https://github.com/suhitaghosh10/emo-stargan/blob/main/overview.pdf).


![Concept of our method. For details we refer to our paper at .....](emo-stargan.png)

## Highlights:
- Emo-StarGAN: An emotion-preserving deep semi-supervised voice conversion-based speaker anonymisation method is proposed.
- Emotion supervision techiniques are proposed: (a) Direct: using emotion classifier (b) Indirect: using losses leveraging acoustic features and deep features which represent the emotional content of the source and converted samples.
- The indirect techniques can also be used in the absence of emotion labels.
- Experiments demonstrate its generalizability on the following benchmark datasets, across different accents, genders, emotions and cross-corpus conversions:
  - [Emotional Speech Dataset (ESD)](https://hltsingapore.github.io/ESD/)
  - [Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
  - [Centre for Speech Technology Voice Cloning Toolkit (VCTK)](https://datashare.ed.ac.uk/handle/10283/2950)

## Samples
Samples can be found [here]()

## Pre-requisites:
1. Python >= 3.7
2. Install the following dependencies mentioned in the requirements.txt

## Training:

### Before Training
1. Before starting the training, please specify the number of target speaskers in `num_speaker_domains` and other details such as training and validation data in `config.yml` file.
2. Download and copy the emotion embeddings [weights]() to the folder Utils/emotion_encoder
3. Download and copy the vocoder [weights]() to the folder Utils/Vocoder

### Train
```bash
python train.py --config_path ./Configs/speaker_domain_config.yml
```
## Inference
coming soon...

