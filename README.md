# StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis

### Yinghao Aaron Li, Cong Han, Nima Mesgarani

> Text-to-Speech (TTS) has recently seen great progress in synthesizing high-quality speech owing to the rapid development of parallel TTS systems, but producing speech with naturalistic prosodic variations, speaking styles and emotional tones remains challenging. Moreover, since duration and speech are generated separately, parallel TTS models still have problems finding the best monotonic alignments that are crucial for naturalistic speech synthesis. Here, we propose StyleTTS, a style-based generative model for parallel TTS that can synthesize diverse speech with natural prosody from a reference speech utterance. With novel Transferable Monotonic Aligner (TMA) and duration-invariant data augmentation schemes, our method significantly outperforms state-of-the-art models on both single and multi-speaker datasets in subjective tests of speech naturalness and speaker similarity. Through self-supervised learning of the speaking styles, our model can synthesize speech with the same prosodic and emotional tone as any given reference speech without the need for explicitly labeling these categories.

Paper: [https://arxiv.org/abs/2107.10394](https://arxiv.org/abs/2205.15439)

Audio samples: [https://styletts.github.io/](https://styletts.github.io/)

## Pre-requisites
1. Python >= 3.7 (Python 3.9 recommended)
```bash
conda create -n styletts python=3.9
```
2. Clone this repository:
```bash
git clone https://github.com/seichi042I/StyleTTS_JP
cd StyleTTS_JP
```
3. Install requirements:
```bash
pip install -r requirements.txt
# install pyopenjtalk
pip install pyopenjtalk --no-build-isolation
```
3. Download and preprocess JSUT dataset
```bash
chmod +x preprocess.sh
./preprocess.sh
```

## Training
Train both stages with:
```bash
chmod +x run.sh
./run.sh
```

## Inference
1. Download pretrained vocoder at [Hifi-GAN Link](https://drive.google.com/file/d/1ujkBWJfwaM2-Aks-ecOvGUbA0chJtZYX/view?usp=drive_link) then unzip it to `Demo/Hifi-gan/Vocoder` folder.
2. Download pretrained StyleTTS at [StyleTTS Link](https://drive.google.com/file/d/1LMIjFk7xTnLDcDgwUMrbr5d2hwKtAmyP/view?usp=sharing) then unzip it to `Models` folder.
3. Run each cell of `Demo/inference_JSUT.ipynb`.
