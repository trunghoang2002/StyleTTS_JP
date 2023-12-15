# StyleTTS: A Style-Based Generative Model for Natural and Diverse Text-to-Speech Synthesis

### Yinghao Aaron Li, Cong Han, Nima Mesgarani

> Text-to-Speech (TTS) has recently seen great progress in synthesizing high-quality speech owing to the rapid development of parallel TTS systems, but producing speech with naturalistic prosodic variations, speaking styles and emotional tones remains challenging. Moreover, since duration and speech are generated separately, parallel TTS models still have problems finding the best monotonic alignments that are crucial for naturalistic speech synthesis. Here, we propose StyleTTS, a style-based generative model for parallel TTS that can synthesize diverse speech with natural prosody from a reference speech utterance. With novel Transferable Monotonic Aligner (TMA) and duration-invariant data augmentation schemes, our method significantly outperforms state-of-the-art models on both single and multi-speaker datasets in subjective tests of speech naturalness and speaker similarity. Through self-supervised learning of the speaking styles, our model can synthesize speech with the same prosodic and emotional tone as any given reference speech without the need for explicitly labeling these categories.

Paper: [https://arxiv.org/abs/2107.10394](https://arxiv.org/abs/2205.15439)

Audio samples: [https://styletts.github.io/](https://styletts.github.io/)

## 前提条件
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/seichi042I/StyleTTS_JP
cd StyleTTS_JP
```

## 学習
```bash
chmod +x run.sh
chmod +x preprocess.sh
./run.sh
```
個別に学習をするときは以下
First stage training:
```bash
python train_first.py --config_path ./Configs/config.yml
```
Second stage training:
```bash
python train_second.py --config_path ./Configs/config.yml
```


## Inference
こちら[Hifi-GAN Link](https://drive.google.com/file/d/1h_h0GFdC6VOiZ-oFDClqy2bVonA1xDiw/view?usp=sharing)をダウンロードして`Vocoder`に解凍し、`Demo/inference_JSUT.ipynb`の各セルを実行してください。zero-shotの部分は対応していません。[元のリポジトリ](https://github.com/yl4579/StyleTTS)を参考に各自で動かしてみてください
