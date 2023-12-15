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
3. Install python requirements: 
```bash
pip install SoundFile torchaudio munch torch pydub pyyaml librosa git+https://github.com/resemble-ai/monotonic_align.git
```
4. Download and extract the [LJSpeech dataset](https://keithito.com/LJ-Speech-Dataset/), unzip to the data folder and upsample the data to 24 kHz. The vocoder, text aligner and pitch extractor are pre-trained on 24 kHz data, but you can easily change the preprocessing and re-train them using your own preprocessing. I will provide more receipes and pre-trained models later if I have time. If you are willing to help, feel free to work on other preprocessing methods. 
For LibriTTS, you will need to combine train-clean-360 with train-clean-100 and rename the folder train-clean-460 (see [val_list_libritts.txt](https://github.com/yl4579/StyleTTS/blob/main/Data/val_list_libritts.txt) as an example).

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
詳細は[inference.ipynb](https://github.com/yl4579/StyleTTS/blob/main/Demo/Inference_LJSpeech.ipynb)を参照されたい。

24kHzのLJSpeechコーパスに対するStyleTTSとHifi-GANの事前学習は、[StyleTTS Link](https://drive.google.com/file/d/1aqOExU7NroGHdIVjgkzqRYrK5q_694cj/view?usp=sharing)と[Hifi-GAN Link](https://drive.google.com/file/d/1h_h0GFdC6VOiZ-oFDClqy2bVonA1xDiw/view?usp=sharing)からダウンロードできます。

LibriTTSコーパスに事前学習されたStyleTTSとHifi-GANは、[StyleTTS Link](https://drive.google.com/file/d/1nm0yB6Y5QWF3FYGfJCwQ6zYNlOAYVSet/view?usp=sharing)と[Hifi-GAN Link](https://drive.google.com/file/d/1RDxYknrzncGzusYeVeDo38ErNdczzbik/view?usp=sharing)からダウンロードできます。また、ゼロショットデモを実行したい場合は、LibriTTSからtest-cleanをダウンロードする必要があります。

HiFi-GANの事前学習モデルを`Vocoder`に解凍し、ノートブックの各セルを実行してください。
