import yaml
from munch import Munch
import torch
import librosa
import sys

import glob
import argparse
import json
import torch
from scipy.io.wavfile import write
from attrdict import AttrDict
sys.path.insert(0, "./Demo/hifi-gan")
from vocoder import Generator
import librosa

from models import *
from utils import *
from meldataset import *

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def compute_style(model, ref_dicts, device):
    reference_embeddings = {}
    for key, path in ref_dicts.items():
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = preprocess(audio).to(device)
        try:
            with torch.no_grad():
                ref = model.style_encoder(mel_tensor.unsqueeze(1))
            reference_embeddings[key] = (ref.squeeze(1), audio)
        except:
            continue
    
    return reference_embeddings

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def load_hifigan(device):
    cp_g = scan_checkpoint("Demo/hifi-gan/Vocoder/LibriTTS/", 'g_')

    config_file = os.path.join(os.path.split(cp_g)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

def load_styletts(config, model_path, device):
    # load pretrained ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)

    # load pretrained F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)

    model = build_model(Munch(config['model_params']), text_aligner, pitch_extractor)

    params = torch.load(model_path, map_location='cpu')
    params = params['net']
    for key in model:
        if key in params:
            if not "discriminator" in key:
                print('%s loaded' % key)
                model[key].load_state_dict(params[key])
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]
    return model

def synthesize_speech(text, reference_embeddings, speed, model, generator, textcleaner, device):
    """
    Generate speech from text using reference audio samples.
    
    Args:
        text (str): Input text to be synthesized.
        reference_embeddings : Dictionary of reference audio samples and their embeddings.
        speed (float): Speed factor for the synthesized speech.
        model: Pretrained TTS model.
        generator: Vocoder model to convert features to waveform.
        textcleaner: Class to clean and tokenize input text.
        device: Computation device ("cuda" or "cpu").

    Returns:
        dict: Synthesized waveforms keyed by reference sample names.
    """
    
    # Tokenize input text
    tokens = textcleaner(text)
    tokens.insert(0, 0)  # Start token
    tokens.append(0)  # End token
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    
    converted_samples = {}
    
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        m = length_to_mask(input_lengths).to(device)
        t_en = model.text_encoder(tokens, input_lengths, m)

        for key, (ref, _) in reference_embeddings.items():
            s = ref.squeeze(1)
            style = s
            
            d = model.predictor.text_encoder(t_en, style, input_lengths, m)
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            # pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            pred_dur = torch.round(duration.squeeze() / speed).clamp(min=1)
            
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data)).to(device)
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0))
            style = s.expand(en.shape[0], en.shape[1], -1)
            
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            out = model.decoder((t_en @ pred_aln_trg.unsqueeze(0)), F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            
            c = out.squeeze()
            y_g_hat = generator(c.unsqueeze(0))
            y_out = y_g_hat.squeeze().cpu().numpy()
            
            converted_samples[key] = y_out
    
    return converted_samples

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    print(torch.cuda.get_device_name())

    # Load configuration
    model_config_path = "./Models/JSUT/config.yml"
    config = yaml.safe_load(open(model_config_path))
    print("Loaded config file...")

    # Load models
    model_path = "./Models/JSUT/epoch_2nd_00100.pth"
    model = load_styletts(config, model_path, device)
    generator = load_hifigan(device)
    textcleaner = TextCleaner()
    
    # Example usage
    text = "昨日、私は公園へ行きました。天気がよくて、空は青く、風が気持ちよかったです。公園にはたくさんの人がいました。子どもたちは楽しそうに遊んでいて、大人たちはベンチに座って話していました。私は木の下で本を読みながら、リラックスしました。そのあと、カフェに行ってコーヒーを飲みました。とても楽しい一日でした。"
    reference_samples = [
        "jsut_24kHz/onomatopee300/wav/ONOMATOPEE300_001.wav",
        "jsut_24kHz/loanword128/wav/LOANWORD128_001.wav",
        "jsut_24kHz/precedent130/wav/PRECEDENT130_001.wav"
    ]

    # Compute style embeddings from reference samples
    ref_dicts = {path.split('/')[-1].replace('.wav', ''): path for path in reference_samples}
    reference_embeddings = compute_style(model, ref_dicts, device)

    # Synthesize speech
    converted_samples = synthesize_speech(text, reference_embeddings, model, generator, textcleaner, device)

    # Save synthesized audio
    for key, audio in converted_samples.items():
        output_path = f"output_{key}.wav"
        write(output_path, 24000, audio)
        print(f"Saved synthesized audio to {output_path}")
    print("Synthesis complete.")