from TTS import TTS
import torch
import torchaudio
import numpy as np
from librosa.feature import melspectrogram
import librosa
import pandas as pd
import os
import pickle
from label import Label_Ref
import math

def wav_features(wav_data, samples, fs=16000, feature_type='wav2vec'): 
    wav_data = torch.from_numpy(wav_data).to(dtype=torch.float32)
    if feature_type == 'wav2vec':
        bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
    elif feature_type == 'hubert':
        bundle = torchaudio.pipelines.HUBERT_BASE
    if feature_type == 'wav2vec' or feature_type == 'hubert':
        if fs!=bundle.sample_rate:
            wav_data = torchaudio.functional.resample(wav_data, fs, bundle.sample_rate)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        model = bundle.get_model().to(device)
    if feature_type == 'mel':
        hop_length = math.ceil(wav_data.shape[1]/(samples-1))
        wav_data = librosa.util.pad_center(wav_data, size=hop_length*(samples-1), axis=1)
        features = trans_mel(wav_data, sample_rate=fs, framesize=512, mel_band=120, 
                                        hop_length=hop_length, is_log=True, is_normal=True) 
    elif feature_type == 'wav2vec':
        wav_data = wav_data.to(device)
        features = trans_wav2vec(wav_data, model, samples)
    elif feature_type == 'hubert':
        wav_data = wav_data.to(device)
        features = trans_hubert(wav_data, model, samples)
    return features

def trans_mel(wav_data, sample_rate, framesize=512, mel_band=120, hop_length=128, is_log=True, is_normal=True):
    wav_data = np.array(wav_data)
    mel_spect = melspectrogram(y=wav_data, sr=sample_rate, n_mels=mel_band, n_fft=framesize, hop_length=hop_length)

    if is_log == True:
        mel_spect = np.log(mel_spect + 1e-5) 

    if is_normal == True:
        mel_spect = torch.from_numpy(mel_spect)
        mel_mean = torch.mean(mel_spect, dim=2, keepdims=True)
        mel_std = torch.std(mel_spect, dim=2, keepdims=True)
        mel_spect = (mel_spect - mel_mean) / mel_std
        
    mel_spect = np.array(mel_spect)
    return mel_spect

def trans_wav2vec(wav_data, model, samples, is_normal=True):
    with torch.inference_mode():
        wav2vec_feature , _ = model.extract_features(wav_data)
        wav2vec_feature = torch.stack(wav2vec_feature)
        wav2vec_feature = wav2vec_feature[20]
    wav2vec_feature = wav2vec_feature.cpu()
    wav2vec_feature = wav2vec_feature.transpose(1, 2).contiguous()
    wav2vec_feature = torchaudio.transforms.Resample(wav2vec_feature.shape[2], samples)(wav2vec_feature)

    if is_normal == True:
        wav2vec_mean = torch.mean(wav2vec_feature, dim=2, keepdims=True)
        wav2vec_std = torch.std(wav2vec_feature, dim=2, keepdims=True)
        wav2vec_feature = (wav2vec_feature - wav2vec_mean) / wav2vec_std

    wav2vec_feature = np.array(wav2vec_feature)
    return wav2vec_feature

def trans_hubert(wav_data, model, samples, is_normal=True):
    with torch.inference_mode():
        hubert_feature , _ = model.extract_features(wav_data)
        hubert_feature = torch.stack(hubert_feature)
        hubert_feature = hubert_feature[8]
    hubert_feature = hubert_feature.cpu()
    hubert_feature = hubert_feature.transpose(1, 2).contiguous()
    hubert_feature = torchaudio.transforms.Resample(hubert_feature.shape[2], samples)(hubert_feature)

    if is_normal == True:
        hubert_mean = torch.mean(hubert_feature, dim=2, keepdims=True)
        hubert_std = torch.std(hubert_feature, dim=2, keepdims=True)
        hubert_feature = (hubert_feature - hubert_mean) / hubert_std
    
    hubert_feature = np.array(hubert_feature)
    return hubert_feature

def readSound(wordlist, save_path='sound'):
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    tts = TTS(PER=0) 
    for word in wordlist:
        tts.generate(text=word, save_path=save_path+f"/{word}.wav")

def load_features(label, samples, features_type, fea_path, save=False):
    if save:
        if not os.path.exists(fea_path):
            os.makedirs(fea_path) 
        data_dict = {}
        soundlist = []
        for word in label:
            wav, sample_rate = librosa.load(ROOT + f"/sound/{word}.wav", sr=None)
            soundlist.append(wav)
        for i in range(len(label)):
            data = np.expand_dims(soundlist[i], axis=0)
            data_dict[label[i]] = wav_features(wav_data=data, fs=sample_rate, samples=samples, feature_type=features_type)
        fea_path = os.path.join(fea_path, f'wav_feature_{features_type}.pkl')
        with open(fea_path, 'wb') as f:
            pickle.dump(data_dict, f)
    else:
        fea_path = os.path.join(fea_path, f'wav_feature_{features_type}.pkl')
        with open(fea_path, 'rb') as f:
            data_dict = pickle.load(f)
    features = []
    for word in label:
        features.append(data_dict[word])
    features = np.concatenate(features)
    return features

if __name__ == '__main__':
    ROOT = os.path.dirname(os.path.abspath(__file__))
    wordlist = list(Label_Ref().label.keys())
    # readSound(wordlist, save_path=os.path.join(ROOT, f'sound'))
    samples = 320
    features_mel = load_features(wordlist, samples, features_type='mel', fea_path=os.path.join(ROOT, f'feature'), save=True)
    features_wav2vec = load_features(wordlist, samples, features_type='wav2vec', fea_path=os.path.join(ROOT, f'feature'), save=True)
    features_hubert = load_features(wordlist, samples, features_type='hubert', fea_path=os.path.join(ROOT, f'feature'), save=True)
    print(features_mel.shape, features_wav2vec.shape, features_hubert.shape)






