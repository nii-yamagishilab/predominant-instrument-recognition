import glob
import json
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import os
from pathlib import Path
import random
import numpy as np
import torchaudio
from tqdm import tqdm
import torchvision
from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import torch.nn as nn
from sklearn.preprocessing import LabelBinarizer
from torch.distributions import Bernoulli
import soundfile as sf
import pyloudnorm as pyln
import librosa


class IRMASDataset(Dataset):
    def __init__(self, meta_path,
                 wav_dir,
                 is_test=False,
                 to_mono='mean',
                 n_classes=11,
                 normalize_amp=False,
                 target_lufs=-12.0,
                 orig_sr=44100,
                 target_sr=16000,
                 ):
        # read manifest
        with open(meta_path, 'r') as f:
            meta_json = json.load(f)
        self.dataset = OrderedDict(meta_json)
        self.datalist = list(self.dataset.keys())

        self.n_classes = n_classes
        self.lb = LabelBinarizer()
        self.lb.fit([*range(self.n_classes)])
        self.dir = wav_dir

        self.is_test = is_test
        self.to_mono = to_mono

        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.resampler = torchaudio.transforms.Resample(self.orig_sr, self.target_sr)

        self.normalize_amp = normalize_amp
        self.loudness_meter = pyln.Meter(self.target_sr) # after down sample!
        self.target_lufs = target_lufs

        if 'start' in list(self.dataset.values())[0]:
            self.slice = True
        else:
            self.slice = False

    def __getitem__(self, index):
        song_meta = self.dataset[self.datalist[index]]
        r_path = song_meta['relative_path']
        label = song_meta['instrument']

        song_path = os.path.join(self.dir, r_path + '.wav')
        wav, _ = torchaudio.load(song_path)
        wav = self.resampler(wav)
        if self.to_mono == 'mean':
            wav = (wav[0] + wav[1]) / 2

        if self.slice:
            st = int(song_meta['start'] * self.target_sr)
            ed = int(song_meta['end'] * self.target_sr)
            wav = wav[st: ed]

        if self.normalize_amp:
            wav = self.loudness_normalize(wav)

        if self.is_test:
            one_hot_label = self.lb.transform(label).astype('float').sum(axis=0)
        else:
            one_hot_label = self.lb.transform([label])[0].astype('float')

        return {'feature': wav,
                'one_hot_label': one_hot_label,
                'label': label,
                }

    def __len__(self):
        return len(self.datalist)

    def get_one_sec(self, excerpt, sr):
        idx = random.randint(0, len(excerpt) - sr)
        clip = excerpt[idx: idx + sr]

        return clip

    def combine_samples(self, sample, noise, snr_db):
        sample_rms = sample.norm(p=2)
        noise_rms = noise.norm(p=2)
        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / sample_rms
        out = (scale * sample + noise) / 2

        return out

    def mixup(self, y1, y2, mix_aplha, ln=False):
        lam = np.random.beta(mix_aplha, mix_aplha)
        mix_wav = lam * y1 + (1 - lam) * y2
        return mix_wav, lam

    def mixup_inside_batch(self, y1, y2, mix_aplha=10, ln=False):
        lam = np.random.beta(mix_alpha, mix_alpha)
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()
        exit()
        return

    def get_bgm_metadata(self, metadata):
        bgm_meta = defaultdict(dict)
        for label_idx in range(11):
            for k, v in metadata.items():
                if v['instrument'] != label_idx:
                    bgm_meta[label_idx].update({k: metadata[k]})

        return bgm_meta

    def load_bgm(self, current_label, bgm_metadata):
        bgm_key = random.choice(list(bgm_metadata[current_label].keys()))
        bgm_meta = self.dataset[bgm_key]
        r_path = bgm_meta['relative_path']
        label = bgm_meta['instrument']

        song_path = os.path.join(self.dir, r_path + '.wav')
        wav, _ = torchaudio.load(song_path)
        wav = self.resampler(wav)
        if self.to_mono == 'mean':
            wav = (wav[0] + wav[1]) / 2

        return wav, label

    def loudness_normalize(self, wav):
        wav = wav.detach().numpy()
        lufs = self.loudness_meter.integrated_loudness(wav)
        wav = pyln.normalize.loudness(wav, lufs, self.target_lufs)
        wav = torch.from_numpy(wav)
        return wav


class testIRMAS(Dataset):
    def __init__(self, csv_path,
                 feat_dir,
                 normalize=True,
                 upsample=True,
                 ch_expand=False,
                 slice_win=43,
                 slice_hop=21,
                 ):
        # read manifest
        self.manifest = pd.read_csv(csv_path)
        self.data_len = len(self.manifest)
        self.audio_data = self.manifest['filename'].tolist()
        self.labels = self.manifest['encoded_label'].tolist()
        self.feat_dir = feat_dir

        self.sf = StickyFingers(normalize=normalize, upsample=upsample, ch_expand=ch_expand)

        self.slice_win = slice_win
        self.slice_hop = slice_hop

    def __getitem__(self, index):
        single_audio_path = self.audio_data[index].split('/')[-1]
        single_audio_label = self.labels[index]
        singe_label_list = single_audio_label.split(', ')
        single_feature_path = os.path.join(self.feat_dir, single_audio_path + '.npy')
        mel = np.load(single_feature_path)

        feat_list = []
        for s_idx, start_index in enumerate(range(0, mel.shape[1] - self.slice_win + 1, self.slice_hop)):
            mel_slice = mel[:, start_index:(start_index + self.slice_win)]
            mel_slice = self.sf.transform(mel_slice)
            feat_list.append(mel_slice)

        feats = torch.cat(feat_list, dim=0)

        return feats, sorted([int(item) for item in singe_label_list])

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    pass
