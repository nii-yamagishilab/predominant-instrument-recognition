# ==============================================================================
# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Lifan Zhong
# All rights reserved.
# ==============================================================================
import json
import torch
from torch.utils.data.dataset import Dataset
import os
import random
import numpy as np
import torchaudio
from collections import OrderedDict, defaultdict
from sklearn.preprocessing import LabelBinarizer
import pyloudnorm as pyln

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

    # def mixup_inside_batch(self, y1, y2, mix_alpha=10, ln=False):
    #     lam = np.random.beta(mix_alpha, mix_alpha)
    #     batch_size = x.size()[0]
    #     index = torch.randperm(batch_size).cuda()
    #     return

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

if __name__ == "__main__":
    pass
