import json
import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelBinarizer


# random.seed(1024)
# np.random.seed(1024)
# torchaudio.set_audio_backend('sox_io')


class NSynthDataset(Dataset):
    """A dataset class for NSynthDataset.
    """

    def __init__(self,
                 meta_path,
                 wav_dir,
                 is_eval=False,
                 n_classes=16,
                 sr=16000,
                 ):
        self.wav_dir = wav_dir
        self.is_eval = is_eval
        self.sr = sr

        self.lb = LabelBinarizer()
        self.lb.fit([*range(n_classes)])

        with open(meta_path, 'r') as f:
            meta_json = json.load(f)
        self.dataset = OrderedDict(meta_json)
        self.datalist = list(self.dataset.keys())

    def __len__(self):
        return 3 * len(self.datalist) if self.is_eval else len(self.datalist)

    def __getitem__(self, index):
        clip_idx, group_idx = index % len(self.datalist), index // len(self.datalist)
        note_str = self.datalist[clip_idx]
        song_meta = self.dataset[note_str]
        label = song_meta['instrument_family_rearrange']
        one_hot_label = self.lb.transform([label])[0].astype('float')

        data_path = os.path.join(self.wav_dir, note_str + '.wav')
        waveform, _ = torchaudio.load(data_path)
        waveform = waveform[0]

        if self.is_eval:
            waveform = waveform.unfold(0, 3 * self.sr, int(self.sr / 2))[group_idx]
        else:
            waveform = self.get_clip(waveform)

        return {
            'feature': waveform,
            'label': label,
            'one_hot_label': one_hot_label,
        }

    def get_clip(self, wav):
        start = random.randint(0, self.sr)
        clip = wav[start: start + 3 * self.sr]
        return clip
