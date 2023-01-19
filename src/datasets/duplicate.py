import glob
import json
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
import librosa
import os
from pathlib import Path
import random
import numpy as np
import torchaudio
from tqdm import tqdm
import torchvision
from PIL import Image
import torch.nn.functional as F
from .sf import StickyFingers


class IRMAS(Dataset):
    def __init__(self, csv_path,
                 feat_dir,
                 normalize=True,
                 upsample=True,
                 ch_expand=False,
                 slice_win=43,
                 slice_hop=43,
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

        self.max_n_slice = (130-self.slice_win)//self.slice_hop + 1

    def __getitem__(self, index):
        group = int(index // self.data_len)
        mel_basename = self.audio_data[index - group * self.data_len].split('/')[-1] + '.npy'
        single_audio_label = self.labels[index - group * self.data_len]
        # get mel
        mel_path = os.path.join(self.feat_dir, mel_basename)
        mel = np.load(mel_path)

        # get slice
        mel_slice = mel[:, group * self.slice_hop : (group * self.slice_hop + self.slice_win)]
        mel_slice = self.sf.transform(mel_slice)

        return (mel_slice, single_audio_label)

    def __len__(self):
        return self.max_n_slice * self.data_len


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
