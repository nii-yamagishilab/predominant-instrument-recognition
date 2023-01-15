import json
import os
import random
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torchaudio
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from torchaudio_augmentations import *


# random.seed(1024)
# np.random.seed(1024)
# torchaudio.set_audio_backend('sox_io')


class NSynthDataset(Dataset):
    """A dataset class for NSynthDataset.
    """

    def __init__(self,
                 meta_path,
                 wav_dir,
                 augment=None,
                 is_eval=False,
                 n_classes=1006,
                 sr=16000,
                 seg_dur=3,
                 aug_p=0.0,
                 ):
        self.wav_dir = wav_dir
        self.is_eval = is_eval
        self.sr = sr
        self.seg_dur = int(seg_dur * self.sr)

        self.lb = LabelBinarizer()
        self.lb.fit([*range(n_classes)])

        with open(meta_path, 'r') as f:
            meta_json = json.load(f)
        self.dataset = OrderedDict(meta_json)
        self.datalist = list(self.dataset.keys())

        self.instr_audio_dict = defaultdict(list)
        self.update_instr_meta()

        self.augment = augment.strip().split(', ') if augment else False
        # resampler for trimmed audio
        self.resampler = torchaudio.transforms.Resample(24000, 16000)

        self.transform = self.augments()
        self.aug_p = aug_p

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, index):
        song_meta = self.dataset[self.datalist[index]]
        note_str = song_meta['note_str']
        label = song_meta['instrument']
        instr_fml_label = song_meta['instrument_family_rearrange']

        one_hot_label = self.lb.transform([label])[0].astype('float')

        if (not self.is_eval) and self.augment and random.random() < 0.5:
            if 'stitch' in self.augment:
                waveform = self.stitch(note_str, song_meta)
                # start_index = random.randint(0, waveform.shape[1] - self.seg_dur)
                # waveform = waveform[0][start_index: start_index + self.seg_dur]

        else:
            data_path = os.path.join(self.wav_dir, 'nsynth-{}'.format(song_meta['official_split']), 'audio',
                                     note_str + '.wav')
            waveform, _ = torchaudio.load(data_path)
        waveform = waveform[0][:self.seg_dur]


        if torch.rand(1).item() < self.aug_p:
            waveform = self.transform(waveform.unsqueeze(0)).squeeze(0)

        return {
            'feature': waveform,
            'label': label,
            'one_hot_label': one_hot_label,
            'fml_label': instr_fml_label
        }

    def get_random_clip(self, wav):
        start = random.randint(0, self.sr)
        clip = wav[start: start + 3 * self.sr]
        return clip

    def stitch(self, note_str, song_meta):
        samples_pool = self.instr_audio_dict[song_meta['instrument']]
        audio_len = 0
        data_file = os.path.join(self.wav_dir, 'nsynth-{}'.format(song_meta['official_split']),
                                 'trim_audio',
                                 note_str + '.wav')
        wav, _ = torchaudio.load(data_file)
        wav = self.resampler(wav)
        wavform = [wav]
        audio_len += wav.shape[1]
        while audio_len < self.seg_dur:
            note_str = random.sample(samples_pool, 1)[0]
            data_file = os.path.join(self.wav_dir, 'nsynth-{}'.format(self.dataset[note_str]['official_split']),
                                     'trim_audio',
                                     note_str + '.wav')
            wav, _ = torchaudio.load(data_file)
            wav = self.resampler(wav)
            wavform.append(wav)
            audio_len += wav.shape[1]
        wavform = torch.cat(wavform, dim=1).to(wav)
        return wavform

    def is_sil(self, wav):
        mean_sig = torch.mean(np.abs(wav))
        return True if mean_sig < 5e-4 else False

    def update_instr_meta(self):
        instr_fml_labels = ['vocal',
                            'string',
                            'bass',
                            'synthetic_bass',
                            'flute',
                            'electronic_guitar',
                            'acoustic_guitar',
                            'synthetic_guitar',
                            'synth_lead',
                            'mallet',
                            'reed',
                            'brass',
                            'organ',
                            'electronic_keyboard',
                            'acoustic_keyboard',
                            'synthetic_keyboard',
                            ]
        le = LabelEncoder()
        le.fit(instr_fml_labels)
        for k, v in self.dataset.items():
            # add audio samples for instrument
            self.instr_audio_dict[v['instrument']].append(k)
            # add revised instrument family labels
            i_f_s = v['instrument_family_str']
            if i_f_s == 'guitar' or i_f_s == 'keyboard':
                i_f_s = v['instrument_source_str'] + '_' + i_f_s

            if i_f_s == 'bass' and v['instrument_source_str'] == 'synthetic':
                i_f_s = 'synthetic_bass'
            v['instrument_family_rearrange'] = int(le.transform([i_f_s])[0])
            v['instrument_family_rearrange_str'] = i_f_s


    def augments(self):
        transforms = [
            # RandomApply([PolarityInversion()], p=0.8),
            RandomApply([Noise(min_snr=0.001, max_snr=0.005)], p=0.3),
            RandomApply([Gain()], p=0.2),
            HighLowPass(sample_rate=16000),  # this augmentation will always be applied in this aumgentation chain!
            RandomApply([Delay(sample_rate=16000)], p=0.5),
            RandomApply([PitchShift(
                n_samples=1,
                sample_rate=16000
            )], p=0.4),
            RandomApply([Reverb(sample_rate=16000)], p=0.3)
        ]
        transform = Compose(transforms=transforms)
        return transform
