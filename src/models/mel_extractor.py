import torchaudio.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import os
from torch.distributions import Bernoulli


class MelExtractor(nn.Module):
    def __init__(self,
                 top_db=None,
                 sample_rate=22050,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=128,
                 power=2,
                 ch_expand='copy',
                 log_mels=True,
                 two_step=True,
                 normalize=False,
                 discretize=False,
                 p_spec_aug=0.0,
                 ):
        super(MelExtractor, self).__init__()
        self.top_db = top_db
        self.sample_rate = sample_rate

        self.ch_expand = ch_expand

        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

        self.f_min = 0.0
        self.f_max = int(self.sample_rate / 2)

        self.log_mels = log_mels
        self.log_offset = 1e-6
        # log_mel is useless!

        self.two_step = two_step

        self.normalize = normalize
        self.discretize = discretize

        if self.two_step:
            self.trans = nn.Sequential(
                transforms.Spectrogram(n_fft=n_fft, hop_length=self.hop_length, power=2),
                transforms.MelScale(
                    self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1, None, 'htk'
                )
            )
        else:
            self.trans = transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_max=int(self.sample_rate / 2))

        self.to_db = transforms.AmplitudeToDB(top_db=top_db)
        self.interpolator = nn.Upsample(size=224, mode='bilinear', align_corners=True)

        self.p_spec_aug = p_spec_aug
        if self.p_spec_aug > 0.0:
            self.spec_aug = True
            print('Spec Augment, p:{}'.format(self.p_spec_aug))
        else:
            self.spec_aug = False
        self.bernoulli = Bernoulli(self.p_spec_aug)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=21, iid_masks=True)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10, iid_masks=True)

    def forward(self, x):
        mel = self.trans(x)
        # if self.log_mels:
        #     out = torch.log(mel + self.log_offset)
        # else:
        if self.power > 2:
            mel = mel ** 2
        out = self.to_db(mel)

        if self.normalize:
            out = self._normalize(out)

        if self.discretize:
            out = self._discretize(out)

        out = out.unsqueeze(1)

        if self.training and self.spec_aug:
            is_apply = self.bernoulli.sample()
            if is_apply:
                out = self.freq_mask(out)
                out = self.time_mask(out)

        if self.ch_expand == 'copy':
            out = torch.cat((out, out, out), 1)
        elif self.ch_expand == 'freq_split':
            h, w = out.shape[-2] // 3, out.shape[-1]
            out = F.unfold(out,
                           kernel_size=(h, w),
                           stride=h,
                           ).permute(0, 2, 1)
            out = out.view(out.shape[0], out.shape[1], h, w)
        else:
            raise NotImplementedError
        out = self.interpolator(out)

        return out

    def _normalize(self, mel_spec):
        """
        perform normalization on mel-spectrogram (B x 1 X M_BIN X T)
        https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/5
        """

        bs, m_bin, t = mel_spec.shape[0], mel_spec.shape[1], mel_spec.shape[2]
        mel_spec = mel_spec.contiguous().view(bs, -1)
        mel_spec -= mel_spec.min(1, True)[0]
        mel_spec /= mel_spec.max(1, True)[0]
        mel_spec = mel_spec.view(bs, m_bin, t)

        return mel_spec

    def _discretize(self, mel_spec):
        '''
        mapping mel-spectrogram to discrete values.
        :param mel_spec:
        :return: discrete mel-spec
        '''

        mel_spec = torch.clamp(mel_spec, 0.0, 1.0)
        boundaries = torch.linspace(start=0.0, end=1.0, steps=256).type_as(mel_spec)
        idx = torch.bucketize(mel_spec, boundaries).type_as(mel_spec)
        mel_spec = idx * (1 / 255)

        return mel_spec


def plot_spectrogram(specgram, out_path, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(specgram, origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.savefig(os.path.join(out_path, '{}.jpg'.format(title)), )
    plt.close()


def plot_hist(mel, title):
    mel = mel.detach().cpu().numpy().ravel()
    plt.hist(mel, color='blue', edgecolor='black',
             bins=20)
    plt.savefig('dist-{}.jpg'.format(title))
    plt.close()


if __name__ == '__main__':
    test_wav = '/home/zhong_lifan/data/IRMAS-TrainingData/cla/152__[cla][nod][cla]0217__3.wav'
    waveform, sample_rate = torchaudio.load(test_wav, normalize=True)
    waveform = (waveform[0] + waveform[1]) / 2
    waveform = waveform[:44100].unsqueeze(0)
    p = MelExtractor(n_fft=2048, hop_length=512, log_mels=False,
                     normalize=True)

    out = p(waveform)
    exit()
    import matplotlib.pyplot as plt

    title = 'mel-log-filter-orig'
    plot_spectrogram(specgram=out, out_path='./', title=title)
    plot_hist(out, title=title)
    print(out.shape)
