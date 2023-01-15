import torchaudio.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F


class MelExtractor(nn.Module):
    def __init__(self,
                 top_db=None,
                 orig_sr=44100,
                 target_sr=22050,
                 hop_length=512,
                 n_mels=128,
                 power=2,
                 ch_expand='copy',
                 ):
        super(MelExtractor, self).__init__()
        self.top_db = top_db
        self.orig_sr = orig_sr
        self.target_sr = target_sr

        self.n_mels = n_mels
        self.hop_length = hop_length
        self.power = power

        self.resampler = transforms.Resample(self.orig_sr, self.target_sr)
        self.trans = transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=2048,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_max=int(self.target_sr / 2))

        self.to_db = transforms.AmplitudeToDB(top_db=top_db)
        self.interpolator = nn.Upsample(size=224,mode='bilinear',align_corners=True)

        self.ch_expand = ch_expand


    def forward(self, x):
        x = self.resampler(x)
        x = self.trans(x)
        if self.power > 2:
            x = x ** 2
        out = self.to_db(x)
        out = out.unsqueeze(1)
        if self.ch_expand == 'copy':
            out = torch.cat((out,out,out),1)
        elif self.ch_expand == 'freq_split':
            h, w = out.shape[-2]//3, out.shape[-1]
            out = F.unfold(out,
                           kernel_size=(h,w),
                           stride=h,
                           ).permute(0,2,1)
            out = out.view(out.shape[0], out.shape[1], h, w)
        else:
            raise NotImplementedError
        out = self.interpolator(out)

        return out

    def scale(self,o):
        xx = o.view(o.size(0), -1)
        xx -= xx.min(1, keepdim=True)[0]
        xx /= xx.max(1, keepdim=True)[0]
        xx = xx.view(o.shape[0], o.shape[1], o.shape[2], o.shape[3])
        return xx