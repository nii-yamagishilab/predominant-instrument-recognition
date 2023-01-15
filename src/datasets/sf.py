import json

from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
import torch
import numpy as np

class StickyFingers():
    def __init__(self,normalize=True, upsample=True, ch_expand=False):
        self.scaler = MinMaxScaler(feature_range=(0, 1),clip=True)
        self.normalize= normalize
        self.upsample = upsample
        self.ch_expand = ch_expand

    def transform(self, x):
        x = x.transpose()
        if self.normalize:
            x = self.scaler.fit_transform(x)

        # upsample and convert to tensor
        if self.upsample:
            x = self.upsampler(x)

        if self.ch_expand:
            x = torch.cat((x,x,x),0)

        return x

    def upsampler(self,orig):
        orig = torch.from_numpy(orig).unsqueeze(0).unsqueeze(0)
        upsampled = F.interpolate(orig, size=224)

        return upsampled.squeeze(0)


def get_bgm_metadata(metadata):
    from collections import defaultdict
    bgm_meta = defaultdict(dict)
    for label_idx in range(11):
        for k, v in metadata.items():
            if v['instrument'] != label_idx:
                bgm_meta[label_idx].update({k: metadata[k]})

    return bgm_meta

if __name__ == '__main__':
    with open('/home/smg/v-zhonglifan/InstrumentRecognition/exp5-aug/metadata/irmas_train.json','r') as f:
        config = json.load(f)
    a = get_bgm_metadata(config)
    print(list(a[0].keys())[1680])
