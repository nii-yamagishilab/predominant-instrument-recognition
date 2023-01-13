import torch
import torch.nn as nn
import torchvision
import numpy as np
from timm.models.layers import trunc_normal_
from doraemon.models.mel_extractor import MelExtractor


class IRResNeXt50(nn.Module):
    def __init__(self, pretrained=True,
                 feature_dim=1024,
                 init_mlp='random',
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
                 ):
        super(IRResNeXt50, self).__init__()
        weights = 'ResNeXt50_32X4D_Weights.IMAGENET1K_V2' if pretrained else None
        self.feature_dim = feature_dim

        self.front = MelExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
            log_mels=log_mels,
            two_step=two_step,
            top_db=top_db,
            ch_expand=ch_expand,
            normalize=normalize,
            discretize=discretize,
        )
        self.backbone = torchvision.models.resnext50_32x4d(weights=weights)
        self.backbone.fc = nn.Linear(2048, self.feature_dim, bias=False)
        self.bn0 = nn.BatchNorm1d(self.feature_dim)
        self.head = nn.Linear(self.feature_dim, 11)

        if init_mlp != 'random':
            self.backbone.fc.apply(self._init_weights)
            self.bn0.apply(self._init_weights)
            self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.front(x)
        emb = self.backbone(x)
        x = self.bn0(emb)
        x = self.head(x)

        return emb, x
