# ==============================================================================
# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Lifan Zhong
# All rights reserved.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchinfo import summary
from src.models.mie.transforms import SincConv


# model implementation of (Han+ 2016)
class IRNet(nn.Module):
    def __init__(self, num_classes=11):
        super(IRNet, self).__init__()
        self.upsampler = torchaudio.transforms.Resample(16000, 22050)

        kernel_size = int(50 / 1000 * 16000)
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        stride = int(12 / 1000 * 16000)
        self.trans = SincConv(
            out_channels=128,
            kernel_size=kernel_size,
            in_channels=1,
            padding='same',
            stride=stride,
            init_type='mel',
            min_low_hz=5,
            min_band_hz=5,
            requires_grad=True,
        )
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(0.25),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(0.25)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout2d(0.25)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(9, 10))
        )

        self.emb = nn.Linear(256, 1024)
        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

        # initialize
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_uniform_(m.weight)
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.upsampler(x)
        x = x / (x.max(1, True)[0])
        # mel = self.mel_spec(x)[:, :, :43]
        mel = self.trans(x)
        # x = torch.log(mel + 1e-6)
        x = mel.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.reshape(x.shape[0], -1)
        emb = self.emb(x)
        x = self.fc(emb)
        return emb, x