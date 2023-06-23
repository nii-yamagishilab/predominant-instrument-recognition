import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import Tensor


__all__ = ['LDE','SAP']

###############################################################################
### LDE
###############################################################################


class LDE(nn.Module):
    def __init__(self, D, input_dim, with_bias=False, distance_type='norm', network_type='att', pooling='mean'):
        super(LDE, self).__init__()
        self.dic = nn.Parameter(torch.randn(D, input_dim)) # input_dim by D (dictionary components)
        nn.init.uniform_(self.dic.data, -1, 1)
        self.wei = nn.Parameter(torch.ones(D)) # non-negative assigning weight in Eq(4) in LDE paper
        if with_bias: # Eq(4) in LDE paper
            self.bias = nn.Parameter(torch.zeros(D))
        else:
            self.bias = 0
        assert distance_type == 'norm' or distance_type == 'sqr'
        if distance_type == 'norm':
            self.dis = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dis = lambda x: torch.sum(x**2, dim=-1)
        assert network_type == 'att' or network_type == 'lde'
        if network_type == 'att':
            self.norm = lambda x: F.softmax(-self.dis(x) * self.wei + self.bias, dim = -2)
        else:
            self.norm = lambda x: F.softmax(-self.dis(x) * (self.wei ** 2) + self.bias, dim = -1)
        assert pooling == 'mean' or pooling == 'mean+std'
        self.pool = pooling
        # regularization maybe

    def forward(self, x):
        r = x.view(x.size(0), x.size(1), 1, x.size(2)) - self.dic # residaul vector
        w = self.norm(r).view(r.size(0), r.size(1), r.size(2), 1) # numerator without r in Eq(5) in LDE paper
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-9) #batch_size, timesteps, component # denominator of Eq(5) in LDE paper
        if self.pool == 'mean':
            x = torch.sum(w * r, dim=1) # Eq(5) in LDE paper
        else:
            x1 = torch.sum(w * r, dim=1) # Eq(5) in LDE paper
            x2 = torch.sqrt(torch.sum(w * r ** 2, dim=1)+1e-8) # std vector
            x = torch.cat([x1, x2], dim=-1)
        return x.view(x.size(0), -1)

## TODO
## MultiHead Implementation
# adapted from: https://github.com/zyzisyz/mfa_conformer/blob/1b9c229948f8dbdbe9370937813ec75d4b06b097/module/_pooling.py
class SAP(nn.Module):
    def __init__(self, dim:int, n_heads=1) -> None:
        """SAP
        Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
        Link： https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
        Args:
            dim (pair): the size of attention weights
        """
        super(SAP, self).__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.FloatTensor(dim, n_heads))
        self.relu = nn.ReLU()

    def forward(self, x:Tensor) -> Tensor:
        """Computes Self-Attentive Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, frames, dim).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        h = self.relu(self.sap_linear(x))
        w = torch.matmul(h, self.attention)
        w = F.softmax(w, dim=1)
        x = torch.sum(x * w, dim=1)
        return x