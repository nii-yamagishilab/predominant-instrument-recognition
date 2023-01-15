# https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self,
                 gamma,
                 alpha=None,
                 ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        loss = self.focal_loss(labels=labels, logits=logits, alpha=self.alpha, gamma=self.gamma)
        return loss

    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.
        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
        Args:
          labels: A float tensor of size [batch, num_classes].
          logits: A float tensor of size [batch, num_classes].
          alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
          gamma: A float scalar modulating loss from hard and easy examples.
        Returns:
          focal_loss: A float32 scalar representing normalized total loss.
        """
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                               torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        if self.alpha:
            loss = alpha * loss
        focal_loss = torch.sum(loss)

        # focal_loss /= torch.sum(labels)
        focal_loss /= labels.shape[0]
        return focal_loss
