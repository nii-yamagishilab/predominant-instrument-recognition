import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from src.models.image_resnet50 import IRNet
from src.datasets.nsynth_dataset import NSynthDataset
from src.optimization import cosine_warmup_restart_exponential_decay, exponential_decay_with_warmup
from torch.utils.data import DataLoader
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.utils import Config
import random


class InstrumentRecognizer(pl.LightningModule):
    def __init__(self, **kwargs,
                 # train_meta_path,
                 # valid_meta_path,
                 # wav_dir,
                 # last_epoch,
                 # loss_fn,
                 # lr=3e-4,
                 # bs=80,
                 # warmup=5,
                 # ch_expand='copy',
                 # lr_sched='cosine',
                 ):
        super(InstrumentRecognizer, self).__init__()

        self.save_hyperparameters()

        self.model = IRNet(n_classes=1006)
        # if self.hparams.nsynth_pretrained:
        #     ckpt = self.update_nsynth_state_dict(self.hparams.nsynth_pretrained)
        #     self.model.load_state_dict(ckpt['state_dict'], strict=False)
        #     print('[Settings]Training with NSynth pretrained weights from {}'.format(self.hparams.nsynth_pretrained))
        # else:
        #     print('[Settings]Without NSynth weights!')

        if self.hparams.loss_fn == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
            print('[Settings]Training criterion: {}'.format(self.hparams.loss_fn))
        elif self.hparams.loss_fn == 'CE':
            self.loss = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
            print('[Settings]Training criterion: CrossEntropyLoss')
        else:
            raise NotImplementedError

    def forward(self, x):
        emb, x = self.model(x)
        return emb, x

    def configure_optimizers(self):
        if self.hparams.optim.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        else:
            raise NotImplementedError
        print('[Settings]Optimizer: {}'.format(self.hparams.optim.lower()))
        w_steps = int(self.hparams.warmup * len(self.train_dataloader()) / self.hparams.gpus)
        max_steps = int(self.hparams.last_epoch * len(self.train_dataloader()) / self.hparams.gpus)
        if self.hparams.lr_sched == 'cosine':
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                     w_steps,
                                                                     max_steps,
                                                                     )
        elif self.hparams.lr_sched.startswith('exp'):
            scheduler = exponential_decay_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=w_steps,
                step_size=len(self.train_dataloader()),
                gamma=float(self.hparams.lr_sched.split('-')[-1])
            )
        else:
            raise NotImplementedError
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        signals = batch['feature']
        one_hot_labels = batch['one_hot_label']
        labels = batch['label']

        if self.hparams.loss_fn == 'CE':
            if torch.rand(1).item() >= self.hparams.mixup_p:
                _, outputs = self(signals)
                loss = self.loss(outputs, labels)
            else:
                mixed_signals, y_a, y_b, lam = self.mixup_data(signals, labels, alpha=self.hparams.mixup_alpha)
                _, outputs = self(mixed_signals)
                loss = lam * self.loss(outputs, y_a) + (1 - lam) * self.loss(outputs, y_b)


        else:
            _, outputs = self(signals)
            loss = self.loss(outputs, one_hot_labels)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = Accuracy().cuda()
        acc = accuracy(predicted, labels)
        pbar = {'train_acc': acc}
        self.log('train_acc', acc,
                 sync_dist=True,
                 )
        self.log('train_loss', loss,
                 sync_dist=True,
                 )

        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        signals = batch['feature']
        one_hot_labels = batch['one_hot_label']
        labels = batch['label']
        _, outputs = self(signals)

        if self.hparams.loss_fn == 'CE':
            loss = self.loss(outputs, labels)
        else:
            loss = self.loss(outputs, one_hot_labels)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = Accuracy().cuda()
        acc = accuracy(predicted, labels)

        return {'val_loss_step': loss, 'val_acc_step': acc}

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['val_loss_step'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['val_acc_step'] for x in val_step_outputs]).mean()
        pbar = {'avg_val_acc': avg_val_acc}
        self.log('valid_acc', avg_val_acc,
                 sync_dist=True,
                 )
        self.log('valid_loss', avg_val_loss,
                 sync_dist=True,
                 )
        return {'val_loss': avg_val_loss, 'progress_bar': pbar}

    def train_dataloader(self):
        train_dataset = NSynthDataset(meta_path='metadata/nsynth_train.json',
                                      wav_dir='nsynth_data',
                                      augment=self.hparams.augment,
                                      seg_dur=self.hparams.seg_dur,
                                      aug_p=self.hparams.aug_p
                                      )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.hparams.bs,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=False)
        print('[INFO]Number of Training Samples: {}'.format(len(train_dataset)))
        return train_loader

    def val_dataloader(self):
        valid_dataset = NSynthDataset(meta_path='metadata/nsynth_valid.json',
                                      wav_dir='nsynth_data',
                                      is_eval=True,
                                      seg_dur=self.hparams.seg_dur,
                                      )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=128,  # fix the valid batch size
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=False)
        print('[INFO]Number of Validation Samples: {}'.format(len(valid_dataset)))
        return valid_loader

    def update_nsynth_state_dict(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        del checkpoint['state_dict']['fc11.fc.weight']
        del checkpoint['state_dict']['fc11.fc.bias']
        del checkpoint['state_dict']['fc12.fc.weight']
        del checkpoint['state_dict']['fc12.fc.bias']
        return checkpoint

    def mixup_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        x_roll = x.roll(1, 0)
        y_roll = y.roll(1, 0)

        mixed_x = lam * x + (1 - lam) * x_roll
        y_a, y_b = y, y_roll
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y, alpha=1.0):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        x_roll = x.roll(1, 0)
        y_roll = y.roll(1, 0)

        m_len = int((1 - lam) * x.shape[1])
        m_begin = random.randint(0, x.shape[1] - m_len)
        x[:, m_begin: m_len] = x_roll[:, m_begin: m_len]

        y_a, y_b = y, y_roll
        return x, y_a, y_b, lam

    def multi_mixup(self, x, y):
        n_samples = random.randint(2, 5)
        lam_list = torch.rand(n_samples).to(x)
        mixed_x = torch.zeros_like(x).to(x)
        y_list = []
        for i in range(n_samples):
            x_roll = x.roll(i, 0)
            y_roll = y.roll(i, 0)
            mixed_x += x_roll * lam_list[i]
            y_list.append(y_roll)
        return mixed_x, y_list, lam_list

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

    def prob(self):
        s = next(iter(self.train_dataloader()))
        print(o.shape)
        print(s.shape)
        # xx = o.view(o.size(0), -1)
        # xx -= xx.min(1, keepdim=True)[0]
        # xx /= xx.max(1, keepdim=True)[0]
        # xx = xx.view(o.shape[0], o.shape[1], o.shape[2], o.shape[3])

        from src.utils import plot_hist
        plot_hist(o[46])


if __name__ == '__main__':
    model = InstrumentRecognizer(data_dir='/home/zhong_lifan/data/IRMAS-Precomputed/ICASSP23/train/1024-log-mel')
    model.prob()
