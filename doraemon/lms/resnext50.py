import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from doraemon.utils import binarize
from doraemon.models.resnext50 import IRResNeXt50
from doraemon.datasets.irmas_dataset import IRMASDataset
from doraemon.optimization import cosine_warmup_restart_exponential_decay, exponential_decay_with_warmup
from torch.utils.data import DataLoader
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class InstrumentRecognizer(pl.LightningModule):
    def __init__(self, **kwargs
                 # train_meta_path,
                 # valid_meta_path,
                 # wav_dir,
                 # last_epoch,
                 # loss_fn,
                 # lr=3e-4,
                 # bs=80,
                 # warmup=5,
                 # top_db=80.0,
                 # ch_expand='copy',
                 # lr_sched='cosine',
                 ):
        super(InstrumentRecognizer, self).__init__()
        self.save_hyperparameters()

        self.model = IRResNeXt50(init_mlp=self.hparams.init_mlp,
                                n_fft=self.hparams.n_fft,
                                normalize=self.hparams.normalize,
                                discretize=self.hparams.discretize
                                )

        if self.hparams.loss_fn == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.hparams.loss_fn == 'focal':
            self.loss = None
        else:
            raise NotImplementedError
        print('Training criterion: {}'.format(self.hparams.loss_fn))

    def forward(self, x):
        emb, x = self.model(x)
        return emb, x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        w_steps = int(self.hparams.warmup * len(self.train_dataloader()) / self.hparams.gpus)
        max_steps = int(self.hparams.last_epoch * len(self.train_dataloader()) / self.hparams.gpus)
        if self.hparams.lr_sched == 'cosine':
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                     w_steps,
                                                                     max_steps,
                                                                     )
        elif self.hparams.lr_sched.startswith('polynomial'):
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=w_steps,
                num_training_steps=max_steps,
                power=int(self.hparams.lr_sched[-1]),
            )
        elif self.hparams.lr_sched.startswith('restart'):
            scheduler = cosine_warmup_restart_exponential_decay(
                optimizer=optimizer,
                num_warmup_steps=w_steps,
                num_training_steps=max_steps,
                num_cycles=int(self.hparams.lr_sched.split('-')[1]),
                gamma=float(self.hparams.lr_sched.split('-')[2]),
            )
        elif self.hparams.lr_sched.startswith('exp'):
            scheduler = exponential_decay_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=w_steps,
                num_training_steps=max_steps,
                num_epoch=self.hparams.last_epoch,
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
        _, outputs = self(signals)
        if self.hparams.loss_fn == 'focal':
            loss = torchvision.ops.sigmoid_focal_loss(outputs,
                                                      one_hot_labels,
                                                      reduction=self.hparams.reduction)
        else:
            loss = self.loss(outputs, one_hot_labels)

        _, predicted = torch.max(outputs.data, 1)
        accuracy = Accuracy().cuda()
        acc = accuracy(predicted, labels)
        pbar = {'train_acc': acc}
        self.log('train_acc', acc,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log('train_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )

        return {'loss': loss, 'progress_bar': pbar}

    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
        del results['progress_bar']['train_acc']
        return results

    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()
        pbar = {'avg_val_acc': avg_val_acc}
        self.log('valid_acc', avg_val_acc,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        self.log('valid_loss', avg_val_loss,
                 on_step=False,
                 on_epoch=True,
                 sync_dist=True,
                 )
        if self.current_epoch >= 1:
            self.log("hp_metric", avg_val_loss, sync_dist=True)
        return {'val_loss': avg_val_loss, 'progress_bar': pbar}

    def train_dataloader(self):
        train_dataset = IRMASDataset(meta_path=self.hparams.train_meta_path,
                                     wav_dir=self.hparams.wav_dir,
                                     normalize_amp=self.hparams.normalize_amp
                                     )
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.hparams.bs,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=False)
        return train_loader

    def val_dataloader(self):
        valid_dataset = IRMASDataset(meta_path=self.hparams.valid_meta_path,
                                     wav_dir=self.hparams.wav_dir,
                                     normalize_amp=self.hparams.normalize_amp
                                     )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=512,  # fix the valid batch size
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=False)
        return valid_loader

    def prob(self):
        s = next(iter(self.train_dataloader()))
        s = s['feature']
        o = self.preprocessor(s)
        print(o.shape)
        print(s.shape)
        # xx = o.view(o.size(0), -1)
        # xx -= xx.min(1, keepdim=True)[0]
        # xx /= xx.max(1, keepdim=True)[0]
        # xx = xx.view(o.shape[0], o.shape[1], o.shape[2], o.shape[3])

        from doraemon.utils import plot_hist
        plot_hist(o[46])


if __name__ == '__main__':
    model = InstrumentRecognizer(data_dir='/home/zhong_lifan/data/IRMAS-Precomputed/ICASSP23/train/1024-log-mel')
    model.prob()
