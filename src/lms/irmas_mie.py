import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from src.loss_fn import FocalLoss, CBLoss
from src.lms.mie import InstrumentRecognizer
from src.datasets.irmas_dataset import IRMASDataset
from src.optimization import cosine_warmup_restart_exponential_decay, exponential_decay_with_warmup, step_with_warmup
from torch.utils.data import DataLoader
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random


class IRMASRecognizer(pl.LightningModule):
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
        super(IRMASRecognizer, self).__init__()
        self.save_hyperparameters()

        feature_extractor = InstrumentRecognizer.load_from_checkpoint(self.hparams.pretrained)
        feature_extractor = feature_extractor.model
        feat_dim = feature_extractor.fc0.in_features
        latent_dim = feature_extractor.fc0.out_features
        out_dim = feature_extractor.fc11.fc.in_features
        self.feature_extractor = feature_extractor
        if self.hparams.csf == 'mlp':
            self.feature_extractor.fc0 = nn.Identity()
            self.feature_extractor.bn0 = nn.Identity()
            self.feature_extractor.fc11 = nn.Identity()
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(feat_dim),
                nn.Dropout(p=0.15),
                nn.Linear(feat_dim, 11)
            )
        elif self.hparams.csf == 'head_only':
            self.feature_extractor.fc11.fc = nn.Linear(in_features=feature_extractor.fc11.fc.in_features
                                                       , out_features=11)
            print('[INFO] finetune with head only')
        else:
            raise NotImplementedError

        if self.hparams.loss_fn == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
            print('Training criterion: {}'.format(self.hparams.loss_fn))
        elif self.hparams.loss_fn == 'focal':
            self.loss = FocalLoss(
                alpha=None,
                gamma=self.hparams.fl_gamma
            )
            print('Training criterion: Focal Loss; Gamma={}'.format(self.hparams.fl_gamma))
        elif self.hparams.loss_fn == 'cb':
            self.loss = CBLoss(
                samples_per_class=[328, 429, 380, 539, 644, 579, 615, 530, 490, 493, 661],
                n_classes=11,
                loss_type=self.hparams.cb_loss_type,
                beta=self.hparams.cb_beta,
                gamma=self.hparams.fl_gamma,
            )
            print('Training criterion: CB Loss with {}'.format(self.hparams.cb_loss_type))
            print('Beta = {} | Gamma = {}'.format(self.hparams.cb_beta, self.hparams.fl_gamma))
        elif self.hparams.loss_fn == 'CE':
            self.loss = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
            print('Training criterion: CrossEntropyLoss with Label Smoothing: {}'.format(self.hparams.label_smoothing))
        else:
            raise NotImplementedError

    def forward(self, x):
        emb = self.feature_extractor(x)[0]
        x = self.classifier(emb)
        return emb, x

    def configure_optimizers(self):
        if self.hparams.optim.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        elif self.hparams.optim.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=float(1e-3 / self.hparams.lr))
        elif self.hparams.optim.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise NotImplementedError
        print('[SETTINGS] Optimizer: {}'.format(self.hparams.optim.lower()))
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
        elif self.hparams.lr_sched.startswith('step'):
            scheduler = step_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=w_steps,
                step_size=(len(self.train_dataloader()) * 50),
                gamma=0.5
            )
        elif self.hparams.lr_sched.startswith('restart'):
            scheduler = cosine_warmup_restart_exponential_decay(
                optimizer=optimizer,
                num_warmup_steps=w_steps,
                num_training_steps=max_steps,
                num_cycles=int(self.hparams.lr_sched.split('-')[1]),
                gamma=float(self.hparams.lr_sched.split('-')[2]),
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
            _, outputs = self(signals)
            loss = self.loss(outputs, labels)
        else:
            _, outputs = self(signals)
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
        train_dataset = IRMASDataset(meta_path=self.hparams.train_meta_path,
                                     wav_dir=self.hparams.wav_dir,
                                     normalize_amp=self.hparams.normalize_amp,
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
                                     normalize_amp=self.hparams.normalize_amp,
                                     )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=512,  # fix the valid batch size
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=False)
        return valid_loader

    # def test_dataloader(self):
    #     test_dataset = IRMASDataset(
    #         meta_path='metadata/irmas_test.json',
    #         wav_dir='irmas_data/IRMAS-TestingData',
    #         normalize_amp=True,
    #         is_eval=True,
    #     )
    #     test_loader = DataLoader(
    #         dataset=test_dataset,
    #         batch_size=1,
    #         num_workers=8,
    #     )
    #     return test_loader


if __name__ == '__main__':
    pass
