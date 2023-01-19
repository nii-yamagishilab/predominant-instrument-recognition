import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from src.loss_fn import FocalLoss, CBLoss
from src.models.instr_emd_sinc_model import InstrEmdSincModel
from src.datasets.irmas_dataset import IRMASDataset
from src.optimization import cosine_warmup_restart_exponential_decay, exponential_decay_with_warmup
from torch.utils.data import DataLoader
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.utils import Config


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
        model_cfg = Config.fromfile(self.hparams.model_cfg_path)

        self.model = InstrEmdSincModel(opt=model_cfg.model)
        if self.hparams.nsynth_pretrained:
            # ckpt = self.update_nsynth_state_dict(self.hparams.nsynth_pretrained)
            # self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Training with NSynth pretrained weights from {}'.format(self.hparams.nsynth_pretrained))
        else:
            print('Without NSynth weights!')

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
            self.loss = nn.CrossEntropyLoss()
            print('Training criterion: CrossEntropyLoss')
        else:
            raise NotImplementedError

    def forward(self, x):
        emb, x = self.model(x)
        return emb, x

    def configure_optimizers(self):
        if self.hparams.optim.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=5e-4)
        elif self.hparams.optim.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
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
        _, outputs = self(signals)

        if self.hparams.loss_fn == 'CE':
            loss = self.loss(outputs, labels)
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
                                     normalize_amp=self.hparams.normalize_amp,
                                     p_aug=self.hparams.p_aug,
                                     mixup_alpha=self.hparams.mixup_alpha,
                                     min_snr_db=self.hparams.min_snr_db,
                                     max_snr_db=self.hparams.max_snr_db,
                                     is_song=True,
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
                                     is_song=True,
                                     )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=512,  # fix the valid batch size
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=False)
        return valid_loader

    def update_nsynth_state_dict(self, ckpt_path):
        checkpoint = torch.load(ckpt_path)
        del checkpoint['state_dict']['fc11.fc.weight']
        del checkpoint['state_dict']['fc11.fc.bias']
        del checkpoint['state_dict']['fc12.fc.weight']
        del checkpoint['state_dict']['fc12.fc.bias']
        return checkpoint


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
