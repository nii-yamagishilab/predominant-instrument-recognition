import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from doraemon.loss_fn import FocalLoss, CBLoss
from doraemon.models.instr_emd_sinc_model import InstrEmdSincModel
from doraemon.datasets.nsynth_dataset import NSynthDataset
from doraemon.optimization import cosine_warmup_restart_exponential_decay, exponential_decay_with_warmup
from torch.utils.data import DataLoader
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from doraemon.utils import Config


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
            ckpt = self.update_nsynth_state_dict(self.hparams.nsynth_pretrained)
            self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('[Settings]Training with NSynth pretrained weights from {}'.format(self.hparams.nsynth_pretrained))
        else:
            print('[Settings]Without NSynth weights!')

        if self.hparams.loss_fn == 'BCE':
            self.loss = nn.BCEWithLogitsLoss()
            print('[Settings]Training criterion: {}'.format(self.hparams.loss_fn))
        elif self.hparams.loss_fn == 'focal':
            self.loss = FocalLoss(
                alpha=None,
                gamma=self.hparams.fl_gamma
            )
            print('[Settings]Training criterion: Focal Loss; Gamma={}'.format(self.hparams.fl_gamma))
        elif self.hparams.loss_fn == 'cb':
            self.loss = CBLoss(
                samples_per_class=[328, 429, 380, 539, 644, 579, 615, 530, 490, 493, 661],
                n_classes=11,
                loss_type=self.hparams.cb_loss_type,
                beta=self.hparams.cb_beta,
                gamma=self.hparams.fl_gamma,
            )
            print('[Settings]Training criterion: CB Loss with {}'.format(self.hparams.cb_loss_type))
            print('[Settings]Beta = {} | Gamma = {}'.format(self.hparams.cb_beta, self.hparams.fl_gamma))
        elif self.hparams.loss_fn == 'CE':
            self.loss = nn.CrossEntropyLoss()
            print('[Settings]Training criterion: CrossEntropyLoss')
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
        print('[Settings]Optimizer: {}'.format(self.hparams.optim.lower()))
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
                 sync_dist=True,
                 )
        self.log('train_loss', loss,
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

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        signals = batch['feature']
        one_hot_labels = batch['one_hot_label']
        labels = batch['label']
        _, outputs = self(signals)
        _, predicted = torch.max(outputs.data, 1)
        correct = torch.sum(preds == labels.data)
        return {'test_correct': correct}

    def test_epoch_end(self, test_step_outputs):
        avg_acc = torch.stack([x['test_correct'].float() for x in outputs]).sum() / len(self.test_dataloader())
        return {'avg_test_acc': avg_acc}

    def train_dataloader(self):
        train_dataset = NSynthDataset(meta_path='metadata/nsynth_train.json',
                                      wav_dir='nsynth_data/nsynth-train/audio',
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
                                      wav_dir='nsynth_data/nsynth-valid/audio',
                                      is_eval=True,
                                      )
        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=128,  # fix the valid batch size
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=False)
        print('[INFO]Number of Validation Samples: {}'.format(len(valid_dataset)))
        return valid_loader

    def test_dataloader(self):
        test_dataset = NSynthDataset(meta_path='metadata/nsynth_test.json',
                                     wav_dir='nsynth_data/nsynth-test/audio',
                                     is_eval=True,
                                     )
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=128,  # fix the valid batch size
                                 shuffle=False,
                                 num_workers=8,
                                 pin_memory=False)
        print('[INFO]Number of Testing Samples: {}'.format(len(test_dataset)))
        return test_loader

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

        from doraemon.utils import plot_hist
        plot_hist(o[46])


if __name__ == '__main__':
    model = InstrumentRecognizer(data_dir='/home/zhong_lifan/data/IRMAS-Precomputed/ICASSP23/train/1024-log-mel')
    model.prob()
