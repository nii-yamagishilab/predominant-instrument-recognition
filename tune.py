import importlib.machinery
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, EarlyStopping, ModelSummary
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import yaml
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# from doraemon import Evaluator


def callback_list():
    cblist = []
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpointer = ModelCheckpoint(
        monitor="valid_acc",
        mode="max",
        filename="{epoch}-{valid_acc:.3f}",
        save_top_k=2,
        save_weights_only=True,
        save_last=True,
    )
    els = EarlyStopping(monitor='valid_acc',
                        patience=30,
                        mode='max')
    cblist.append(lr_monitor)
    cblist.append(checkpointer)
    cblist.append(els)
    cblist.append(
        ModelSummary(max_depth=-1)
    )
    return cblist

def get_args():
    # 引数の導入
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_meta_path', type=str, required=True)
    parser.add_argument('--valid_meta_path', type=str, required=True)
    parser.add_argument('--wav_dir', type=str, required=True)
    parser.add_argument('--bs', type=int, required=True)
    parser.add_argument('--lr', type=float, required=True)
    parser.add_argument('--warmup', type=int, required=True)
    parser.add_argument('--last_epoch', type=int, required=True)
    parser.add_argument('--lr_sched', type=str, required=True)
    parser.add_argument('--pretrained', type=str, required=True)

    parser.add_argument('--optim', type=str, default='adam')

    parser.add_argument('--loss_fn', type=str, required=True)

    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--result_folder', type=str, required=True)

    parser.add_argument('--prefix', type=str)

    parser.add_argument('--normalize_amp', action='store_true')
    parser.add_argument('--no-normalize_amp', dest='normalize_amp', action='store_false')
    parser.set_defaults(normalize_amp=False)

    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--no-sync_bn', dest='sync_bn', action='store_false')
    parser.set_defaults(sync_bn=False)

    # # Instrument Encoder CFG
    # parser.add_argument('--model_cfg_path', type=str, required=True)
    # parser.add_argument('--nsynth_pretrained', type=str, required=True)

    # temperature
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    # structure
    parser.add_argument('--csf', type=str, default='mlp')

    parser.add_argument('--mixup_p', type=float, default=0.0)
    parser.add_argument('--mixup_alpha', type=float, default=1.0)



    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    start_time = time.time()

    pm = importlib.machinery.SourceFileLoader('pm', args.model_path).load_module()
    pl.seed_everything(int(args.seed), workers=True)
    model = pm.IRMASRecognizer(**vars(args))

    exp_name = args.result_folder.split('/')[-1]

    if args.prefix:
        v = args.prefix
    else:
        v = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    logger = TensorBoardLogger(
        save_dir='logs_finetune',
        name=exp_name,
        version=v,
    )

    callbacks = callback_list()
    if args.gpus > 1:
        trainer = pl.Trainer(
            max_epochs=args.last_epoch,
            accelerator='gpu',
            devices=args.gpus,
            strategy="ddp",
            callbacks=callbacks,
            deterministic=True,
            logger=logger,
            enable_progress_bar=False,
            sync_batchnorm=args.sync_bn
        )
        trainer.fit(model)
    else:
        trainer = pl.Trainer(
            max_epochs=args.last_epoch,
            accelerator='gpu',
            devices=1,
            callbacks=callbacks,
            deterministic=True,
            logger=logger,
            enable_progress_bar=True
        )
        trainer.fit(model)

    end_time = time.time()
    time_spent = (end_time - start_time)/3600
    print('total training time = {:.3f} h'.format(time_spent))
    # for ckpt_path in glob.glob('{}/**/*.ckpt'.format(result_folder), recursive=True):
    #     e = Evaluator(model_path=model_path,
    #                   feat_dir=test_dir)
    #     e.test(ckpt_path,result_folder)
