# ==============================================================================
# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Lifan Zhong
# All rights reserved.
# ==============================================================================

import os
import glob
import torch
import shutil


def move_ckpt(tem_dir):
    # irnet4irmas.ckpt & irnet4nsynth.ckpt
    ckpt_lst = glob.glob(os.path.join(tem_dir, '*.ckpt'))
    for ckpt_path in ckpt_lst:
        rename_model_file(ckpt_path)

def rename_model_file(ckpt_path):
    ckpt = torch.load(ckpt_path)
    basename = os.path.basename(ckpt_path)
    dir_name = os.path.join(ckpt['hyper_parameters']['result_folder'][2:], ckpt['hyper_parameters']['prefix'], 'checkpoints')
    if 'nsynth' in basename:
        ckpt['hyper_parameters']['model_path'] = './src/lms/mie.py'
        save_dir = os.path.join('tb_logs', dir_name)
        os.makedirs(save_dir, exist_ok=True)
    elif 'irmas' in basename:
        ckpt['hyper_parameters']['model_path'] = './src/lms/irmas_mie.py'
        ckpt['hyper_parameters']['pretrained'] = './tb_logs' + \
                                                 ckpt['hyper_parameters']['pretrained'].strip().split('tb_logs')[1]
        save_dir = os.path.join('logs_finetune', dir_name)
        os.makedirs(save_dir, exist_ok=True)
    else:
        print('ok but.. at least indicate the stage, nsynth or imras?')
        exit()
    torch.save(ckpt, os.path.join(save_dir, 'last.ckpt'))

if __name__ == '__main__':
    move_ckpt(tem_dir='model_ckpt_tmp')