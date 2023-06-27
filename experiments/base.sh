# ==============================================================================
# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Lifan Zhong
# All rights reserved.
# ==============================================================================
#!/bin/sh
#SBATCH -p qgpu
#SBATCH --job-name=no-aug
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --begin=now

/bin/hostname

source {your_env}

seed=2233
LR=1e-3
seg_dur=1
mixup_alpha=0.4
label_smoothing=0.05
LR_SCHED="cosine"
RESULT_DIR="irnet-no-aug-no-mixup-no-spec"
PREFIX="seed-$seed-mixup_alpha-$mixup_alpha"
model_name="mie"

python train.py \
--seed $seed \
--gpus 2 \
--prefix $PREFIX \
--model_cfg_path './model_configs/irnet_no_specbn.py' \
--nsynth_pretrained '' \
--model_path "./src/lms/$model_name.py" \
--result_folder "./$RESULT_DIR" \
--last_epoch 30 \
--warmup 3 \
--lr $LR \
--bs 128 \
--loss_fn 'CE' \
--optim 'adam' \
--lr_sched $LR_SCHED \
--sync_bn \
--augment 'stitch' \
--seg_dur $seg_dur \
--mixup_p 0.0 \
--mixup_alpha $mixup_alpha \
--label_smoothing $label_smoothing \
--aug_p 0.0 \
