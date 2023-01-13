#!/bin/sh
#SBATCH -p qgpu
#SBATCH --job-name=red-3463
#SBATCH --time=72:00:00
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:tesla_a100:2
#SBATCH --begin=now

/bin/hostname

source /home/smg/v-zhonglifan/miniconda3/bin/activate ir

seed=1231
LR=1e-3
seg_dur=1
mixup_alpha=0.4
label_smoothing=0.05
LR_SCHED="cosine"
RESULT_DIR="reduced-3-4-6-3"
PREFIX="seed-$seed-mixup_alpha-$mixup_alpha"
model_name="mie"

python train.py \
--seed $seed \
--gpus 2 \
--prefix $PREFIX \
--model_cfg_path './model_configs/reduced_3-4-6-3.py' \
--nsynth_pretrained '' \
--model_path "./doraemon/lms/$model_name.py" \
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
--mixup_p 0.5 \
--mixup_alpha $mixup_alpha \
--label_smoothing $label_smoothing \
--aug_p 0.3 \
