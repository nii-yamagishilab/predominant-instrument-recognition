# ==============================================================================
# Copyright (c) 2023, Yamagishi Laboratory, National Institute of Informatics
# Author: Lifan Zhong
# All rights reserved.
# ==============================================================================
#!/bin/sh
#SBATCH -p qgpu
#SBATCH --job-name=scratch
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --begin=now

/bin/hostname

source {your_env}

seed=906
epoch=40
lr=3.5e-3
warmup=5
bs=64
LR_SCHED="cosine"
RESULT_DIR="no-pre-train-no-specbn"
PREFIX="seed-$seed-lr-$lr-bs-$bs-epoch-$epoch-warmup-$warmup-lkrelu"
model_name="no_pre_irmas_mie"

python tune.py \
--seed $seed \
--gpus 1 \
--prefix $PREFIX \
--pretrained "tb_logs/irnet-no-specbn/seed-906-mixup_alpha-0.4/hparams.yaml" \
--model_path "./src/lms/$model_name.py" \
--result_folder "./$RESULT_DIR" \
--train_meta_path 'metadata/irmas_slice_train.json' \
--valid_meta_path 'metadata/irmas_slice_valid.json' \
--wav_dir 'irmas_data/IRMAS-TrainingData/' \
--last_epoch $epoch \
--warmup $warmup \
--lr $lr \
--bs $bs \
--loss_fn 'BCE' \
--lr_sched $LR_SCHED \
--normalize_amp \
--csf 'mlp' \
--label_smoothing 0.0 \
--mixup_p 0.0 \
--mixup_alpha 0.2 \

python evaluate.py --model_name $model_name --ckpt_dir "logs_finetune/$RESULT_DIR/$PREFIX" --mode test --act sigmoid --seg_dur 1