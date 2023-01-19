#!/bin/sh
#SBATCH -p qgpu
#SBATCH --job-name=finetune
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --begin=now

/bin/hostname

source /home/smg/v-zhonglifan/miniconda3/bin/activate ir

seed=1231
epoch=400
lr=1e-3
warmup=5
bs=128
LR_SCHED="restart-75-0.75"
RESULT_DIR="irnet-start-sgd"
PREFIX="seed-$seed-lr-$lr-bs-$bs-epoch-$epoch-warmup-$warmup-5e-4-$LR_SCHED"
model_name="irmas_mie"

python tune.py \
--seed $seed \
--gpus 1 \
--prefix $PREFIX \
--pretrained "/home/smg/v-zhonglifan/InstrumentRecognition/exp22-maybefinal/tb_logs/irnet/seed-$seed-mixup_alpha-0.4/checkpoints/last.ckpt" \
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
--optim "sgd" \
--mixup_p 0.0 \
--mixup_alpha 0.2 \

python evaluate.py --model_name $model_name --ckpt_dir "logs_finetune/$RESULT_DIR/$PREFIX" --mode test --act sigmoid --seg_dur 1