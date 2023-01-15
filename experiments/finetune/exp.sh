#!/bin/sh
#SBATCH -p qgpu
#SBATCH --job-name=finetune
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --begin=now

/bin/hostname

source /home/smg/v-zhonglifan/miniconda3/bin/activate ir

seed=906
epoch=50
lr=2.5e-4
warmup=4
bs=64
LR_SCHED="cosine"
RESULT_DIR="mel-aug-mixup-modify"
PREFIX="seed-$seed-lr-$lr-bs-$bs-epoch-$epoch-warmup-$warmup"
model_name="irmas_mie"

python tune.py \
--seed $seed \
--gpus 1 \
--prefix $PREFIX \
--pretrained "/home/smg/v-zhonglifan/InstrumentRecognition/exp21-finalfinal/tb_logs/$RESULT_DIR/seed-$seed-mixup_alpha-0.4/checkpoints/last.ckpt" \
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