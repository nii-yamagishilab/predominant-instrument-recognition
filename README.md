Source codes for NSynth-Pretrained Predominant Instrument Recognition project.

## Requirements
Here's a list of some key pakages for our experiments, for more details please refer to `requirements.txt` file or `environment.yml` file.
- Python 3.9.12
- PyTorch 1.13.0
- torchaudio 0.13.0
- torchaudio-augmentations 0.2.4
- scikit-learn 1.0.2
- pytorch-lightning 1.8.0.post1
- numpy 1.22.3

## Data Preparation

If you are a member of Yamagishi-lab,NII just do:
```
ln -s /home/smg/v-zhonglifan/data/IRMAS ./irmas_data
ln -s /home/smg/v-xuanshi/DATA/nsynth ./nsynth_data
cp -r '/home/smg/v-zhonglifan/InstrumentRecognition/exp22-maybefinal/metadata' ./metadata
cp /home/smg/v-zhonglifan/InstrumentRecognition/c-IR/IRModels.tar.gz .
```

### Download Data
For pre-training, download the [NSynth](https://magenta.tensorflow.org/nsynth) data and put it in `nsyth_data`. The structure of the directory is expected to be:
```
- nsynth_data
    - nsynth-test
        - audio
            - bass_electronic_018-022-100.wav
            - ...
        - trim_audio (if you're going to concatenate the samples, please remove the silence parts)
            - ...
    - nsynth-train
        ...
    - nsynth-valid
        ...
```

For fine-tuning, download the [IRMAS](https://www.upf.edu/web/mtg/irmas) data and put it in `irmas_data` . The structure of the directory is expected to be:
```
- irmas_data
    - IRMAS-TestingData
        - Part1
            - (02) dont kill the whale-1.txt
            - (02) dont kill the whale-1.wav
        - Part2
        - Part3
    - IRMAS-TrainingData
        - cel
            - 008__[cel][nod][cla]0058__1.wav
        - cla
        - ...

```
### Write Metadata
To write metadata for both datasets, run:
```
python write_metadata_irmas.py
python write_metadata_nsynth.py
```
It takes some time to process NSynth, since we have to load the audio to check if it's a 'silence' clip.

You can find the processed manifest in `./metadata`.

## Trained Weights & Inference
Firstly, download the [weights](TODO), unzip them:
```
tar -xvf IRModels.tar.gz
```
You can find the unzipped pretrained weights in pair in `./model_ckpt_tmp`, and the folder looks like:
```
- model_ckpt_tmp
    - irnet4irmas.ckpt
    - irnet4nsynth.ckpt
    - ...
```
Then run:
```
python before_evaluate.py
```
This will proceed the checkpoint files to the form as our codes expect, by moving them to a proper place and rewrite some dict values to help our code find the model structure config. 
This is because our code expect the checkpoints to be in `{logs_dir}/{RESULT_DIR}/{PREFIX}/checkpoints/last.ckpt`, and rely on the config value to construct the model.

Sorry this code is far from clean yet (・・；)

After processing, run:
```
(evaluate all the checkpoints in a specifc folder)
python evaluate.py --model_name irmas_mie --ckpt_dir "logs_finetune" --mode test --act sigmoid --seg_dur 1

(evaluate one specific checkpoint)
python evaluate.py --model_name irmas_mie --ckpt_dir "logs_finetune/XXXXX/XXXX.ckpt" --mode test --act sigmoid --seg_dur 1
```

The results should be (let IRNet indicates our best model):

| Model                       | F1-micro | LRAP  | 
|:----------------------------|:---------|:------|
| IRNet                       | 0.682    | 0.816 |
| IRNet - w/o Effect Augments | 0.668    | 0.809 |
| IRNet - w/o Concatenation   | 0.671    | 0.814 |
| IRNet - w/o Mixup Augments  | 0.654    | 0.799 |
| IRNet - w/o Effect & Mixup  | 0.637    | 0.708 |


Note that this may slightly differ from our results, since we report the average results of three experiments with the same setting except for random seed.

If you are a member of Yamagishi-lab,NII, you can find all the trained weights in my folder `v-zhonglifan/InstrumentRecognition/exp22-maybefinal`.


## Pre-training
Please refer to the scripts in `./experiments`.
To pre-train a model on NSynth, run:
```
(for slurm) sbatch experiments/irnet.sh
(for private) sh experiments/irnet.sh
```
or you can directly run the code below. The pre-training results will be saved in `tb_logs/{RESULT_DIR}/{PREFIX}`.
```
python train.py \
--seed 233333 \
--gpus 2 \
--prefix PREFIX \
--model_cfg_path './model_configs/irnet.py' \
--nsynth_pretrained '' \
--model_path "./src/lms/mie.py" \
--result_folder "./RESULT_FOLDER" \
--last_epoch 30 \
--warmup 3 \
--lr 1e-3 \
--bs 128 \
--loss_fn 'CE' \
--optim 'adam' \
--lr_sched 'cosine' \
--sync_bn \
--augment 'stitch' \
--seg_dur 1 \
--mixup_p 0.5 \
--mixup_alpha 0.4 \
--label_smoothing 0.05 \
--aug_p 0.3 \
```
## Fine-tuning
Please refer to the scripts in `./experiments/finetune`.
To pre-train a model on NSynth, run:
```
(for slurm) sbatch experiments/finetune/irnet.sh
(for private) sh experiments/finetune/irnet.sh
```
or you can directly run the code below. The fine-tuning results will be saved in `logs_finetune/{RESULT_DIR}/{PREFIX}`.
```
python tune.py \
--seed 233333 \
--gpus 1 \
--prefix $PREFIX \
--pretrained "./tb_logs/$RESULT_DIR/seed-$seed-mixup_alpha-0.4/checkpoints/last.ckpt" \
--model_path "./src/lms/irmas_mie.py" \
--result_folder "./$RESULT_DIR" \
--train_meta_path 'metadata/irmas_slice_train.json' \
--valid_meta_path 'metadata/irmas_slice_valid.json' \
--wav_dir 'irmas_data/IRMAS-TrainingData/' \
--last_epoch 40 \
--warmup 5 \
--lr 2.5e-4 \
--bs 64 \
--loss_fn 'BCE' \
--lr_sched 'cosine' \
--normalize_amp \
--csf 'mlp' \
--label_smoothing 0.0 \
--mixup_p 0.0 \
--mixup_alpha 0.2 \
```

## Additional Experiments
If you're interested in the parameter reduction experiments, you can look into `experiments/reduced_pretrain` and `experiments/reduced_finetune`.

My implementation of shared residual blocks in PyTorch can be found in `src/models/mie/backbones/resnet.py`, which is a re-implementation of [ShaResNet](https://github.com/aboulch/sharesnet) [[arxiv](https://arxiv.org/abs/1702.08782)].

## *notes
If the codes run into errors when modified, or you find something confusing there. Here are some messages that might help:
* The `src` directory was previously named `doraemon` when the released model checkpoints were trained.
* Training was done by Pytorch-lightning modules, but inference was done separately with an Evaluator class in `src/evaluator`.

If you have any further question, please contact: [zhong_lifan@yahoo.co.jp](zhong_lifan@yahoo.co.jp)



