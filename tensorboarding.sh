#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh && conda activate ir
tensorboard --logdir=$1 --port=12345 --bind_all