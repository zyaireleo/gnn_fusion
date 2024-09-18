#!/bin/bash
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:1
source activate referformer
nvidia-smi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
export PYTHONPATH='/public/home/lfzh/LYF/ReferFormer'

python3 /public/home/lfzh/LYF/ReferFormer/datasets/ytvos.py
