#!/bin/bash
#SBATCH --gres=gpu:NVIDIAA100-PCIE-40GB:2

## NVIDIAA100-PCIE-40GB
## TeslaV100-SXM2-32GB
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
source activate referformer
#module load anaconda
#module load cuda/12.1
#module load gcc
#module list
#source activate foundmental
#GPUS=${GPUS:-4}
#module load cuda/11.7
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_HOME=/usr/local/cuda-11.7
nvcc
GPUS=2
PORT=${PORT:-29500}
if [ $GPUS -lt 4 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-4}

OUTPUT_DIR=${1:-'./output/ytvos/20240813/'}
PRETRAINED_WEIGHTS=${2:-'/public/home/lfzh/LYF/ReferFormer/output/coco/20240730/checkpoint.pth'}
PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

CHECKPOINT=${OUTPUT_DIR}/checkpoint.pth

# train
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
#main.py --with_box_refine --binary --freeze_text_encoder \
#--epochs 6 --lr_drop 3 5 \
#--output_dir=${OUTPUT_DIR} --pretrained_weights=${PRETRAINED_WEIGHTS} ${PY_ARGS}

## inference
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder \
--output_dir=${OUTPUT_DIR} --resume=${CHECKPOINT}  ${PY_ARGS}

echo "Working path is: ${OUTPUT_DIR}"

