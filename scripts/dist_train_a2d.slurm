#!/bin/bash
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:2

## NVIDIAA100-PCIE-40GB
## TeslaV100-SXM2-32GB
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=5
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

GPUS=${GPUS:-8}
PORT=${PORT:-29500}
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

OUTPUT_DIR=${1:-'./output/a2d/20240912/swint-scratch-hyper-node'}
PRETRAINED_WEIGHTS=${2:-'/public/home/lfzh/LYF/ReferFormer/pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth'}

PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this
echo "Load pretrained weights from: ${PRETRAINED_WEIGHTS}"

# train & test
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
#main.py --dataset_file a2d --with_box_refine --freeze_text_encoder --batch_size 2 \
#--epochs 6 --lr_drop 3 5 \
#--output_dir=${OUTPUT_DIR} --pretrained_weights=${PRETRAINED_WEIGHTS} ${PY_ARGS}

#from scratch
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --use_env main.py --dataset_file a2d --with_box_refine \
--epochs 15 --lr_drop 8 10 12 --dropout 0.1 --weight_decay 1e-4 --batch_size 2 \
--output_dir=${OUTPUT_DIR} --num_frames 6 --vision_backbone resnet50 \
--backbone_pretrained pretrained_weights/swin_tiny_patch244_window877_kinetics400_1k.pth
#--backbone_pretrained /public/home/lfzh/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth \
#--resume /public/home/lfzh/LYF/ReferFormer/output/a2d/20240830/resnet50-scratch-new-gnn-query-backbone-change/checkpoint.pth \

echo "Working path is: ${OUTPUT_DIR}"




