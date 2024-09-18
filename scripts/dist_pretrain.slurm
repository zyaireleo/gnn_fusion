#!/bin/bash
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:2
#opts -- NVIDIAA100-PCIE-40GB TeslaV100-SXM2-32GB
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
##SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --time=72:00:00
#SBATCH --mem=32G
source activate referformer
export CUDA_HOME=/usr/local/cuda-11.7
export TORCH_DISTRIBUTED_DEBUG=DETAIL
nvcc

GPUS=2
CPUS_PER_TASK=2
PORT=${PORT:-29002}
if [ $GPUS -lt 4 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-4}
fi
CPUS_PER_TASK=${CPUS_PER_TASK:-2}

OUTPUT_DIR=${1:-'./output/coco/20240730/'}

PY_ARGS=${@:3}  # Any arguments from the forth one are captured by this

# train
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${PORT} --use_env \
main_pretrain.py --with_box_refine --dataset_file all  --binary --batch_size 2 \
 --num_frames=1 --epochs=12 --lr_drop 6 8 10 --output_dir=${OUTPUT_DIR} ${PY_ARGS}\
 --resume=${OUTPUT_DIR}/checkpoint.pth
# do not freeze text encoder\

echo "Working path is: ${OUTPUT_DIR}"

