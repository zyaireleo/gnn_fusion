#!/usr/bin/env bash
module load cuda/12.1
module load gcc
module list
source activate foundmental
nvidia-smi

# Default values
GPUS=4
OUTPUT_DIR='/HOME/scz0rv9/run/codes/ReferFormer/output/ref-davis'
CHECKPOINT='/HOME/scz0rv9/run/codes/ReferFormer/output/ytvos/checkpoint.pth'

# Parse command line options
TEMP=$(getopt -o g:o:c: --long gpus:,output_dir:,checkpoint: -n 'script.sh' -- "$@")
eval set -- "$TEMP"

# extract options and their arguments into variables.
while true ; do
    case "$1" in
        -g|--gpus)
            GPUS="$2"; shift 2 ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -c|--checkpoint)
            CHECKPOINT="$2"; shift 2 ;;
        --) shift ; break ;;
        *) echo "Internal error!" ; exit 1 ;;
    esac
done

# Other parameters (after options)
PY_ARGS="$@"

echo "Load model weights from: ${CHECKPOINT}"

# test using the model trained on ref-youtube-vos directly
python3 inference_davis.py --with_box_refine --binary --freeze_text_encoder \
--ngpu=${GPUS} --output_dir="${OUTPUT_DIR}" --resume="${CHECKPOINT}" ${PY_ARGS}

# evaluation
ANNO0_DIR="${OUTPUT_DIR}/valid/anno_0"
ANNO1_DIR="${OUTPUT_DIR}/valid/anno_1"
ANNO2_DIR="${OUTPUT_DIR}/valid/anno_2"
ANNO3_DIR="${OUTPUT_DIR}/valid/anno_3"
python3 eval_davis.py --results_path="${ANNO0_DIR}"
python3 eval_davis.py --results_path="${ANNO1_DIR}"
python3 eval_davis.py --results_path="${ANNO2_DIR}"
python3 eval_davis.py --results_path="${ANNO3_DIR}"

echo "Working path is: ${OUTPUT_DIR}"
