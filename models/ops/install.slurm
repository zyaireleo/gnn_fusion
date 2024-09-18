#!/bin/bash
#SBATCH --gres=gpu:TeslaV100-SXM2-32GB:1
source activate referformer
#module load cuda/12.1
module load compiler/gcc/7.3.1
module load cuda/11.7
nvidia-smi
nvcc
#module list
python -c "import torch; print('CUDA is available.' if torch.cuda.is_available() else 'CUDA is not available.')"
#python -c "import torch; print(torch.__version__)"

#conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

python setup.py build install
