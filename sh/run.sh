#! /bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python clipdraw_DCgan.py --output-path './results/dcgan_full4/' --num-epochs 50 --d-lr 0.000001 --g-lr 0.00002