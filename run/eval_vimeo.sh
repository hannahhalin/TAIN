#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu
#SBATCH -o ../outputs/eval_vimeo.out
cd PATH_TO_DIRECTORY
CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name TAIN \
    --dataset vimeo90k \
    --data_root PATH_TO_DATASET/vimeo_triplet \
    --mode test \
    --resume \
    --resume_exp TAIN
