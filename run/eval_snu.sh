#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu
#SBATCH -o ../outputs/eval_snu_extreme.out
cd PATH_TO_DIRECTORY
CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name TAIN \
    --dataset snufilm \
    --data_root PATH_TO_DATASET/SNU-FILM \
    --mode test \
    --resume \
    --resume_exp TAIN \
    --test_mode extreme
