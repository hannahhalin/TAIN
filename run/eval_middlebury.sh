#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu 
#SBATCH -o  ../outputs/eval_middlebury.out
cd PATH_TO_DIRECTORY
CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name TAIN \
    --dataset middlebury \
    --data_root PATH_TO_DATASET/middlebury/ \
    --mode test \
    --resume \
    --resume_exp TAIN
