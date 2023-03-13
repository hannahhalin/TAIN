#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -p compsci-gpu 
#SBATCH -o ../outputs/eval_ucf.out
cd PATH_TO_DIRECTORY
CUDA_VISIBLE_DEVICES=0 python main.py \
    --exp_name TAIN \
    --dataset ucf \
    --data_root PATH_TO_DATASET/UCF101_results/ucf101_interp_ours/ \
    --mode test \
    --resume \
    --resume_exp TAIN
