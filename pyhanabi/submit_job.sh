#!/bin/bash
#SBATCH --partition=csc494-proj
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G 
#SBATCH --time=4:00:00
#SBATCH --output=logs/r3d2_%j.out


FREQ=$1
SEED=$2
LM_WEIGHT=$3
PLAYER=$4
ADDLAYER=$5
WANDB="1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate r3d3_hanabi

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python r3d2_main.py --config configs/r3d2.yaml --update_freq_text_enc $FREQ --seed $SEED --lm_weights $LM_WEIGHT --num_player $PLAYER --wandb $WANDB --num_of_additional_layer $ADDLAYER