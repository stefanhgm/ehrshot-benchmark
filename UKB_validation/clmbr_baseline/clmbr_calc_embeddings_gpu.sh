#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --mem=280000
#SBATCH --gpus=1
#SBATCH --job-name=clmbr-baseline
#SBATCH -N 1
#SBATCH -e ./logs/job_%j.err
#SBATCH -o ./logs/job_%j.out


CODE_DIR=~/Documents/ehrshot-benchmark/UKB_validation/clmbr_baseline

cd $CODE_DIR

source ~/.bashrc

conda activate /sc-projects/sc-proj-ukb-cvd/environments/LLM

python3 ./clmbr_baseline_create_embeddings.py 