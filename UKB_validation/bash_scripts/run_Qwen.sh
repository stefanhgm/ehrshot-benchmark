#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=200000
#SBATCH --job-name=QWEN2
#SBATCH -e ./logs/job_%j.err
#SBATCH -o ./logs/job_%j.out

# Retrieve disease and phecode from command line arguments
disease="$1"
phecode="$2"
minyears="$3"
maxyears="$4"
modelname="$5"

# Run the program (replace "your_program" with the actual command)
echo "Running program for disease: $disease with phecode: $phecode"

CODE_DIR=~/Documents/LLM2Vec_project

cd $CODE_DIR

export HF_HOME=/sc-projects/sc-proj-dh-ag-eils-ml/shared_hf_cache

cmd="conda run -p LLM2Vec python3 ./LLM2Vec.py \
  --indication \"$disease\" \
  --phecode \"$phecode\" \
  --model \"$modelname\" \
  --minyears \"$minyears\" \
  --maxyears \"$maxyears\" \
  --batch_size 5 \
  --calculate_embeddings \
  --tokenlength 8192" #\
  #--keep_all_codes" #\
  #--disable_wandb"
  #--clmbrcodes

echo "Running command: $cmd"
eval $cmd