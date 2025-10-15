#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --mem=200000
#SBATCH --job-name=QWEN
#SBATCH -N 1
#SBATCH -e /home/gear11/logs/job_%j.err
#SBATCH -o /home/gear11/logs/job_%j.out

# Retrieve disease and phecode from command line arguments
disease="$1"
phecode="$2"
minyears="$3"
maxyears="$4"
modelname="$5"
num_PCA="$6"
queryinclusion="$7"
dateinclusion="$8"
dataset="$9"
start="${10}"
end="${11}"

# Run the program (replace "your_program" with the actual command)
echo "Running program for disease: $disease with phecode: $phecode"

CODE_DIR=/home/gear11/Documents/ehrshot-benchmark/UKB_validation

cd $CODE_DIR

#source /home/gear11/.bashrc

export HF_HOME=/sc-projects/sc-proj-dh-ag-eils-ml/shared_hf_cache

#conda activate /sc-projects/sc-proj-ukb-cvd/environments/LLM2Vec
#conda activate /sc-projects/sc-proj-ukb-cvd/environments/Qwen

cmd="conda run -p /sc-projects/sc-proj-ukb-cvd/environments/LLM2Vec python3 ./LLM2Vec.py \
  --includequeries \"$queryinclusion\" \
  --indication \"$disease\" \
  --phecode \"$phecode\" \
  --model \"$modelname\" \
  --minyears \"$minyears\" \
  --maxyears \"$maxyears\" \
  --ehr_format True \
  --batch_size 5 \
  --include_dates \"$dateinclusion\" \
  --withPCA \"$num_PCA\" \
  --diseaseunspecific \"$dataset\" \
  --save_embeddings \
  --calculate_embeddings \
  --tokenlength 8192 \
  --start \"$start\" \
  --end \"$end\"" #\
  #--disable_wandb"
  #--clmbrcodes

echo "Running command: $cmd"
eval $cmd