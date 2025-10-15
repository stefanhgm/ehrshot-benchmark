#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=pgpu
#SBATCH --gpus=2 
#SBATCH --mem=130000
#SBATCH --job-name=NVEmbed
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

# Run the program (replace "your_program" with the actual command)
echo "Running program for disease: $disease with phecode: $phecode"

CODE_DIR=~/Documents/ehrshot-benchmark/UKB_validation

cd $CODE_DIR

source ~/.bashrc

#conda activate /sc-projects/sc-proj-ukb-cvd/environments/nvembed
conda activate /sc-projects/sc-proj-ukb-cvd/environments/LLM2Vec

python3 ./LLM2Vec.py --includequeries "$queryinclusion" --indication "$disease" --phecode "$phecode" --model "$modelname" --minyears "$minyears" --maxyears "$maxyears" --ehr_format --batch_size 15 --include_dates "$dateinclusion" --withPCA "$num_PCA" --diseaseunspecific "$dataset" --save_embeddings --calculate_embeddings