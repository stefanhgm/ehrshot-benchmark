#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=compute
#SBATCH --mem=300GB
#SBATCH --cpus-per-task=32
#SBATCH -e /home/gear11/logs/job_%j.err
#SBATCH -o /home/gear11/logs/job_%j.out

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

source ~/.bashrc

conda activate /sc-projects/sc-proj-dh-ag-eils-ml/shared_envs/EHRSHOT_ENV_Georg

python3 ./LLM2Vec.py --indication "$disease" --phecode "$phecode" --model "$modelname" --minyears "$minyears" --maxyears "$maxyears" --infer_all --tokenlength 8192 #--clmbrcodes True 
