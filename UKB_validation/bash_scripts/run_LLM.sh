#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=200000
#SBATCH --job-name=LLM2Vec
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

# Load configuration from the repo-root .env file (also inherited via `sbatch --export=ALL`)
SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[ -f "$SCRIPT_DIR_SELF/../../.env" ] && source "$SCRIPT_DIR_SELF/../../.env"
set +a

CODE_DIR="${EHRSHOT_BENCHMARK_DIR%/}/UKB_validation"

cd $CODE_DIR

conda activate LLM2Vec

python3 ./LLM2Vec.py --indication "$disease" --phecode "$phecode" --model "$modelname" --minyears "$minyears" --maxyears "$maxyears" --batch_size 5 --calculate_embeddings --tokenlength 8192 
