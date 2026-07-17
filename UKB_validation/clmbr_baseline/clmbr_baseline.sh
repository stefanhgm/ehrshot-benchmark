#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH --partition=compute
#SBATCH --mem=350000
#SBATCH --job-name=clmbr_baseline
#SBATCH -N 1
#SBATCH -e ./logs/job_%j.err
#SBATCH -o ./logs/job_%j.out


# Load configuration from the repo-root .env file (also inherited via `sbatch --export=ALL`)
SCRIPT_DIR_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
[ -f "$SCRIPT_DIR_SELF/../../.env" ] && source "$SCRIPT_DIR_SELF/../../.env"
set +a

CODE_DIR="${EHRSHOT_BENCHMARK_DIR%/}/UKB_validation/clmbr_baseline"

cd $CODE_DIR


conda activate LLM2Vec

python3 ./clmbr_baseline.py 