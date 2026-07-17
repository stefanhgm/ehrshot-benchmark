#!/bin/bash
#SBATCH --job-name=11_tune_finetuning_params
#SBATCH --output=logs/11_tune_finetuning_params_%A_%a.out
#SBATCH --error=logs/11_tune_finetuning_params_%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Time to run: ~1-3 hrs per array task
#
# Hyperparameter search over the tuning grid. Run from ehrshot/bash_scripts/ so
# the ../.. relative paths (including the config's ../../EHRSHOT_ASSETS/... paths,
# which 11_tune_finetuning_params.py uses verbatim) resolve.
#
# The grid enumerates (27 encoder + 27 decoder configs) x 9 tasks x 2 k-values
# x 1 replicate = 972 jobs, driven by a plain sbatch array over --run-job-index.
#
# Three-phase flow:
#   1. Plan (writes the job manifest):
#        python3 ../11_tune_finetuning_params.py --config ../configs/tuning_grid.yaml --plan-only
#   2. Run the array (one GPU job per grid point):
#        sbatch --array=0-971 11_tune_finetuning_params.sh
#   3. Collect results + write best_params_{encoder,decoder}.json:
#        python3 ../11_tune_finetuning_params.py --config ../configs/tuning_grid.yaml --collect-only

python3 ../11_tune_finetuning_params.py \
    --config ../configs/tuning_grid.yaml \
    --run-job-index ${SLURM_ARRAY_TASK_ID}
