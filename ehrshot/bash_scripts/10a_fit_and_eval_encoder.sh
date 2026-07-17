#!/bin/bash
#SBATCH --job-name=10a_fit_and_eval_encoder
#SBATCH --output=logs/10a_fit_and_eval_encoder_%A.out
#SBATCH --error=logs/10a_fit_and_eval_encoder_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: ~1-3 hrs per (task, k, replicate)
#
# Single encoder finetune + eval run. Run from ehrshot/bash_scripts/ so the
# ../.. relative paths resolve. Edit --sub_task / --k / --replicate to select a
# single run (these are the array-job knobs used by 11_tune_finetuning_params.py).

mkdir -p ../../EHRSHOT_ASSETS/experiments/llm_variants

python3 ../10a_fit_and_eval_encoder.py \
    --sub_task guo_los \
    --k 32 \
    --replicate 0 \
    --output_dir ../../EHRSHOT_ASSETS/experiments/llm_variants
