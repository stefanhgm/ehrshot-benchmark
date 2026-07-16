#!/bin/bash
#SBATCH --job-name=10b_fit_and_eval_decoder
#SBATCH --output=logs/10b_fit_and_eval_decoder_%A.out
#SBATCH --error=logs/10b_fit_and_eval_decoder_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: ~1-3 hrs per (task, k, replicate)
#
# Single decoder finetune + eval run. Run from ehrshot/bash_scripts/ so the
# ../.. relative paths resolve. Edit --sub_task / --k / --replicate to select a
# single run (these are the array-job knobs used by 11_tune_finetuning_params.py).

mkdir -p ../../EHRSHOT_ASSETS/experiments/llm_variants

python3 ../10b_fit_and_eval_decoder.py \
    --sub_task guo_los \
    --k 32 \
    --replicate 0 \
    --output_dir ../../EHRSHOT_ASSETS/experiments/llm_variants

# In-context learning (ICL) evaluation instead of finetuning:
# --icl_shots selects the number of in-context examples ({0, 2, 4, 6}).
# For long-context ICL the prompt reaches ~128k tokens; the Qwen3 SDPA path then
# materializes a [B, 1, L, L] attention mask (~214 GiB at B=8), so --batch_size 1
# is REQUIRED here.
#
# python3 ../10b_fit_and_eval_decoder.py \
#     --sub_task guo_los \
#     --k 32 \
#     --replicate 0 \
#     --icl_shots 0 \
#     --batch_size 1 \
#     --max_input_length 131072 \
#     --output_dir ../../EHRSHOT_ASSETS/experiments/llm_variants
#
# python3 ../10b_fit_and_eval_decoder.py \
#     --sub_task guo_los --k 32 --replicate 0 \
#     --icl_shots 2 --batch_size 1 --max_input_length 131072 \
#     --output_dir ../../EHRSHOT_ASSETS/experiments/llm_variants
#
# python3 ../10b_fit_and_eval_decoder.py \
#     --sub_task guo_los --k 32 --replicate 0 \
#     --icl_shots 4 --batch_size 1 --max_input_length 131072 \
#     --output_dir ../../EHRSHOT_ASSETS/experiments/llm_variants
#
# python3 ../10b_fit_and_eval_decoder.py \
#     --sub_task guo_los --k 32 --replicate 0 \
#     --icl_shots 6 --batch_size 1 --max_input_length 131072 \
#     --output_dir ../../EHRSHOT_ASSETS/experiments/llm_variants
