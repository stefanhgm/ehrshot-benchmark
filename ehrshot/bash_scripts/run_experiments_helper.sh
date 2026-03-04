#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --output=logs/ehrshot_%A.log
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=40
# --qos=long_job 

# 8-GPU setups (pgpu partition) 
# Available node types:
#   - DGX A100 80GB (128 cpus, 2TB):  s-sc-dgx01, s-sc-dgx02
#   - H100 80GB (96 cpus, 2TB):       s-sc-pgpu08
#   - H200 141GB (96 cpus, 2TB):      s-sc-pgpu11–15
# 4-GPU setups (pgpu partition)
# Available node types:
#   - A100 SXM4 40GB (128 cpus, 493GB):  s-sc-pgpu01, s-sc-pgpu02
#   - A100 SXM4 80GB (128 cpus, 493GB):  s-sc-pgpu03, s-sc-pgpu04, s-sc-pgpu05, s-sc-pgpu06, s-sc-pgpu07

# Any 8 GPUs node
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1950G
#SBATCH --exclusive

# Only H200
##SBATCH --partition=pgpu
##SBATCH --gres=gpu:8
##SBATCH --cpus-per-task=96
##SBATCH --mem=1950G
##SBATCH --exclusive
##SBATCH --exclude=s-sc-dgx01,s-sc-dgx02,s-sc-pgpu08

# Only DGX A100
##SBATCH --partition=pgpu
##SBATCH --gres=gpu:8
##SBATCH --cpus-per-task=96
##SBATCH --mem=1950G
##SBATCH --exclusive
##SBATCH --exclude=s-sc-pgpu08,s-sc-pgpu11,s-sc-pgpu12,s-sc-pgpu13,s-sc-pgpu14,s-sc-pgpu15

# Any 4 GPUs node with 80GB GPU
##SBATCH --partition=pgpu
##SBATCH --gres=gpu:4
##SBATCH --cpus-per-task=128
##SBATCH --mem=480G
##SBATCH --exclusive
##SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02,s-sc-pgpu03,s-sc-pgpu04

# Single GPU resources: 
##SBATCH --partition=gpu
##SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
##SBATCH --mem=480GB

# CPU
##SBATCH --partition=compute

# Disable Python output buffering
export PYTHONUNBUFFERED=1

# Check if instruction file at last position, then set instructions_file_arg
if [ -z "${12}" ]
then
    instructions_file_arg=""
else
    instructions_file_arg="--task_to_instructions ${12}"
fi

# TODO: For multi-gpu setup
# CUDA_VISIBLE_DEVICES=0 
python /home/sthe14/ehrshot-benchmark/ehrshot/run_experiments.py \
    --base_dir $1 \
    --experiment_folder $2 \
    --path_to_database $3 \
    --path_to_labels_dir $4 \
    --path_to_split_csv $5 \
    --num_threads $6 \
    --text_encoder $7 \
    --serialization_strategy $8 \
    --excluded_ontologies $9 \
    --num_aggregated ${10} \
    --time_window_days ${11} \
    $instructions_file_arg
