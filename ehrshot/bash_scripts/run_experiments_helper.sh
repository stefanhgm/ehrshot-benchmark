#!/bin/bash
#SBATCH --job-name=ehrshot
#SBATCH --output=logs/ehrshot_%A.log
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=40

# Any 8 GPUs node
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1950G
#SBATCH --exclusive

# Single GPU resources: 
##SBATCH --partition=gpu
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
python "$1/ehrshot/run_experiments.py" \
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
