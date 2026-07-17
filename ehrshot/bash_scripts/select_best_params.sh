#!/bin/bash
#SBATCH --job-name=select_best_params
#SBATCH --output=logs/select_best_params_%A.out
#SBATCH --error=logs/select_best_params_%A.err
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Time to run: < 1 min
#
# Pick the best hyperparameter config per model type from the tuning sweep and
# write ../configs/best_params_encoder.json and ../configs/best_params_decoder.json.
# Run from ehrshot/bash_scripts/ so the ../.. relative paths resolve.
#
# --input_csv is the aggregated tuning results CSV produced by the step-11
# --collect-only phase.

python3 ../select_best_params.py \
    --input_csv ../../EHRSHOT_ASSETS/experiments/tuning/tuning_results_raw.csv \
    --output_dir ../configs \
    --config ../configs/tuning_grid.yaml
