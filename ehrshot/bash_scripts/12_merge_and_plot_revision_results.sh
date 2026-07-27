#!/bin/bash
#SBATCH --job-name=12_merge_and_plot_revision_results
#SBATCH --output=logs/12_merge_and_plot_revision_results_%A.out
#SBATCH --error=logs/12_merge_and_plot_revision_results_%A.err
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Time to run: ~10 mins
#
# Merge the per-run result CSVs, join the tuning summary, and produce the
# revision comparison figures/tables. Run from ehrshot/bash_scripts/ so the
# ../.. relative paths resolve. All five arguments below are required.

mkdir -p ../../EHRSHOT_ASSETS/figures

python3 ../12_merge_and_plot_revision_results.py \
    --results_dir ../../EHRSHOT_ASSETS/experiments/llm_variants \
    --extra_results_dir ../../EHRSHOT_ASSETS/experiments/tuning/run_outputs \
    --output_file ../../EHRSHOT_ASSETS/figures/merged_results.csv \
    --tuning_results_csv ../../EHRSHOT_ASSETS/experiments/tuning/tuning_results_raw.csv \
    --baseline_dir ../../EHRSHOT_ASSETS/results
