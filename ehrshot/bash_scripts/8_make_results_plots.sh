#!/bin/bash
#SBATCH --job-name=8_make_figures
#SBATCH --output=logs/8_make_figures_%A.out
#SBATCH --error=logs/8_make_figures_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

mkdir -p ../../EHRSHOT_ASSETS/figures

# --model_heads "[('clmbr', 'lr_lbfgs'), ('llm', 'lr_lbfgs'), ('agr', 'lr_lbfgs'), ('count', 'lr_lbfgs'), ('count', 'gbm'), ('count', 'rf')]" \
# --path_to_results_dir ../../EHRSHOT_ASSETS/results \

python3 ../8_make_results_plots.py \
    --path_to_labels_and_feats_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_results_dir ../../EHRSHOT_ASSETS/experiments/full_run_codes_list/qwen3_embedding_8b_unique_codes_list_recent_8k_no_unres_3_0_full_with_baselines \
    --path_to_output_dir ../../EHRSHOT_ASSETS/figures \
    --model_heads "[('llm', 'lr_lbfgs'), ('count', 'gbm'), ('clmbr', 'lr_lbfgs')]" \
    --shot_strat all