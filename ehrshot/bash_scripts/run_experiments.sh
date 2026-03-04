# Options
num_threads=32
EHRSHOT_ENV="/sc-projects/sc-proj-dh-ag-eils-ml/shared_envs/EHRSHOT_ENV_QWEN3" # Set to EHRSHOT_ENV_QWEN3 for Qwen3, EHRSHOT_ENV for Llama3 and GteQwen2
# Make conda activate work in non-interactive shells
source /opt/miniforge/etc/profile.d/conda.sh
conda activate "$EHRSHOT_ENV"

# avoid ~/.local site-packages interfering
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Fix libnotify issue
# export LD_PRELOAD="$CONDA_PREFIX/lib/libittnotify.so"

# Debug outputs
set -euo pipefail
echo "=== DEBUG ENV ==="
echo "HOST=$(hostname)"
echo "USER=$USER"
echo "SHELL=$SHELL"
echo "PWD=$(pwd)"
echo "CONDA_EXE=${CONDA_EXE:-<unset>}"
echo "CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"
echo "which python: $(which python || true)"
python -V || true
python -c "import sys; print('sys.executable:', sys.executable); print('sys.path:', sys.path[:5])" || true
python -c "import wandb; print('wandb:', wandb.__version__)" || true
echo "==============="

# Paths
EXPERIMENT_IDENTIFIER="full_run_codes_list"
BASE_DIR="/home/sthe14/ehrshot-benchmark"
SCRIPT_DIR="$BASE_DIR/ehrshot"
INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions_list.json"

# INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions_list_w_time.json"
# INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions_neutral_list.json"
# Markdown
# INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions.json"
# INSTRUCTIONS_FILE="${BASE_DIR}/ehrshot/serialization/task_to_instructions_neutral.json"
# NOTE: Set to experiment path --> only LLM
# NOTE: Set to final_exp path --> LR and CLIMBR as well
EXPERIMENTS_DIR="$BASE_DIR/EHRSHOT_ASSETS/experiments/$EXPERIMENT_IDENTIFIER"
mkdir -p $EXPERIMENTS_DIR


# Define the different options to iterate over
text_encoders=(
    "qwen3_embedding_8b"
    
    # "qwen3_embedding_4b"
    # "qwen3_embedding_0_6b"
    # "gteqwen2_7b_instruct"
    # "gteqwen2_7b_instruct_chunked_2k"
    # "gteqwen2_7b_instruct_chunked_1k"
    # "gteqwen2_7b_instruct_chunked_512"
    # "gteqwen2_1_5b_instruct"    
    # "llm2vec_llama3_1_7b_instruct_supervised"
    # "llm2vec_llama3_1_7b_instruct_supervised_chunked_2k"
    # "llm2vec_llama3_1_7b_instruct_supervised_chunked_1k"
    # "llm2vec_llama3_1_7b_instruct_supervised_chunked_512"
    # "llm2vec_llama2_sheared_1_3b_supervised"
    # "bioclinicalbert"
    # "medbert"
    # "deberta_v3_base"
    # "deberta_v3_large"
    # "bert_base"
    # "bert_large"
    # "bioclinicalbert-concat"
    # "medbert-concat"
    # "deberta_v3_base-concat"
    # "deberta_v3_large-concat"
    # "bert_base-concat"
    # "bert_large-concat"
)

serialization_strategies=(
    "unique_codes_list_recent_8k"

#     "unique_codes_list_recent_4k"
#     "unique_codes_list_recent_2k"
#     "unique_codes_list_recent_1k"
#     "unique_codes_list_recent_512"
#     "unique_codes_list_w_time_8k"
#     "unique_codes_list_8k"
#     "unique_codes_list_recent_w_time_8k"

#     "unique_then_list_visits_wo_allconds_w_values"
#     "unique_then_list_visits_wo_allconds_w_values_8k_json"
#     "unique_then_list_visits_wo_allconds_w_values_8k_xml"
#     "unique_then_list_visits_wo_allconds_w_values_8k_yaml"

#     "unique_codes_list_recent_8k_neutral"
#     # (custom run for no prompt)
#
#     "unique_codes_list_recent_8k_only_demographics"
#     "unique_codes_list_recent_8k_only_visits"
#     "unique_codes_list_recent_8k_only_conditions"
#     "unique_codes_list_recent_8k_only_medications"
#     "unique_codes_list_recent_8k_only_procedures"
#     "unique_codes_list_recent_8k_only_labs"
# 
#     "unique_codes_list_recent_8k_no_demographics"
#     "unique_codes_list_recent_8k_no_visits"
#     "unique_codes_list_recent_8k_no_conditions"
#     "unique_codes_list_recent_8k_no_medications"
#     "unique_codes_list_recent_8k_no_procedures"
#     "unique_codes_list_recent_8k_no_labs"
)

# Fixed options
instructions_options=("true")
excluded_ontologies=("no_unres")
num_aggregated=(3)
time_window_days=(0)
# time_window_days=(-1 1 7 30 365 1095)

# Labels = Dataset subset
DATASET="full"

if [ $DATASET == "full" ]; then
    LABELS_DIR=$BASE_DIR/EHRSHOT_ASSETS/benchmark
elif [ $DATASET == "new_guo_chexpert" ]; then
    LABELS_DIR=$BASE_DIR/EHRSHOT_ASSETS/benchmark_subsets/new_guo_chexpert # new_guo_chexpert
elif [ $DATASET == "new_guo" ]; then
    LABELS_DIR=$BASE_DIR/EHRSHOT_ASSETS/benchmark_subsets/new_guo # new_guo
fi

for text_encoder in "${text_encoders[@]}"; do
    for serialization_strategy in "${serialization_strategies[@]}"; do
        for excluded_ontology in "${excluded_ontologies[@]}"; do
            for num_aggregated_val in "${num_aggregated[@]}"; do
                for time_window_days_val in "${time_window_days[@]}"; do
                    for use_instructions in "${instructions_options[@]}"; do
                        # Define experiment name based on concatenation of options and create timestamped directory
                        instructions_suffix="_no_instr"
                        instructions_file_arg=""
                        if [ $use_instructions == "true" ]; then
                            instructions_suffix=""
                            instructions_file_arg="$INSTRUCTIONS_FILE"
                        fi

                        experiment_name="${text_encoder}_${serialization_strategy}_${excluded_ontology}_${num_aggregated_val}_${time_window_days_val}_${DATASET}${instructions_suffix}"
                        experiment_dir="${EXPERIMENTS_DIR}/${experiment_name}"
                        mkdir -p $experiment_dir

                        # Check if the experiment has already been run by testing if all_results.csv exists in the experiment directory
                        if [ -f "${experiment_dir}/all_results.csv" ]; then
                            echo "Experiment $experiment_name already exists. Skipping..."
                            continue
                        fi

                        # Run the experiment with bash or slurm
                        cmd="bash"
                        [[ " $* " == *" --is_use_slurm "* ]] && cmd="sbatch"

                        $cmd /home/sthe14/ehrshot-benchmark/ehrshot/bash_scripts/run_experiments_helper.sh \
                            $BASE_DIR \
                            $experiment_dir \
                            $BASE_DIR/EHRSHOT_ASSETS/femr/extract \
                            $LABELS_DIR \
                            $BASE_DIR/EHRSHOT_ASSETS/splits/person_id_map.csv \
                            $num_threads \
                            $text_encoder \
                            $serialization_strategy \
                            $excluded_ontology \
                            $num_aggregated_val \
                            $time_window_days_val \
                            $instructions_file_arg
                    done
                done
            done
        done
    done
done