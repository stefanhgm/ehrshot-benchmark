#!/bin/bash
#SBATCH --job-name=5_generate_clmbr_features
#SBATCH --output=logs/5_generate_clmbr_features_%A.out
#SBATCH --error=logs/5_generate_clmbr_features_%A.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=20

# Time to run: 20 mins

# Based on EHRSHOT_ENV_ORIG enviroment cloned into EHRSHOT_ENV_CLMBR:
# Fix for JAX CUDA/cuDNN error:
# - jaxlib here expects CUDA 11 + cuDNN 8
# - install missing cuDNN runtime in this env:
#     pip install nvidia-cudnn-cu11==8.6.0.163
# - ensure libs are visible at runtime:
#     export CUDA_HOME=/usr/local/cuda-11.8
#     export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:\
#$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cublas/lib:\
#$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:\
#/usr/local/cuda-11.8/targets/x86_64-linux/lib
#
# Otherwise JAX fails with:
#   "CUDNN_STATUS_INTERNAL_ERROR" / "DNN library initialization failed"


python3 ../5_generate_clmbr_features.py \
    --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    --path_to_labels_dir ../../EHRSHOT_ASSETS/benchmark \
    --path_to_features_dir ../../EHRSHOT_ASSETS/features \
    --path_to_models_dir ../../EHRSHOT_ASSETS/models \
    --model clmbr  \
    --is_force_refresh

# Time to run: XXXX mins

# python3 ../5_generate_clmbr_features.py \
#     --path_to_database ../../EHRSHOT_ASSETS/femr/extract \
    # --path_to_labels_dir ../../EHRSHOT_ASSETS/benchmark \
    # --path_to_features_dir ../../EHRSHOT_ASSETS/features \
#     --path_to_models_dir ../../EHRSHOT_ASSETS/models \
#     --model motor \
#     --is_force_refresh