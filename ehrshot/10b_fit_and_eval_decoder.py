#!/usr/bin/env python3
"""
Run LLM variants experiments with command line arguments support
"""

import argparse
import sys
import pickle
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Protocol, List, Dict, Any, Optional
import os
import collections
import torch
import tqdm
import math
import random
import logging
from dataclasses import dataclass
import re

import sklearn
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from scipy.sparse import issparse
import scipy
import lightgbm as lgb
import femr
import femr.datasets
from femr.labelers import load_labeled_patients, LabeledPatients
from loguru import logger

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset

from serialization.text_encoder import (
    TextEncoder,
    Qwen3Embedding_8B_Encoder,
    Qwen3Embedding_4B_Encoder,
    Qwen3Embedding_0_6B_Encoder,
)

from utils import (
    LABELING_FUNCTION_2_PAPER_NAME,
    SHOT_STRATS,
    MODEL_2_INFO,
    get_labels_and_features,
    process_chexpert_labels,
    convert_multiclass_to_binary_labels,
    CHEXPERT_LABELS,
    LR_PARAMS,
    XGB_PARAMS,
    RF_PARAMS,
    ProtoNetCLMBRClassifier,
    get_patient_splits_by_idx,
)

# %%
def tune_hyperparams(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    model,
    param_grid: Dict[str, List],
    n_jobs: int = 1,
):
    """Hyperparameter tuning with explicit train/val split using PredefinedSplit"""
    X: np.ndarray = (
        scipy.sparse.vstack([X_train, X_val])
        if issparse(X_train)
        else np.concatenate((X_train, X_val), axis=0)
    )
    y: np.ndarray = np.concatenate((y_train, y_val), axis=0)
    
    test_fold: np.ndarray = -np.ones(X.shape[0])
    test_fold[X_train.shape[0] :] = 0

    clf = GridSearchCV(
        model,
        param_grid,
        scoring="roc_auc",
        n_jobs=n_jobs,
        verbose=0,
        cv=PredefinedSplit(test_fold),
        refit=False,
    )
    clf.fit(X, y)
    best_model = model.__class__(**clf.best_params_)
    best_model.fit(X_train, y_train)
    return best_model

def run_evaluation_lr(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    model_head: str,
    n_jobs: int = 1,
    test_patient_ids: np.ndarray = None,
) -> Tuple[Any, Dict[str, float]]:
    logger.critical(f"Start | Training {model_head}")
    logger.info(f"Train shape: X = {X_train.shape}, Y = {y_train.shape}")
    logger.info(f"Val shape: X = {X_val.shape}, Y = {y_val.shape}")
    logger.info(f"Test shape: X = {X_test.shape}, Y = {y_test.shape}")
    logger.info(f"Train prevalence:  {np.mean(y_train)}")
    logger.info(f"Val prevalence:  {np.mean(y_val)}")
    logger.info(f"Test prevalence:  {np.mean(y_test)}")
    logger.info(
        f"Test pids:  {len(test_patient_ids)} | {len(y_test)} | {len(set(test_patient_ids))}"
    )

    np.random.seed(X_train.shape[0])
    train_shuffle_idx = np.arange(X_train.shape[0])
    np.random.shuffle(train_shuffle_idx)
    X_train = X_train[train_shuffle_idx]
    y_train = y_train[train_shuffle_idx]

    logger.critical(f"Start | Fitting {model_head}...")
    model_head_parts: List[str] = model_head.split("_")
    model_head_base: str = model_head_parts[0]
    
    if model_head_base == "gbm":
        model = lgb.LGBMClassifier(random_state=0)
        XGB_PARAMS["min_child_samples"] = [1]
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, XGB_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "rf":
        RF_PARAMS["min_samples_leaf"] = [1]
        RF_PARAMS["min_samples_split"] = [2]
        model = RandomForestClassifier(random_state=0)
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, RF_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "lr":
        solver: str = model_head_parts[1]
        scaler = MaxAbsScaler().fit(X_train)
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        model = LogisticRegression(
            n_jobs=1, penalty="l2", tol=0.0001, solver=solver, max_iter=1000, random_state=0
        )
        model = tune_hyperparams(X_train, X_val, y_train, y_val, model, LR_PARAMS, n_jobs=n_jobs)
        logger.info(f"Best hparams: {model.get_params()}")
    elif model_head_base == "protonet":
        model = ProtoNetCLMBRClassifier()
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Model head `{model_head}` not supported.")
    
    logger.critical(f"Finish | Fitting {model_head}...")

    y_train_proba = model.predict_proba(X_train)[::, 1]
    y_val_proba = model.predict_proba(X_val)[::, 1]
    y_test_proba = model.predict_proba(X_test)[::, 1]

    metric_dict = {
        "auroc": metrics.roc_auc_score,
        "brier": metrics.brier_score_loss,
        "auprc": metrics.average_precision_score,
    }

    scores = {}
    for metric, func in metric_dict.items():
        scores[metric] = {}
        train_score = func(y_train, y_train_proba)
        val_score = func(y_val, y_val_proba)
        test_score = func(y_test, y_test_proba)

        logger.info(f"Train {metric} score: {train_score}")
        logger.info(f"Val {metric} score:   {val_score}")
        logger.info(f"Test {metric} score:  {test_score}")

        test_set = sorted(list(set(test_patient_ids)))
        score_list = []
        
        for i in range(1000):
            sample = sklearn.utils.resample(test_set, random_state=i)
            counts = collections.Counter(sample)
            weights = np.zeros_like(test_patient_ids)
            for i, p in enumerate(test_patient_ids):
                weights[i] = counts[p]
            score_val = func(y_test, y_test_proba, sample_weight=weights)
            score_list.append(score_val)

        lower, upper = np.percentile(score_list, [2.5, 97.5])
        std = np.std(score_list, ddof=1)

        scores[metric]["score"] = test_score
        scores[metric]["std"] = std
        scores[metric]["lower"] = lower
        scores[metric]["mean"] = np.mean(score_list)
        scores[metric]["upper"] = upper

    return model, scores

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run LLM decoder experiments for medical prediction tasks")
    
    # Paths
    parser.add_argument("--output_dir", type=str, 
                        default="/sc-projects/sc-proj-ukb-cvd/projects/ehrshot-benchmark/EHRSHOT_ASSETS/experiments/llm_variants",
                        help="Output directory for results (filename will be auto-generated)")
    parser.add_argument("--splits_path", type=str,
                        default="/sc-projects/sc-proj-ukb-cvd/projects/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark/ehrshot_splits_to_serializations.csv",
                        help="Path to splits to serializations CSV file")
    parser.add_argument("--serializations_path", type=str,
                        default="/sc-projects/sc-proj-ukb-cvd/projects/ehrshot-benchmark/EHRSHOT_ASSETS/benchmark/tasks_serializations.pkl",
                        help="Path to tasks serializations pickle file")
    
    # Task selection (for array jobs - specify single task, k, replicate)
    parser.add_argument("--sub_task", type=str, default=None,
                        help="Single task to run (for array jobs)")
    parser.add_argument("--k", type=int, default=None,
                        help="Single k value to run (for array jobs)")
    parser.add_argument("--replicate", type=int, default=None,
                        help="Single replicate to run (for array jobs)")
    
    # Task lists (for running multiple)
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="List of tasks to run (overrides default list)")
    parser.add_argument("--ks", type=int, nargs="*", default=[128],
                        help="List of k values to run")
    parser.add_argument("--replicates", type=int, nargs="*", default=[0],
                        help="List of replicates to run")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name (Qwen/Qwen3-0.6B, Qwen/Qwen3-4B, Qwen/Qwen3-8B)")
    parser.add_argument("--max_input_length", type=int, default=4096,
                        help="Maximum input length in tokens")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training and inference")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    
    # Training parameters
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="Warmup ratio")
    parser.add_argument("--num_train_epochs_cap", type=int, default=20,
                        help="Maximum number of training epochs")
    parser.add_argument("--effective_batch_size", type=int, default=None,
                        help="Effective batch size (if None, uses min(k, 8))")
    
    # Other parameters
    parser.add_argument("--num_threads", type=int, default=40,
                        help="Number of threads for parallel processing")
    parser.add_argument("--labeling_function", type=str, default="llm_decoder_ft",
                        help="Name for the labeling function")
    parser.add_argument("--show_progress", action="store_true", default=True,
                        help="Show progress bars and detailed output")
    parser.add_argument("--quiet", action="store_true", default=False,
                        help="Suppress progress output")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="Overwrite existing output files instead of skipping")
    parser.add_argument("--eval_train_val", action="store_true", default=False,
                        help="Also calculate and log scores for train and validation sets during evaluation")
    parser.add_argument("--val_limit", type=int, default=-1,
                        help="Maximum number of validation examples to use; -1 keeps the full set")
    parser.add_argument("--test_limit", type=int, default=-1,
                        help="Maximum number of test examples to use; -1 keeps the full set")
    parser.add_argument("--subset_seed", type=int, default=42,
                        help="Random seed used when subsetting validation/test splits")

    return parser.parse_args()

# %%
def _maybe_limit_split(
    split_df: pd.DataFrame,
    limit: int,
    rng: np.random.Generator,
    split_name: str,
    seed: int,
) -> pd.DataFrame:
    """Optionally down-sample a split to the requested limit with reproducibility."""
    if limit is None or limit < 0:
        return split_df

    available = len(split_df)
    if available == 0:
        return split_df

    if limit == 0:
        logger.info(f"{split_name.title()} limit set to 0; returning empty split")
        return split_df.iloc[0:0].copy()

    if limit >= available:
        logger.info(f"{split_name.title()} limit {limit} >= available {available}; using full split")
        return split_df

    indices = rng.choice(available, size=limit, replace=False)
    limited_df = split_df.iloc[np.sort(indices)].copy()
    logger.info(
        f"Applying {split_name} limit: selected {limit} of {available} rows (seed={seed})"
    )
    return limited_df

# %%
# Default task list
DEFAULT_TASKS = [
    "guo_los",
    "guo_readmission",
    "guo_icu",
    "lab_thrombocytopenia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_anemia",
    "new_hypertension",
    "new_hyperlipidemia",
    "new_pancan",
    "new_celiac",
    "new_lupus",
    "new_acutemi",
    "chexpert_Lung Lesion",
    "chexpert_Pneumothorax",
    "chexpert_Fracture",
    "chexpert_Consolidation",
    "chexpert_Cardiomegaly",
    "chexpert_Enlarged Cardiomediastinum",
    "chexpert_Edema",
    "chexpert_Pneumonia",
    "chexpert_Pleural Other",
    "chexpert_Lung Opacity",
    "chexpert_Atelectasis",
    "chexpert_Pleural Effusion",
    "chexpert_No Finding",
    "chexpert_Support Devices",
]

def main():
    """Main experiment function"""
    args = parse_args()
    
    # Validate required arguments
    if args.sub_task is None or args.k is None or args.replicate is None:
        print("Error: --sub_task, --k, and --replicate are required arguments")
        print("Example: python fit_and_eval_decoder.py --sub_task guo_los --k 128 --replicate 0 --model_name Qwen/Qwen3-0.6B")
        sys.exit(1)
    
    # Configure progress display
    show_progress = args.show_progress and not args.quiet
    
    print(f"Running single experiment: task={args.sub_task}, k={args.k}, replicate={args.replicate}, model={args.model_name}")

    # Generate output filename
    model_safe = re.sub(r'[^\w\-_\.]', '_', args.model_name.replace('/', '_'))
    output_filename = f"results_{model_safe}_{args.sub_task}_k{args.k}_r{args.replicate}.csv"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Check if output file already exists
    if os.path.exists(output_path) and not args.overwrite:
        print(f"Output file already exists: {output_path}")
        print("Skipping experiment to avoid overwriting results.")
        print("Use --overwrite flag to overwrite existing results.")
        sys.exit(0)
    elif os.path.exists(output_path) and args.overwrite:
        print(f"Output file already exists: {output_path}")
        print("Overwriting existing results due to --overwrite flag.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    splits_to_serializations, tasks_serializations = load_data(args)
    
    # Enable optimizations
    logger.info("Enabling PyTorch optimizations...")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Run single experiment
    logger.info("Starting single experiment...")
    results = run_single_experiment(args, splits_to_serializations, tasks_serializations, show_progress)
    
    # Save results
    print(f"Saving results to: {output_path}")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print("Done!")

# %%
def load_data(args):
    """Load data splits and serializations"""
    logger.info(f"Loading splits data from: {args.splits_path}")
    dtype_dict = {
        "task": str,
        "split_name": str,
        "shot_size": int,
        "fold": int,
        "patient_id": int,
        "prediction_time": str,
        "label_type": str,
        "label_value": str,
        "serialization_idx": int,
    }

    splits_to_serializations = pd.read_csv(args.splits_path, dtype=dtype_dict, parse_dates=["prediction_time"])
    splits_to_serializations["label_value"] = splits_to_serializations["label_value"].apply(
        lambda x: x == "True"
    )
    logger.info(f"Loaded {len(splits_to_serializations)} split records")

    logger.info(f"Loading serializations data from: {args.serializations_path}")
    with open(args.serializations_path, "rb") as f:
        tasks_serializations = pickle.load(f)
    logger.info(f"Loaded {len(tasks_serializations)} serialized samples")

    return splits_to_serializations, tasks_serializations

# %%
class llm_classifier(Protocol):
    """Interface for all LLM variant classifiers"""
    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: list[str],
        X_val_texts: list[str],
        X_test_texts: list[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        n_jobs: int = 1,
        test_patient_ids: np.ndarray | None = None,
        **kwargs,
    ) -> Tuple[object, dict]: ...

# %%
class llm_encoder:
    """LLM encoder using Qwen3-Embedding with logistic regression head"""
    
    _INSTR_PATH = "/sc-projects/sc-proj-ukb-cvd/projects/ehrshot-benchmark/ehrshot/serialization/task_to_instructions.json"

    def __init__(self, model_size: str = "8B", max_input_length: int = 4096):
        self.model_size = model_size
        self.max_input_length = max_input_length
        self._instructions = self._load_instructions()

        if model_size == "8B":
            self._backbone = Qwen3Embedding_8B_Encoder(max_input_length=max_input_length)
        elif model_size == "4B":
            self._backbone = Qwen3Embedding_4B_Encoder(max_input_length=max_input_length)
        elif model_size in {"0.6B", "0_6B", "0.6b"}:
            self._backbone = Qwen3Embedding_0_6B_Encoder(max_input_length=max_input_length)
        else:
            raise ValueError(
                f"Unsupported Qwen3 size: {model_size}. Use '8B' (default), '4B', or '0.6B'."
            )
        self._encoder = TextEncoder(self._backbone)

    def _load_instructions(self) -> dict:
        path = Path(self._INSTR_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Instruction file not found: {self._INSTR_PATH}")
        with open(path, "r") as f:
            return json.load(f)

    def _instruction_for(self, sub_task: str) -> str:
        instruction_prefix = self._instructions.get("instruction_prefix", "")
        instruction = self._instructions.get(sub_task, "")
        if instruction_prefix != "":
            instruction = f"{instruction_prefix} {instruction}"
        assert isinstance(instruction, str), f"Instruction for task {sub_task} must be a string"
        return instruction

    def _encode_texts_with_instruction(self, texts: list[str], sub_task: str) -> np.ndarray:
        instr = self._instruction_for(sub_task)
        instructions = [instr] * len(texts)
        return self._encoder.encode_texts(instructions=instructions, texts=texts, cache_dir=None)

    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: list[str],
        X_val_texts: list[str],
        X_test_texts: list[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        n_jobs: int = 1,
        test_patient_ids: np.ndarray | None = None,
        *,
        lr_solver: str = "lbfgs",
        eval_train_val: bool = False,
    ) -> Tuple[object, dict]:
        X_train = self._encode_texts_with_instruction(X_train_texts, sub_task=sub_task)
        X_val = self._encode_texts_with_instruction(X_val_texts, sub_task=sub_task)
        X_test = self._encode_texts_with_instruction(X_test_texts, sub_task=sub_task)

        _ = eval_train_val  # Unused; included for API compatibility with decoder variants

        model_head = f"lr_{lr_solver}"
        best_model, scores = run_evaluation_lr(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            model_head=model_head,
            n_jobs=n_jobs,
            test_patient_ids=test_patient_ids,
        )
        return best_model, scores

# %%
def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Find the starting index of needle subsequence in haystack"""
    if not needle or len(needle) > len(haystack):
        return None
    first = needle[0]
    for i, tok in enumerate(haystack):
        if tok == first and haystack[i : i + len(needle)] == needle:
            return i
    return None

class LLMDecoderQwen3:
    """Qwen3 decoder for Yes/No classification using next-token probabilities"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_context_tokens: int = 4096,
        cache_dir: Optional[str] = None,
        answer_after_think: bool = True,
        show_progress: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map, cache_dir=cache_dir
        )
        self.model.eval()

        self.max_context_tokens = max_context_tokens
        self.answer_after_think = answer_after_think
        self.show_progress = show_progress
        self._literal_yes = "Yes"
        self._literal_no = "No"
        self.enable_thinking = False
        self._assistant_prefill = "Answer: "

        # Build mapping from surface forms to token IDs
        self._build_token_mappings()

    def _build_token_mappings(self):
        """Build single-token surface form to ID mappings for Yes/No variants"""
        self._surface_to_ids: Dict[str, List[int]] = {}
        special = set(self.tokenizer.all_special_ids or [])
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab_size = max(self.tokenizer.get_vocab().values()) + 1

        for tid in range(vocab_size):
            if tid in special:
                continue
            try:
                s = self.tokenizer.decode([tid], skip_special_tokens=True)
            except Exception:
                continue
            if s is not None:
                self._surface_to_ids.setdefault(s, []).append(tid)

        def _variants(base: str) -> List[str]:
            cases = {base, base.upper(), base.lower(), base.capitalize()}
            punct = {"", ".", "!", "?"}
            lead = {"", " "}
            out = set()
            for c in cases:
                for p in punct:
                    for l in lead:
                        out.add(f"{l}{c}{p}")
            return sorted(out, key=len)

        self._yes_surfaces = _variants("Yes")
        self._no_surfaces = _variants("No")

        def _collect_ids(surfaces: List[str]) -> List[int]:
            ids = []
            for s in surfaces:
                for tid in self._surface_to_ids.get(s, []):
                    ids.append(tid)
            seen = set()
            uniq = []
            for tid in ids:
                if tid not in seen:
                    seen.add(tid)
                    uniq.append(tid)
            return uniq

        self._yes_token_ids_static: List[int] = _collect_ids(self._yes_surfaces)
        self._no_token_ids_static: List[int] = _collect_ids(self._no_surfaces)

        if self.show_progress:
            print(
                f"[Decoder] Found {len(self._yes_token_ids_static)} single-token YES variants "
                f"and {len(self._no_token_ids_static)} single-token NO variants."
            )
            dbg_yes = [(tid, self.tokenizer.decode([tid])) for tid in self._yes_token_ids_static]
            dbg_no = [(tid, self.tokenizer.decode([tid])) for tid in self._no_token_ids_static]
            print("YES token ids:", dbg_yes[:6], " ...")
            print("NO token ids:", dbg_no[:6], " ...")

    def _build_messages(self, ehr_text: str, question: str) -> List[Dict[str, str]]:
        sys = f"Question: {question}\n\nAnswer STRICTLY with a single token: Yes or No. No punctuation, no extra words."
        user = f"{ehr_text}"
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    def _apply_template(self, messages: List[Dict[str, str]], enable_thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def _render_with_budget(
        self, ehr_text: str, task_instruction: str, enable_thinking: bool
    ) -> str:
        """Render prompt with proper truncation to fit within max_context_tokens"""
        sys = f"Question: {task_instruction}\n\nAnswer STRICTLY with a single token: Yes or No. No punctuation, no extra words."
        user_prefix = ""

        # Calculate available budget for EHR text
        messages_base = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_prefix},
        ]
        text_base = self.tokenizer.apply_chat_template(
            messages_base,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        base_len = len(self.tokenizer(text_base, add_special_tokens=False).input_ids)
        budget = self.max_context_tokens - base_len

        if budget <= 0:
            return text_base

        # Tokenize EHR text with truncation to prevent the warning
        ehr_ids = self.tokenizer(
            ehr_text, 
            add_special_tokens=False,
            truncation=True,
            max_length=budget
        ).input_ids
        
        ehr_trimmed = self.tokenizer.decode(ehr_ids, skip_special_tokens=True)
        
        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_prefix + ehr_trimmed},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=enable_thinking
        )

    def _first_candidate_token_id(self, prompt_text: str, candidate: str) -> int:
        """Fallback method to find token ID for a candidate word"""
        ctx_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
        cand_ids = self.tokenizer(prompt_text + candidate, add_special_tokens=False).input_ids
        if len(cand_ids) <= len(ctx_ids):
            cand_ids = self.tokenizer(
                prompt_text + " " + candidate, add_special_tokens=False
            ).input_ids
        if len(cand_ids) <= len(ctx_ids):
            candidate_only_ids = self.tokenizer(candidate, add_special_tokens=False).input_ids
            if len(candidate_only_ids) > 0:
                return candidate_only_ids[0]
            candidate_with_space = self.tokenizer(
                " " + candidate, add_special_tokens=False
            ).input_ids
            if len(candidate_with_space) > 0:
                return candidate_with_space[-1]
            raise RuntimeError(f"Could not tokenize candidate '{candidate}'")
        return cand_ids[len(ctx_ids)]

    @torch.no_grad()
    def _last_step_probs_from_texts(self, texts: List[str]) -> torch.Tensor:
        """Get probability distribution over vocabulary for the last token position"""
        device = self.model.device
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        last_indices = attention_mask.sum(dim=1) - 1
        last_logits = logits[torch.arange(logits.size(0), device=device), last_indices]
        probs = torch.softmax(last_logits, dim=-1)
        return probs

    @torch.no_grad()
    def predict_proba(
        self,
        ehr_texts: List[str],
        question: str,
        batch_size: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """Compute Yes/No probabilities for a batch of EHR texts"""
        device = self.model.device
        all_p_yes, all_p_no = [], []

        if len(ehr_texts) == 0:
            empty = torch.empty(0, dtype=torch.float32, device=device)
            return {
                "p_yes": empty,
                "p_no": empty,
                "p_yes_2way": empty,
                "p_no_2way": empty,
            }

        if batch_size <= 0:
            batch_size = 1

        it = range(0, len(ehr_texts), batch_size)
        if self.show_progress:
            it = tqdm.trange(0, len(ehr_texts), batch_size)

        for i in it:
            batch_ehrs = ehr_texts[i : i + batch_size]
            prompts_no_think = []
            prompts_think = []
            
            for ehr in batch_ehrs:
                prompts_think.append(
                    self._render_with_budget(ehr, question, enable_thinking=self.enable_thinking)
                )
                prompts_no_think.append(
                    self._render_with_budget(ehr, question, enable_thinking=False)
                )

            if self.answer_after_think:
                contexts_for_scoring = [ptxt + self._assistant_prefill for ptxt in prompts_think]
            else:
                contexts_for_scoring = prompts_no_think

            probs = self._last_step_probs_from_texts(contexts_for_scoring)

            yes_ids_list = self._yes_token_ids_static
            no_ids_list = self._no_token_ids_static

            if not yes_ids_list:
                yes_ids_list = [
                    self._first_candidate_token_id(contexts_for_scoring[0], self._literal_yes)
                ]
            if not no_ids_list:
                no_ids_list = [
                    self._first_candidate_token_id(contexts_for_scoring[0], self._literal_no)
                ]

            y_idx = torch.tensor(yes_ids_list, device=probs.device, dtype=torch.long)
            n_idx = torch.tensor(no_ids_list, device=probs.device, dtype=torch.long)

            p_yes = probs.index_select(dim=1, index=y_idx).sum(dim=1)
            p_no = probs.index_select(dim=1, index=n_idx).sum(dim=1)

            all_p_yes.append(p_yes)
            all_p_no.append(p_no)

        p_yes = torch.cat(all_p_yes, dim=0).cpu()
        p_no = torch.cat(all_p_no, dim=0).cpu()
        denom = (p_yes + p_no).clamp_min(1e-12)
        p_yes_2way = p_yes / denom
        p_no_2way = p_no / denom

        return {
            "p_yes": p_yes,
            "p_no": p_no,
            "p_yes_2way": p_yes_2way,
            "p_no_2way": p_no_2way,
        }

    @torch.no_grad()
    def score(
        self,
        ehr_texts: List[str],
        question: str,
        batch_size: int = 4,
    ) -> torch.Tensor:
        out = self.predict_proba(ehr_texts, question, batch_size=batch_size)
        return out["p_yes_2way"]

# %%
class llm_decoder:
    """LLM decoder classifier without fine-tuning"""
    
    _INSTR_PATH = "/sc-projects/sc-proj-ukb-cvd/projects/ehrshot-benchmark/ehrshot/serialization/task_to_instructions.json"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_input_length: int = 4096,
        batch_size: int = 4,
        cache_dir: str | None = None,
        answer_after_think: bool = True,
        show_progress: bool = True,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self._decoder = LLMDecoderQwen3(
            model_name=model_name,
            device_map="auto",
            torch_dtype="auto",
            max_context_tokens=max_input_length,
            cache_dir=cache_dir,
            answer_after_think=answer_after_think,
            show_progress=show_progress,
        )
        self._instructions = self._load_instructions()

    def _load_instructions(self) -> dict:
        try:
            if os.path.exists(self._INSTR_PATH):
                with open(self._INSTR_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load instructions JSON: {e}")
        return {}

    def _instruction_for(self, sub_task: str) -> str:
        instr = self._instructions.get(sub_task)
        if isinstance(instr, dict):
            instr = instr.get("instruction", None)
        return (
            instr
            or f"Does the patient have the target condition/event for '{sub_task}' at prediction time?"
        )

    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: list[str],
        X_val_texts: list[str],
        X_test_texts: list[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        n_jobs: int = 1,
        test_patient_ids: np.ndarray | None = None,
        eval_train_val: bool = False,
        **kwargs,
    ) -> Tuple[object, dict]:
        question = self._instruction_for(sub_task)

        logger.critical(f"Start | Evaluating llm_decoder with {self.model_name} on '{sub_task}'")
        logger.info(
            f"Train N={len(X_train_texts)}  Val N={len(X_val_texts)}  Test N={len(X_test_texts)}"
        )
        def _safe_mean(arr: np.ndarray) -> float:
            return float(np.mean(arr)) if arr.size > 0 else float("nan")

        train_prev = _safe_mean(y_train)
        val_prev = _safe_mean(y_val)
        test_prev = _safe_mean(y_test)

        logger.info(
            f"Prevalence: train={train_prev:.4f} val={val_prev:.4f} test={test_prev:.4f}"
        )

        # Always calculate test probabilities
        logger.info("Computing test set probabilities...")
        y_test_proba = (
            self._decoder.score(X_test_texts, question=question, batch_size=self.batch_size)
            .cpu()
            .float()
            .numpy()
        )

        # Only calculate train/val probabilities if requested
        y_train_proba = None
        y_val_proba = None
        train_metrics_available = False
        val_metrics_available = False

        if eval_train_val:
            if len(X_train_texts) > 0 and y_train.size > 0:
                logger.info("Computing train set probabilities...")
                y_train_proba = (
                    self._decoder.score(
                        X_train_texts, question=question, batch_size=self.batch_size
                    )
                    .cpu()
                    .float()
                    .numpy()
                )
                train_metrics_available = True
            else:
                logger.info("Skipping train set probabilities; no training samples provided")

            if len(X_val_texts) > 0 and y_val.size > 0:
                logger.info("Computing validation set probabilities...")
                y_val_proba = (
                    self._decoder.score(
                        X_val_texts, question=question, batch_size=self.batch_size
                    )
                    .cpu()
                    .float()
                    .numpy()
                )
                val_metrics_available = True
            else:
                logger.info("Skipping validation set probabilities; no validation samples provided")

        y_test_pred = (y_test_proba >= 0.5).astype(int)

        metric_dict = {
            "auroc": metrics.roc_auc_score,
            "brier": metrics.brier_score_loss,
            "auprc": metrics.average_precision_score,
        }

        scores = {}
        for metric, func in metric_dict.items():
            scores[metric] = {}
            test_score = func(y_test, y_test_proba)

            log_parts = [f"test={test_score:.4f}"]

            if eval_train_val and train_metrics_available and y_train_proba is not None:
                train_score = func(y_train, y_train_proba)
                log_parts.insert(0, f"train={train_score:.4f}")

            if eval_train_val and val_metrics_available and y_val_proba is not None:
                val_score = func(y_val, y_val_proba)
                insert_idx = 1 if train_metrics_available else 0
                log_parts.insert(insert_idx, f"val={val_score:.4f}")

            logger.info(f"{metric.upper()} | {' '.join(log_parts)}")

            if test_patient_ids is None:
                test_patient_ids = np.arange(len(y_test))
            unique_ids = sorted(set(test_patient_ids))

            boots = []
            for i in range(1000):
                sample = sklearn.utils.resample(unique_ids, random_state=i)
                counts = collections.Counter(sample)
                weights = np.array([counts.get(pid, 0) for pid in test_patient_ids], dtype=float)
                if weights.sum() == 0:
                    continue
                boots.append(func(y_test, y_test_proba, sample_weight=weights))

            lower, upper = np.percentile(boots, [2.5, 97.5])
            scores[metric].update(
                score=float(test_score),
                std=float(np.std(boots, ddof=1)),
                lower=float(lower),
                mean=float(np.mean(boots)),
                upper=float(upper),
            )

        model_like = {
            "head": (
                "decoder_yesno_after_think" if self._decoder.answer_after_think else "decoder_yesno"
            ),
            "backbone": self.model_name,
            "sub_task": sub_task,
            "batch_size": self.batch_size,
            "max_input_length": self.max_input_length,
        }
        return model_like, scores

# %%
class _NextTokenYesNoDataset(Dataset):
    """Dataset for single-token next-token prediction training"""
    
    def __init__(
        self,
        decoder: LLMDecoderQwen3,
        ehr_texts: list[str],
        labels: np.ndarray,
        question: str,
        reserve_extra_tokens: int = 8,
        answer_after_think: bool | None = None,
    ):
        self.decoder = decoder
        self.tokenizer = decoder.tokenizer
        self.model = decoder.model
        self.ehrs = ehr_texts
        self.labels = labels.astype(int).tolist()
        self.question = question
        self.reserve = max(2, reserve_extra_tokens)
        self.answer_after_think = (
            decoder.answer_after_think if answer_after_think is None else answer_after_think
        )

        self.yes_ids = decoder._yes_token_ids_static or [
            decoder._first_candidate_token_id("X", decoder._literal_yes)
        ]
        self.no_ids = decoder._no_token_ids_static or [
            decoder._first_candidate_token_id("X", decoder._literal_no)
        ]

        self.ctx_max = max(16, decoder.max_context_tokens - self.reserve)

        # Pre-compute prompts to avoid redundant processing
        self._contexts = []
        for ehr in self.ehrs:
            txt = self.decoder._render_with_budget(
                ehr_text=ehr,
                task_instruction=self.question,
                enable_thinking=self.decoder.enable_thinking if self.answer_after_think else False,
            )
            ctx = txt + self.decoder._assistant_prefill
            self._contexts.append(ctx)

    def __len__(self):
        return len(self.ehrs)

    def __getitem__(self, i: int):
        ctx = self._contexts[i]
        y = self.labels[i]
        # Use literal token IDs instead of random choice from variants
        if y == 1:
            tgt_id = self.decoder._first_candidate_token_id(ctx, self.decoder._literal_yes)
        else:
            tgt_id = self.decoder._first_candidate_token_id(ctx, self.decoder._literal_no)

        ctx_ids = self.tokenizer(
            ctx,
            add_special_tokens=False,
            truncation=True,
            max_length=self.decoder.max_context_tokens - 1,
        ).input_ids

        input_ids = ctx_ids + [tgt_id]
        attn = [1] * len(input_ids)
        
        # Only compute loss on the final token
        labels = [-100] * len(input_ids)
        labels[-1] = tgt_id

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

@dataclass
class _PadCollator:
    """Data collator with padding support"""
    pad_token_id: int
    label_pad_id: int = -100
    max_len: int | None = None

    def __call__(self, batch):
        if self.max_len is not None:
            for x in batch:
                for k in ("input_ids", "attention_mask", "labels"):
                    seq = x[k].tolist()
                    if len(seq) > self.max_len:
                        seq = seq[-self.max_len :]
                    x[k] = torch.tensor(seq, dtype=torch.long)
        maxlen = max(len(x["input_ids"]) for x in batch)

        def pad(seq, val, L):
            return seq + [val] * (L - len(seq))

        input_ids = torch.tensor(
            [pad(x["input_ids"].tolist(), self.pad_token_id, maxlen) for x in batch],
            dtype=torch.long,
        )
        attn = torch.tensor(
            [pad(x["attention_mask"].tolist(), 0, maxlen) for x in batch], dtype=torch.long
        )
        labels = torch.tensor(
            [pad(x["labels"].tolist(), self.label_pad_id, maxlen) for x in batch], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

# %%
class llm_decoder_ft(llm_decoder):
    """LLM decoder with LoRA fine-tuning"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        max_input_length: int = 4096,
        batch_size: int = 4,
        cache_dir: str | None = None,
        answer_after_think: bool = True,
        show_progress: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.03,
        num_train_epochs_cap: int = 20,
        effective_batch_size: int = 8,
        fp16: bool | None = None,
        bf16: bool | None = None,
        seed: int = 42,
        output_dir: str = None,
        early_stopping_threshold: float = 0.0,
        early_stopping_patience: int = 5,
    ):
        super().__init__(
            model_name=model_name,
            max_input_length=max_input_length,
            batch_size=batch_size,
            cache_dir=cache_dir,
            answer_after_think=answer_after_think,
            show_progress=show_progress,
        )

        self.show_progress = show_progress

        # Configure and apply LoRA adapter
        self._decoder.model.train()
        self._lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
        )
        self._decoder.model = get_peft_model(self._decoder.model, self._lora_cfg)
        if self.show_progress:
            self._decoder.model.print_trainable_parameters()

        # Set output directory first (needed for training args)
        self.output_dir = output_dir if output_dir is not None else "./_llm_decoder_ft"
        
        # Auto-select mixed precision based on hardware
        if bf16 is None:
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if fp16 is None:
            fp16 = torch.cuda.is_available() and not bf16

        # Calculate gradient accumulation steps
        gradient_accumulation_steps = max(1, effective_batch_size // batch_size)

        self._train_args_tpl = dict(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            num_train_epochs=num_train_epochs_cap,
            logging_strategy="steps",
            logging_steps=1,
            eval_strategy="epoch",
            optim="adamw_torch",
            save_strategy="epoch",
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            fp16=fp16,
            bf16=bf16,
            dataloader_pin_memory=True,
            report_to=[],
            seed=seed,
            output_dir=self.output_dir,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            torch_empty_cache_steps=50,
        )
        self._early_cb = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )

    def _fit_lora(
        self,
        train_texts: list[str],
        val_texts: list[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        question: str,
    ):
        """Fine-tune the model using LoRA with early stopping"""
        logger.info(f"Starting LoRA fine-tuning with {len(train_texts)} train and {len(val_texts)} val samples")
        logger.info(f"Question/instruction: {question[:100]}..." if len(question) > 100 else f"Question/instruction: {question}")

        logger.info("Creating training and validation datasets...")
        ds_tr = _NextTokenYesNoDataset(
            self._decoder, train_texts, y_train, question, reserve_extra_tokens=8
        )
        ds_va = _NextTokenYesNoDataset(
            self._decoder, val_texts, y_val, question, reserve_extra_tokens=8
        )

        collator = _PadCollator(
            self._decoder.tokenizer.pad_token_id,
            label_pad_id=-100,
            max_len=self._decoder.max_context_tokens,
        )

        def _peek(ds, n=64):
            mx = 0
            arg = -1
            for i in range(min(n, len(ds))):
                L = len(ds[i]["input_ids"])
                if L > mx:
                    mx, arg = L, i
            print(f"[FT] peek max seq_len among first {min(n,len(ds))}: {mx} (idx={arg})")

        _peek(ds_tr)
        _peek(ds_va)

        # Workaround for Accelerate optimizer wrapping issue
        import torch.optim as _optim
        def _noop(self, *args, **kwargs):
            return None
        if not hasattr(_optim.AdamW, "train"):
            _optim.AdamW.train = _noop
        if not hasattr(_optim.AdamW, "eval"):
            _optim.AdamW.eval = _noop

        logger.info("Setting up Trainer with early stopping...")
        args = TrainingArguments(**self._train_args_tpl)
        trainer = Trainer(
            model=self._decoder.model,
            args=args,
            train_dataset=ds_tr,
            eval_dataset=ds_va,
            data_collator=collator,
        )
        trainer.add_callback(self._early_cb)

        logger.info("Starting training...")
        train_out = trainer.train()
        logger.info("Training completed!")

        if self.show_progress:
            print(
                f"[LoRA] Finished at epoch={train_out.metrics.get('epoch', None)}  best eval_loss={trainer.state.best_metric}"
            )
        self._decoder.model.eval()

    def _sanity_check_probabilities(self, texts: list[str], labels: np.ndarray, question: str, phase: str):
        """Print average Yes/No probabilities for a small batch as a sanity check"""
        if len(texts) == 0:
            print(f"Sanity check skipped {phase}: no samples provided")
            return
        try:
            probs_dict = self._decoder.predict_proba(texts, question, batch_size=len(texts))
            p_yes = probs_dict["p_yes_2way"].float().cpu().numpy()
            p_no = probs_dict["p_no_2way"].float().cpu().numpy()

            # Calculate averages by true label
            pos_indices = labels == 1
            neg_indices = labels == 0

            if pos_indices.sum() > 0:
                avg_yes_for_pos = p_yes[pos_indices].mean()
                avg_no_for_pos = p_no[pos_indices].mean()
            else:
                avg_yes_for_pos = avg_no_for_pos = 0.0

            if neg_indices.sum() > 0:
                avg_yes_for_neg = p_yes[neg_indices].mean()
                avg_no_for_neg = p_no[neg_indices].mean()
            else:
                avg_yes_for_neg = avg_no_for_neg = 0.0

            print(f"\n=== SANITY CHECK {phase} ===")
            print(f"Sample size: {len(texts)} (pos: {pos_indices.sum()}, neg: {neg_indices.sum()})")
            print(f"For TRUE POSITIVE samples:")
            print(f"  Avg P(Yes) = {avg_yes_for_pos:.4f}, Avg P(No) = {avg_no_for_pos:.4f}")
            print(f"For TRUE NEGATIVE samples:")
            print(f"  Avg P(Yes) = {avg_yes_for_neg:.4f}, Avg P(No) = {avg_no_for_neg:.4f}")
            print(f"Overall averages:")
            print(f"  Avg P(Yes) = {p_yes.mean():.4f}, Avg P(No) = {p_no.mean():.4f}")
            print("=" * 40)

        except Exception as e:
            print(f"Sanity check failed {phase}: {e}")

    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: list[str],
        X_val_texts: list[str],
        X_test_texts: list[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        n_jobs: int = 1,
        test_patient_ids: np.ndarray | None = None,
        eval_train_val: bool = False,
        **kwargs,
    ):
        question = self._instruction_for(sub_task)

        logger.critical(
            f"Start | Evaluating llm_decoder_ft (LoRA) with {self.model_name} on '{sub_task}'"
        )
        logger.info(
            f"Train N={len(X_train_texts)}  Val N={len(X_val_texts)}  Test N={len(X_test_texts)}"
        )

        def _safe_mean(arr: np.ndarray) -> float:
            return float(np.mean(arr)) if arr.size > 0 else float("nan")

        train_empty = len(X_train_texts) == 0 or y_train.size == 0
        val_empty = len(X_val_texts) == 0 or y_val.size == 0

        logger.info(
            f"Prevalence: train={_safe_mean(y_train):.4f} val={_safe_mean(y_val):.4f} test={_safe_mean(y_test):.4f}"
        )

        if train_empty:
            logger.warning(
                "No training samples provided; skipping LoRA fine-tuning and falling back to zero-shot evaluation."
            )
            return super().run_evaluation(
                sub_task,
                X_train_texts,
                X_val_texts,
                X_test_texts,
                y_train,
                y_val,
                y_test,
                n_jobs=n_jobs,
                test_patient_ids=test_patient_ids,
                eval_train_val=eval_train_val,
                **kwargs,
            )

        if val_empty:
            logger.warning(
                "No validation samples provided; skipping LoRA fine-tuning and falling back to zero-shot evaluation."
            )
            return super().run_evaluation(
                sub_task,
                X_train_texts,
                X_val_texts,
                X_test_texts,
                y_train,
                y_val,
                y_test,
                n_jobs=n_jobs,
                test_patient_ids=test_patient_ids,
                eval_train_val=eval_train_val,
                **kwargs,
            )

        # Sanity check: probabilities before fine-tuning
        if self.show_progress:
            batch_samples = min(self.batch_size, len(X_train_texts))
            if batch_samples > 0:
                self._sanity_check_probabilities(
                    X_train_texts[:batch_samples],
                    y_train[:batch_samples],
                    question,
                    "BEFORE fine-tuning",
                )

        # Fine-tune with early stopping
        self._fit_lora(X_train_texts, X_val_texts, y_train, y_val, question)

        # Sanity check: probabilities after fine-tuning
        if self.show_progress:
            batch_samples = min(self.batch_size, len(X_train_texts))
            if batch_samples > 0:
                self._sanity_check_probabilities(
                    X_train_texts[:batch_samples],
                    y_train[:batch_samples],
                    question,
                    "AFTER fine-tuning",
                )

        # Score with fine-tuned model - always calculate test probabilities
        logger.info("Computing test set probabilities with fine-tuned model...")
        y_test_proba = (
            self._decoder.score(X_test_texts, question=question, batch_size=self.batch_size)
            .cpu()
            .float()
            .numpy()
        )

        # Only calculate train/val probabilities if requested
        y_train_proba = None
        y_val_proba = None
        if eval_train_val:
            logger.info("Computing train set probabilities with fine-tuned model...")
            y_train_proba = (
                self._decoder.score(X_train_texts, question=question, batch_size=self.batch_size)
                .cpu()
                .float()
                .numpy()
            )
            logger.info("Computing validation set probabilities with fine-tuned model...")
            y_val_proba = (
                self._decoder.score(X_val_texts, question=question, batch_size=self.batch_size)
                .cpu()
                .float()
                .numpy()
            )

        y_test_pred = (y_test_proba >= 0.5).astype(int)
        metric_dict = {
            "auroc": metrics.roc_auc_score,
            "brier": metrics.brier_score_loss,
            "auprc": metrics.average_precision_score,
        }
        scores = {}
        for metric, func in metric_dict.items():
            scores[metric] = {}
            test_score = func(y_test, y_test_proba)

            if eval_train_val and y_train_proba is not None and y_val_proba is not None:
                train_score = func(y_train, y_train_proba)
                val_score = func(y_val, y_val_proba)
                logger.info(
                    f"{metric.upper()} | train={train_score:.4f} val={val_score:.4f} test={test_score:.4f}"
                )
            else:
                logger.info(f"{metric.upper()} | test={test_score:.4f}")

            if test_patient_ids is None:
                test_patient_ids = np.arange(len(y_test))
            unique_ids = sorted(set(test_patient_ids))

            boots = []
            for i in range(1000):
                sample = sklearn.utils.resample(unique_ids, random_state=i)
                counts = collections.Counter(sample)
                weights = np.array([counts.get(pid, 0) for pid in test_patient_ids], dtype=float)
                if weights.sum() == 0:
                    continue
                boots.append(func(y_test, y_test_proba, sample_weight=weights))

            lower, upper = np.percentile(boots, [2.5, 97.5])
            scores[metric].update(
                score=float(test_score),
                std=float(np.std(boots, ddof=1)),
                lower=float(lower),
                mean=float(np.mean(boots)),
                upper=float(upper),
            )

        model_like = {
            "head": (
                "decoder_yesno_after_think_ft"
                if self._decoder.answer_after_think
                else "decoder_yesno_ft"
            ),
            "backbone": self.model_name,
            "sub_task": sub_task,
            "batch_size": self.batch_size,
            "max_input_length": self.max_input_length,
            "lora": {
                "r": self._lora_cfg.r,
                "alpha": self._lora_cfg.lora_alpha,
                "dropout": self._lora_cfg.lora_dropout,
                "targets": self._lora_cfg.target_modules,
            },
        }
        return model_like, scores

def run_single_experiment(args, splits_to_serializations, tasks_serializations, show_progress):
    """Run a single experiment configuration"""
    results = []
    model = "llm"
    zero_shot = args.k == 0
    
    # Calculate effective batch size
    effective_batch_size = args.effective_batch_size
    if effective_batch_size is None:
        effective_batch_size = min(args.k, 8) if args.k > 0 else 8
    
    # Create unique output directory for this experiment
    model_safe = re.sub(r'[^\w\-_\.]', '_', args.model_name.replace('/', '_'))
    unique_output_dir = None

    if zero_shot:
        logger.info(
            "k=0 detected; running zero-shot evaluation without LoRA fine-tuning"
        )
        logger.info(f"Creating llm_decoder classifier with model={args.model_name}")
        clf = llm_decoder(
            model_name=args.model_name,
            max_input_length=args.max_input_length,
            batch_size=args.batch_size,
            show_progress=show_progress,
        )
    else:
        unique_output_dir = (
            f"./tmp_llm_decoder_ft_{model_safe}_{args.sub_task}_k{args.k}_r{args.replicate}_{os.getpid()}"
        )

        logger.info(f"Creating llm_decoder_ft classifier with model={args.model_name}")
        logger.info(
            f"LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )
        logger.info(
            f"Training configuration: lr={args.lr}, warmup_ratio={args.warmup_ratio}, max_epochs={args.num_train_epochs_cap}"
        )
        logger.info(
            f"Batch configuration: batch_size={args.batch_size}, effective_batch_size={effective_batch_size}"
        )

        clf = llm_decoder_ft(
            model_name=args.model_name,
            max_input_length=args.max_input_length,
            batch_size=args.batch_size,
            show_progress=show_progress,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lr=args.lr,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs_cap=args.num_train_epochs_cap,
            effective_batch_size=effective_batch_size,
            output_dir=unique_output_dir,
        )

    print(f"Model: {model} | Task: {args.sub_task} | k: {args.k} | replicate: {args.replicate}")

    if args.k == -1:
        # Use all available training data (no shot_size filter)
        logger.info("Using all available training data (no shot_size filter)")
        task_split = splits_to_serializations[
            (splits_to_serializations["task"] == args.sub_task)
            & (splits_to_serializations["fold"] == args.replicate)
        ]
    else:
        # Use specific shot size
        logger.info(f"Using shot_size={args.k} for training data")
        task_split = splits_to_serializations[
            (splits_to_serializations["task"] == args.sub_task)
            & (splits_to_serializations["shot_size"] == args.k)
            & (splits_to_serializations["fold"] == args.replicate)
        ]

    logger.info("Extracting train, validation, and test splits...")
    train_split = task_split[task_split["split_name"] == "train"]
    val_split = task_split[task_split["split_name"] == "val"]
    test_split = splits_to_serializations[
        (splits_to_serializations["task"] == args.sub_task)
        & (splits_to_serializations["split_name"] == "test")
    ]

    rng = np.random.default_rng(args.subset_seed)
    val_split = _maybe_limit_split(val_split, args.val_limit, rng, "validation", args.subset_seed)
    test_split = _maybe_limit_split(test_split, args.test_limit, rng, "test", args.subset_seed)

    logger.info(f"Split sizes - Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")

    logger.info("Loading serialized data for train/val/test splits...")
    X_train_k = [
        tasks_serializations[idx][1] for idx in train_split["serialization_idx"].values
    ]
    X_val_k = [
        tasks_serializations[idx][1] for idx in val_split["serialization_idx"].values
    ]
    X_test = [
        tasks_serializations[idx][1] for idx in test_split["serialization_idx"].values
    ]
    y_train_k = np.array(train_split["label_value"].values)
    y_val_k = np.array(val_split["label_value"].values)
    y_test = np.array(test_split["label_value"].values)
    test_patient_ids = test_split["patient_id"].values

    logger.info(f"Label prevalences - Train: {np.mean(y_train_k):.4f}, Val: {np.mean(y_val_k):.4f}, Test: {np.mean(y_test):.4f}")
    logger.info(f"Starting model evaluation with eval_train_val={args.eval_train_val}...")

    best_model, scores = clf.run_evaluation(
        args.sub_task,
        X_train_k,
        X_val_k,
        X_test,
        y_train_k,
        y_val_k,
        y_test,
        n_jobs=args.num_threads,
        test_patient_ids=test_patient_ids,
        eval_train_val=args.eval_train_val,
    )

    for score_name, score_value in scores.items():
        results.append(
            {
                "labeling_function": args.labeling_function,
                "sub_task": args.sub_task,
                "model": model,
                "replicate": args.replicate,
                "k": args.k,
                "score": score_name,
                "value": score_value["score"],
                "std": score_value["std"],
                "lower": score_value["lower"],
                "mean": score_value["mean"],
                "upper": score_value["upper"],
            }
        )
    
    print(f"Scores: {scores}")
    
    # Clean up temporary checkpoint directory
    if unique_output_dir is not None:
        import shutil
        try:
            if os.path.exists(unique_output_dir):
                shutil.rmtree(unique_output_dir)
                print(f"Cleaned up temporary directory: {unique_output_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {unique_output_dir}: {e}")
    
    return results

if __name__ == "__main__":
    main()
