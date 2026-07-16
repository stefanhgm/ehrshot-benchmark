#!/usr/bin/env python3
"""
Run LLM encoder experiments with LoRA fine-tuning and classification head
"""

import argparse
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Protocol, List, Dict, Any, Optional, Iterable, Sequence
import os
import collections
import torch
import torch.nn as nn
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

from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from serialization.text_encoder import (
    Qwen3Embedding_8B_Encoder,
    Qwen3Embedding_4B_Encoder, 
    Qwen3Embedding_0_6B_Encoder,
    TextEncoder,
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _REPO_ROOT / "EHRSHOT_ASSETS"

def resolve_instruction_from_serializations(
    tasks_serializations: Sequence[Any],
    serialization_indices: Iterable[int],
    sub_task: str,
) -> str:
    """Resolve one strict task instruction from serialization tuples."""
    instructions: list[str] = []
    for idx in serialization_indices:
        entry = tasks_serializations[int(idx)]
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            raise ValueError(
                "Unexpected serialization entry format for "
                f"sub_task='{sub_task}', idx={int(idx)}: {type(entry)}"
            )
        instruction = entry[0]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "Missing non-empty instruction in serialization entry for "
                f"sub_task='{sub_task}', idx={int(idx)}"
            )
        instructions.append(instruction.strip())

    if not instructions:
        raise ValueError(f"No serialization indices were provided for sub_task='{sub_task}'")

    unique_instructions = sorted(set(instructions))
    if len(unique_instructions) > 1:
        raise ValueError(
            "Conflicting instructions found in serializations for "
            f"sub_task='{sub_task}': {unique_instructions[:3]}"
        )

    return unique_instructions[0]

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run LLM encoder experiments for medical prediction tasks")
    
    # Paths
    parser.add_argument("--output_dir", type=str, 
                        default=str(_ASSETS / "experiments" / "llm_variants"),
                        help="Output directory for results (filename will be auto-generated)")
    parser.add_argument("--splits_path", type=str,
                        default=str(_ASSETS / "benchmark" / "ehrshot_splits_to_serializations.csv"),
                        help="Path to splits to serializations CSV file")
    parser.add_argument("--serializations_path", type=str,
                        default=str(_ASSETS / "benchmark" / "tasks_serializations.pkl"),
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
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-Embedding-0.6B",
                        help="Model name (Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B)")
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
    parser.add_argument("--labeling_function", type=str, default="llm_encoder_ft",
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

# Default task list (same as decoder script)
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

class EncoderClassifier(nn.Module):
    """Encoder model with classification head"""
    
    def __init__(
        self,
        encoder_model,
        embedding_size: int,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_size, num_classes)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get embeddings from encoder
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use last token pooling (same as Qwen3 encoders)
        embeddings = self.last_token_pool(outputs.last_hidden_state, attention_mask)
        
        # Apply dropout and classify
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.to(dtype=self.classifier.weight.dtype)
        logits = self.classifier(embeddings)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        # Return in the format expected by Trainer
        return {"loss": loss, "logits": logits} if loss is not None else logits
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the encoder"""
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the encoder"""
        if hasattr(self.encoder, 'gradient_checkpointing_disable'):
            self.encoder.gradient_checkpointing_disable()
    
    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Last token pooling (same as Qwen3 encoders)"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class ClassificationDataset(Dataset):
    """Dataset for text classification"""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 4096,
        instruction: str = "",
    ):
        self.texts = texts
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Add instruction if provided (same format as Qwen3 encoders)
        if self.instruction and len(self.instruction) > 0:
            text = f'Instruct: {self.instruction}\nQuery:\n{text}'
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

@dataclass
class ClassificationCollator:
    """Data collator for classification"""
    tokenizer: Any
    max_length: int = 4096
    
    def __call__(self, batch):
        # Extract components
        input_ids = [item['input_ids'] for item in batch]
        attention_masks = [item['attention_mask'] for item in batch]
        labels = torch.tensor([item['labels'] for item in batch])
        
        # For left padding, we need to pad to the left
        max_len = min(max(len(seq) for seq in input_ids), self.max_length)
        
        padded_input_ids = []
        padded_attention_masks = []
        
        for input_id, attention_mask in zip(input_ids, attention_masks):
            # Truncate from the right (keep beginning of text)
            if len(input_id) > self.max_length:
                input_id = input_id[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            
            # Left pad to max_len
            pad_len = max_len - len(input_id)
            padded_input_id = torch.cat([
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_id.dtype),
                input_id
            ])
            padded_attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=attention_mask.dtype),
                attention_mask
            ])
            
            padded_input_ids.append(padded_input_id)
            padded_attention_masks.append(padded_attention_mask)
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_masks),
            'labels': labels
        }

class llm_encoder_ft:
    """LLM encoder with LoRA fine-tuning and classification head"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        max_input_length: int = 4096,
        batch_size: int = 8,
        cache_dir: str | None = None,
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
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.show_progress = show_progress
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        # Force left padding for Qwen3 + Flash Attention compatibility
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine embedding size based on model
        if "0.6B" in model_name or "0_6B" in model_name:
            embedding_size = 1024
        elif "4B" in model_name:
            embedding_size = 2560
        elif "8B" in model_name:
            embedding_size = 4096
        else:
            raise ValueError(f"Unknown model size for {model_name}")
        
        # Load base encoder model
        base_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        
        # Create encoder + classifier model
        self.model = EncoderClassifier(
            encoder_model=base_model,
            embedding_size=embedding_size,
            num_classes=2,
        )
        
        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Configure LoRA
        self._lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,  # For encoder models
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
        
        # Apply LoRA to encoder part only
        self.model.encoder = get_peft_model(self.model.encoder, self._lora_cfg)
        if self.show_progress:
            self.model.encoder.print_trainable_parameters()
        
        # Set output directory first (needed for training args)
        self.output_dir = output_dir if output_dir is not None else "./_llm_encoder_ft"
        
        # Auto-select mixed precision
        if bf16 is None:
            bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if fp16 is None:
            fp16 = torch.cuda.is_available() and not bf16
        
        # Calculate gradient accumulation steps
        gradient_accumulation_steps = max(1, effective_batch_size // batch_size)

        # Training arguments
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

    def _fit_model(
        self,
        train_texts: List[str],
        val_texts: List[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        instruction: str,
    ):
        """Fine-tune the model with LoRA and train classification head"""
        logger.info(f"Starting LoRA fine-tuning with {len(train_texts)} train and {len(val_texts)} val samples")
        logger.info(f"Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"Instruction: {instruction}")

        # Create datasets
        logger.info("Creating training and validation datasets...")
        train_dataset = ClassificationDataset(
            texts=train_texts,
            labels=y_train,
            tokenizer=self.tokenizer,
            max_length=self.max_input_length,
            instruction=instruction,
        )
        
        val_dataset = ClassificationDataset(
            texts=val_texts,
            labels=y_val,
            tokenizer=self.tokenizer,
            max_length=self.max_input_length,
            instruction=instruction,
        )
        
        # Create data collator
        collator = ClassificationCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_input_length,
        )
        
        if self.show_progress:
            print(f"[FT] Train dataset size: {len(train_dataset)}")
            print(f"[FT] Val dataset size: {len(val_dataset)}")
        
        # Workaround for Accelerate optimizer wrapping issue
        import torch.optim as _optim
        def _noop(self, *args, **kwargs):
            return None
        if not hasattr(_optim.AdamW, "train"):
            _optim.AdamW.train = _noop
        if not hasattr(_optim.AdamW, "eval"):
            _optim.AdamW.eval = _noop
        
        # Create trainer
        logger.info("Setting up Trainer with early stopping...")
        args = TrainingArguments(**self._train_args_tpl)
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
        )
        trainer.add_callback(self._early_cb)

        # Train
        logger.info("Starting training...")
        train_out = trainer.train()
        logger.info("Training completed!")
        
        if self.show_progress:
            print(
                f"[LoRA] Finished at epoch={train_out.metrics.get('epoch', None)} "
                f"best eval_loss={trainer.state.best_metric}"
            )
        
        self.model.eval()
    
    @torch.no_grad()
    def predict_proba(
        self,
        texts: List[str],
        instruction: str = "",
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Get prediction probabilities"""
        if batch_size is None:
            batch_size = self.batch_size
        
        self.model.eval()
        all_probs = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Add instruction if provided
            if instruction and len(instruction) > 0:
                batch_texts = [f'Instruct: {instruction}\nQuery:\n{text}' for text in batch_texts]
            
            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.max_input_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask)
            # Handle both dict and tensor returns
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            probs = F.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
        
        return np.concatenate(all_probs, axis=0)
    
    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: List[str],
        X_val_texts: List[str],
        X_test_texts: List[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        n_jobs: int = 1,
        test_patient_ids: np.ndarray | None = None,
        eval_train_val: bool = False,
        **kwargs,
    ) -> Tuple[object, dict]:
        """Run evaluation pipeline"""
        instruction = kwargs.get("instruction_override")
        if not instruction:
            raise ValueError(
                "instruction_override is required: instructions must come from the "
                "serializations pickle (entry[0]), never from a generated default"
            )
        
        logger.critical(
            f"Start | Evaluating llm_encoder_ft (LoRA) with {self.model_name} on '{sub_task}'"
        )
        logger.info(
            f"Train N={len(X_train_texts)}  Val N={len(X_val_texts)}  Test N={len(X_test_texts)}"
        )
        logger.info(
            f"Prevalence: train={np.mean(y_train):.4f} val={np.mean(y_val):.4f} test={np.mean(y_test):.4f}"
        )
        
        # Fine-tune model
        self._fit_model(X_train_texts, X_val_texts, y_train, y_val, instruction)

        # Always calculate test probabilities
        logger.info("Computing test set probabilities with fine-tuned model...")
        y_test_proba = self.predict_proba(X_test_texts, instruction)[:, 1]  # Positive class

        # Only calculate train/val probabilities if requested
        y_train_proba = None
        y_val_proba = None
        if eval_train_val:
            logger.info("Computing train set probabilities with fine-tuned model...")
            y_train_proba = self.predict_proba(X_train_texts, instruction)[:, 1]
            logger.info("Computing validation set probabilities with fine-tuned model...")
            y_val_proba = self.predict_proba(X_val_texts, instruction)[:, 1]
        
        # Calculate metrics
        metric_dict = {
            "auroc": metrics.roc_auc_score,
            "brier": metrics.brier_score_loss,
            "auprc": metrics.average_precision_score,
        }
        
        scores = {}
        for metric, func in metric_dict.items():
            scores[metric] = {}
            test_score = func(y_test, y_test_proba)
            train_score = None
            val_score = None

            if eval_train_val and y_train_proba is not None and y_val_proba is not None:
                train_score = func(y_train, y_train_proba)
                val_score = func(y_val, y_val_proba)
                logger.info(
                    f"{metric.upper()} | train={train_score:.4f} val={val_score:.4f} test={test_score:.4f}"
                )
            else:
                logger.info(f"{metric.upper()} | test={test_score:.4f}")
            
            # Bootstrap confidence intervals
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
            if train_score is not None:
                scores[metric]["train_score"] = float(train_score)
            if val_score is not None:
                scores[metric]["val_score"] = float(val_score)
        
        model_like = {
            "head": "encoder_classification_ft",
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

def load_data(args):
    """Load data splits and serializations (same as decoder script)"""
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

def run_single_experiment(args, splits_to_serializations, tasks_serializations, show_progress):
    """Run a single experiment configuration"""
    results = []
    model = "llm_encoder"
    
    # Calculate effective batch size
    effective_batch_size = args.effective_batch_size
    if effective_batch_size is None:
        effective_batch_size = min(args.k, 8) if args.k > 0 else 8
    
    # Create unique output directory for this experiment
    model_safe = re.sub(r'[^\w\-_\.]', '_', args.model_name.replace('/', '_'))
    unique_output_dir = f"./tmp_llm_encoder_ft_{model_safe}_{args.sub_task}_k{args.k}_r{args.replicate}_{os.getpid()}"
    
    # Create classifier
    logger.info(f"Creating llm_encoder_ft classifier with model={args.model_name}")
    logger.info(f"LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    logger.info(f"Training configuration: lr={args.lr}, warmup_ratio={args.warmup_ratio}, max_epochs={args.num_train_epochs_cap}")
    logger.info(f"Batch configuration: batch_size={args.batch_size}, effective_batch_size={effective_batch_size}")

    clf = llm_encoder_ft(
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

    # Load task splits (same as decoder script)
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

    candidate_instruction_indices: list[int] = []
    candidate_instruction_indices.extend(train_split["serialization_idx"].values.tolist())
    candidate_instruction_indices.extend(val_split["serialization_idx"].values.tolist())
    candidate_instruction_indices.extend(test_split["serialization_idx"].values.tolist())
    resolved_instruction = resolve_instruction_from_serializations(
        tasks_serializations=tasks_serializations,
        serialization_indices=candidate_instruction_indices,
        sub_task=args.sub_task,
    )
    logger.info(f"Resolved encoder instruction from serializations: {resolved_instruction}")

    logger.info(f"Label prevalences - Train: {np.mean(y_train_k):.4f}, Val: {np.mean(y_val_k):.4f}, Test: {np.mean(y_test):.4f}")
    logger.info(f"Starting model evaluation with eval_train_val={args.eval_train_val}...")

    # Run evaluation
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
        instruction_override=resolved_instruction,
    )

    # Format results (same as decoder script)
    for score_name, score_value in scores.items():
        row = {
            "labeling_function": args.labeling_function,
            "sub_task": args.sub_task,
            "model": model,
            "replicate": args.replicate,
            "k": args.k,
            "score": score_name,
            "selection_metric": "auroc",
            "value": score_value["score"],
            "value_test": score_value["score"],
            "value_val": score_value.get("val_score"),
            "std": score_value["std"],
            "lower": score_value["lower"],
            "mean": score_value["mean"],
            "upper": score_value["upper"],
        }
        if "train_score" in score_value:
            row["value_train"] = score_value["train_score"]
        results.append(row)
    
    print(f"Scores: {scores}")
    
    # Clean up temporary checkpoint directory
    import shutil
    try:
        if os.path.exists(unique_output_dir):
            shutil.rmtree(unique_output_dir)
            print(f"Cleaned up temporary directory: {unique_output_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up {unique_output_dir}: {e}")
    
    return results

def main():
    """Main experiment function"""
    args = parse_args()
    
    # Validate required arguments
    if args.sub_task is None or args.k is None or args.replicate is None:
        print("Error: --sub_task, --k, and --replicate are required arguments")
        print("Example: python fit_and_eval_encoder.py --sub_task guo_los --k 128 --replicate 0 --model_name Qwen/Qwen3-Embedding-0.6B")
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

if __name__ == "__main__":
    main()
