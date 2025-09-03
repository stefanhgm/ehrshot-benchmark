import argparse
import os
import json
import pickle
from typing import List, Dict, Any, Tuple, Protocol
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Imports aligned with your codebase
from serialization.text_encoder import (
    TextEncoder,
    Qwen3Embedding_8B_Encoder,
    Qwen3Embedding_4B_Encoder,
    Qwen3Embedding_0_6B_Encoder,
)

# ----------------------
# CLI
# ----------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM variants experiments (encoder + LR head) in EHRSHOT style.")
    parser.add_argument("--splits_to_serializations", required=True, type=str,
                        help="CSV with columns [task, split_name, shot_size, fold, patient_id, prediction_time, label_type, label_value, serialization_idx].")
    parser.add_argument("--tasks_serializations", required=True, type=str,
                        help="Pickle with list/array mapping serialization_idx -> (metadata, serialization_text).")
    parser.add_argument("--output_csv", required=True, type=str,
                        help="Where to write/append experiment results as CSV.")
    parser.add_argument("--eval_py", required=True, type=str,
                        help="Path to ehrshot/7_eval.py to import run_evaluation() from.")
    parser.add_argument("--instructions_json", type=str, default=None,
                        help="Path to task_to_instructions.json. If omitted, will try to infer or run without instructions.")
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated list of tasks to run. If omitted, uses the default EHRSHOT set from the notebook.")
    parser.add_argument("--ks", type=str, default="8",
                        help="Comma-separated shot sizes. Example: '1,2,4,8'")
    parser.add_argument("--replicates", type=str, default="0",
                        help="Comma-separated replicate IDs. Example: '0,1,2'")
    parser.add_argument("--n_jobs", type=int, default=40, help="Number of parallel jobs for LR evaluation.")
    parser.add_argument("--model_size", type=str, default="0.6B", choices=["0.6B","4B","8B"], help="Qwen3Embedding backbone size.")
    parser.add_argument("--max_input_length", type=int, default=4096, help="Max tokens for the encoder input.")
    parser.add_argument("--lr_solver", type=str, default="lbfgs", help="LR solver name to match 7_eval.py naming (e.g., 'lbfgs', 'liblinear').")
    return parser.parse_args()

# ----------------------
# Dynamic import of 7_eval.run_evaluation
# ----------------------
def import_run_evaluation(eval_py: str):
    import importlib.util
    eval_path = os.path.abspath(eval_py)
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Could not find eval file at {eval_path}")
    spec = importlib.util.spec_from_file_location("ehrshot_eval", eval_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "run_evaluation"):
        raise AttributeError("Expected 'run_evaluation' in the provided eval module.")
    return mod.run_evaluation

# ----------------------
# Classifier interface and implementation
# ----------------------
class LLMClassifier(Protocol):
    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: List[str],
        X_val_texts: List[str],
        X_test_texts: List[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        run_evaluation_lr,
        lr_solver: str,
        n_jobs: int,
        test_patient_ids: List[int],
    ) -> Tuple[Any, Dict[str, Dict[str, float]]]: ...

class LLMEncoder:
    """
    Encoder-only variant: get embeddings via TextEncoder(Qwen3Embedding_*), then feed to LR via 7_eval.run_evaluation().
    """
    def __init__(self, model_size: str, max_input_length: int, instructions_json: str | None):
        self.model_size = model_size
        self.max_input_length = max_input_length
        self._instructions = self._load_instructions(instructions_json)

        if model_size == "8B":
            backbone = Qwen3Embedding_8B_Encoder(max_input_length=max_input_length)
        elif model_size == "4B":
            backbone = Qwen3Embedding_4B_Encoder(max_input_length=max_input_length)
        elif model_size == "0.6B":
            backbone = Qwen3Embedding_0_6B_Encoder(max_input_length=max_input_length)
        else:
            raise ValueError(f"Unsupported model_size {model_size}")
        self._encoder = TextEncoder(backbone)

    @staticmethod
    def _load_instructions(path: str | None) -> Dict[str, str] | None:
        if not path:
            logger.warning("No instructions JSON provided; proceeding without instructions.")
            return None
        p = Path(path)
        if not p.exists():
            logger.warning(f"Instructions file not found at {p}. Proceeding without instructions.")
            return None
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if not all(isinstance(v, str) for v in obj.values()):
            raise ValueError("All values in task_to_instructions must be strings.")
        logger.info(f"Loaded instructions from {p}")
        return obj

    def _encode_texts_with_instruction(self, texts: List[str], sub_task: str) -> np.ndarray:
        # Attach instruction if present
        if self._instructions and sub_task in self._instructions:
            instruction = self._instructions[sub_task]
            texts = [f"{instruction}\n\n{text}" for text in texts]
        embeddings = self._encoder.encode_texts(texts)  # expects (n_samples, emb_dim)
        return embeddings

    def run_evaluation(
        self,
        sub_task: str,
        X_train_texts: List[str],
        X_val_texts: List[str],
        X_test_texts: List[str],
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        run_evaluation_lr,
        lr_solver: str,
        n_jobs: int,
        test_patient_ids: List[int],
    ) -> Tuple[Any, Dict[str, Dict[str, float]]]:
        # 1) Build embeddings
        X_train = self._encode_texts_with_instruction(X_train_texts, sub_task=sub_task)
        X_val   = self._encode_texts_with_instruction(X_val_texts,   sub_task=sub_task)
        X_test  = self._encode_texts_with_instruction(X_test_texts,  sub_task=sub_task)

        # 2) Reuse original training/metrics/CI from 7_eval.py
        model_head = f"lr_{lr_solver}"
        best_model, scores = run_evaluation_lr(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            model_head=model_head, n_jobs=n_jobs, test_patient_ids=test_patient_ids,
        )
        return best_model, scores

# ----------------------
# Main
# ----------------------
def main():
    args = parse_args()

    # Parse lists
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else [
        'guo_los', 'guo_readmission', 'guo_icu',
        'lab_thrombocytopenia', 'lab_hyperkalemia', 'lab_hypoglycemia', 'lab_hyponatremia', 'lab_anemia',
        'new_hypertension', 'new_hyperlipidemia', 'new_pancan', 'new_celiac', 'new_lupus', 'new_acutemi',
        'chexpert_Lung Lesion', 'chexpert_Pneumothorax', 'chexpert_Pleural Effusion', 'chexpert_No Finding', 'chexpert_Support Devices'
    ]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    replicates = [int(x) for x in args.replicates.split(",") if x.strip()]

    # Load evaluation routine
    run_evaluation_lr = import_run_evaluation(args.eval_py)
    logger.info(f"Using evaluation from: {os.path.abspath(args.eval_py)}")

    # Read split mapping
    dtype_dict = {'task': str, 'split_name': str, 'shot_size': int, 'fold': int, 'patient_id': int,
                  'prediction_time': str, 'label_type': str, 'label_value': str, 'serialization_idx': int}
    splits_to_serializations = pd.read_csv(args.splits_to_serializations, dtype=dtype_dict, parse_dates=['prediction_time'])  # type: ignore
    splits_to_serializations['label_value'] = splits_to_serializations['label_value'].apply(lambda x: x == 'True')

    # Load serialized texts
    with open(args.tasks_serializations, 'rb') as f:
        tasks_serializations = pickle.load(f)

    logger.info(f"Loaded {len(splits_to_serializations)} rows of splits and {len(tasks_serializations)} serializations.")

    # Classifier
    clf: LLMClassifier = LLMEncoder(model_size=args.model_size, max_input_length=args.max_input_length, instructions_json=args.instructions_json)
    LABELING_FUNCTION = "llm_encoder"

    # Run loop
    results: List[Dict[str, Any]] = []
    model = "llm"

    for sub_task in tasks:
        for k in ks:
            for replicate in replicates:
                logger.info(f"Task={sub_task} | k={k} | replicate={replicate}")

                task_split = splits_to_serializations[
                    (splits_to_serializations['task'] == sub_task) &
                    (splits_to_serializations['shot_size'] == k) &
                    (splits_to_serializations['fold'] == replicate)
                ]
                train_split = task_split[task_split['split_name'] == 'train']
                val_split   = task_split[task_split['split_name'] == 'val']
                test_split  = splits_to_serializations[
                    (splits_to_serializations['task'] == sub_task) & (splits_to_serializations['split_name'] == 'test')
                ]

                X_train_texts = [tasks_serializations[idx][1] for idx in train_split['serialization_idx'].values]
                X_val_texts   = [tasks_serializations[idx][1] for idx in val_split['serialization_idx'].values]
                X_test_texts  = [tasks_serializations[idx][1] for idx in test_split['serialization_idx'].values]
                y_train = np.array(train_split['label_value'].values)
                y_val   = np.array(val_split['label_value'].values)
                y_test  = np.array(test_split['label_value'].values)

                test_patient_ids = test_split['patient_id'].astype(int).tolist()

                best_model, scores = clf.run_evaluation(
                    sub_task=sub_task,
                    X_train_texts=X_train_texts, X_val_texts=X_val_texts, X_test_texts=X_test_texts,
                    y_train=y_train, y_val=y_val, y_test=y_test,
                    run_evaluation_lr=run_evaluation_lr,
                    lr_solver=args.lr_solver,
                    n_jobs=args.n_jobs,
                    test_patient_ids=test_patient_ids,
                )

                # Flatten scores into rows
                for score_name, score_value in scores.items():
                    results.append({
                        'labeling_function': LABELING_FUNCTION,
                        'sub_task': sub_task,
                        'model': model,
                        'replicate': replicate,
                        'k': k,
                        'score': score_name,
                        'value': score_value.get('score'),
                        'std': score_value.get('std'),
                        'lower': score_value.get('lower'),
                        'mean': score_value.get('mean'),
                        'upper': score_value.get('upper'),
                    })
                logger.info(f"Scores: {scores}")

    # Save
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    if out_path.exists():
        # Append but avoid duplicates by a simple concat + drop_duplicates, preserving first occurrence
        prev = pd.read_csv(out_path)
        df = pd.concat([prev, df], ignore_index=True)
        df = df.drop_duplicates(subset=['labeling_function','sub_task','model','replicate','k','score'], keep='first')
    df.to_csv(out_path, index=False)
    logger.success(f"Wrote results to {out_path} with {len(df)} rows.")

if __name__ == "__main__":
    main()
