#!/usr/bin/env python3
"""
Paired, patient-level bootstrap significance tests for ΔAUROC between two models
from EHRSHOT all_probas.csv outputs.

Expected structure:
  EHRSHOT: <EXP_DIR>/<TASK>/all_probas.csv
  UKBB:    <EXP_DIR>/<TaskName>_all_probas.csv  (all models in one file)

Each all_probas.csv must have columns:
  patient_id,sub_task,model,head,replicate,k,label,proba

Comparison spec format (A|B):
  "Aname:Amodel_in_file:Ahead:k:Aexp_dir|Bname:Bmodel_in_file:Bhead:k:Bexp_dir"

Notes:
- k is NOT inferred. We filter rows with exactly that k. If k=-1, we use k==-1 rows.
- For a given k, we use ALL replicates by averaging probabilities across replicates
  (after validating the test rows match across replicates).
- Paired, patient-level (cluster) bootstrap on ΔAUROC = AUROC(A) - AUROC(B)
- Optional collapse of CheXpert subtasks into a single sub_task 'chexpert'
  When collapsed, the bootstrap computes macro-averaged AUROC (mean of per-subtask
  AUROCs) inside each replicate, matching the macro-average metric reported in the
  paper. This avoids a Simpson's paradox where the pooled (micro) AUROC can disagree
  with the macro-averaged AUROC due to varying subtask prevalences.
- Holm correction across tasks per comparison
- Parallel per-task with --num_threads
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from utils import LABELING_FUNCTION_2_PAPER_NAME

import numpy as np
import pandas as pd
# roc_auc_score is not called directly — we use _fast_weighted_auroc which
# reimplements the same computation with precomputed sort order for speed.
# from sklearn.metrics import roc_auc_score

from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# EHRSHOT tasks
# ============================================================

TASKS_DEFAULT_EHRSHOT = [
    "guo_icu",
    "guo_los",
    "guo_readmission",
    "lab_anemia",
    "lab_hyperkalemia",
    "lab_hypoglycemia",
    "lab_hyponatremia",
    "lab_thrombocytopenia",
    "new_acutemi",
    "new_celiac",
    "new_hyperlipidemia",
    "new_hypertension",
    "new_lupus",
    "new_pancan",
    "chexpert",
]

TASK_DISPLAY_ORDER_EHRSHOT = [
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
    "chexpert",
]

TASK_DISPLAY_NAME_OVERRIDE_EHRSHOT = {
    "guo_icu": "ICU Transfer",
}


# ============================================================
# UKBB tasks
# ============================================================

# Internal task keys (used for CLI, lookups, filtering).
# These match the filename prefixes in the results directory.
UKBB_TASKS_DEFAULT = [
    "Hypertension",
    "Diabetes mellitus",
    "Atrial fibrillation",
    "Pneumonia",
    "Chronic obstructive pulmonary disease [COPD]",
    "Chronic kidney disease",
    "Ischemic heart disease",
    "Myocardial infarction [Heart attack]",
    "Cerebral infarction [Ischemic stroke]",
    "Heart failure",
    "Cardiac arrest",
    "Abdominal aortic aneurysm",
    "Pulmonary embolism",
    "Aortic stenosis",
    "Mitral valve insufficiency",
    "Endocarditis",
    "Rheumatic fever and chronic rheumatic heart diseases",
    "Anemia",
    "Back pain",
    "Parkinson's disease (Primary)",
    "Rheumatoid arthritis",
    "Psoriasis",
    "Suicide ideation and attempt or self harm",
    "death",
    "hospitalization",
]

TASK_DISPLAY_ORDER_UKBB = list(UKBB_TASKS_DEFAULT)  # same order

# Display names for UKBB (key = internal task key, value = display name).
UKBB_TASK_DISPLAY_NAMES: Dict[str, str] = {
    "Hypertension": "Hypertension",
    "Diabetes mellitus": "Diabetes Mellitus",
    "Atrial fibrillation": "Atrial Fibrillation",
    "Pneumonia": "Pneumonia",
    "Chronic obstructive pulmonary disease [COPD]": "COPD",
    "Chronic kidney disease": "Chronic Kidney Disease",
    "Ischemic heart disease": "Ischemic Heart Disease",
    "Myocardial infarction [Heart attack]": "Myocardial Infarction",
    "Cerebral infarction [Ischemic stroke]": "Cerebral Infarction",
    "Heart failure": "Heart Failure",
    "Cardiac arrest": "Cardiac Arrest",
    "Abdominal aortic aneurysm": "Abdominal Aortic Aneurysm",
    "Pulmonary embolism": "Pulmonary Embolism",
    "Aortic stenosis": "Aortic Stenosis",
    "Mitral valve insufficiency": "Mitral Valve Insufficiency",
    "Endocarditis": "Endocarditis",
    "Rheumatic fever and chronic rheumatic heart diseases": "Rheumatic Fever",
    "Anemia": "Anemia",
    "Back pain": "Back Pain",
    "Parkinson's disease (Primary)": "Parkinson's Disease",
    "Rheumatoid arthritis": "Rheumatoid Arthritis",
    "Psoriasis": "Psoriasis",
    "Suicide ideation and attempt or self harm": "Suicide Ideation / Self Harm",
    "death": "Death",
    "hospitalization": "Hospitalization",
}

# Maximum k at which each UKBB task was evaluated (exclusive upper bound for
# *few-shot* settings). Tasks not listed here have no restriction.
# For k=-1 (all), every task is included regardless.
UKBB_TASK_MAX_K: Dict[str, int] = {
    "Cardiac arrest": 32,
    "Abdominal aortic aneurysm": 24,
    "Aortic stenosis": 48,
    "Mitral valve insufficiency": 64,
    "Endocarditis": 24,
    "Rheumatic fever and chronic rheumatic heart diseases": 64,
    "Parkinson's disease (Primary)": 32,
    "Suicide ideation and attempt or self harm": 64,
}


# ============================================================
# Dataset-aware helpers
# ============================================================

# Global dataset mode, set once in main() and read everywhere.
_DATASET: str = "ehrshot"


def task_display_name(task: str) -> str:
    if _DATASET == "ukbb":
        return UKBB_TASK_DISPLAY_NAMES.get(task, task)
    return TASK_DISPLAY_NAME_OVERRIDE_EHRSHOT.get(task, LABELING_FUNCTION_2_PAPER_NAME[task])


def get_task_display_order() -> List[str]:
    if _DATASET == "ukbb":
        return TASK_DISPLAY_ORDER_UKBB
    return TASK_DISPLAY_ORDER_EHRSHOT


def sort_results_by_display_order(results: List) -> List:
    """Sort DeltaResult list according to the active display order."""
    order = {t: i for i, t in enumerate(get_task_display_order())}
    return sorted(results, key=lambda r: order.get(r.task, 999))


def filter_ukbb_tasks_for_k(tasks: List[str], k: int) -> List[str]:
    """Remove UKBB tasks that were not evaluated at shot size k.

    For k=-1 (all data), every task is included.
    For few-shot k>0, exclude tasks whose max evaluated k is < requested k.
    """
    if k == -1:
        return tasks
    return [t for t in tasks if k <= UKBB_TASK_MAX_K.get(t, float("inf"))]


@dataclass(frozen=True)
class ModelSpec:
    display_name: str         # label in output
    model_in_file: str        # value in CSV 'model' column
    head: str                 # value in CSV 'head' column
    k: int                    # fixed k to filter (k=-1 allowed)
    exp_dir: str              # experiment folder


@dataclass(frozen=True)
class ComparisonSpec:
    name: str
    a: ModelSpec
    b: ModelSpec


@dataclass
class DeltaResult:
    task: str
    k: int
    delta: float
    ci_low: float
    ci_high: float
    p: float
    p_adj: float
    n_examples: int
    n_patients: int


# ----------------------------
# Multiple testing: Holm
# ----------------------------

def holm_adjust(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for j, idx in enumerate(order):
        raw = pvals[idx]
        val = (m - j) * raw
        val = min(1.0, max(val, prev))
        adj[idx] = val
        prev = val
    return adj.tolist()


# ----------------------------
# Fast weighted AUROC with precomputed sort order
# ----------------------------

def _precompute_auroc_order(y, proba):
    """Sort predictions once in descending order and find distinct-score boundaries.

    Returns (desc_order, y_sorted, proba_sorted, threshold_idxs) to be reused
    across all bootstrap iterations.
    """
    desc_order = np.argsort(-proba, kind='mergesort')
    y_sorted = y[desc_order].astype(np.float64)
    proba_sorted = proba[desc_order].astype(np.float64)
    distinct = np.nonzero(np.diff(proba_sorted))[0]
    threshold_idxs = np.concatenate([distinct, [len(y_sorted) - 1]])
    return desc_order, y_sorted, proba_sorted, threshold_idxs


def _fast_weighted_auroc(y_sorted, proba_sorted, desc_order, threshold_idxs, sample_weight=None):
    """Compute AUROC using precomputed sort order. Numerically identical to
    sklearn's roc_auc_score but avoids redundant sorting.

    When sample_weight has zeros (rare in patient-bootstrap), the distinct-
    threshold indices are recomputed for the filtered array to match sklearn's
    zero-weight filtering behavior exactly.
    """
    if sample_weight is not None:
        w = sample_weight[desc_order]
        if np.any(w == 0):
            nonzero = w != 0
            ys = y_sorted[nonzero]
            ps = proba_sorted[nonzero]
            w = w[nonzero]
            distinct = np.nonzero(np.diff(ps))[0]
            tidx = np.concatenate([distinct, [len(ys) - 1]])
        else:
            ys = y_sorted
            tidx = threshold_idxs
        tps = np.cumsum(ys * w, dtype=np.float64)[tidx]
        fps = np.cumsum((1.0 - ys) * w, dtype=np.float64)[tidx]
    else:
        tps = np.cumsum(y_sorted, dtype=np.float64)[threshold_idxs]
        fps = 1.0 + threshold_idxs.astype(np.float64) - tps

    tps = np.concatenate([[0.0], tps])
    fps = np.concatenate([[0.0], fps])

    if fps[-1] <= 0 or tps[-1] <= 0:
        raise ValueError("Only one class present or all weight on one class")

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]
    return float(np.trapz(tpr, fpr))


# ----------------------------
# Macro-averaged AUROC helper
# ----------------------------

def _macro_auroc_fast(y, sub_tasks, precomp_a, precomp_b, sample_weight=None):
    """Compute macro-averaged ΔAUROC: mean of per-subtask AUROCs for each model.

    precomp_a / precomp_b are dicts mapping subtask -> (mask, desc_order,
    y_sorted, proba_sorted, threshold_idxs), precomputed once before the
    bootstrap loop.

    Returns (auroc_a, auroc_b) as a tuple.
    """
    aurocs_a = []
    aurocs_b = []
    for s, (mask, do_a, ys_a, ps_a, ti_a) in precomp_a.items():
        _, do_b, ys_b, ps_b, ti_b = precomp_b[s]
        ys = y[mask]
        if len(np.unique(ys)) < 2:
            continue
        ws = sample_weight[mask] if sample_weight is not None else None
        if ws is not None:
            if np.all(ws[ys == 0] == 0) or np.all(ws[ys == 1] == 0):
                continue
        aurocs_a.append(_fast_weighted_auroc(ys_a, ps_a, do_a, ti_a, sample_weight=ws))
        aurocs_b.append(_fast_weighted_auroc(ys_b, ps_b, do_b, ti_b, sample_weight=ws))
    if len(aurocs_a) == 0:
        raise RuntimeError("No subtask had both classes present for macro AUROC.")
    return float(np.mean(aurocs_a)), float(np.mean(aurocs_b))


# ----------------------------
# Bootstrap test
# ----------------------------

def paired_patient_bootstrap_delta_auroc(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    patient_ids: np.ndarray,
    B: int,
    seed: int,
    sub_tasks: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Paired, patient-level (cluster) bootstrap on ΔAUROC = AUROC(A) - AUROC(B).

    If sub_tasks is provided, AUROC is computed as the macro average over subtasks
    (mean of per-subtask AUROCs) inside each bootstrap replicate. This is used for
    collapsed CheXpert so the bootstrap statistic matches the macro-averaged AUROC
    reported in the paper, avoiding Simpson's paradox from pooled/micro AUROC.

    Sort orders are precomputed once before the loop so each bootstrap iteration
    only performs O(n) cumulative sums instead of O(n log n) sorting.
    """
    use_macro = sub_tasks is not None

    # Precompute sort orders once
    if use_macro:
        unique_subs = np.unique(sub_tasks)
        precomp_a = {}
        precomp_b = {}
        for s in unique_subs:
            mask = sub_tasks == s
            precomp_a[s] = (mask, *_precompute_auroc_order(y[mask], proba_a[mask]))
            precomp_b[s] = (mask, *_precompute_auroc_order(y[mask], proba_b[mask]))
        da_obs, db_obs = _macro_auroc_fast(y, sub_tasks, precomp_a, precomp_b)
        delta_obs = da_obs - db_obs
    else:
        do_a, ys_a, ps_a, ti_a = _precompute_auroc_order(y, proba_a)
        do_b, ys_b, ps_b, ti_b = _precompute_auroc_order(y, proba_b)
        delta_obs = (
            _fast_weighted_auroc(ys_a, ps_a, do_a, ti_a)
            - _fast_weighted_auroc(ys_b, ps_b, do_b, ti_b)
        )

    unique_pids = np.unique(patient_ids)
    P = len(unique_pids)
    rng = np.random.default_rng(seed)

    pid_to_int = {pid: i for i, pid in enumerate(unique_pids)}
    pid_ints = np.array([pid_to_int[pid] for pid in patient_ids], dtype=np.int32)

    deltas = []
    for _ in range(B):
        sampled = rng.choice(P, size=P, replace=True)
        counts = np.bincount(sampled, minlength=P).astype(np.float64)
        w = counts[pid_ints]

        # Guard against degenerate single-class data (rare)
        if np.all(y == 0) or np.all(y == 1):
            continue

        try:
            if use_macro:
                da, db = _macro_auroc_fast(y, sub_tasks, precomp_a, precomp_b, sample_weight=w)
            else:
                da = _fast_weighted_auroc(ys_a, ps_a, do_a, ti_a, sample_weight=w)
                db = _fast_weighted_auroc(ys_b, ps_b, do_b, ti_b, sample_weight=w)
            deltas.append(da - db)
        except (ValueError, RuntimeError):
            # bootstrap sample may have only one class in a subtask; skip
            continue

    if len(deltas) < max(100, B // 10):
        raise RuntimeError(f"Too few valid bootstrap replicates ({len(deltas)}/{B}).")

    deltas = np.array(deltas, dtype=np.float64)
    lo, hi = np.percentile(deltas, [2.5, 97.5])

    # two-sided sign test on bootstrap deltas (+1 correction)
    p_le = (1.0 + np.sum(deltas <= 0.0)) / (len(deltas) + 1.0)
    p_ge = (1.0 + np.sum(deltas >= 0.0)) / (len(deltas) + 1.0)
    p = min(1.0, 2.0 * min(p_le, p_ge))

    return float(delta_obs), float(lo), float(hi), float(p)


# ----------------------------
# IO / filtering / alignment
# ----------------------------

def read_all_probas(exp_dir: str, task: str, dataset: str = None) -> pd.DataFrame:
    """Read all_probas CSV. Handles both EHRSHOT and UKBB layouts."""
    ds = dataset if dataset is not None else _DATASET
    if ds == "ukbb":
        # UKBB: flat directory with <TaskName>_all_probas.csv
        path = os.path.join(exp_dir, f"{task}_all_probas.csv")
    else:
        # EHRSHOT: subdirectory per task
        path = os.path.join(exp_dir, task, "all_probas.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    req = {"patient_id", "sub_task", "model", "head", "replicate", "k", "label", "proba"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")

    # UKBB labels may be True/False strings — convert to int
    if df["label"].dtype == object or df["label"].dtype == bool:
        df["label"] = df["label"].map({True: 1, False: 0, "True": 1, "False": 0})
        if df["label"].isna().any():
            raise ValueError(f"Unexpected label values in {path}: {df['label'].unique()}")
        df["label"] = df["label"].astype(int)

    return df


def collapse_chexpert(df: pd.DataFrame, task: str, do_collapse: bool) -> pd.DataFrame:
    if not do_collapse or task != "chexpert":
        return df
    df = df.copy()
    # Preserve original subtask labels for macro-averaged bootstrap.
    # The "sub_task" column is overwritten to "chexpert" so that filter_df
    # treats all rows as one task, but "original_sub_task" is kept so we
    # can compute per-subtask AUROCs inside bootstrap replicates.
    df["original_sub_task"] = df["sub_task"]
    df["sub_task"] = "chexpert"
    return df


def filter_df(df: pd.DataFrame, spec: ModelSpec, task: str, collapse_chexpert_flag: bool) -> pd.DataFrame:
    df = collapse_chexpert(df, task, collapse_chexpert_flag)
    df = df[(df["model"] == spec.model_in_file) & (df["head"] == spec.head) & (df["k"] == spec.k)].copy()
    if df.empty:
        raise ValueError(
            f"No rows after filtering task={task}, model={spec.model_in_file}, head={spec.head}, k={spec.k} "
            f"in {spec.exp_dir}"
        )

    # pick the dominant sub_task (except chexpert where we forced 'chexpert')
    sub = df["sub_task"].value_counts().idxmax()
    df = df[df["sub_task"] == sub].copy()
    if df.empty:
        raise ValueError(f"No rows left after sub_task filter for task={task} in {spec.exp_dir}")
    return df


def align_two(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align rows A vs B to (y, proba_a, proba_b, patient_ids).

    We first try exact row order. If that fails, we try sorting by (patient_id, label).
    If that fails, we error (no example_id available).
    """
    a = df_a.reset_index(drop=True)
    b = df_b.reset_index(drop=True)

    if len(a) != len(b):
        raise RuntimeError(f"Row count mismatch: {len(a)} vs {len(b)}")

    pid_a = a["patient_id"].to_numpy()
    pid_b = b["patient_id"].to_numpy()

    y_a = a["label"].to_numpy()
    y_b = b["label"].to_numpy()

    if not np.array_equal(pid_a, pid_b):
        raise RuntimeError("patient_id mismatch between models")

    if not np.array_equal(y_a, y_b):
        raise RuntimeError("label mismatch between models")

    return y_a.astype(int), a["proba"].to_numpy(float), b["proba"].to_numpy(float), pid_a

def mean_proba_over_replicates(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Returns (y, patient_ids, mean_proba, original_sub_tasks) averaged over all replicates.
    Requires that all replicates contain the same (patient_id, label) rows in the same order
    (or at least sortable to the same order).

    original_sub_tasks is non-None only when the "original_sub_task" column exists
    (i.e. for collapsed CheXpert), and is used downstream for macro-averaged bootstrap.
    """
    has_orig_sub = "original_sub_task" in df.columns

    reps = sorted(df["replicate"].unique().tolist())
    ys, pids, probas, orig_subs = [], [], [], []

    for r in reps:
        d = df[df["replicate"] == r].copy()
        d = d.reset_index(drop=True)
        ys.append(d["label"].to_numpy())
        pids.append(d["patient_id"].to_numpy())
        probas.append(d["proba"].to_numpy(float))
        if has_orig_sub:
            orig_subs.append(d["original_sub_task"].to_numpy())

    # Validate identical rows across replicates (strong check)
    y0, pid0 = ys[0], pids[0]
    orig_sub0 = orig_subs[0] if has_orig_sub else None
    for i in range(1, len(reps)):
        yr, pidr = ys[i], pids[i]
        if not (np.array_equal(y0, yr) and np.array_equal(pid0, pidr)):
            # try sorting within each replicate by (patient_id,label) to recover
            sort_cols = ["patient_id", "label"]
            if has_orig_sub:
                sort_cols.append("original_sub_task")
            ys2, pids2, probas2, orig_subs2 = [], [], [], []
            for r in reps:
                d = df[df["replicate"] == r].sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
                ys2.append(d["label"].to_numpy())
                pids2.append(d["patient_id"].to_numpy())
                probas2.append(d["proba"].to_numpy(float))
                if has_orig_sub:
                    orig_subs2.append(d["original_sub_task"].to_numpy())
            y0, pid0 = ys2[0], pids2[0]
            orig_sub0 = orig_subs2[0] if has_orig_sub else None
            ok = True
            for j in range(1, len(reps)):
                yr2, pidr2 = ys2[j], pids2[j]
                if not (np.array_equal(y0, yr2) and np.array_equal(pid0, pidr2)):
                    ok = False
                    break
            if not ok:
                raise RuntimeError(
                    "Replicates do not share the same test rows (patient_id/label). "
                    "Need example_id to average per-example across replicates."
                )
            ys, pids, probas, orig_subs = ys2, pids2, probas2, orig_subs2
            break

    # CAREFUL: Average probabilities across replicates (not labels) to get mean proba across shot splits
    mean_proba = np.mean(np.stack(probas, axis=0), axis=0)
    return y0.astype(int), pid0, mean_proba, orig_sub0


# ----------------------------
# Per-task computation (worker)
# ----------------------------

def compute_one_task(
    comp_name: str,
    task: str,
    spec_a: ModelSpec,
    spec_b: ModelSpec,
    bootstrap: int,
    seed: int,
    collapse_chexpert_flag: bool,
    dataset: str = "ehrshot",
) -> Optional[DeltaResult]:
    try:
        df_a = filter_df(read_all_probas(spec_a.exp_dir, task, dataset=dataset), spec_a, task, collapse_chexpert_flag)
        df_b = filter_df(read_all_probas(spec_b.exp_dir, task, dataset=dataset), spec_b, task, collapse_chexpert_flag)
    except (ValueError, FileNotFoundError) as e:
        print(f"  [SKIP] {comp_name} / {task}: {e}", file=sys.stderr)
        return None

    # Use all replicates by averaging probabilities per test row
    y_a, pid_a, proba_a, orig_sub_a = mean_proba_over_replicates(df_a)
    y_b, pid_b, proba_b, orig_sub_b = mean_proba_over_replicates(df_b)

    # Align A vs B (should match; if not, attempt sort-based alignment)
    tmp_a = pd.DataFrame({"patient_id": pid_a, "label": y_a, "proba": proba_a})
    tmp_b = pd.DataFrame({"patient_id": pid_b, "label": y_b, "proba": proba_b})
    y, pa, pb, pids = align_two(tmp_a, tmp_b)

    # For collapsed CheXpert, pass original subtask labels so the bootstrap
    # computes macro-averaged AUROC (mean of per-subtask AUROCs) per replicate,
    # matching the metric reported in the paper.
    sub_tasks_for_bootstrap = None
    if orig_sub_a is not None:
        # Validate that both models share the same subtask labels in the same order
        if orig_sub_b is None or not np.array_equal(orig_sub_a, orig_sub_b):
            raise RuntimeError("original_sub_task mismatch between models A and B for chexpert")
        sub_tasks_for_bootstrap = orig_sub_a

    delta, lo, hi, p = paired_patient_bootstrap_delta_auroc(
        y=y,
        proba_a=pa,
        proba_b=pb,
        patient_ids=pids,
        B=bootstrap,
        seed=seed,
        sub_tasks=sub_tasks_for_bootstrap,
    )

    return DeltaResult(
        task=task,
        k=spec_a.k,
        delta=delta,
        ci_low=lo,
        ci_high=hi,
        p=p,
        p_adj=np.nan,
        n_examples=int(len(y)),
        n_patients=int(len(np.unique(pids))),
    )


# ----------------------------
# Parsing / output
# ----------------------------

def parse_model_spec(s: str) -> ModelSpec:
    # "Name:model_in_file:head:k:exp_dir"  (exp_dir may contain ':', so split max 4)
    parts = s.split(":", 4)
    if len(parts) != 5:
        raise ValueError(f"Bad model spec: '{s}'. Expected 5 fields: Name:model:head:k:exp_dir")
    name, model_in_file, head, k_str, exp_dir = parts
    try:
        k = int(k_str)
    except ValueError:
        raise ValueError(f"Bad k in spec '{s}': '{k_str}' is not an int")
    return ModelSpec(name, model_in_file, head, k, exp_dir)


def parse_comparison(s: str) -> ComparisonSpec:
    # "A...|B..."
    if "|" not in s:
        raise ValueError(f"Bad comparison: '{s}'. Missing '|' between A and B.")
    left, right = s.split("|", 1)
    a = parse_model_spec(left)
    b = parse_model_spec(right)
    if a.k != b.k:
        raise ValueError(f"k mismatch in comparison: {a.k} vs {b.k}. (You said one k per run.)")
    name = f"{a.display_name}_vs_{b.display_name}_k{a.k}"
    return ComparisonSpec(name=name, a=a, b=b)


def latex_escape(x: str) -> str:
    return x.replace("_", r"\_")


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}"


def fmt_compact_p(p: float) -> str:
    """Format p-value for the compact table."""
    if p < 0.001:
        return "{<}0.001"
    return f"{p:.3f}"


def autodiscover_tasks(exp_dir: str) -> List[str]:
    if _DATASET == "ukbb":
        # UKBB: flat directory, files named <TaskName>_all_probas.csv
        tasks = []
        suffix = "_all_probas.csv"
        for f in sorted(os.listdir(exp_dir)):
            if f.endswith(suffix):
                task_name = f[: -len(suffix)]
                tasks.append(task_name)
        if not tasks:
            raise RuntimeError(f"Autodiscover found no UKBB tasks under {exp_dir}")
        return tasks
    else:
        tasks = []
        for t in sorted(os.listdir(exp_dir)):
            if os.path.isfile(os.path.join(exp_dir, t, "all_probas.csv")):
                tasks.append(t)
        if not tasks:
            raise RuntimeError(f"Autodiscover found no tasks under {exp_dir}")
        return tasks


def compute_comparison(
    comp: ComparisonSpec,
    tasks: List[str],
    bootstrap: int,
    seed: int,
    collapse_chexpert_flag: bool,
    num_threads: int,
) -> List[DeltaResult]:
    results: List[DeltaResult] = []

    # For UKBB, filter out tasks that were not evaluated at this k
    if _DATASET == "ukbb":
        tasks = filter_ukbb_tasks_for_k(tasks, comp.a.k)

    if num_threads <= 1:
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(tasks, desc=comp.name, unit="task")
        for task in iterator:
            results.append(compute_one_task(comp.name, task, comp.a, comp.b, bootstrap, seed, collapse_chexpert_flag, dataset=_DATASET))
    else:
        with ProcessPoolExecutor(max_workers=num_threads) as ex:
            futs = [
                ex.submit(compute_one_task, comp.name, task, comp.a, comp.b, bootstrap, seed, collapse_chexpert_flag, _DATASET)
                for task in tasks
            ]
            iterator = as_completed(futs)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futs), desc=comp.name, unit="task")
            for fut in iterator:
                results.append(fut.result())

    return results


def main() -> int:
    global _DATASET

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--comparisons",
        nargs="+",
        required=True,
        help="One or more comparison strings: Aname:model:head:k:dir|Bname:model:head:k:dir",
    )
    ap.add_argument("--tasks", nargs="+", default=["autodiscover"], help="Task folders or 'autodiscover'")
    ap.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap replicates (B)")
    ap.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed")
    ap.add_argument("--num_threads", type=int, default=1, help="Parallel workers over tasks")
    ap.add_argument("--collapse_chexpert", action="store_true", default=True, help="Collapse chexpert subtasks")
    ap.add_argument("--no_collapse_chexpert", dest="collapse_chexpert", action="store_false")

    # Dataset selection
    ds_group = ap.add_mutually_exclusive_group(required=True)
    ds_group.add_argument("--ehrshot", action="store_const", const="ehrshot", dest="dataset")
    ds_group.add_argument("--ukbb", action="store_const", const="ukbb", dest="dataset")

    args = ap.parse_args()
    _DATASET = args.dataset

    comps = [parse_comparison(s) for s in args.comparisons]

    if args.tasks == ["autodiscover"]:
        if _DATASET == "ukbb":
            tasks = list(UKBB_TASKS_DEFAULT)
        else:
            # autodiscover from first model exp_dir
            tasks = autodiscover_tasks(comps[0].a.exp_dir)
    else:
        tasks = args.tasks

    # 1) compute all results first
    all_results_by_comp: Dict[str, List[DeltaResult]] = {}
    flat: List[DeltaResult] = []

    for comp in comps:
        res = compute_comparison(
            comp=comp,
            tasks=tasks,
            bootstrap=args.bootstrap,
            seed=args.seed,
            collapse_chexpert_flag=args.collapse_chexpert,
            num_threads=args.num_threads,
        )
        all_results_by_comp[comp.name] = res
        flat.extend(res)

    # 2) Per-k Holm correction: group all comparisons sharing the same k and
    #    correct across (comparisons × tasks) within each k-level. This treats
    #    each shot setting as a separate family of tests (e.g. 3 baselines ×
    #    15 tasks = 45 tests per k), which is appropriate when different k
    #    values represent distinct experimental conditions.
    k_groups: Dict[int, List[DeltaResult]] = {}
    for r in flat:
        k_groups.setdefault(r.k, []).append(r)
    for k, results_in_k in k_groups.items():
        pvals = [r.p for r in results_in_k]
        print(f"\n% Holm correction for k={k} for {len(pvals)} tests.")
        padj = holm_adjust(pvals)
        for r, pa in zip(results_in_k, padj):
            r.p_adj = float(pa)

    # 4) Compact tables: one per k, one row per task, one column per baseline.
    result_index: Dict[Tuple[str, int, str], DeltaResult] = {}
    baseline_names_seen: Dict[int, List[str]] = {}
    for comp in comps:
        bname = comp.b.display_name
        k = comp.a.k
        baseline_names_seen.setdefault(k, [])
        if bname not in baseline_names_seen[k]:
            baseline_names_seen[k].append(bname)
        for r in all_results_by_comp[comp.name]:
            result_index[(bname, k, r.task)] = r

    display_order = get_task_display_order()

    for k in sorted(k_groups.keys()):
        baselines = baseline_names_seen.get(k, [])
        if not baselines:
            continue
        k_label = "All" if k == -1 else f"{k}-shot"
        print(f"\n% Compact table: k={k} ({k_label})")
        header_parts = " & ".join([f"$\\Delta$ vs {b}" for b in baselines])
        print(f"% Task & {header_parts} \\\\")
        for task_key in display_order:
            tname = task_display_name(task_key)
            cells = []
            for bname in baselines:
                r = result_index.get((bname, k, task_key))
                if r is None:
                    cells.append("---")
                    continue
                d = f"{r.delta:+.3f}"
                ci_low = f"{r.ci_low:+.3f}"
                ci_high = f"{r.ci_high:+.3f}"
                p = fmt_compact_p(r.p_adj)

                if r.p_adj < 0.05:
                    cells.append(f"\\bestcip{{{d}}}{{{ci_low}}}{{{ci_high}}}{{{p}}}")
                else:
                    cells.append(f"\\estcip{{{d}}}{{{ci_low}}}{{{ci_high}}}{{{p}}}")
            print(f"{tname} & {' & '.join(cells)} \\\\")

    return 0


if __name__ == "__main__":
    sys.exit(main())