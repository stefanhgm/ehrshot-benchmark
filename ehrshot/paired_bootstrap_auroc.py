#!/usr/bin/env python3
"""
Paired, patient-level bootstrap significance tests for ΔAUROC between two models
from EHRSHOT all_probas.csv outputs.

Expected structure:
  <EXP_DIR>/<TASK>/all_probas.csv

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
from sklearn.metrics import roc_auc_score

from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


TASKS_DEFAULT = [
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
# Bootstrap test
# ----------------------------

def paired_patient_bootstrap_delta_auroc(
    y: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    patient_ids: np.ndarray,
    B: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    delta_obs = roc_auc_score(y, proba_a) - roc_auc_score(y, proba_b)

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

        # roc_auc_score errors if only one class overall (rare, but guard anyway)
        if np.all(y == 0) or np.all(y == 1):
            continue

        da = roc_auc_score(y, proba_a, sample_weight=w)
        db = roc_auc_score(y, proba_b, sample_weight=w)
        deltas.append(da - db)

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

def read_all_probas(exp_dir: str, task: str) -> pd.DataFrame:
    path = os.path.join(exp_dir, task, "all_probas.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    req = {"patient_id", "sub_task", "model", "head", "replicate", "k", "label", "proba"}
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")
    return df


def collapse_chexpert(df: pd.DataFrame, task: str, do_collapse: bool) -> pd.DataFrame:
    if not do_collapse or task != "chexpert":
        return df
    df = df.copy()
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

def mean_proba_over_replicates(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y, patient_ids, mean_proba) averaged over all replicates.
    Requires that all replicates contain the same (patient_id, label) rows in the same order
    (or at least sortable to the same order).
    """
    reps = sorted(df["replicate"].unique().tolist())
    ys, pids, probas = [], [], []

    for r in reps:
        d = df[df["replicate"] == r].copy()
        d = d.reset_index(drop=True)
        ys.append(d["label"].to_numpy())
        pids.append(d["patient_id"].to_numpy())
        probas.append(d["proba"].to_numpy(float))

    # Validate identical rows across replicates (strong check)
    y0, pid0 = ys[0], pids[0]
    for yr, pidr in zip(ys[1:], pids[1:]):
        if not (np.array_equal(y0, yr) and np.array_equal(pid0, pidr)):
            # try sorting within each replicate by (patient_id,label) to recover
            ys2, pids2, probas2 = [], [], []
            for r in reps:
                d = df[df["replicate"] == r].sort_values(["patient_id", "label"], kind="mergesort").reset_index(drop=True)
                ys2.append(d["label"].to_numpy())
                pids2.append(d["patient_id"].to_numpy())
                probas2.append(d["proba"].to_numpy(float))
            y0, pid0 = ys2[0], pids2[0]
            ok = True
            for yr2, pidr2 in zip(ys2[1:], pids2[1:]):
                if not (np.array_equal(y0, yr2) and np.array_equal(pid0, pidr2)):
                    ok = False
                    break
            if not ok:
                raise RuntimeError(
                    "Replicates do not share the same test rows (patient_id/label). "
                    "Need example_id to average per-example across replicates."
                )
            ys, pids, probas = ys2, pids2, probas2
            break

    # CAREFUL: Average probabilities across replicates (not labels) to get mean proba across shot splits
    mean_proba = np.mean(np.stack(probas, axis=0), axis=0)
    return y0.astype(int), pid0, mean_proba


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
) -> DeltaResult:
    df_a = filter_df(read_all_probas(spec_a.exp_dir, task), spec_a, task, collapse_chexpert_flag)
    df_b = filter_df(read_all_probas(spec_b.exp_dir, task), spec_b, task, collapse_chexpert_flag)

    # Use all replicates by averaging probabilities per test row
    y_a, pid_a, proba_a = mean_proba_over_replicates(df_a)
    y_b, pid_b, proba_b = mean_proba_over_replicates(df_b)

    # Align A vs B (should match; if not, attempt sort-based alignment)
    tmp_a = pd.DataFrame({"patient_id": pid_a, "label": y_a, "proba": proba_a})
    tmp_b = pd.DataFrame({"patient_id": pid_b, "label": y_b, "proba": proba_b})
    y, pa, pb, pids = align_two(tmp_a, tmp_b)

    delta, lo, hi, p = paired_patient_bootstrap_delta_auroc(
        y=y,
        proba_a=pa,
        proba_b=pb,
        patient_ids=pids,
        B=bootstrap,
        seed=seed,
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
    name = f"{a.display_name}_vs_{b.display_name}"
    return ComparisonSpec(name=name, a=a, b=b)


def latex_escape(x: str) -> str:
    return x.replace("_", r"\_")


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return r"$<10^{-4}$"
    return f"{p:.4f}"


def autodiscover_tasks(exp_dir: str) -> List[str]:
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

    if num_threads <= 1:
        iterator = tasks
        if tqdm is not None:
            iterator = tqdm(tasks, desc=comp.name, unit="task")
        for task in iterator:
            results.append(compute_one_task(comp.name, task, comp.a, comp.b, bootstrap, seed, collapse_chexpert_flag))
    else:
        with ProcessPoolExecutor(max_workers=num_threads) as ex:
            futs = [
                ex.submit(compute_one_task, comp.name, task, comp.a, comp.b, bootstrap, seed, collapse_chexpert_flag)
                for task in tasks
            ]
            iterator = as_completed(futs)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futs), desc=comp.name, unit="task")
            for fut in iterator:
                results.append(fut.result())

    return results


def main() -> int:
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
    args = ap.parse_args()

    comps = [parse_comparison(s) for s in args.comparisons]

    if args.tasks == ["autodiscover"]:
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

    # 2) global Holm across ALL (task × comparison) tests
    all_p = [r.p for r in flat]
    all_p_adj = holm_adjust(all_p)
    for r, pa in zip(flat, all_p_adj):
        r.p_adj = float(pa)

    # 3) print (now r.p_adj is globally adjusted)
    for comp in comps:
        res = all_results_by_comp[comp.name]

        print(f"\n% Comparison: {comp.name}  (Δ = A - B)")
        print(f"% A={comp.a}  B={comp.b}")
        print(r"% Columns: task & ΔAUROC & [CI_low, CI_high] & p & p_adj(Holm-global) & N_examples & N_patients \\")
        for r in res:
            task = LABELING_FUNCTION_2_PAPER_NAME[r.task]
            p_adj = fmt_p(r.p_adj)
            if r.p_adj < 0.05:
                p_adj = r"\textbf{" + p_adj + "}"
            print(
                f"{task} & {r.delta:+.4f} & "
                f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}] & "
                f"{fmt_p(r.p)} & {p_adj} & {r.n_examples} & {r.n_patients} \\\\"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())