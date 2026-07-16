#!/usr/bin/env python3
"""
Merge individual experiment results into a single CSV file and create comparison plots
"""

import argparse
import glob
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from contextlib import contextmanager
import pandas as pd
import seaborn as sns
from itertools import cycle

import warnings

warnings.filterwarnings('ignore')


MODEL_METADATA: Dict[str, Dict[str, str]] = {
    "Qwen_Qwen3-0.6B": {
        "hf_model": "Qwen/Qwen3-0.6B",
        "model_family": "Decoder",
        "model_size": "0.6B",
        "model_display": "Qwen3-0.6B",
    },
    "Qwen_Qwen3-8B": {
        "hf_model": "Qwen/Qwen3-8B",
        "model_family": "Decoder",
        "model_size": "8B",
        "model_display": "Qwen3-8B",
        "model_variant": "Qwen3-8B (LoRA)",
    },
    "Qwen_Qwen3-Embedding-0.6B": {
        "hf_model": "Qwen/Qwen3-Embedding-0.6B",
        "model_family": "Encoder",
        "model_size": "0.6B",
        "model_display": "Qwen3-Embedding-0.6B",
    },
    "Qwen_Qwen3-Embedding-8B": {
        "hf_model": "Qwen/Qwen3-Embedding-8B",
        "model_family": "Encoder",
        "model_size": "8B",
        "model_display": "Qwen3-Embedding-8B",
        "model_variant": "Qwen3-Emb-8B (LoRA)",
    },
    "Qwen_Qwen3-Embedding-8B_Frozen": {
        "hf_model": "Qwen/Qwen3-Embedding-8B",
        "model_family": "Encoder",
        "model_size": "8B",
        "model_display": "Qwen3-Embedding-8B",
        "model_variant": "Qwen3-Emb-8B",
    },
}

MODEL_TYPE_MAPPING: Dict[str, str] = {
    "llm_encoder_ft": "Encoder Fine-tune",
    "llm_decoder_ft": "Decoder Fine-tune",
    "llm_encoder_frozen": "Encoder Frozen",
    "llm_decoder_zero_shot": "Decoder Zero-Shot",
    "llm_decoder_icl": "Decoder ICL",
}

TABLEAU_PALETTE_SEQUENCE: List[str] = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

MARKER_SEQUENCE: List[str] = ['X', 'o', 'h', '^', '*']

PAPER_STYLE = 'seaborn-v0_8-paper'
PAPER_RC_PARAMS = {
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
}

DEFAULT_FIGSIZE_SINGLE = (7.0, 4.0)
FACET_WIDTH_PER_COL = 4.135
FACET_HEIGHT_PER_ROW = 3.6
BAR_FIG_WIDTH = 6.0
BAR_HEIGHT_PER_TASK = 0.35

SINGLE_PANEL_RC = {
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 180,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
}

GROUP_FACET_RC = {
    'font.size': 8,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'lines.linewidth': 2.0,
}


GROUP_TITLE_FONT_SIZE = 12
GROUP_AXIS_LABEL_FONT_SIZE = 10
GROUP_TICK_FONT_SIZE = 9
GROUP_LEGEND_FONT_SIZE = 9
GROUP_SUPTITLE_FONT_SIZE = 18


@dataclass(frozen=True)
class MetricPlotConfig:
    score: str
    display_name: str
    file_suffix: str
    overall_title: str
    task_group_title: str
    per_task_bar_title: str
    per_task_line_title: str


METRIC_PLOT_CONFIGS: Tuple[MetricPlotConfig, ...] = (
    MetricPlotConfig(
        score='auroc',
        display_name='AUROC',
        file_suffix='',
        overall_title='Macro AUROC Across EHRSHOT Task Groups',
        task_group_title='Mean AUROC by Task Group',
        per_task_bar_title='Per-task Comparison (Qwen3 8B, 1-shot)',
        per_task_line_title='Per-task Trajectories (Qwen3 8B)',
    ),
    MetricPlotConfig(
        score='auprc',
        display_name='AUPRC',
        file_suffix='_auprc',
        overall_title='Macro AUPRC Across EHRSHOT Task Groups',
        task_group_title='Mean AUPRC by Task Group',
        per_task_bar_title='Per-task Comparison (Qwen3 8B, 1-shot, AUPRC)',
        per_task_line_title='Per-task Trajectories (Qwen3 8B, AUPRC)',
    ),
    MetricPlotConfig(
        score='brier',
        display_name='Brier Score',
        file_suffix='_brier',
        overall_title='Macro Brier Score Across EHRSHOT Task Groups',
        task_group_title='Mean Brier Score by Task Group',
        per_task_bar_title='Per-task Comparison (Qwen3 8B, 1-shot, Brier Score)',
        per_task_line_title='Per-task Trajectories (Qwen3 8B, Brier Score)',
    ),
)

TITLE_FONT_SIZE = 24
AXIS_LABEL_FONT_SIZE = 18
TICK_FONT_SIZE = 16
LEGEND_FONT_SIZE = 16
SUBPLOT_TITLE_FONT_SIZE = 20
SUPTITLE_FONT_SIZE = 24


@contextmanager
def paper_style(rc_override: Optional[Dict[str, object]] = None):
    """Context manager applying the manuscript plot style consistently."""
    params = PAPER_RC_PARAMS.copy()
    if rc_override:
        params.update(rc_override)
    with plt.style.context(PAPER_STYLE), mpl.rc_context(params):
        yield


def _set_seaborn_theme():
    """Apply seaborn theme aligned with manuscript styling."""
    sns.set_theme(style='ticks', context='paper', rc=PAPER_RC_PARAMS)

SHOT_ORDER = [-1, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128, 256]

K128_TARGET_MODELS = {
    "Qwen_Qwen3-8B",
    "Qwen_Qwen3-Embedding-8B",
    "Qwen_Qwen3-Embedding-8B_Frozen",
}

TASK_GROUP_DISPLAY_NAMES = {
    "Critical Care": "Operational Outcomes",
    "Laboratory": "Anticipating Lab Test Results",
    "New Diagnoses": "Assignment of New Diagnoses",
    "CheXpert": "Anticipating Chest X-ray Findings",
}

TABLE_COLUMN_ORDER: List[str] = [
    "Operational Outcomes",
    "Anticipating Lab Test Results",
    "Assignment of New Diagnoses",
    "Anticipating Chest X-ray Findings",
]

TABLE_SECTION_ORDER: List[str] = [
    "Encoder models",
    "Decoder models",
]

QWEN_EMB_BASE = "Qwen3-Emb-8B"
QWEN_EMB_LORA = "Qwen3-Emb-8B (LoRA)"
QWEN_DECODER_ICL = "Qwen3-8B (ICL)"
QWEN_DECODER_LORA = "Qwen3-8B (LoRA)"

MODEL_VARIANT_DISPLAY_ORDER: List[str] = [
    QWEN_EMB_BASE,
    QWEN_EMB_LORA,
    QWEN_DECODER_ICL,
    QWEN_DECODER_LORA,
]

TABLE_VARIANT_ORDER = {label: idx for idx, label in enumerate(MODEL_VARIANT_DISPLAY_ORDER)}

MODEL_VARIANT_ORDER = TABLE_VARIANT_ORDER

TABLE_VARIANT_CONFIG: Dict[str, Dict[str, object]] = {
    QWEN_EMB_BASE: {"section": "Encoder models"},
    QWEN_EMB_LORA: {"section": "Encoder models"},
    QWEN_DECODER_ICL: {"section": "Decoder models"},
    QWEN_DECODER_LORA: {"section": "Decoder models"},
}

TUNING_SECTION_DISPLAY: Dict[str, str] = {
    "llm_encoder_ft": "Encoder models",
    "llm_decoder_ft": "Decoder models",
}

TUNING_SECTION_ORDER: List[str] = [
    "llm_encoder_ft",
    "llm_decoder_ft",
]

MODEL_TYPE_ORDER: List[str] = [
    "Decoder Fine-tune",
    "Decoder ICL",
    "Encoder Fine-tune",
    "Encoder Frozen",
    "Decoder Zero-Shot",
]

MODEL_TYPE_DISPLAY_NAMES = {
    "Decoder Fine-tune": "Decoder Fine-tune",
    "Decoder ICL": "Decoder ICL",
    "Encoder Fine-tune": "Encoder Fine-tune",
    "Encoder Frozen": "Encoder Frozen",
    "Decoder Zero-Shot": "Decoder Zero-Shot",
}

TABLE_VARIANT_RENAMES: Dict[tuple, str] = {
    ("Decoder Fine-tune", "Decoder (8B)"): QWEN_DECODER_LORA,
    ("Encoder Fine-tune", "Encoder (8B)"): QWEN_EMB_LORA,
    ("Encoder Frozen", "Encoder Frozen (8B)"): QWEN_EMB_BASE,
}

LEGACY_VARIANT_NAME_MAP: Dict[str, str] = {
    "Qwen3-Emb-8B": QWEN_EMB_BASE,
    "Qwen3-Emb-8B (LoRA)": QWEN_EMB_LORA,
    "Qwen-Emb-8B": QWEN_EMB_BASE,
    "Qwen-Emb-8B (LoRA)": QWEN_EMB_LORA,
    "Qwen3-8B (Zero Shot)": QWEN_DECODER_ICL,
    "Qwen3-8B (LoRA)": QWEN_DECODER_LORA,
    "Qwen3-8b (Zero Shot)": QWEN_DECODER_ICL,
    "Qwen3-8b (LoRA)": QWEN_DECODER_LORA,
    "Qwen3-8B (Zero-Shot)": QWEN_DECODER_ICL,
    "Qwen3-8B (ICL 2-shot)": QWEN_DECODER_ICL,
    "Qwen3-8B (ICL 4-shot)": QWEN_DECODER_ICL,
    "Qwen3-8B (ICL 6-shot)": QWEN_DECODER_ICL,
}


def _normalize_result_frame(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
    """Attach revision-specific source metadata and normalize decoder ICL rows."""

    df = df.copy()
    source_path = Path(file_path)
    experiment_source = source_path.parent.name
    df["experiment_source"] = experiment_source

    if experiment_source == "decoder_icl":
        if "icl_shots" in df.columns:
            df["icl_shots"] = pd.to_numeric(df["icl_shots"], errors="coerce")
            decoder_shots = df["icl_shots"].fillna(0).astype(int)
            df["k"] = decoder_shots
            df["model_type"] = "Decoder ICL"
            df["model_variant"] = QWEN_DECODER_ICL
        else:
            df["model_type"] = "Decoder ICL"
            df["model_variant"] = QWEN_DECODER_ICL

    return df


def _order_variants_for_style(variants: Iterable[str]) -> List[str]:
    """Return a stable order for style assignment using configured variant priorities."""

    seen: List[str] = []
    for variant in variants:
        if variant and variant not in seen:
            seen.append(variant)

    return sorted(
        seen,
        key=lambda v: (MODEL_VARIANT_ORDER.get(v, len(MODEL_VARIANT_ORDER)), v),
    )


def _build_variant_styles(variants: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Assign Tableau palette colors and configured markers to each variant."""

    ordered_variants = _order_variants_for_style(variants)
    color_cycle = cycle(TABLEAU_PALETTE_SEQUENCE)
    marker_cycle = cycle(MARKER_SEQUENCE)

    styles: Dict[str, Dict[str, str]] = {}
    for variant in ordered_variants:
        styles[variant] = {
            'color': next(color_cycle),
            'marker': next(marker_cycle),
        }

    return styles


def _variant_color(variant: str, variant_styles: Dict[str, Dict[str, str]]) -> str:
    return variant_styles.get(variant, {}).get('color', TABLEAU_PALETTE_SEQUENCE[0])


def _variant_marker(variant: str, variant_styles: Dict[str, Dict[str, str]]) -> str:
    return variant_styles.get(variant, {}).get('marker', MARKER_SEQUENCE[0])


def _deduplicate_legend(handles: List, labels: List[str]):
    seen = set()
    filtered_handles = []
    filtered_labels = []

    for handle, label in zip(handles, labels):
        if not label or label in seen:
            continue
        seen.add(label)
        filtered_handles.append(handle)
        filtered_labels.append(label)

    return filtered_handles, filtered_labels


def _make_variant_legend_handle(
    variant: str,
    variant_styles: Dict[str, Dict[str, str]],
    *,
    linewidth: float = 2.0,
    markersize: float = 8.0,
    markeredgewidth: float = 1.5,
    markeredgecolor: str = 'white',
    linestyle_override: Optional[str] = None,
    marker_override: Optional[str] = None,
) -> Line2D:
    color = _variant_color(variant, variant_styles)
    marker = marker_override if marker_override is not None else _variant_marker(variant, variant_styles)
    linestyle = linestyle_override if linestyle_override is not None else '-'

    if variant == QWEN_DECODER_ICL and linestyle_override is None:
        linestyle = '--'

    return Line2D(
        [0],
        [0],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor,
    )


def _extract_model_safe(file_path: str, sub_task: Optional[str]) -> Optional[str]:
    """Infer the model identifier slug from the result filename."""

    stem = Path(file_path).stem
    if stem.startswith("results_"):
        stem = stem[len("results_"):]

    if "_k" not in stem:
        return None

    prefix, _ = stem.rsplit("_k", 1)
    if sub_task:
        suffix = f"_{sub_task}"
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]

    return prefix or None


def _slug_to_variant(model_safe: str) -> Dict[str, str]:
    meta = MODEL_METADATA.get(model_safe, {})
    if not meta:
        hf_model = model_safe.replace("_", "/", 1)
        return {
            "hf_model": hf_model,
            "model_family": "Unknown",
            "model_size": "",
            "model_display": hf_model,
            "model_variant": hf_model,
        }

    model_family = meta.get("model_family", "Unknown")
    model_size = meta.get("model_size", "")
    display = meta.get("model_display", meta.get("hf_model", model_safe))
    variant = meta.get("model_variant")
    if not variant:
        variant = f"{model_family} ({model_size})".strip()

    return {
        "hf_model": meta.get("hf_model", model_safe.replace("_", "/", 1)),
        "model_family": model_family,
        "model_size": model_size,
        "model_display": display,
        "model_variant": variant,
    }


def _infer_task_group(sub_task: Optional[str]) -> str:
    if not isinstance(sub_task, str):
        return "Unknown"

    if sub_task.startswith("chexpert_"):
        return "CheXpert"
    if sub_task.startswith("lab_"):
        return "Laboratory"
    if sub_task.startswith("guo_"):
        return "Critical Care"
    if sub_task.startswith("new_"):
        return "New Diagnoses"

    if " " in sub_task:
        prefix = sub_task.split(" ", 1)[0]
        return prefix.replace("_", " ").title()

    prefix = sub_task.split("_", 1)[0]
    return prefix.title()


def _apply_variant_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize model_variant labels for manuscript tables and plots."""

    if 'model_variant' not in df.columns or 'model_type' not in df.columns:
        return df

    df = df.copy()

    for (model_type, original), new_label in TABLE_VARIANT_RENAMES.items():
        mask = (df['model_type'] == model_type) & (df['model_variant'] == original)
        if mask.any():
            df.loc[mask, 'model_variant'] = new_label

    df['model_variant'] = df['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)

    return df


def _load_baseline_results(baseline_dir: str):
    """Load legacy embedding-based results for inclusion in merged outputs."""

    baseline_path = Path(baseline_dir)
    if not baseline_path.exists():
        print(f"Baseline directory not found: {baseline_dir}")
        return []

    slug = "Qwen_Qwen3-Embedding-8B_Frozen"
    meta = _slug_to_variant(slug)
    if "model_variant" not in meta or not meta["model_variant"]:
        meta["model_variant"] = "Qwen3-Emb-8B (LoRA)"

    dataframes = []
    files = sorted(baseline_path.glob("*/all_results.csv"))
    if not files:
        print(f"No task-level results found in baseline directory: {baseline_dir}")
        return []

    for csv_path in files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Error reading baseline {csv_path}: {exc}")
            continue

        drop_cols = [col for col in df.columns if col.startswith("Unnamed")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if 'sub_task' not in df.columns:
            print(f"Baseline file missing 'sub_task': {csv_path}")
            continue

        if 'model' in df.columns:
            df['baseline_source_model'] = df['model']
            df = df[df['baseline_source_model'] == 'llm'].copy()
            if df.empty:
                print(f"No llm baseline rows found in {csv_path}")
                continue
        else:
            df['baseline_source_model'] = None

        if 'k' in df.columns:
            df['k'] = pd.to_numeric(df['k'], errors='coerce')
            df = df[df['k'] >= 1].copy()
            if df.empty:
                print(f"No positive-shot baseline rows found in {csv_path}")
                continue

        if 'labeling_function' in df.columns:
            df['baseline_labeling_function'] = df['labeling_function']
        else:
            df['baseline_labeling_function'] = None

        label_func = df['baseline_labeling_function'].ffill().bfill().iloc[0]
        if isinstance(label_func, str) and label_func.lower() == 'chexpert':
            df['sub_task'] = df['sub_task'].apply(
                lambda name: name if str(name).startswith('chexpert_') else f"chexpert_{name}"
            )
        else:
            df['sub_task'] = df['sub_task'].astype(str)

        df['labeling_function'] = 'llm_encoder_frozen'
        df['model'] = 'llm_encoder_frozen'
        df['source_file'] = str(csv_path.relative_to(baseline_path.parent))
        df['model_safe'] = slug
        df['model_type'] = 'Encoder Frozen'
        df['task_group'] = df['sub_task'].apply(_infer_task_group)

        for key, value in meta.items():
            df[key] = value

        dataframes.append(df)

    if not dataframes:
        print(f"No usable baseline data found in {baseline_dir}")

    return dataframes

def merge_results(
    results_dir,
    output_file,
    pattern="results_*.csv",
    plot_results=True,
    plot_dir=None,
    overall_show_replicates: bool = False,
    baseline_dirs=None,
    extra_results_dirs=None,
    manuscript_root: Optional[str] = None,
    manuscript_table_name: Optional[str] = None,
    manuscript_figure_name: Optional[str] = None,
    tuning_results_csv: Optional[str] = None,
):
    """
    Merge all individual result CSV files into a single file and optionally create plots

    Args:
        results_dir: Directory containing individual result files
        output_file: Path to output merged CSV file
        pattern: Glob pattern to match result files
        plot_results: Whether to create comparison plots
        plot_dir: Directory to save plots (if None, uses same as output_file)
        overall_show_replicates: Overlay per-task trajectories on the overall plot when True
        baseline_dirs: Optional legacy result directories
        manuscript_root: Optional path to manuscript repository for copying key assets
        manuscript_table_name: Optional override for the copied k=128 LaTeX table name
        manuscript_figure_name: Optional override for the copied overall trend figure name
        tuning_results_csv: Optional path to the raw tuning results CSV for the tuning summary table
    """
    
    if extra_results_dirs is None:
        extra_results_dirs = []

    result_dirs = [results_dir, *extra_results_dirs]
    result_files: List[str] = []
    for directory in result_dirs:
        search_pattern = os.path.join(directory, pattern)
        result_files.extend(glob.glob(search_pattern))

    result_files = sorted(set(result_files))

    if not result_files:
        print(f"No result files found matching pattern: {pattern} in {result_dirs}")
        return
    
    print(f"Found {len(result_files)} result files to merge")
    
    # Read and concatenate all result files
    all_results = []
    
    for file_path in result_files:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Warning: Empty file {os.path.basename(file_path)}")
                continue

            sub_task = df["sub_task"].iloc[0] if "sub_task" in df.columns and not df.empty else None
            model_safe = _extract_model_safe(file_path, sub_task)

            meta = _slug_to_variant(model_safe) if model_safe else {}

            df["source_file"] = os.path.basename(file_path)
            if model_safe:
                df["model_safe"] = model_safe
            for key, value in meta.items():
                df[key] = value

            if "labeling_function" in df.columns:
                df["model_type"] = df["labeling_function"].map(MODEL_TYPE_MAPPING).fillna(df["labeling_function"])
                if "model_family" not in df.columns:
                    df["model_family"] = df["labeling_function"].map({
                        "llm_encoder_ft": "Encoder",
                        "llm_decoder_ft": "Decoder",
                        "llm_decoder_zero_shot": "Decoder",
                        "llm_encoder_frozen": "Encoder",
                    }).fillna("Unknown")
            else:
                df["model_type"] = "Unknown"
                if "model_family" not in df.columns:
                    df["model_family"] = "Unknown"

            if "model_variant" not in df.columns:
                def _variant(row):
                    size = row.get("model_size", "")
                    family = row.get("model_family", "Unknown")
                    return f"{family} ({size})" if size else family

                df["model_variant"] = df.apply(_variant, axis=1)

            if "sub_task" in df.columns:
                df["task_group"] = df["sub_task"].apply(_infer_task_group)
            else:
                df["task_group"] = "Unknown"

            df = _normalize_result_frame(df, file_path)
            all_results.append(df)
            print(f"Added {len(df)} rows from {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if baseline_dirs is None:
        baseline_dirs = []
    env_baselines = os.environ.get("MERGE_RESULTS_BASELINES")
    if env_baselines:
        baseline_dirs = list(baseline_dirs) + [p for p in env_baselines.split(os.pathsep) if p]

    for baseline_dir in baseline_dirs:
        baseline_dir = baseline_dir.strip()
        if not baseline_dir:
            continue
        baseline_data = _load_baseline_results(baseline_dir)
        if baseline_data:
            all_results.extend(baseline_data)

    if not all_results:
        print("No valid result files found")
        return
    
    # Concatenate all DataFrames
    merged_df = pd.concat(all_results, ignore_index=True)
    
    # Ensure model metadata columns are populated consistently
    if 'model_type' in merged_df.columns:
        merged_df['model_type'] = merged_df['model_type'].fillna(
            merged_df.get('labeling_function')
        )
    elif 'labeling_function' in merged_df.columns:
        merged_df['model_type'] = merged_df['labeling_function'].map(MODEL_TYPE_MAPPING).fillna(
            merged_df['labeling_function']
        )
    else:
        merged_df['model_type'] = 'Unknown'

    if 'model_variant' not in merged_df.columns:
        merged_df['model_variant'] = merged_df.apply(
            lambda row: f"{row.get('model_family', 'Unknown')} ({row.get('model_size', '')})".strip()
            if row.get('model_size') else row.get('model_family', 'Unknown'),
            axis=1,
        )

    if 'task_group' not in merged_df.columns and 'sub_task' in merged_df.columns:
        merged_df['task_group'] = merged_df['sub_task'].apply(_infer_task_group)

    if 'k' in merged_df.columns:
        merged_df['k'] = pd.to_numeric(merged_df['k'], errors='coerce')

    merged_df = _apply_variant_overrides(merged_df)

    # Sort by task, model, k, replicate for consistent ordering
    sort_columns = []
    if 'sub_task' in merged_df.columns:
        sort_columns.append('sub_task')
    if 'model_variant' in merged_df.columns:
        sort_columns.append('model_variant')
    elif 'model' in merged_df.columns:
        sort_columns.append('model') 
    if 'k' in merged_df.columns:
        sort_columns.append('k')
    if 'replicate' in merged_df.columns:
        sort_columns.append('replicate')
    if 'score' in merged_df.columns:
        sort_columns.append('score')
    
    if sort_columns:
        merged_df = merged_df.sort_values(sort_columns)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged results
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merged {len(merged_df)} total rows into {output_file}")
    print(f"Columns: {list(merged_df.columns)}")

    # Print summary statistics
    if 'sub_task' in merged_df.columns:
        print(f"Unique tasks: {merged_df['sub_task'].nunique()}")
    if 'k' in merged_df.columns:
        print(f"Unique k values: {sorted(merged_df['k'].unique())}")
    if 'replicate' in merged_df.columns:
        print(f"Unique replicates: {sorted(merged_df['replicate'].unique())}")
    if 'score' in merged_df.columns:
        print(f"Score types: {sorted(merged_df['score'].unique())}")

    table_assets = export_k_tables(merged_df, output_path)

    if tuning_results_csv:
        tuning_path = Path(tuning_results_csv)
        if tuning_path.exists():
            try:
                tuning_df = pd.read_csv(tuning_path)
                export_tuning_summary_table(tuning_df, output_path)
            except Exception as exc:
                print(f"Failed to export tuning summary table from {tuning_path}: {exc}")
        else:
            print(f"Skipping tuning summary table export; file not found: {tuning_path}")

    # Create plots if requested
    plot_assets = None
    if plot_results and not merged_df.empty:
        if plot_dir is None:
            plot_dir = str(Path(output_file).parent)
        plot_assets = create_comparison_plots(
            merged_df,
            plot_dir,
            show_overall_replicates=overall_show_replicates,
        )

    if manuscript_root:
        table_targets = table_assets.get(128, {}) if table_assets else {}
        plot_targets = plot_assets or {}

        manuscript_root_path = Path(manuscript_root)
        tables_dir = manuscript_root_path / "tables"
        figures_dir = manuscript_root_path / "figures"

        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

        if 'latex' in table_targets:
            src_table = table_targets['latex']
            dest_name = manuscript_table_name or src_table.name
            dest_table = tables_dir / dest_name
            shutil.copy2(src_table, dest_table)
            print(f"Copied manuscript table to {dest_table}")

        overall_plot = plot_targets.get('overall', {})
        if isinstance(overall_plot, dict) and 'pdf' in overall_plot:
            src_fig = overall_plot['pdf']
            dest_name = manuscript_figure_name or src_fig.name
            dest_fig = figures_dir / dest_name
            shutil.copy2(src_fig, dest_fig)
            print(f"Copied manuscript figure to {dest_fig}")

        overall_auprc_plot = plot_targets.get('overall_auprc', {})
        if isinstance(overall_auprc_plot, dict) and 'pdf' in overall_auprc_plot:
            src_fig = overall_auprc_plot['pdf']
            dest_fig = figures_dir / 'encoder_decoder_overall_auprc.pdf'
            shutil.copy2(src_fig, dest_fig)
            print(f"Copied manuscript AUPRC figure to {dest_fig}")

        figure_basenames = [
            'encoder_decoder_by_group.pdf',
            'encoder_decoder_by_group_auprc.pdf',
        ]
        plot_root = Path(plot_dir if plot_dir else output_path.parent)
        for basename in figure_basenames:
            source_fig = plot_root / basename
            if source_fig.exists():
                dest_fig = figures_dir / basename
                shutil.copy2(source_fig, dest_fig)
                print(f"Copied task group comparison figure to {dest_fig}")

    return merged_df


def _prepare_metric_dataframe(
    df: pd.DataFrame,
    metric_config: MetricPlotConfig,
) -> Optional[pd.DataFrame]:
    """Filter and normalize results for a specific evaluation metric."""

    metric_df = df[df['score'] == metric_config.score].copy() if 'score' in df.columns else df.copy()
    if metric_df.empty:
        print(f"No {metric_config.display_name} data found for plotting")
        return None

    required_columns = {'k', 'mean', 'model_variant', 'model_type'}
    missing = required_columns - set(metric_df.columns)
    if missing:
        print(f"Missing columns for plotting ({metric_config.display_name}): {missing}")
        return None

    metric_df = metric_df.dropna(subset=['mean', 'k'])
    metric_df['k'] = pd.to_numeric(metric_df['k'], errors='coerce')
    metric_df = metric_df.dropna(subset=['k'])

    metric_df['model_type'] = metric_df['model_type'].map(MODEL_TYPE_MAPPING).fillna(metric_df['model_type'])
    metric_df['model_variant'] = metric_df['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)

    if 'task_group' not in metric_df.columns:
        metric_df['task_group'] = metric_df.get('sub_task', 'Unknown').apply(_infer_task_group)

    return metric_df


ZERO_SHOT_PLOT_X = 0.75


def _plot_k_value(k: float) -> float:
    """Keep k=0 visible on a log axis while preserving the original tick styling."""

    return ZERO_SHOT_PLOT_X if float(k) == 0.0 else float(k)


def _plot_k_values(values):
    return [_plot_k_value(v) for v in values]


def _configure_log_shot_axis(
    ax,
    ticks,
    *,
    xlabel: str,
    fontsize: int,
    tick_fontsize: int,
    tick_mode: str = "all",
    base: Optional[int] = None,
) -> None:
    """Configure shot-count axis to match the original log-axis behavior."""

    if base is None:
        ax.set_xscale('log')
    else:
        ax.set_xscale('log', base=base)
    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight='bold')
    if ticks:
        visible_ticks = []
        visible_labels = []
        for tick in ticks:
            tick_value = int(tick)
            if tick_mode == "all":
                visible_ticks.append(_plot_k_value(tick_value))
                visible_labels.append(str(tick_value))
            elif tick_mode == "powers":
                if tick_value == 0 or tick_value == 1 or (tick_value > 0 and (tick_value & (tick_value - 1) == 0)):
                    visible_ticks.append(_plot_k_value(tick_value))
                    visible_labels.append(str(tick_value))
            else:
                raise ValueError(f"Unsupported tick_mode: {tick_mode}")
        ax.set_xticks(visible_ticks)
        ax.set_xticklabels(visible_labels, fontsize=tick_fontsize)


def create_comparison_plots(
    df,
    plot_dir,
    show_overall_replicates: bool = False,
):
    """Create comparison plots focused on encoder vs decoder fine-tuning."""

    plot_path = Path(plot_dir)
    plot_path.mkdir(parents=True, exist_ok=True)

    generated_outputs: Dict[str, Dict[str, Path]] = {}

    for metric_config in METRIC_PLOT_CONFIGS:
        metric_df = _prepare_metric_dataframe(df, metric_config)
        if metric_df is None:
            continue

        value_column = f'mean_{metric_config.score}'

        variant_candidates: List[str] = metric_df['model_variant'].dropna().unique().tolist()
        variant_styles = _build_variant_styles(variant_candidates)

        eight_b_slugs = {
            "Qwen_Qwen3-8B",
            "Qwen_Qwen3-Embedding-8B",
            "Qwen_Qwen3-Embedding-8B_Frozen",
        }
        if 'model_safe' in metric_df.columns:
            eight_b_df = metric_df[
                metric_df['model_safe'].isin(eight_b_slugs)
                | (metric_df['model_variant'] == QWEN_DECODER_ICL)
            ]
            if not eight_b_df.empty:
                metric_df = eight_b_df
        if metric_df.empty:
            print(f"No shot sizes available for plotting ({metric_config.display_name})")
            continue

        metric_df = metric_df.sort_values(['model_variant', 'k'])

        metric_outputs: Dict[str, Dict[str, Path]] = {}

        overall_paths = _plot_overall_trend(
            metric_df,
            plot_path,
            variant_styles,
            metric_config,
            value_column,
            show_subtask_lines=show_overall_replicates,
        )
        if overall_paths:
            metric_outputs['overall'] = overall_paths

        _plot_task_group_facets(
            metric_df,
            plot_path,
            variant_styles,
            metric_config,
            value_column,
        )
        _plot_per_task_bars(
            metric_df,
            plot_path,
            variant_styles,
            metric_config,
            value_column,
        )
        _plot_per_task_lines(
            metric_df,
            plot_path,
            variant_styles,
            metric_config,
            value_column,
        )

        if metric_outputs:
            generated_outputs[metric_config.score] = metric_outputs
            if metric_config.score == 'auroc':
                generated_outputs.update(metric_outputs)
            else:
                generated_outputs.update(
                    {f"{key}_{metric_config.score}": value for key, value in metric_outputs.items()}
                )

    return generated_outputs or None

def _plot_overall_trend(
    df: pd.DataFrame,
    plot_path: Path,
    variant_styles: Dict[str, Dict[str, str]],
    metric_config: MetricPlotConfig,
    value_column: str,
    show_subtask_lines: bool = False,
) -> Optional[Dict[str, Path]]:
    """Plot macro trends across shot counts for a specific evaluation metric."""

    # First compute per-task means
    per_task = (
        df.groupby(['model_variant', 'model_type', 'task_group', 'sub_task', 'k'], dropna=False)['mean']
        .mean()
        .reset_index(name='per_task_mean')
    )

    # Then compute macro average per task group
    group_macro = (
        per_task.groupby(['model_variant', 'model_type', 'task_group', 'k'], dropna=False)
        .agg(
            group_macro=('per_task_mean', 'mean'),
            per_task_std=('per_task_mean', 'std'),
        )
        .reset_index()
    )

    # Finally compute macro average across task groups (matching table computation)
    summary = (
        group_macro.groupby(['model_variant', 'model_type', 'k'], dropna=False)
        .agg(
            metric=('group_macro', 'mean'),
            std_across_tasks=('group_macro', 'std'),
            tasks=('task_group', 'nunique'),
        )
        .reset_index()
    )

    if summary.empty:
        print(f"Insufficient data for overall trend plot ({metric_config.display_name})")
        return None

    summary['std_across_tasks'] = summary['std_across_tasks'].fillna(0.0)
    summary['tasks'] = summary['tasks'].fillna(0).astype(int)
    summary['model_variant'] = summary['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)
    summary = summary.rename(columns={'metric': value_column})
    summary['model_variant'] = pd.Categorical(
        summary['model_variant'],
        categories=MODEL_VARIANT_DISPLAY_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(['model_variant', 'k'])

    replicate_trends: Optional[pd.DataFrame] = None
    if show_subtask_lines:
        if 'replicate' not in df.columns:
            print(
                f"[WARN] Cannot overlay replicate trajectories; 'replicate' column missing ({metric_config.display_name})"
            )
        else:
            # Compute per-task means per replicate
            rep_per_task = (
                df.dropna(subset=['replicate'])
                .groupby(['model_variant', 'replicate', 'task_group', 'sub_task', 'k'], dropna=False)['mean']
                .mean()
                .reset_index(name='per_task_mean')
            )

            # Compute macro average per task group per replicate
            rep_group_macro = (
                rep_per_task.groupby(['model_variant', 'replicate', 'task_group', 'k'], dropna=False)
                .agg(group_macro=('per_task_mean', 'mean'))
                .reset_index()
            )

            # Compute macro average across task groups per replicate
            replicate_trends = (
                rep_group_macro.groupby(['model_variant', 'replicate', 'k'], dropna=False)
                .agg(metric=('group_macro', 'mean'))
                .reset_index()
            )
            if not replicate_trends.empty:
                replicate_trends = replicate_trends.rename(columns={'metric': value_column})
                replicate_trends['model_variant'] = replicate_trends['model_variant'].replace(
                    LEGACY_VARIANT_NAME_MAP
                )
                replicate_trends['replicate'] = replicate_trends['replicate'].astype(str)

    with paper_style(SINGLE_PANEL_RC):
        fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_SINGLE)

        for variant in MODEL_VARIANT_DISPLAY_ORDER:
            group = summary[summary['model_variant'] == variant]
            if group.empty:
                continue
            color = _variant_color(variant, variant_styles)
            marker = _variant_marker(variant, variant_styles)

            if replicate_trends is not None:
                variant_trends = replicate_trends[replicate_trends['model_variant'] == variant]
                if not variant_trends.empty:
                    for rep in variant_trends['replicate'].unique():
                        rep_line = variant_trends[variant_trends['replicate'] == rep]
                        rep_line = rep_line.sort_values('k')
                        ax.plot(
                            _plot_k_values(rep_line['k']),
                            rep_line[value_column],
                            color=color,
                            linestyle='-',
                            linewidth=1.0,
                            alpha=0.15,
                            zorder=1,
                        )

            ax.plot(
                _plot_k_values(group['k']),
                group[value_column],
                label=variant,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=8,
                markeredgewidth=1.5,
                markeredgecolor='white',
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(1.0)

        ax.set_title(
            metric_config.overall_title,
            fontsize=14,
            fontweight='bold',
            pad=10,
        )
        ax.set_ylabel(
            f"Macro {metric_config.display_name}",
            fontsize=12,
            fontweight='bold',
        )
        _configure_log_shot_axis(
            ax,
            sorted(df['k'].unique()),
            xlabel='# of Train Examples per Class',
            fontsize=12,
            tick_fontsize=10,
            tick_mode="all",
        )

        ax.tick_params(axis='y', labelsize=8)
        ax.grid(visible=True, which='major', axis='y', linestyle='-', alpha=0.3)
        ax.grid(visible=True, which='minor', axis='x', linestyle=':', alpha=0.3)

        present_variants = set(summary['model_variant'].dropna())

        legend_variants = [variant for variant in MODEL_VARIANT_DISPLAY_ORDER if variant in present_variants]

        if legend_variants:
            handles = [
                _make_variant_legend_handle(
                    variant,
                    variant_styles,
                    linewidth=2.0,
                    markersize=8.0,
                )
                for variant in legend_variants
            ]

            ax.legend(
                handles,
                legend_variants,
                loc='lower center',
                ncol=len(legend_variants),
                frameon=False,
                fontsize=10,
                bbox_to_anchor=(0.5, -0.4),
                title_fontsize=10,
                handlelength=2.5,
            )

        plt.tight_layout()

        output_png = plot_path / f"encoder_decoder_overall{metric_config.file_suffix}.png"
        output_pdf = plot_path / f"encoder_decoder_overall{metric_config.file_suffix}.pdf"
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved overall comparison plot ({metric_config.display_name}): {output_png}")

    return {"png": output_png, "pdf": output_pdf}


def _plot_task_group_facets(
    df: pd.DataFrame,
    plot_path: Path,
    variant_styles: Dict[str, Dict[str, str]],
    metric_config: MetricPlotConfig,
    value_column: str,
) -> None:
    """Create small multiples comparing encoder/decoder trends per task group."""

    if 'task_group' not in df.columns:
        print("Task group column missing; skipping facet plot")
        return

    df_means = (
        df.groupby(
            [
                'task_group',
                'sub_task',
                'model_variant',
                'model_type',
                'k',
            ],
            dropna=False,
        )['mean']
        .mean()
        .reset_index()
    )

    if df_means.empty:
        print(f"Insufficient per-task data for task group plot ({metric_config.display_name})")
        return

    group_summary = (
        df_means.groupby(['task_group', 'model_variant', 'model_type', 'k'], dropna=False)
        .agg(
            metric=('mean', 'mean'),
            std_across_tasks=('mean', 'std'),
            tasks=('sub_task', 'nunique'),
        )
        .reset_index()
    )

    if group_summary.empty:
        print(f"Insufficient data for task group plot ({metric_config.display_name})")
        return

    group_summary['std_across_tasks'] = group_summary['std_across_tasks'].fillna(0.0)
    group_summary = group_summary.rename(columns={'metric': value_column})

    group_summary['task_group_display'] = group_summary['task_group'].map(
        TASK_GROUP_DISPLAY_NAMES
    ).fillna(group_summary['task_group'])
    df_means['task_group_display'] = df_means['task_group'].map(
        TASK_GROUP_DISPLAY_NAMES
    ).fillna(df_means['task_group'])

    task_group_order = [
        "Operational Outcomes",
        "Anticipating Lab Test Results",
        "Assignment of New Diagnoses",
        "Anticipating Chest X-ray Findings",
    ]

    groups = [g for g in task_group_order if g in group_summary['task_group_display'].unique()]
    n_groups = len(groups)
    if n_groups == 0:
        print(f"No task groups available for plotting ({metric_config.display_name})")
        return

    n_cols = 2
    n_rows = math.ceil(n_groups / n_cols)

    with paper_style(GROUP_FACET_RC):
        figsize = (
            max(FACET_WIDTH_PER_COL, n_cols * FACET_WIDTH_PER_COL),
            max(FACET_HEIGHT_PER_ROW, n_rows * FACET_HEIGHT_PER_ROW),
        )
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=False,
            sharey=False,
        )
        axes = axes.flatten()
        plotted_variants = set()

        for ax, group_name in zip(axes, groups):
            subset_mean = group_summary[group_summary['task_group_display'] == group_name]
            subset_mean = subset_mean.sort_values('k')

            subset_mean['model_variant'] = subset_mean['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)
            subset_subtasks = df_means[df_means['task_group_display'] == group_name]
            subset_subtasks['model_variant'] = subset_subtasks['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)

            for variant in MODEL_VARIANT_DISPLAY_ORDER:
                if variant not in subset_mean['model_variant'].unique():
                    continue
                grp_mean = subset_mean[subset_mean['model_variant'] == variant]
                color = _variant_color(variant, variant_styles)
                marker = _variant_marker(variant, variant_styles)

                grp_subtasks = subset_subtasks[subset_subtasks['model_variant'] == variant]
                for subtask in grp_subtasks['sub_task'].unique():
                    subtask_data = grp_subtasks[grp_subtasks['sub_task'] == subtask].sort_values('k')
                    ax.plot(
                        _plot_k_values(subtask_data['k']),
                        subtask_data['mean'],
                        color=color,
                        linestyle='-',
                        linewidth=1.1,
                        alpha=0.08,
                    )

                ax.plot(
                    _plot_k_values(grp_mean['k']),
                    grp_mean[value_column],
                    label=variant,
                    color=color,
                    marker=marker,
                    linewidth=2.5,
                    markersize=10,
                    markeredgewidth=2.0,
                    markeredgecolor='white',
                )
                plotted_variants.add(variant)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_title(group_name, size=GROUP_TITLE_FONT_SIZE, pad=8, fontweight='bold')
            ax.set_ylabel(
                f"Mean {metric_config.display_name}",
                fontsize=GROUP_AXIS_LABEL_FONT_SIZE,
                fontweight='bold',
            )
            _configure_log_shot_axis(
                ax,
                sorted(df['k'].unique()),
                xlabel='# of Train Examples per Class',
                fontsize=GROUP_AXIS_LABEL_FONT_SIZE,
                tick_fontsize=GROUP_TICK_FONT_SIZE,
                tick_mode="powers",
            )
            ax.grid(True, which="both", ls="-", alpha=0.15)

            ax.tick_params(axis='both', which='major', labelsize=GROUP_TICK_FONT_SIZE)
            ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, length=3)
            ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, length=3)

            ax.grid(True, which='minor', linestyle=':', alpha=0.12)

        for ax in axes[n_groups:]:
            ax.axis('off')

        legend_variants = [variant for variant in MODEL_VARIANT_DISPLAY_ORDER if variant in plotted_variants]

        if legend_variants:
            legend_handles = [
                _make_variant_legend_handle(
                    variant,
                    variant_styles,
                    linewidth=2.5,
                    markersize=9.0,
                    markeredgewidth=2.0,
                )
                for variant in legend_variants
            ]

            fig.legend(
                legend_handles,
                legend_variants,
                loc='lower center',
                ncol=len(legend_variants),
                frameon=False,
                fontsize=GROUP_LEGEND_FONT_SIZE,
                bbox_to_anchor=(0.5, -0.04),
                handlelength=2.3,
            )

        plt.tight_layout(rect=(0, 0, 1, 0.92))

        fig.suptitle(
            metric_config.task_group_title,
            fontsize=GROUP_SUPTITLE_FONT_SIZE,
            fontweight='bold',
            y=0.96,
        )

        output_png = plot_path / f"encoder_decoder_by_group{metric_config.file_suffix}.png"
        output_pdf = plot_path / f"encoder_decoder_by_group{metric_config.file_suffix}.pdf"
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved task group comparison plot ({metric_config.display_name}): {output_png}")


def _plot_per_task_bars(
    df: pd.DataFrame,
    plot_path: Path,
    variant_styles: Dict[str, Dict[str, str]],
    metric_config: MetricPlotConfig,
    value_column: str,
) -> None:
    """Create horizontal bar chart comparing encoder vs decoder per task (k=1)."""

    if 'k' not in df.columns or 'sub_task' not in df.columns or 'model_variant' not in df.columns:
        print("Missing columns for per-task bar plot; skipping")
        return

    eight_b_slugs = {
        "Qwen_Qwen3-8B",
        "Qwen_Qwen3-Embedding-8B",
        "Qwen_Qwen3-Embedding-8B_Frozen",
    }
    if 'model_safe' not in df.columns:
        print("model_safe column missing; skipping per-task bars")
        return

    subset = df[(df['model_safe'].isin(eight_b_slugs)) & (df['k'] == 1)].copy()
    if subset.empty:
        print(f"No overlapping k=1 data for per-task comparison ({metric_config.display_name})")
        return

    subset['model_variant'] = subset['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)

    summary = (
        subset.groupby(['sub_task', 'model_variant'])['mean']
        .mean()
        .reset_index()
    )

    summary['model_variant'] = summary['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)

    pivot = summary.pivot(index='sub_task', columns='model_variant', values='mean')
    pivot = pivot.rename(columns=LEGACY_VARIANT_NAME_MAP)
    variants = [v for v in MODEL_VARIANT_DISPLAY_ORDER if v in pivot.columns]

    if len(variants) < 2:
        print(f"Per-task plot requires at least two model variants ({metric_config.display_name}); skipping")
        return

    pivot = pivot.dropna(subset=variants)
    if pivot.empty:
        print(f"No tasks with complete per-task coverage at k=1 ({metric_config.display_name})")
        return

    if QWEN_EMB_BASE in pivot.columns and QWEN_DECODER_LORA in pivot.columns:
        pivot['difference'] = pivot[QWEN_EMB_BASE] - pivot[QWEN_DECODER_LORA]
    else:
        pivot['difference'] = 0
    pivot = pivot.sort_values('difference')

    long_df = pivot.reset_index().melt(
        id_vars=['sub_task', 'difference'],
        value_vars=variants,
        var_name='model_variant',
        value_name=value_column,
    )

    long_df['task_label'] = long_df['sub_task'].str.replace('_', ' ').str.title()
    long_df['model_variant'] = long_df['model_variant'].astype('category')
    long_df['model_variant'] = long_df['model_variant'].cat.reorder_categories(variants, ordered=True)

    variant_colors = {variant: _variant_color(variant, variant_styles) for variant in variants}

    fig_height = max(6, BAR_HEIGHT_PER_TASK * pivot.shape[0])

    with paper_style():
        _set_seaborn_theme()
        fig, ax = plt.subplots(figsize=(BAR_FIG_WIDTH, fig_height))

        sns.barplot(
            data=long_df,
            y='task_label',
            x=value_column,
            hue='model_variant',
            palette=variant_colors,
            hue_order=variants,
            ax=ax,
        )

        ax.set_xlabel(f'Macro {metric_config.display_name} (k = 1)', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_ylabel('Task', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
        ax.set_title(metric_config.per_task_bar_title, fontsize=TITLE_FONT_SIZE, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = _deduplicate_legend(handles, labels)
        ax.legend(
            handles,
            labels,
            frameon=False,
            fontsize=LEGEND_FONT_SIZE,
            title_fontsize=LEGEND_FONT_SIZE,
        )
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

        plt.tight_layout()
        output_png = plot_path / f"encoder_decoder_per_task_k1{metric_config.file_suffix}.png"
        output_pdf = plot_path / f"encoder_decoder_per_task_k1{metric_config.file_suffix}.pdf"
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved per-task comparison plot ({metric_config.display_name}): {output_png}")


def _plot_per_task_lines(
    df: pd.DataFrame,
    plot_path: Path,
    variant_styles: Dict[str, Dict[str, str]],
    metric_config: MetricPlotConfig,
    value_column: str,
) -> None:
    """Plot per-task trajectories with k on the x-axis for 8B encoder/decoder variants."""

    if 'model_safe' not in df.columns or 'sub_task' not in df.columns:
        print("Missing columns for per-task trajectory plot; skipping")
        return

    eight_b_slugs = {
        "Qwen_Qwen3-8B",
        "Qwen_Qwen3-Embedding-8B",
        "Qwen_Qwen3-Embedding-8B_Frozen",
    }
    subset = df[df['model_safe'].isin(eight_b_slugs)].copy()
    if subset.empty:
        print(f"No k data for per-task trajectories ({metric_config.display_name})")
        return

    subset['plot_k'] = subset['k'].map(_plot_k_value)
    subset['Task'] = subset['sub_task'].str.replace('_', ' ').str.title()
    subset['model_variant'] = subset['model_variant'].replace(LEGACY_VARIANT_NAME_MAP)

    available_variants = subset['model_variant'].dropna().unique()
    variant_colors = {variant: _variant_color(variant, variant_styles) for variant in available_variants}
    if not variant_colors:
        print(f"No variants available for per-task trajectory plot ({metric_config.display_name})")
        return

    hue_order = [v for v in MODEL_VARIANT_DISPLAY_ORDER if v in variant_colors]
    marker_map = {variant: _variant_marker(variant, variant_styles) for variant in hue_order}
    if not hue_order:
        print(f"No ordered variants found for per-task trajectory plot ({metric_config.display_name})")
        return

    with paper_style():
        _set_seaborn_theme()
        g = sns.relplot(
            data=subset,
            x='plot_k',
            y='mean',
            hue='model_variant',
            style='model_variant',
            kind='line',
            col='Task',
            col_wrap=4,
            palette=variant_colors,
            hue_order=hue_order,
            style_order=hue_order,
            markers=marker_map,
            dashes=False,
            linewidth=1.8,
            height=2.4,
            aspect=1.1,
            facet_kws={'sharey': False},
        )

        for ax in g.axes.flatten():
            if ax is None:
                continue
            _configure_log_shot_axis(
                ax,
                sorted(subset['k'].unique()),
                xlabel='k (shots)',
                fontsize=AXIS_LABEL_FONT_SIZE,
                tick_fontsize=TICK_FONT_SIZE,
                tick_mode="all",
                base=2,
            )
            ax.set_ylabel(f'Macro {metric_config.display_name}', fontsize=AXIS_LABEL_FONT_SIZE, fontweight='bold')
            ax.grid(axis='both', linestyle=':', alpha=0.25)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)

        g.set_titles('{col_name}', size=SUBPLOT_TITLE_FONT_SIZE)
        legend_handles: List[Line2D] = []
        legend_labels: List[str] = []

        for variant in hue_order:
            if variant not in variant_colors:
                continue
            legend_handles.append(
                _make_variant_legend_handle(
                    variant,
                    variant_styles,
                    linewidth=1.8,
                    markersize=6.0,
                    markeredgewidth=1.5,
                )
            )
            legend_labels.append(variant)

        if g._legend is not None:
            g._legend.remove()

        if legend_handles:
            g.figure.legend(
                legend_handles,
                legend_labels,
                loc='upper center',
                ncol=min(len(legend_handles), 3),
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                title_fontsize=LEGEND_FONT_SIZE,
            )

        g.figure.subplots_adjust(top=0.88)
        g.figure.suptitle(
            metric_config.per_task_line_title,
            fontsize=SUPTITLE_FONT_SIZE,
            fontweight='bold',
        )

        output_png = plot_path / f"encoder_decoder_per_task_lines{metric_config.file_suffix}.png"
        output_pdf = plot_path / f"encoder_decoder_per_task_lines{metric_config.file_suffix}.pdf"
        g.figure.savefig(output_png, dpi=300, bbox_inches='tight')
        g.figure.savefig(output_pdf, dpi=300, bbox_inches='tight')
        plt.close(g.figure)

    print(f"Saved per-task trajectory plot ({metric_config.display_name}): {output_png}")


def _format_decimal(value: float, drop_leading_zero: bool = False) -> str:
    if pd.isna(value):
        return "--"

    formatted = f"{value:.3f}"

    if drop_leading_zero:
        if formatted.startswith("-0"):
            return f"-{formatted[2:]}"
        if formatted.startswith("0"):
            return formatted[1:]

    return formatted


def _format_ci_latex(mean: float, lower: float, upper: float) -> str:
    return (
        f"$\\ci{{{_format_decimal(mean)}}}"
        f"{{{_format_decimal(lower, drop_leading_zero=True)}}}"
        f"{{{_format_decimal(upper, drop_leading_zero=True)}}}$"
    )


def _format_ci_csv(mean: float, lower: float, upper: float) -> str:
    if pd.isna(mean) or pd.isna(lower) or pd.isna(upper):
        return "--"
    return f"{mean:.3f} ({lower:.3f}, {upper:.3f})"


def _parse_tuning_lr_token(token: str) -> Tuple[float, str]:
    match = re.fullmatch(r"(\d+)e(\d+)", token)
    if not match:
        try:
            value = float(token)
        except ValueError:
            return math.inf, token
        return value, token

    base = int(match.group(1))
    exponent = int(match.group(2))
    value = base * (10 ** (-exponent))
    return value, f"{base}e-{exponent}"


def _parse_tuning_dropout_token(token: str) -> Tuple[float, str]:
    try:
        value = int(token) / 100.0
    except ValueError:
        return math.inf, token
    return value, f"{value:.2f}"


def _parse_tuning_config_id(config_id: str) -> Dict[str, object]:
    match = re.fullmatch(r"(enc|dec)_lr([^_]+)_r(\d+)_d(\d+)", str(config_id))
    if not match:
        return {
            "learning_rate": "--",
            "learning_rate_value": math.inf,
            "lora_r": "--",
            "lora_r_value": math.inf,
            "lora_dropout": "--",
            "lora_dropout_value": math.inf,
        }

    lr_value, lr_display = _parse_tuning_lr_token(match.group(2))
    dropout_value, dropout_display = _parse_tuning_dropout_token(match.group(4))
    lora_r_value = int(match.group(3))

    return {
        "learning_rate": lr_display,
        "learning_rate_value": lr_value,
        "lora_r": str(lora_r_value),
        "lora_r_value": lora_r_value,
        "lora_dropout": dropout_display,
        "lora_dropout_value": dropout_value,
    }


def export_tuning_summary_table(
    tuning_df: pd.DataFrame,
    output_path: Path,
    target_ks: Tuple[int, ...] = (8, 16),
) -> Optional[Dict[str, Path]]:
    """Create a LaTeX and CSV summary table for the step-03 tuning sweep."""

    required_columns = {"labeling_function", "config_id", "k", "score", "value_val"}
    missing_columns = required_columns - set(tuning_df.columns)
    if missing_columns:
        print(f"Missing columns for tuning table export: {sorted(missing_columns)}")
        return None

    subset = tuning_df.copy()
    subset = subset[subset["score"] == "auroc"].copy()
    subset = subset[subset["labeling_function"].isin(TUNING_SECTION_DISPLAY)].copy()
    subset["k"] = pd.to_numeric(subset["k"], errors="coerce")
    subset["value_val"] = pd.to_numeric(subset["value_val"], errors="coerce")
    subset = subset[subset["k"].isin(target_ks)]
    subset = subset.dropna(subset=["k", "value_val"])

    if subset.empty:
        print("No AUROC tuning rows available for tuning table export")
        return None

    k_means = (
        subset.groupby(["labeling_function", "config_id", "k"], dropna=False)["value_val"]
        .mean()
        .reset_index()
    )
    overall_means = (
        subset.groupby(["labeling_function", "config_id"], dropna=False)["value_val"]
        .mean()
        .reset_index()
        .rename(columns={"value_val": "macro_avg_mean_val_auroc"})
    )

    pivot = (
        k_means.pivot(index=["labeling_function", "config_id"], columns="k", values="value_val")
        .reset_index()
    )
    pivot.columns.name = None

    summary = pivot.merge(overall_means, on=["labeling_function", "config_id"], how="outer")

    parsed = summary["config_id"].apply(_parse_tuning_config_id).apply(pd.Series)
    summary = pd.concat([summary, parsed], axis=1)
    summary["section_rank"] = summary["labeling_function"].map(
        {name: idx for idx, name in enumerate(TUNING_SECTION_ORDER)}
    ).fillna(len(TUNING_SECTION_ORDER))

    summary = summary.sort_values(
        by=[
            "section_rank",
            "learning_rate_value",
            "lora_r_value",
            "lora_dropout_value",
            "config_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    best = (
        summary.sort_values(
            by=[
                "section_rank",
                "macro_avg_mean_val_auroc",
                "learning_rate_value",
                "lora_r_value",
                "lora_dropout_value",
                "config_id",
            ],
            ascending=[True, False, True, True, True, True],
            kind="mergesort",
        )
        .groupby("labeling_function", as_index=False)
        .first()[["labeling_function", "config_id"]]
    )
    best["selected"] = "yes"
    summary = summary.merge(best, on=["labeling_function", "config_id"], how="left")
    summary["selected"] = summary["selected"].fillna("")

    csv_columns = [
        "Model Type",
        "Config ID",
        "Learning Rate",
        "LoRA r",
        "LoRA Dropout",
        "k=8 Mean Val. AUROC",
        "k=16 Mean Val. AUROC",
        "Macro Avg. Mean Val. AUROC",
        "Selected",
    ]

    csv_rows = []
    for _, row in summary.iterrows():
        csv_rows.append(
            {
                "Model Type": TUNING_SECTION_DISPLAY.get(str(row["labeling_function"]), str(row["labeling_function"])),
                "Config ID": row["config_id"],
                "Learning Rate": row["learning_rate"],
                "LoRA r": row["lora_r"],
                "LoRA Dropout": row["lora_dropout"],
                "k=8 Mean Val. AUROC": _format_decimal(row.get(8)),
                "k=16 Mean Val. AUROC": _format_decimal(row.get(16)),
                "Macro Avg. Mean Val. AUROC": _format_decimal(row["macro_avg_mean_val_auroc"]),
                "Selected": row["selected"],
            }
        )

    csv_df = pd.DataFrame(csv_rows, columns=csv_columns)

    latex_lines = [
        "\\begin{table}[]",
        "    \\caption{\\textbf{Hyperparameter tuning results for revision fine-tuning runs.} Mean validation AUROC across the tuning sweep for the Guo and new-diagnosis tasks at $k \\in \\{8, 16\\}$. Rows marked in bold denote the selected configuration for each model family.}",
        "    \\label{tab:ehrshot_tuning_hparams}",
        "    \\centering",
        "    \\footnotesize",
        "    \\setlength{\\tabcolsep}{3.2pt}",
        "    \\begin{tabular}{>{\\raggedright\\arraybackslash}p{2.8cm}",
        "                >{\\raggedright\\arraybackslash}p{1.35cm}",
        "                >{\\raggedright\\arraybackslash}p{0.8cm}",
        "                >{\\raggedright\\arraybackslash}p{1.1cm}",
        "                >{\\raggedright\\arraybackslash}p{1.45cm}",
        "                >{\\raggedright\\arraybackslash}p{1.55cm}",
        "                >{\\raggedright\\arraybackslash}p{1.7cm}}",
        "    \\toprule",
        "\\textbf{Config ID} & \\textbf{LR} & \\textbf{$r$} & \\textbf{Dropout} & \\textbf{$k=8$ Val. AUROC} & \\textbf{$k=16$ Val. AUROC} & \\textbf{Macro Avg.} \\\\ \\midrule",
    ]

    active_sections = [section for section in TUNING_SECTION_ORDER if section in set(summary["labeling_function"])]
    for section_idx, section_name in enumerate(active_sections):
        latex_lines.append(
            f"\\multicolumn{{7}}{{l}}{{\\textbf{{{TUNING_SECTION_DISPLAY[section_name]}}}}} \\\\ \\midrule"
        )
        section_df = summary[summary["labeling_function"] == section_name]
        for _, row in section_df.iterrows():
            config_id = str(row["config_id"])
            if row["selected"] == "yes":
                config_id = f"\\textbf{{{config_id}}}"

            latex_lines.append(
                " & ".join(
                    [
                        config_id,
                        str(row["learning_rate"]),
                        str(row["lora_r"]),
                        str(row["lora_dropout"]),
                        _format_decimal(row.get(8)),
                        _format_decimal(row.get(16)),
                        _format_decimal(row["macro_avg_mean_val_auroc"]),
                    ]
                )
                + " \\\\"
            )

        if section_idx < len(active_sections) - 1:
            latex_lines.append("\\midrule")

    latex_lines.extend(
        [
            "    \\bottomrule",
            "    \\end{tabular}",
            "\\end{table}",
        ]
    )

    table_basename = f"{output_path.stem}_tuning_summary"
    latex_path = output_path.parent / f"{table_basename}.tex"
    csv_path = output_path.parent / f"{table_basename}.csv"

    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")
    csv_df.to_csv(csv_path, index=False)

    print(f"Saved tuning LaTeX table to {latex_path}")
    print(f"Saved tuning CSV table to {csv_path}")

    return {"latex": latex_path, "csv": csv_path}


def _export_single_shot_table(
    df: pd.DataFrame,
    output_path: Path,
    target_k: int,
) -> Optional[Dict[str, Path]]:
    """Create LaTeX and CSV summaries for a specific shot size."""

    required_columns = {
        'score',
        'k',
        'model_safe',
        'model_variant',
        'model_type',
        'mean',
        'lower',
        'upper',
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(f"Missing columns for k={target_k} export: {sorted(missing_columns)}")
        return None

    subset = df[df['score'] == 'auroc'].copy()

    if subset.empty:
        print(f"No AUROC rows available for k={target_k} table export")
        return None

    if 'k' in subset.columns:
        subset['k'] = pd.to_numeric(subset['k'], errors='coerce')

    subset = subset.dropna(subset=['k'])
    subset = _apply_variant_overrides(subset)

    target_variants = set(TABLE_VARIANT_CONFIG.keys())
    subset = subset[subset['model_variant'].isin(target_variants)].copy()

    if subset.empty:
        print(f"No matching model variants found for k={target_k} table export")
        return None

    filtered_frames = []
    for variant, config in TABLE_VARIANT_CONFIG.items():
        variant_df = subset[subset['model_variant'] == variant].copy()
        if variant_df.empty:
            print(f"Skipping {variant} due to missing rows")
            continue

        shot = config.get('fixed_k') if config else None
        if shot is None:
            shot = target_k

        if shot is not None and 'k' in variant_df.columns:
            variant_df = variant_df[variant_df['k'] == shot]
            if variant_df.empty:
                print(f"Skipping {variant} due to missing k={shot} rows when assembling table for k={target_k}")
                continue

        filtered_frames.append(variant_df)

    if not filtered_frames:
        print(f"No data remaining after filtering for k={target_k} table export")
        return None

    subset = pd.concat(filtered_frames, ignore_index=True)

    if 'task_group' in subset.columns:
        missing_task_group = subset['task_group'].isna()
        if missing_task_group.any() and 'sub_task' in subset.columns:
            subset.loc[missing_task_group, 'task_group'] = subset.loc[missing_task_group, 'sub_task'].apply(_infer_task_group)
    elif 'sub_task' in subset.columns:
        subset['task_group'] = subset['sub_task'].apply(_infer_task_group)
    else:
        subset['task_group'] = 'Unknown'

    for metric in ('mean', 'lower', 'upper'):
        subset[metric] = pd.to_numeric(subset[metric], errors='coerce')

    subset = subset.dropna(subset=['mean', 'lower', 'upper'])
    if subset.empty:
        print(f"No valid metric values for k={target_k} table export")
        return None

    subset['task_group_display'] = subset['task_group'].map(TASK_GROUP_DISPLAY_NAMES).fillna(subset['task_group'])

    if 'sub_task' not in subset.columns:
        subset['sub_task'] = 'Unknown'

    per_task_summary = (
        subset.groupby(
            ['model_variant', 'model_type', 'task_group_display', 'sub_task'],
            dropna=False,
        )[['mean', 'lower', 'upper']]
        .mean()
        .reset_index()
    )

    aggregated = (
        per_task_summary.groupby(
            ['model_variant', 'model_type', 'task_group_display'],
            dropna=False,
        )[['mean', 'lower', 'upper']]
        .mean()
        .reset_index()
    )

    aggregated = aggregated[aggregated['task_group_display'].isin(TABLE_COLUMN_ORDER)]
    if aggregated.empty:
        print(f"No aggregated k={target_k} data for configured task groups; skipping export")
        return None

    rows = []
    for (model_variant, model_type), group in aggregated.groupby(['model_variant', 'model_type']):
        config = TABLE_VARIANT_CONFIG.get(model_variant, {})
        section = config.get('section', model_type)
        group = group.set_index('task_group_display')

        values = {}
        for column in TABLE_COLUMN_ORDER:
            if column in group.index:
                metrics = group.loc[column]
                values[column] = (
                    float(metrics['mean']),
                    float(metrics['lower']),
                    float(metrics['upper']),
                )

        if not values:
            print(f"Skipping {model_variant} due to missing task group coverage at k={target_k}")
            continue

        macro_mean = sum(val[0] for val in values.values()) / len(values)
        macro_lower = sum(val[1] for val in values.values()) / len(values)
        macro_upper = sum(val[2] for val in values.values()) / len(values)
        values['Macro Avg. Across Task Groups'] = (macro_mean, macro_lower, macro_upper)

        rows.append({
            'model_type': model_type,
            'section': section,
            'model_label': model_variant,
            'values': values,
        })

    if not rows:
        print(f"No rows to export for k={target_k} tables")
        return None

    rows.sort(
        key=lambda row: (
            TABLE_SECTION_ORDER.index(row['section']) if row['section'] in TABLE_SECTION_ORDER else len(TABLE_SECTION_ORDER),
            TABLE_VARIANT_ORDER.get(row['model_label'], len(TABLE_VARIANT_ORDER)),
        )
    )

    csv_columns = ['Model Type', 'Model'] + TABLE_COLUMN_ORDER + ['Macro Avg. Across Task Groups']
    csv_data = []

    for row in rows:
        csv_row = {
            'Model Type': row['section'],
            'Model': row['model_label'],
        }
        for column in TABLE_COLUMN_ORDER + ['Macro Avg. Across Task Groups']:
            metrics = row['values'].get(column)
            csv_row[column] = _format_ci_csv(*metrics) if metrics else "--"
        csv_data.append(csv_row)

    csv_df = pd.DataFrame(csv_data, columns=csv_columns)

    if target_k == 128:
        caption_line = (
            "    \\caption{\\textbf{Performance for All Examples on EHRSHOT.} Macro averaged area under receiver operating characteristic curve (AUROC) performance and bootstrapped 95\\% confidence intervals for LLM embedding model and decoder variant Qwen3-8B at $k=128$. Both model variants were also fine tuned via low-rank adaption (LoRA). Fine-tuning via LoRA does not lead to an improvement for the LLM embedding model Qwen3-Emb compared to only tuning the prediction head. For the decoder, the fine-tuned Qwen3-8B slightly outperforms Qwen3-Emb-8B, but for a much higher computational cost.}"
        )
        label_line = "    \\label{tab:ehrshot_k128_encoder_decoder}"
    else:
        caption_line = (
            f"    \\caption{{\\textbf{{EHRSHOT performance for Qwen encoder and decoder variants.}} Macro-averaged AUROC at $k={target_k}$ for Qwen3 encoder variants, the Qwen3 decoder ICL baseline, and the LoRA-tuned Qwen3 decoder.}}"
        )
        label_line = f"    \\label{{tab:ehrshot_k{target_k}_encoder_decoder}}"

    latex_lines = [
        "\\begin{table}[]",
        caption_line,
        label_line,
        "    \\centering",
        "    \\footnotesize",
        "    \\setlength{\\tabcolsep}{2.6pt}",
        "    \\begin{tabular}{>{\\raggedright\\arraybackslash}p{3.05cm}",
        "                >{\\raggedright\\arraybackslash}p{1.8cm}",
        "                >{\\raggedright\\arraybackslash}p{1.8cm}",
        "                >{\\raggedright\\arraybackslash}p{1.8cm}",
        "                >{\\raggedright\\arraybackslash}p{1.8cm}",
        "                >{\\raggedright\\arraybackslash}p{1.8cm}}",
        "    \\toprule",
        "\\textbf{Model}                           & \\textbf{Operational Outcomes} & \\textbf{Anticipating Lab Test Results} & \\textbf{Assignment of New Diagnoses} & \\textbf{Anticipating Chest X-ray Findings} & \\textbf{Macro Avg. Across Task Groups} \\\\ \\midrule",
    ]

    active_groups = []
    for section in TABLE_SECTION_ORDER:
        group_rows = [row for row in rows if row['section'] == section]
        if group_rows:
            active_groups.append((section, group_rows))

    remaining_rows = [row for row in rows if row['section'] not in TABLE_SECTION_ORDER]
    for row in remaining_rows:
        active_groups.append((row['section'], [row]))

    total_groups = len(active_groups)

    for group_index, (section, group_rows) in enumerate(active_groups):
        display_name = section
        latex_lines.append(f"\\multicolumn{{6}}{{l}}{{\\textbf{{{display_name}}}}} \\\\ \\midrule")

        for row in group_rows:
            cell_parts = []
            for column in TABLE_COLUMN_ORDER + ['Macro Avg. Across Task Groups']:
                metrics = row['values'].get(column)
                cell_parts.append(_format_ci_latex(*metrics) if metrics else "--")

            row_line = " & ".join([row['model_label'], *cell_parts]) + " \\\\"
            latex_lines.append(row_line)

        if group_index < total_groups - 1:
            latex_lines.append("\\midrule")

    latex_lines.extend([
        "    \\bottomrule",
        "    \\end{tabular}",
        "\\end{table}",
    ])

    table_basename = f"{output_path.stem}_k{target_k}_summary"
    latex_path = output_path.parent / f"{table_basename}.tex"
    csv_path = output_path.parent / f"{table_basename}.csv"

    latex_path.write_text("\n".join(latex_lines) + "\n", encoding='utf-8')
    csv_df.to_csv(csv_path, index=False)

    print(f"Saved k={target_k} LaTeX table to {latex_path}")
    print(f"Saved k={target_k} CSV table to {csv_path}")

    return {"latex": latex_path, "csv": csv_path}


def export_k_tables(
    df: pd.DataFrame,
    output_path: Path,
    target_shots: Optional[List[int]] = None,
) -> Dict[int, Dict[str, Path]]:
    """Generate manuscript-ready tables for each requested shot size."""

    auroc_df = df[df['score'] == 'auroc'] if 'score' in df.columns else df

    if target_shots is None:
        if 'k' not in auroc_df.columns:
            print("No k column found; skipping per-k table export")
            return {}

        numeric_k = pd.to_numeric(auroc_df['k'], errors='coerce').dropna()
        available_shots = sorted({int(k) for k in numeric_k if k > 0})

        ordered_shots = [shot for shot in SHOT_ORDER if shot in available_shots and shot > 0]
        remaining_shots = [shot for shot in available_shots if shot not in ordered_shots]
        target_shots = ordered_shots + remaining_shots

    assets: Dict[int, Dict[str, Path]] = {}

    for shot in target_shots:
        try:
            shot_int = int(shot)
        except (TypeError, ValueError):
            print(f"Skipping invalid k value: {shot}")
            continue

        export = _export_single_shot_table(df, output_path, shot_int)
        if export:
            assets[shot_int] = export

    return assets

def main():
    parser = argparse.ArgumentParser(description="Merge individual experiment result files")
    parser.add_argument("--results_dir", type=str,
                       required=True,
                       help="Directory containing individual result files")
    parser.add_argument(
        "--extra_results_dir",
        action="append",
        required=True,
        help="Additional directory containing result files to merge",
    )
    parser.add_argument("--output_file", type=str,
                       required=True,
                       help="Path to output merged CSV file")
    parser.add_argument(
        "--tuning_results_csv",
        type=str,
        required=True,
        help="Path to the raw step-03 tuning sweep CSV used for the hyperparameter tuning summary table",
    )
    parser.add_argument("--pattern", type=str, default="results_*.csv",
                       help="Glob pattern to match result files")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="Create comparison plots (default: True)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                       help="Skip creating plots")
    parser.add_argument("--plot_dir", type=str, default=None,
                       help="Directory to save plots (default: same as output file directory)")
    parser.add_argument("--manuscript_root", type=str, default=None,
                       help="Path to manuscript repository root for copying key outputs")
    parser.add_argument("--manuscript_table_name", type=str, default="merged_results_k128_summary.tex",
                       help="Filename to use when copying the k=128 summary table into the manuscript tables directory")
    parser.add_argument("--manuscript_figure_name", type=str, default="encoder_decoder_overall.pdf",
                       help="Filename to use when copying the overall comparison figure into the manuscript figures directory")
    parser.add_argument(
        "--baseline_dir",
        action="append",
        required=True,
        help="Path(s) to legacy results directories with per-task all_results.csv files",
    )
    parser.add_argument(
        "--overall-show-replicates",
        action="store_true",
        default=True,
        help="Overlay per-task trajectories behind the overall trends",
    )
    parser.add_argument(
        "--no-overall-show-replicates",
        dest="overall_show_replicates",
        action="store_false",
        help="Disable replicate trajectory overlays in the overall trends",
    )

    args = parser.parse_args()

    merge_results(
        args.results_dir,
        args.output_file,
        args.pattern,
        plot_results=args.plot,
        plot_dir=args.plot_dir,
        overall_show_replicates=args.overall_show_replicates,
        baseline_dirs=args.baseline_dir,
        extra_results_dirs=args.extra_results_dir,
        manuscript_root=args.manuscript_root,
        manuscript_table_name=args.manuscript_table_name,
        manuscript_figure_name=args.manuscript_figure_name,
        tuning_results_csv=args.tuning_results_csv,
    )

if __name__ == "__main__":
    main()
