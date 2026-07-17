from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TARGET_PREFIXES = ("guo_", "new_")
TARGET_KS = (8, 16)
TARGET_SCORE = "auroc"


def _rank_map(config_order: list[str]) -> dict[str, int]:
    return {config_id: idx for idx, config_id in enumerate(config_order)}


def _build_selected_configs(
    filtered_df: pd.DataFrame,
    labeling_function: str,
    config_id: str,
) -> list[dict[str, Any]]:
    subset = filtered_df[
        (filtered_df["labeling_function"].astype(str) == str(labeling_function))
        & (filtered_df["config_id"].astype(str) == str(config_id))
    ].copy()
    if subset.empty:
        return []

    has_model = "model_name" in subset.columns

    rows: list[dict[str, Any]] = []
    for (sub_task, k), group in subset.groupby(["sub_task", "k"], dropna=False, sort=True):
        row: dict[str, Any] = {
            "sub_task": str(sub_task),
            "k": int(k),
        }
        if has_model:
            model_values = sorted(
                {
                    str(v)
                    for v in group["model_name"].tolist()
                    if pd.notna(v) and str(v) != ""
                }
            )
            if model_values:
                row["model_name"] = model_values[0]
        rows.append(row)

    return rows


def select_best_configs(
    results_df: pd.DataFrame,
    config_order: list[str],
    task_prefixes: tuple[str, ...] = TARGET_PREFIXES,
    ks: tuple[int, ...] = TARGET_KS,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    required = {"labeling_function", "config_id", "sub_task", "k", "score", "value_val"}
    missing = required.difference(results_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = results_df.copy()
    filtered = filtered[filtered["score"] == TARGET_SCORE]
    filtered = filtered[filtered["sub_task"].astype(str).str.startswith(task_prefixes)]
    filtered = filtered[filtered["k"].isin(ks)]
    filtered = filtered[filtered["value_val"].notna()]

    aggregated = (
        filtered.groupby(["labeling_function", "config_id"], as_index=False)["value_val"]
        .mean()
        .rename(columns={"value_val": "mean_val_auroc"})
    )

    rank = _rank_map(config_order)
    default_rank = len(rank)
    aggregated["_order_rank"] = aggregated["config_id"].map(lambda x: rank.get(str(x), default_rank))
    aggregated = aggregated.sort_values(
        by=["labeling_function", "mean_val_auroc", "_order_rank", "config_id"],
        ascending=[True, False, True, True],
        kind="mergesort",
    )

    best: dict[str, dict[str, Any]] = {}
    for labeling_function, group in aggregated.groupby("labeling_function", sort=False):
        top = group.iloc[0]
        selected_config_id = str(top["config_id"])
        best[str(labeling_function)] = {
            "config_id": selected_config_id,
            "mean_val_auroc": float(top["mean_val_auroc"]),
            "selection_metric": TARGET_SCORE,
            "task_prefixes": list(task_prefixes),
            "ks": list(ks),
            "selected_configs": _build_selected_configs(filtered, str(labeling_function), selected_config_id),
        }

    return best, aggregated.drop(columns=["_order_rank"]).reset_index(drop=True)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("YAML config must be a mapping")
        return loaded
    except ImportError:
        raise RuntimeError("PyYAML is required to load tuning config")


def _write_best_jsons(best: dict[str, dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder = best.get("llm_encoder_ft", {})
    decoder = best.get("llm_decoder_ft", {})

    (output_dir / "best_params_encoder.json").write_text(json.dumps(encoder, indent=2, sort_keys=True))
    (output_dir / "best_params_decoder.json").write_text(json.dumps(decoder, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Select best finetuning params by mean validation AUROC")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--config_order", type=str, nargs="*", default=None)
    args = parser.parse_args()

    config_order: list[str]
    if args.config_order:
        config_order = list(args.config_order)
    elif args.config is not None:
        config = _load_simple_yaml(args.config)
        config_order = [str(x) for x in config.get("config_order", [])]
    else:
        config_order = []

    df = pd.read_csv(args.input_csv)
    best, aggregated = select_best_configs(df, config_order=config_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregated_path = args.output_dir / "tuning_results_aggregated.csv"
    aggregated.to_csv(aggregated_path, index=False)
    _write_best_jsons(best, args.output_dir)

    print(f"Wrote aggregated results to {aggregated_path}")
    print(f"Wrote best encoder/decoder params to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
