from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from select_best_params import select_best_configs


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Config YAML must be a mapping")
        return loaded
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for tuning config") from exc


def _model_safe(model_name: str) -> str:
    return model_name.replace("/", "_")


def _build_jobs(config: dict[str, Any], output_dir: Path, max_input_length: int) -> list[dict[str, Any]]:
    tasks = [str(x) for x in config["tasks"]]
    ks = [int(x) for x in config["ks"]]
    replicates = [int(x) for x in config.get("replicates", [0])]
    model_names = dict(config["model_names"])

    jobs: list[dict[str, Any]] = []
    for labeling_function, script_name, config_list_key in (
        ("llm_encoder_ft", "10a_fit_and_eval_encoder.py", "encoder_configs"),
        ("llm_decoder_ft", "10b_fit_and_eval_decoder.py", "decoder_configs"),
    ):
        model_name = str(model_names[labeling_function])
        for cfg in config.get(config_list_key, []):
            config_id = str(cfg["id"])
            params = dict(cfg.get("params", {}))
            for task in tasks:
                for k in ks:
                    for replicate in replicates:
                        output_filename = (
                            f"results_{_model_safe(model_name)}_{task}_k{k}_r{replicate}.csv"
                        )
                        command = [
                            "python",
                            str(REPO_ROOT / "ehrshot" / script_name),
                            "--sub_task",
                            task,
                            "--k",
                            str(k),
                            "--replicate",
                            str(replicate),
                            "--model_name",
                            model_name,
                            "--output_dir",
                            str(output_dir / "run_outputs" / config_id),
                            "--max_input_length",
                            str(max_input_length),
                            "--eval_train_val",
                            "--overwrite",
                            "--labeling_function",
                            labeling_function,
                        ]
                        for key, value in params.items():
                            command.extend([f"--{key}", str(value)])
                        jobs.append(
                            {
                                "labeling_function": labeling_function,
                                "config_id": config_id,
                                "sub_task": task,
                                "k": k,
                                "replicate": replicate,
                                "output_csv": str(output_dir / "run_outputs" / config_id / output_filename),
                                "command": command,
                            }
                        )
    return jobs


def _run_jobs(jobs: list[dict[str, Any]], dry_run: bool) -> None:
    for idx, job in enumerate(jobs, start=1):
        cmd_text = " ".join(job["command"])
        print(
            f"[{idx}/{len(jobs)}] {job['labeling_function']} {job['config_id']} "
            f"task={job['sub_task']} k={job['k']} r={job['replicate']}"
        )
        print(f"CMD: {cmd_text}")
        if not dry_run:
            subprocess.run(job["command"], check=True)


def _run_single_job(jobs: list[dict[str, Any]], job_index: int) -> None:
    if job_index < 0 or job_index >= len(jobs):
        raise IndexError(f"job_index={job_index} out of range for {len(jobs)} jobs")
    job = jobs[job_index]
    cmd_text = " ".join(job["command"])
    print(
        f"[job {job_index + 1}/{len(jobs)}] {job['labeling_function']} {job['config_id']} "
        f"task={job['sub_task']} k={job['k']} r={job['replicate']}"
    )
    print(f"CMD: {cmd_text}")
    subprocess.run(job["command"], check=True)


def _collect_raw_results(jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for job in jobs:
        output_csv = Path(str(job["output_csv"]))
        if not output_csv.exists():
            continue
        df = pd.read_csv(output_csv)
        df["config_id"] = job["config_id"]
        df["labeling_function"] = job["labeling_function"]
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _write_best_params(best: dict[str, dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "best_params_encoder.json").write_text(
        json.dumps(best.get("llm_encoder_ft", {}), indent=2, sort_keys=True)
    )
    (output_dir / "best_params_decoder.json").write_text(
        json.dumps(best.get("llm_decoder_ft", {}), indent=2, sort_keys=True)
    )


def _config_params_by_id(config: dict[str, Any], config_key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cfg in config.get(config_key, []):
        cfg_id = str(cfg.get("id", ""))
        if not cfg_id:
            continue
        out[cfg_id] = dict(cfg.get("params", {}))
    return out


def _attach_tuned_hparams(best: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    encoder_params = _config_params_by_id(config, "encoder_configs")
    decoder_params = _config_params_by_id(config, "decoder_configs")
    out: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []

    for labeling_function, payload in best.items():
        row = dict(payload)
        config_id = str(row.get("config_id", ""))
        if labeling_function == "llm_encoder_ft" and config_id in encoder_params:
            row["tuned_hparams"] = encoder_params[config_id]
        elif labeling_function == "llm_decoder_ft" and config_id in decoder_params:
            row["tuned_hparams"] = decoder_params[config_id]
        else:
            unresolved.append(f"{labeling_function}:{config_id}")
        out[labeling_function] = row

    if unresolved:
        raise ValueError(
            "Selected best config_id could not be resolved to tuned_hparams: "
            + ", ".join(unresolved)
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune finetuning params for guo/new tasks at k=8/16")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--serializations-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=4096,
        help="Per-record token budget passed to the fit_and_eval scripts",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--run-job-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_modes = sum(
        [
            bool(args.plan_only),
            bool(args.collect_only),
            args.run_job_index is not None,
        ]
    )
    if selected_modes > 1:
        raise ValueError("Use at most one of --plan-only, --collect-only, or --run-job-index")
    if args.dry_run and (args.collect_only or args.run_job_index is not None):
        raise ValueError("--dry-run cannot be combined with --collect-only or --run-job-index")

    config = _load_yaml(args.config)
    output_dir = Path(str(args.output_dir or config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    serializations_path = Path(
        str(
            args.serializations_path
            or config.get("serializations_path", REPO_ROOT / "EHRSHOT_ASSETS/benchmark/tasks_serializations.pkl")
        )
    )

    jobs = _build_jobs(config=config, output_dir=output_dir, max_input_length=args.max_input_length)
    for job in jobs:
        job["command"].extend(["--serializations_path", str(serializations_path)])
    print(f"Planned jobs: {len(jobs)}")

    manifest_path = args.manifest_path or (output_dir / "tuning_jobs_manifest.json")
    manifest_path.write_text(json.dumps(jobs, indent=2, sort_keys=True))
    print(f"Wrote job manifest to {manifest_path}")

    if args.plan_only:
        print("PLAN ONLY: manifest generated.")
        return 0

    if args.run_job_index is not None:
        _run_single_job(jobs=jobs, job_index=args.run_job_index)
        return 0

    if args.collect_only:
        raw = _collect_raw_results(jobs)
        raw_path = output_dir / "tuning_results_raw.csv"
        raw.to_csv(raw_path, index=False)
        print(f"Wrote raw tuning results to {raw_path}")

        best, aggregated = select_best_configs(
            raw,
            config_order=[str(x) for x in config.get("config_order", [])],
        )
        best = _attach_tuned_hparams(best=best, config=config)

        aggregated_path = output_dir / "tuning_results_aggregated.csv"
        aggregated.to_csv(aggregated_path, index=False)
        _write_best_params(best, output_dir)

        print(f"Wrote aggregated tuning results to {aggregated_path}")
        print(f"Wrote best parameter JSON files to {output_dir}")
        return 0

    _run_jobs(jobs=jobs, dry_run=args.dry_run)

    if args.dry_run:
        print("DRY RUN: no experiments executed.")
        return 0

    raw = _collect_raw_results(jobs)
    raw_path = output_dir / "tuning_results_raw.csv"
    raw.to_csv(raw_path, index=False)
    print(f"Wrote raw tuning results to {raw_path}")

    best, aggregated = select_best_configs(
        raw,
        config_order=[str(x) for x in config.get("config_order", [])],
    )
    best = _attach_tuned_hparams(best=best, config=config)

    aggregated_path = output_dir / "tuning_results_aggregated.csv"
    aggregated.to_csv(aggregated_path, index=False)
    _write_best_params(best, output_dir)

    print(f"Wrote aggregated tuning results to {aggregated_path}")
    print(f"Wrote best parameter JSON files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
