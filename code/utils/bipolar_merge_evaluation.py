"""Run configured benchmarks for one bipolar interpolation point."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import yaml

from CPVM.code.utils.batch_evaluation import run_models


BENCHMARKS = {
    "trait": ("CPVM.code.test.trait_test", "TRAIT"),
    "mmlu_pro": ("CPVM.code.test.mmlu_pro_test", "MMLU-Pro"),
    "gpqa": ("CPVM.code.test.gpqa_test", "GPQA-Diamond"),
    "aime25": ("CPVM.code.test.aime25_test", "AIME 2025"),
    "gsm8k": ("CPVM.code.test.gsm8k_test", "GSM8K"),
    "math500": ("CPVM.code.test.math500_test", "MATH-500"),
    "medmcqa": ("CPVM.code.test.medmcqa_test", "MedMCQA"),
}


def load_evaluations(evaluations: dict | None) -> dict:
    """Read benchmark YAMLs and retain a hash of scoring settings."""
    if evaluations is None:
        return {}
    if not isinstance(evaluations, dict):
        raise ValueError("evaluations must map dataset names to config paths")
    loaded = {}
    for name, spec in evaluations.items():
        if name not in BENCHMARKS:
            raise ValueError(f"Unknown evaluation {name!r}; choose from {sorted(BENCHMARKS)}")
        if isinstance(spec, str):
            config_path, overrides = spec, {}
        elif isinstance(spec, dict):
            unknown = set(spec) - {"config_path", "overrides"}
            if unknown:
                raise ValueError(f"Unknown {name} evaluation options: {sorted(unknown)}")
            config_path, overrides = spec.get("config_path"), spec.get("overrides", {})
        else:
            raise ValueError(f"evaluations.{name} must be a config path or mapping")
        if not isinstance(config_path, str) or not config_path.strip():
            raise ValueError(f"evaluations.{name}.config_path must be a path")
        if not isinstance(overrides, dict):
            raise ValueError(f"evaluations.{name}.overrides must be a mapping")
        source = Path(config_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Evaluation config does not exist: {source}")
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError(f"Evaluation config must be a mapping: {source}")
        config.update(overrides)
        for key in ("model_paths", "model_path", "output_path"):
            config.pop(key, None)
        stable = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str)
        loaded[name] = {
            "config_path": str(source),
            "config": config,
            "config_hash": hashlib.sha256(stable.encode("utf-8")).hexdigest()[:12],
        }
    return loaded


def evaluate_curve_point(name, specification, model_path, run_directory,
                         run_name, path_method, trait, coordinate_label):
    """Run one benchmark in a spawned process and return its metrics."""
    config = dict(specification["config"])
    config["model_paths"] = [str(model_path)]
    result_directory = (run_directory / "evaluation" / name /
                        f"t-{coordinate_label}" / specification["config_hash"])
    config["output_path"] = str(result_directory)
    config["test_name"] = f"{run_name}-t-{coordinate_label}"
    module_name, benchmark_title = BENCHMARKS[name]
    benchmark = importlib.import_module(module_name)
    print(f"Evaluating {benchmark_title}: {path_method}, t={coordinate_label}", flush=True)
    rows = run_models(config, benchmark.run, benchmark_title)
    if len(rows) != 1 or rows[0]["status"] != "OK":
        raise RuntimeError(f"{benchmark_title} did not complete for t={coordinate_label}")
    files = sorted(result_directory.glob("*.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one {benchmark_title} report in {result_directory}")
    metrics = rows[0]["metrics"]
    if name == "trait":
        primary = f"{trait} (%)"
    elif name == "mmlu_pro":
        primary = "Accuracy (%)"
    elif name == "aime25":
        primary = "Mean accuracy (%)"
    else:
        primary = f"Avg@{config['samples_per_question']} (%)"
    if primary not in metrics:
        raise ValueError(f"{benchmark_title} did not report {primary!r}")
    return {
        "benchmark": benchmark_title,
        "config_path": specification["config_path"],
        "config_hash": specification["config_hash"],
        "primary_metric": primary,
        "primary_value": metrics[primary],
        "metrics": metrics,
        "result_path": str(files[0]),
        "model_path_used": str(model_path),
        "model_path_is_temporary": any(
            part.startswith(".temporary-models-") for part in model_path.parts
        ),
    }
