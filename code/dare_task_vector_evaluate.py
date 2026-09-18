"""Evaluate a DaRE seed's retained checkpoints on TRAIT only."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from CPVM.code.dare_task_vector_merge import (
    dare_output_path,
    task_vector_output_path,
)
from CPVM.code.utils.arguments import parse_args_yaml


PERSONALITIES = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
]


def load_yaml(path: str) -> dict:
    """Load one existing benchmark configuration."""

    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def model_specs(config: dict) -> list[dict]:
    """Describe every configured checkpoint in stable evaluation order."""

    seed = int(config["seed"])
    specs = []
    for retain in config["retain_percentages"]:
        specs.append(
            {
                "name": f"DaRE seed {seed} retain {retain:g}%",
                "method": "DaRE",
                "seed": seed,
                "retain_percent": retain,
                "path": dare_output_path(config["output_dir"], retain, seed),
            }
        )
    if config.get("include_task_vector", False):
        alpha = float(config["task_vector_alpha"])
        specs.append(
            {
                "name": f"Task Vector alpha={alpha:g}",
                "method": "Task Vector",
                "seed": None,
                "retain_percent": None,
                "path": task_vector_output_path(config["output_dir"], alpha),
            }
        )
    return specs


def trait_result_path(config: dict, model_path: Path) -> Path:
    """Return the unique TRAIT report path for one merged model."""

    return (
        Path(config["result_output_dir"])
        / "trait"
        / f"trait-{model_path.name}.json"
    )


def result_has_test(path: Path, test_name: str) -> bool:
    """Return whether a valid TRAIT report already contains this run."""

    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as file:
            report = json.load(file)
    except (OSError, json.JSONDecodeError):
        return False
    return any(test.get("test_name") == test_name for test in report.get("tests", []))


def run_benchmark(
    module: str,
    base_config_path: str,
    model_path: Path,
    output_path: Path,
    test_name: str,
    expected_result: Path,
) -> None:
    """Run one existing test in an isolated process, unless already complete."""

    if result_has_test(expected_result, test_name):
        print(f"Skipping existing result: {expected_result}", flush=True)
        return

    benchmark_config = load_yaml(base_config_path)
    benchmark_config["model_path"] = str(model_path)
    benchmark_config["output_path"] = str(output_path)
    benchmark_config["test_name"] = test_name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8"
    ) as file:
        yaml.safe_dump(benchmark_config, file, sort_keys=False, allow_unicode=True)
        file.flush()
        subprocess.run(
            [sys.executable, "-m", module, "--config", file.name], check=True
        )


def collect_row(config: dict, spec: dict) -> dict | None:
    """Read one model's finished TRAIT report into one summary row."""

    path = trait_result_path(config, spec["path"])
    if not result_has_test(path, config["test_name"]):
        return None

    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    trait_record = next(
        test
        for test in reversed(report["tests"])
        if test["test_name"] == config["test_name"]
    )
    row = {
        "model": spec["name"],
        "method": spec["method"],
        "seed": spec["seed"],
        "retain_percent": spec["retain_percent"],
        "model_path": str(spec["path"]),
        "result_path": str(path),
    }
    row.update(trait_record["scores"])
    return row


def write_summary(config: dict, specs: list[dict]) -> None:
    """Write CSV, JSON, and Markdown tables for all completed models."""

    rows = [row for spec in specs if (row := collect_row(config, spec))]
    if not rows:
        return

    root = Path(config["result_output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "method",
        "seed",
        "retain_percent",
        *PERSONALITIES,
        "model_path",
        "result_path",
    ]
    with (root / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    columns = ["seed", "retain_percent", *PERSONALITIES]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---:"] * len(columns)) + " |",
    ]
    for row in rows:
        values = [
            str(row["seed"]),
            f"{row['retain_percent']:g}",
            *(f"{row[name]:.2f}" for name in PERSONALITIES),
        ]
        lines.append("| " + " | ".join(values) + " |")
    table = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(table, encoding="utf-8")
    print("\n=== COMPLETED RESULTS ===\n" + table, flush=True)


def run(config: dict) -> None:
    """Evaluate each configured checkpoint on TRAIT, then summarize."""

    specs = model_specs(config)
    root = Path(config["result_output_dir"])
    for index, spec in enumerate(specs, start=1):
        model_path = spec["path"]
        print(
            f"\n{'=' * 72}\n[{index}/{len(specs)}] {spec['name']}\n"
            f"{model_path}\n{'=' * 72}",
            flush=True,
        )
        result_path = trait_result_path(config, model_path)
        run_benchmark(
            "CPVM.code.test.trait_test",
            config["trait_config_path"],
            model_path,
            root / "trait",
            config["test_name"],
            result_path,
        )
        write_summary(config, specs)


def main() -> None:
    """Load the YAML interface and run the TRAIT evaluation sweep."""

    run(parse_args_yaml("Evaluate MergeLM DaRE models on TRAIT"))


if __name__ == "__main__":
    main()
