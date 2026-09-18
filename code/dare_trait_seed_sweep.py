"""Run five DaRE seeds serially while retaining at most four checkpoints."""

from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

from CPVM.code.dare_task_vector_evaluate import (
    PERSONALITIES,
    model_specs,
    result_has_test,
    trait_result_path,
)
from CPVM.code.dare_task_vector_merge import (
    checkpoint_exists,
    dare_output_path,
)
from CPVM.code.utils.arguments import parse_args_yaml


WORKSPACE = Path(__file__).resolve().parents[2]
EXPECTED_RETAINS = [1, 5, 10, 20]


def validate_config(config: dict) -> list[int]:
    """Fail before model work if the bounded sweep contract is violated."""

    seeds = [int(seed) for seed in config["seeds"]]
    retains = list(config["retain_percentages"])
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(seed == 0 for seed in seeds):
        raise ValueError("seeds must contain five distinct, non-zero integers")
    if retains != EXPECTED_RETAINS:
        raise ValueError(f"retain_percentages must be exactly {EXPECTED_RETAINS}")
    if config.get("include_task_vector", False):
        raise ValueError("This sweep is DaRE-only; include_task_vector must be false")
    return seeds


def config_for_seed(config: dict, seed: int) -> dict:
    """Build one seed's merge/evaluation config and unique result namespace."""

    seed_config = deepcopy(config)
    seed_config["seed"] = seed
    seed_config["include_task_vector"] = False
    seed_config["skip_existing"] = True
    seed_config["result_output_dir"] = str(
        Path(config["result_output_dir"]) / f"seed-{seed}"
    )
    seed_config["test_name"] = f"{config['test_name']}-seed{seed}"
    return seed_config


def model_paths(config: dict, seed: int) -> list[Path]:
    """Return the four checkpoint paths owned by one configured seed."""

    return [
        dare_output_path(config["output_dir"], retain, seed)
        for retain in config["retain_percentages"]
    ]


def remove_checkpoint(output_root: Path, checkpoint: Path) -> None:
    """Remove one known direct child of the dedicated working directory."""

    root = output_root.resolve()
    target = checkpoint.resolve()
    if target.parent != root:
        raise ValueError(f"Refusing to remove checkpoint outside {root}: {target}")
    if target.exists():
        print(f"Removing temporary checkpoint: {target}", flush=True)
        shutil.rmtree(target)


def remove_seed_models(config: dict, seed: int) -> None:
    """Delete exactly the four temporary DaRE checkpoints for one seed."""

    output_root = Path(config["output_dir"])
    for checkpoint in model_paths(config, seed):
        remove_checkpoint(output_root, checkpoint)


def remove_other_seed_models(config: dict, seeds: list[int], keep_seed: int) -> None:
    """Ensure configured sweep checkpoints from other seeds do not accumulate."""

    for seed in seeds:
        if seed != keep_seed:
            remove_seed_models(config, seed)


def checkpoints_complete(config: dict, seed: int) -> bool:
    """Return whether all four models for a seed finished saving."""

    return all(checkpoint_exists(path) for path in model_paths(config, seed))


def results_complete(seed_config: dict) -> bool:
    """Return whether all four unique TRAIT result files are complete."""

    return all(
        result_has_test(
            trait_result_path(seed_config, spec["path"]),
            seed_config["test_name"],
        )
        for spec in model_specs(seed_config)
    )


def run_module(module: str, config: dict) -> None:
    """Run one existing module with an isolated generated YAML file."""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8"
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
        file.flush()
        subprocess.run(
            [sys.executable, "-m", module, "--config", file.name],
            check=True,
            cwd=WORKSPACE,
        )


def aggregate_rows(config: dict, seeds: list[int]) -> list[dict]:
    """Load every completed per-seed summary in requested seed order."""

    rows = []
    result_root = Path(config["result_output_dir"])
    for seed in seeds:
        summary_path = result_root / f"seed-{seed}" / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open(encoding="utf-8") as file:
            rows.extend(json.load(file))
    return rows


def write_aggregate_summary(config: dict, seeds: list[int]) -> None:
    """Write a durable 20-row JSON/CSV/Markdown summary outside scratch."""

    rows = aggregate_rows(config, seeds)
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
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Aggregate summary now contains {len(rows)} rows: {root}", flush=True)


def run(config: dict) -> None:
    """Generate, test, and retire each seed before moving to the next one."""

    seeds = validate_config(config)
    final_seed = seeds[-1]

    for index, seed in enumerate(seeds, start=1):
        seed_config = config_for_seed(config, seed)
        is_final = seed == final_seed
        print(
            f"\n{'#' * 72}\nSEED [{index}/{len(seeds)}]: {seed}\n{'#' * 72}",
            flush=True,
        )

        seed_results_done = results_complete(seed_config)
        seed_models_done = checkpoints_complete(config, seed)
        seed_summary = Path(seed_config["result_output_dir"]) / "summary.json"
        if seed_results_done and not seed_summary.exists():
            # Every benchmark call is skipped; this only rebuilds the summary.
            run_module("CPVM.code.dare_task_vector_evaluate", seed_config)

        if seed_results_done and not is_final:
            print(f"Seed {seed} TRAIT results already complete; skipping it", flush=True)
            remove_seed_models(config, seed)
            write_aggregate_summary(config, seeds)
            continue

        remove_other_seed_models(config, seeds, keep_seed=seed)

        if not seed_models_done:
            run_module("CPVM.code.dare_task_vector_merge", seed_config)
        else:
            print(f"Reusing four complete seed {seed} checkpoints", flush=True)

        if not seed_results_done:
            run_module("CPVM.code.dare_task_vector_evaluate", seed_config)
        else:
            print(f"Seed {seed} TRAIT results already complete", flush=True)

        if not results_complete(seed_config):
            raise RuntimeError(f"Seed {seed} did not produce all four TRAIT results")

        write_aggregate_summary(config, seeds)
        if not is_final:
            remove_seed_models(config, seed)

    remove_other_seed_models(config, seeds, keep_seed=final_seed)
    write_aggregate_summary(config, seeds)
    print(
        f"\nSweep complete. Only seed {final_seed}'s four checkpoints are retained.",
        flush=True,
    )


def main() -> None:
    """Load the YAML interface and run the bounded five-seed sweep."""

    run(parse_args_yaml("Five-seed DaRE to TRAIT sweep"))


if __name__ == "__main__":
    main()
