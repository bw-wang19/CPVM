#!/usr/bin/env python3
"""Generate the Top-k ranking ablations and test each model on TRAIT."""

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml


WORKSPACE = Path("/home/wbw/workspace")
PROJECT = WORKSPACE / "CPVM"
TOPK_CONFIG = PROJECT / "config/topk_merge.yaml"
TRAIT_CONFIG = PROJECT / "config/trait_test.yaml"
RESULT_DIR = Path("/mnt/nvme2/models/amadeus/cpvm_output/test/topk_ranking_sweep/trait")
TEST_NAME = "topk-ranking-ablation-20260905"

# Set KEEP_MODELS=1 when launching if all merged checkpoints should be retained.
KEEP_MODELS = os.environ.get("KEEP_MODELS", "0") == "1"

RUNS = [
    ("contribution", k_percent, True)
    for k_percent in (1, 5, 20)
] + [
    (ranking_method, k_percent, use_weight_rescale)
    for ranking_method in ("delta_magnitude", "finetuned_magnitude")
    for k_percent in (1, 5, 10, 20)
    for use_weight_rescale in (False, True)
]

sys.path.insert(0, str(WORKSPACE))
from CPVM.code.topk_merge import build_run_name  # noqa: E402


def load_yaml(path: Path) -> dict:
    """Load one existing experiment configuration."""

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def run_module(module: str, config: dict) -> None:
    """Run an existing CPVM entry point with a temporary YAML configuration."""

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8"
    ) as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
        file.flush()
        subprocess.run(
            [sys.executable, "-m", module, "--config", file.name],
            cwd=WORKSPACE,
            check=True,
        )


def model_is_complete(model_path: Path) -> bool:
    """Return whether this sweep finished writing a checkpoint."""

    return (model_path / ".merge_complete").exists()


def result_has_test(result_path: Path) -> bool:
    """Return whether this sweep's TRAIT record is already present."""

    if not result_path.exists():
        return False
    with result_path.open(encoding="utf-8") as file:
        report = json.load(file)
    return any(
        test.get("test_name") == TEST_NAME
        for test in report.get("tests", [])
    )


def remove_model(model_path: Path) -> None:
    """Delete only the checkpoint that has just finished its TRAIT test."""

    if model_path.exists():
        print(f"Removing tested checkpoint: {model_path}", flush=True)
        shutil.rmtree(model_path)


def main() -> None:
    """Run all 19 merge-and-TRAIT jobs sequentially."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    topk_template = load_yaml(TOPK_CONFIG)
    trait_template = load_yaml(TRAIT_CONFIG)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for index, (ranking_method, k_percent, use_weight_rescale) in enumerate(
        RUNS, start=1
    ):
        merge_config = deepcopy(topk_template)
        merge_config.update(
            ranking_method=ranking_method,
            k_percent=k_percent,
            alpha=1.0,
            use_weight_rescale=use_weight_rescale,
        )
        if ranking_method == "contribution":
            merge_config.update(
                contribution_channel="L_plus",
                selection_mode="positive",
            )

        run_name = build_run_name(
            experiment_name=merge_config["experiment_name"],
            k_percent=k_percent,
            alpha=merge_config["alpha"],
            selection_mode=merge_config["selection_mode"],
            contribution_channel=merge_config["contribution_channel"],
            use_weight_rescale=use_weight_rescale,
            ranking_method=ranking_method,
        )
        model_path = Path(merge_config["output_dir"]) / run_name
        result_path = RESULT_DIR / f"trait-{run_name}.json"

        print(
            f"\n{'=' * 78}\n"
            f"[{index}/{len(RUNS)}] {run_name}\n"
            f"{'=' * 78}",
            flush=True,
        )

        if result_has_test(result_path):
            print(f"Skipping completed TRAIT result: {result_path}", flush=True)
            if not KEEP_MODELS:
                remove_model(model_path)
            continue

        if model_is_complete(model_path):
            print(f"Reusing merged checkpoint: {model_path}", flush=True)
        else:
            if model_path.exists():
                remove_model(model_path)
            run_module("CPVM.code.topk_merge", merge_config)
            (model_path / ".merge_complete").touch()

        trait_config = deepcopy(trait_template)
        trait_config.update(
            model_path=str(model_path),
            adapter_path=None,
            output_path=str(RESULT_DIR),
            result_model_name=run_name,
            test_name=TEST_NAME,
            use_vllm=False,
            data_parallel_size=2,
        )
        run_module("CPVM.code.test.trait_test", trait_config)

        if not KEEP_MODELS:
            remove_model(model_path)

    print("\nAll Top-k ranking TRAIT evaluations are complete.", flush=True)


if __name__ == "__main__":
    main()
