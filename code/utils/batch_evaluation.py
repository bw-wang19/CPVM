"""Run model evaluations sequentially and print one comparison table."""

import json
import multiprocessing as mp
from pathlib import Path
from tempfile import TemporaryDirectory


TRAIT_SUMMARY_ORDER = (
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
)


SUMMARY_METRIC_SPECS = (
    ("accuracy", "Accuracy (%)", 100),
    ("strict_accuracy", "Strict accuracy (%)", 100),
    ("flexible_accuracy", "Flexible accuracy (%)", 100),
    ("hendrycks_accuracy", "Hendrycks accuracy (%)", 100),
    ("mean_accuracy", "Mean accuracy (%)", 100),
    ("parse_rate", "Parse rate (%)", 100),
    ("flexible_parse_rate", "Flexible parse rate (%)", 100),
    ("grader_timeout_count", "Grader timeouts", 1),
    ("samples_per_question", "Samples/question", 1),
    ("average_output_tokens", "Avg output tokens", 1),
    ("median_output_tokens", "Median output tokens", 1),
    ("length_truncated_rate", "Length-truncated (%)", 100),
    ("length_truncated_accuracy", "Truncated accuracy (%)", 100),
    ("non_truncated_accuracy", "Non-truncated accuracy (%)", 100),
    ("strict_completion_accuracy", "Strict completion accuracy (%)", 100),
    ("length_truncated_token_rate", "Truncated token share (%)", 100),
)


def _summary_metrics(summary):
    """Select and scale comparable metrics for the all-models table."""

    metrics = {}
    if "average_accuracy" in summary:
        sample_count = summary.get("samples_per_question")
        label = (
            f"Avg@{sample_count} (%)"
            if sample_count is not None else "Average accuracy (%)"
        )
        metrics[label] = 100 * summary["average_accuracy"]
    if "test_at_n" in summary:
        sample_count = summary.get("samples_per_question")
        label = (
            f"Test@{sample_count} (%)"
            if sample_count is not None else "Test@N (%)"
        )
        metrics[label] = 100 * summary["test_at_n"]
    metrics.update({
        label: scale * summary[key]
        for key, label, scale in SUMMARY_METRIC_SPECS
        if key in summary and summary[key] is not None
    })
    metrics["Questions"] = summary["num_questions"]
    return metrics


def ordered_trait_names(names):
    """Return measured traits in the canonical TRAIT summary order."""
    names = list(names)
    ordered = [name for name in TRAIT_SUMMARY_ORDER if name in names]
    return ordered + [name for name in names if name not in TRAIT_SUMMARY_ORDER]


def _evaluate_model(config, run_one, result_path):
    """Keep model/GPU lifetime inside one process and return only summary data."""
    report = run_one(config)
    if "scores" in report:
        metrics = {
            f"{personality} (%)": report["scores"][personality]
            for personality in ordered_trait_names(report["scores"])
        }
        metrics["Questions"] = report["num_questions"]
    else:
        metrics = _summary_metrics(report["summary"])
    Path(result_path).write_text(
        json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
    )


def _print_table(benchmark, rows):
    columns = ["Model path", "Status"]
    for row in rows:
        for name in row["metrics"]:
            if name not in columns:
                columns.append(name)
    if benchmark == "TRAIT":
        canonical_columns = [
            f"{personality} (%)" for personality in TRAIT_SUMMARY_ORDER
        ]
        present_columns = set(columns[2:])
        columns = (
            columns[:2]
            + [name for name in canonical_columns if name in present_columns]
            + [name for name in columns[2:] if name not in canonical_columns]
        )

    values = []
    for row in rows:
        cells = [row["model_path"], row["status"]]
        for name in columns[2:]:
            value = row["metrics"].get(name)
            cells.append(
                "-" if value is None else
                str(value) if name in {"Questions", "Samples/question"}
                else f"{value:.2f}"
            )
        values.append(cells)
    widths = [
        max(len(column), *(len(row[index]) for row in values))
        for index, column in enumerate(columns)
    ]
    def format_row(cells):
        return "| " + " | ".join(
            cell.ljust(width) for cell, width in zip(cells, widths)
        ) + " |"
    print(f"\n=== {benchmark}: ALL MODELS SUMMARY ===", flush=True)
    print(format_row(columns), flush=True)
    print("| " + " | ".join("-" * width for width in widths) + " |", flush=True)
    for row in values:
        print(format_row(row), flush=True)


def run_models(config, run_one, benchmark):
    """Evaluate model_paths in order; model_path remains a single-model alias.

    Returns rows containing model_path, status and numeric metrics. Each model
    runs in a fresh process so Transformers/vLLM GPU memory is fully released
    before the next model starts. Existing run_one(config) APIs stay unchanged.
    """
    model_paths = config.get("model_paths")
    if model_paths is None:
        model_paths = config.get("model_path")
    if isinstance(model_paths, (str, Path)):
        model_paths = [model_paths]
    if not isinstance(model_paths, (list, tuple)) or not model_paths:
        raise ValueError("Set model_paths to a nonempty list of model paths.")
    if any(not isinstance(path, (str, Path)) or not str(path).strip()
           for path in model_paths):
        raise ValueError("Each model_paths entry must be a nonempty model path.")

    context = mp.get_context("spawn")
    rows = []
    with TemporaryDirectory(prefix="cpvm-model-evaluation-") as temporary_dir:
        for index, model_path in enumerate(model_paths, start=1):
            single_config = dict(config)
            single_config.pop("model_paths", None)
            single_config["model_path"] = str(model_path)
            summary_path = Path(temporary_dir) / f"{index}.json"
            print(
                f"\n=== {benchmark} [{index}/{len(model_paths)}]: {model_path} ===",
                flush=True,
            )
            process = context.Process(
                target=_evaluate_model,
                args=(single_config, run_one, str(summary_path)),
            )
            process.start()
            try:
                process.join()
            except KeyboardInterrupt:
                process.terminate()
                process.join()
                raise
            success = process.exitcode == 0 and summary_path.exists()
            rows.append({
                "model_path": str(model_path),
                "status": "OK" if success else f"FAILED (exit {process.exitcode})",
                "metrics": json.loads(summary_path.read_text(encoding="utf-8"))
                if success else {},
            })
            process.close()
    _print_table(benchmark, rows)
    failures = sum(row["status"] != "OK" for row in rows)
    if failures:
        raise RuntimeError(f"{failures} model evaluation(s) failed; see the log above.")
    return rows
