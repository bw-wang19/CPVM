"""Write per-benchmark tables and figures for a bipolar merge sweep."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_AGGREGATE_TERMS = (
    "average",
    "overall",
    "composite",
    "aggregate",
    "parse",
    "truncat",
    "accuracy",
    "success",
    "strict",
    "token",
    "length",
)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _model_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("model_path", "path", "name"):
            if value.get(key):
                return _model_label(value[key])
        return ""
    return Path(str(value).rstrip("/")).name


def _source_labels(source_models: Any) -> tuple[str, str]:
    """Return the center-model label and a compact subtitle for the figure."""
    if not isinstance(source_models, dict):
        label = _model_label(source_models)
        return label, label

    aliases = (
        ("base", ("base", "base_model", "base_model_path", "center", "center_model")),
        ("plus", ("plus", "plus_model", "plus_model_path", "high", "high_model")),
        ("star", ("star", "star_model", "star_model_path", "low", "low_model")),
    )
    labels: list[str] = []
    center = ""
    for role, keys in aliases:
        label = next((_model_label(source_models[key]) for key in keys if source_models.get(key)), "")
        if label:
            labels.append(f"{role}: {label}")
            if role == "base":
                center = label
    if not labels:
        labels = [f"{key}: {_model_label(value)}" for key, value in source_models.items()]
    return center, "  |  ".join(labels)


def _expected_hash(protocol: Any) -> str:
    if isinstance(protocol, dict):
        return str(protocol.get("config_hash") or "")
    return str(protocol) if isinstance(protocol, str) else ""


def _trait_metric_names(metric_names: list[str]) -> list[str]:
    return [
        name
        for name in metric_names
        if name.endswith(" (%)")
        and not any(term in name.casefold() for term in _AGGREGATE_TERMS)
    ]


def _is_target_trait(metric_name: str, target_trait: str) -> bool:
    target = re.sub(r"[^a-z0-9]", "", target_trait.casefold())
    metric = re.sub(r"[^a-z0-9]", "", metric_name.removesuffix(" (%)").casefold())
    return bool(target) and target in metric


def _safe_stem(dataset: str, used: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset).strip("._") or "dataset"
    candidate = stem
    suffix = 2
    while candidate in used:
        candidate = f"{stem}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _compact_number(value: Any) -> str:
    number = _number(value)
    return f"{number:g}" if number is not None else str(value)


def _selection_label(settings: Any) -> str:
    if not isinstance(settings, dict):
        return "K=?, rescale=?"
    k = settings.get("k_percent")
    k_label = f"{_compact_number(k)}%" if k is not None else "?"
    rescale = settings.get("use_weight_rescale")
    rescale_label = "?" if rescale is None else ("on" if rescale else "off")
    return f"K={k_label}, rescale={rescale_label}"


def _merge_settings(report: dict) -> str:
    """Return only the effective K and rescale setting shown on figures."""
    if report.get("sparsify") is False:
        return "K=100%, rescale=off"

    selection = report.get("selection")
    if not isinstance(selection, dict):
        return "K=?, rescale=?"

    if "high" in selection and "low" in selection:
        high_label = _selection_label(selection["high"])
        low_label = _selection_label(selection["low"])
        if high_label == low_label:
            return high_label
        return f"high: {high_label}; low: {low_label}"

    return _selection_label(selection)


def _comparison_caption(reports: list[dict]) -> str:
    labels = [_merge_settings(report) for report in reports]
    return labels[0] if len(set(labels)) == 1 else "K/rescale vary by path; see legend and CSV"


def write_evaluation_plots(report: dict, run_directory: Path) -> dict:
    """Write CSV, SVG and PNG for complete, current-protocol evaluation series.

    ``report['evaluation_protocols']`` selects datasets for the current run and
    supplies their ``config_hash``. Historical point results remain untouched
    but cannot enter a plot unless their hash matches that selected protocol.
    """
    output_directory = Path(run_directory) / "evaluation"
    output_directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "output_directory": str(output_directory),
        "datasets": {},
        "skipped_datasets": {},
    }

    protocols = report.get("evaluation_protocols") or {}
    requested = sorted({float(t) for t in report.get("requested_t_values", [])})
    points = report.get("points", [])
    path_method = str(report.get("path_method") or "")
    run_name = str(report.get("run_name") or "")
    target_trait = str(report.get("trait") or "")
    base_model, source_subtitle = _source_labels(report.get("source_models"))
    center_description = str(report.get("center_model") or "")
    center_model = base_model if center_description == "base" else center_description
    merge_settings = _merge_settings(report)
    used_stems: set[str] = set()

    for dataset, protocol in protocols.items():
        expected_hash = _expected_hash(protocol)
        if not expected_hash:
            result["skipped_datasets"][dataset] = "current protocol has no config_hash"
            continue
        if not requested:
            result["skipped_datasets"][dataset] = "no requested t values"
            continue

        series: list[tuple[float, dict]] = []
        missing: list[float] = []
        for t in requested:
            matches = [
                point
                for point in points
                if _number(point.get("t")) is not None
                and math.isclose(float(point["t"]), t, rel_tol=0.0, abs_tol=1e-9)
                and isinstance(point.get("evaluations"), dict)
                and dataset in point["evaluations"]
                and point["evaluations"][dataset].get("config_hash") == expected_hash
            ]
            if matches:
                series.append((t, matches[-1]["evaluations"][dataset]))
            else:
                missing.append(t)
        if missing:
            result["skipped_datasets"][dataset] = f"missing current-protocol results for t={missing}"
            continue

        primary_metrics = {str(item.get("primary_metric") or "") for _, item in series}
        if len(primary_metrics) != 1 or "" in primary_metrics:
            result["skipped_datasets"][dataset] = "primary_metric is missing or differs across t values"
            continue
        primary_metric = next(iter(primary_metrics))
        metrics_by_t = [item.get("metrics") or {} for _, item in series]
        metric_names = sorted({name for metrics in metrics_by_t for name in metrics})
        if str(dataset).casefold() == "trait":
            plotted_metrics = _trait_metric_names(metric_names)
        else:
            plotted_metrics = [primary_metric]
        if not plotted_metrics or not any(
            _number(metrics.get(name)) is not None
            for metrics in metrics_by_t
            for name in plotted_metrics
        ):
            result["skipped_datasets"][dataset] = "no plottable numeric metrics"
            continue

        stem = _safe_stem(str(dataset), used_stems)
        csv_path = output_directory / f"{stem}.csv"
        svg_path = output_directory / f"{stem}.svg"
        png_path = output_directory / f"{stem}.png"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "t", "path_method", "run_name", "center_model",
                    "merge_settings", "sparsify",
                    *metric_names,
                ],
            )
            writer.writeheader()
            for (t, _), metrics in zip(series, metrics_by_t):
                row: dict[str, Any] = {
                    "t": t,
                    "path_method": path_method,
                    "run_name": run_name,
                    "center_model": center_model,
                    "merge_settings": merge_settings,
                    "sparsify": report.get("sparsify"),
                }
                row.update({name: _number(metrics.get(name)) for name in metric_names})
                writer.writerow(row)

        fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
        xs = [t for t, _ in series]
        target_found = any(_is_target_trait(name, target_trait) for name in plotted_metrics)
        for name in plotted_metrics:
            is_target = str(dataset).casefold() == "trait" and _is_target_trait(name, target_trait)
            ys = [_number(metrics.get(name)) for metrics in metrics_by_t]
            ax.plot(
                xs,
                [float("nan") if value is None else value for value in ys],
                marker="o",
                linewidth=3.0 if is_target else 1.8,
                markersize=6.5 if is_target else 4.5,
                alpha=1.0 if is_target or not target_found else 0.75,
                zorder=3 if is_target else 2,
                label=name,
            )
        ax.set_xticks(xs)
        ax.set_xlabel("t")
        ax.set_ylabel("Score (%)")
        ax.set_title(
            f"{dataset} | {path_method}\n{merge_settings}\n{source_subtitle}",
            fontsize=9,
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.savefig(svg_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

        result["datasets"][dataset] = {
            "csv": str(csv_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "primary_metric": primary_metric,
            "plotted_metrics": plotted_metrics,
            "merge_settings": merge_settings,
        }

    return result


def _target_trait_metric(metric_names: list[str], target_trait: str) -> str | None:
    candidates = [name for name in _trait_metric_names(metric_names) if _is_target_trait(name, target_trait)]
    if not candidates:
        return None
    exact_target = re.sub(r"[^a-z0-9]", "", target_trait.casefold())
    exact = [
        name
        for name in candidates
        if re.sub(r"[^a-z0-9]", "", name.removesuffix(" (%)").casefold()) == exact_target
    ]
    return exact[0] if exact else candidates[0]


def write_path_comparison_plots(reports: list[dict], output_directory: Path) -> dict:
    """Compare complete evaluation series from multiple interpolation paths.

    A dataset is included only when every report selected the same evaluation
    protocol hash and every requested t value has a matching point result.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "output_directory": str(output_directory),
        "datasets": {},
        "skipped_datasets": {},
    }
    if not reports:
        result["skipped_datasets"]["*"] = "no reports"
        return result

    settings_by_path = {
        str(report.get("path_method") or ""): _merge_settings(report)
        for report in reports
    }
    settings_caption = _comparison_caption(reports)
    shared_settings = len(set(settings_by_path.values())) == 1
    dataset_names = sorted(
        {
            str(dataset)
            for report in reports
            for dataset in (report.get("evaluation_protocols") or {})
        }
    )
    used_stems: set[str] = set()

    for dataset in dataset_names:
        protocols: list[Any] = []
        if any(dataset not in (report.get("evaluation_protocols") or {}) for report in reports):
            result["skipped_datasets"][dataset] = "dataset is not selected by every report"
            continue
        for report in reports:
            protocols.append((report.get("evaluation_protocols") or {})[dataset])

        protocol_hashes = [_expected_hash(protocol) for protocol in protocols]
        if any(not value for value in protocol_hashes):
            result["skipped_datasets"][dataset] = "one or more protocols have no config_hash"
            continue
        if len(set(protocol_hashes)) != 1:
            result["skipped_datasets"][dataset] = "config_hash differs across reports"
            continue

        report_traits = {str(report.get("trait") or "") for report in reports}
        if dataset.casefold() == "trait" and (len(report_traits) != 1 or "" in report_traits):
            result["skipped_datasets"][dataset] = "target trait is missing or differs across reports"
            continue

        rows: list[dict[str, Any]] = []
        metric_names: set[str] = set()
        declared_primary_metrics: set[str] = set()
        failure: str | None = None
        for report, report_hash in zip(reports, protocol_hashes):
            requested = sorted({float(t) for t in report.get("requested_t_values", [])})
            if not requested:
                failure = f"{report.get('path_method')}: no requested t values"
                break
            points = report.get("points") or []
            base_model, _ = _source_labels(report.get("source_models"))
            center_description = str(report.get("center_model") or "")
            center_model = base_model if center_description == "base" else center_description
            merge_settings = _merge_settings(report)
            for t in requested:
                matches: list[dict] = []
                for point in points:
                    point_t = _number(point.get("t"))
                    evaluations = point.get("evaluations")
                    if point_t is None or not math.isclose(point_t, t, rel_tol=0.0, abs_tol=1e-9):
                        continue
                    if not isinstance(evaluations, dict) or not isinstance(evaluations.get(dataset), dict):
                        continue
                    evaluation = evaluations[dataset]
                    if evaluation.get("config_hash") == report_hash:
                        matches.append(evaluation)
                if not matches:
                    failure = f"{report.get('path_method')}: missing current-protocol result for t={t}"
                    break

                evaluation = matches[-1]
                metrics = evaluation.get("metrics") or {}
                if not isinstance(metrics, dict):
                    failure = f"{report.get('path_method')}: invalid metrics for t={t}"
                    break
                declared_primary = str(evaluation.get("primary_metric") or "")
                if declared_primary:
                    declared_primary_metrics.add(declared_primary)
                numeric_metrics = {
                    str(name): value
                    for name, raw_value in metrics.items()
                    if (value := _number(raw_value)) is not None
                }
                metric_names.update(numeric_metrics)
                rows.append(
                    {
                        "path_method": str(report.get("path_method") or ""),
                        "t": t,
                        "run_name": str(report.get("run_name") or ""),
                        "center_model": center_model,
                        "merge_settings": merge_settings,
                        "sparsify": report.get("sparsify"),
                        "metrics": numeric_metrics,
                    }
                )
            if failure:
                break
        if failure:
            result["skipped_datasets"][dataset] = failure
            continue

        ordered_metric_names = sorted(metric_names)
        if dataset.casefold() == "trait":
            primary_metric = _target_trait_metric(ordered_metric_names, next(iter(report_traits)))
            if primary_metric is None:
                result["skipped_datasets"][dataset] = "target trait percentage metric is unavailable"
                continue
        else:
            if len(declared_primary_metrics) != 1:
                result["skipped_datasets"][dataset] = (
                    "primary_metric is missing or differs across reports or t values"
                )
                continue
            primary_metric = next(iter(declared_primary_metrics))

        if any(_number(row["metrics"].get(primary_metric)) is None for row in rows):
            result["skipped_datasets"][dataset] = (
                f"primary metric {primary_metric!r} is unavailable at one or more points"
            )
            continue
        for row in rows:
            row["primary_metric"] = primary_metric
            row["primary_value"] = row["metrics"][primary_metric]

        stem = _safe_stem(dataset, used_stems)
        csv_path = output_directory / f"{stem}.csv"
        svg_path = output_directory / f"{stem}.svg"
        png_path = output_directory / f"{stem}.png"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = [
                "path_method",
                "t",
                "primary_metric",
                "primary_value",
                "run_name",
                "center_model",
                "merge_settings",
                "sparsify",
                *ordered_metric_names,
            ]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                output_row = {key: row[key] for key in fieldnames if key in row}
                output_row.update({name: row["metrics"].get(name) for name in ordered_metric_names})
                writer.writerow(output_row)

        fig, ax = plt.subplots(figsize=(9, 5.4), layout="constrained")
        path_order = list(dict.fromkeys(row["path_method"] for row in rows))
        for path_method in path_order:
            path_rows = sorted(
                (row for row in rows if row["path_method"] == path_method),
                key=lambda row: row["t"],
            )
            ax.plot(
                [row["t"] for row in path_rows],
                [row["primary_value"] for row in path_rows],
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                label=path_method if shared_settings else f"{path_method} | {settings_by_path[path_method]}",
            )
        ax.set_xticks(sorted({row["t"] for row in rows}))
        ax.set_xlabel("t")
        ax.set_ylabel(primary_metric if primary_metric.endswith("(%)") else f"{primary_metric} (%)")
        center_models = list(
            dict.fromkeys(row["center_model"] for row in rows if row["center_model"])
        )
        title = f"{dataset}: path comparison\n{settings_caption}"
        if center_models:
            title += f"\ncenter model: {', '.join(center_models)}"
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(title="Path", loc="best", fontsize=8)
        fig.savefig(svg_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)

        result["datasets"][dataset] = {
            "csv": str(csv_path),
            "svg": str(svg_path),
            "png": str(png_path),
            "primary_metric": primary_metric,
            "path_methods": path_order,
            "config_hash": protocol_hashes[0],
            "merge_settings": settings_caption,
        }

    return result


def _dataset_key(report: dict, normalized_name: str) -> str | None:
    protocols = report.get("evaluation_protocols") or {}
    for name in protocols:
        if re.sub(r"[^a-z0-9]", "", str(name).casefold()) == normalized_name:
            return str(name)
    return None


def _matching_evaluation(
    report: dict,
    dataset: str,
    t: float,
    config_hash: str,
) -> dict | None:
    matches: list[dict] = []
    for point in report.get("points") or []:
        point_t = _number(point.get("t"))
        evaluations = point.get("evaluations")
        if point_t is None or not math.isclose(point_t, t, rel_tol=0.0, abs_tol=1e-9):
            continue
        if not isinstance(evaluations, dict) or not isinstance(evaluations.get(dataset), dict):
            continue
        evaluation = evaluations[dataset]
        if evaluation.get("config_hash") == config_hash:
            matches.append(evaluation)
    return matches[-1] if matches else None


def write_pareto_tradeoff_plot(reports: list[dict], output_directory: Path) -> dict:
    """Plot target-trait strength against MMLU-Pro accuracy for all paths."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    def skipped(reason: str) -> dict:
        return {
            "status": "skipped",
            "reason": reason,
            "output_directory": str(output_directory),
        }

    if not reports:
        return skipped("no reports")

    settings_by_path = {
        str(report.get("path_method") or ""): _merge_settings(report)
        for report in reports
    }
    settings_caption = _comparison_caption(reports)
    shared_settings = len(set(settings_by_path.values())) == 1
    trait_keys = [_dataset_key(report, "trait") for report in reports]
    mmlu_keys = [_dataset_key(report, "mmlupro") for report in reports]
    if any(key is None for key in trait_keys):
        return skipped("trait is not selected by every report")
    if any(key is None for key in mmlu_keys):
        return skipped("mmlu_pro is not selected by every report")

    trait_hashes = [
        _expected_hash((report.get("evaluation_protocols") or {})[key])
        for report, key in zip(reports, trait_keys)
    ]
    mmlu_hashes = [
        _expected_hash((report.get("evaluation_protocols") or {})[key])
        for report, key in zip(reports, mmlu_keys)
    ]
    if any(not value for value in trait_hashes):
        return skipped("one or more trait protocols have no config_hash")
    if any(not value for value in mmlu_hashes):
        return skipped("one or more mmlu_pro protocols have no config_hash")
    if len(set(trait_hashes)) != 1:
        return skipped("trait config_hash differs across reports")
    if len(set(mmlu_hashes)) != 1:
        return skipped("mmlu_pro config_hash differs across reports")

    traits = {str(report.get("trait") or "") for report in reports}
    if len(traits) != 1 or "" in traits:
        return skipped("target trait is missing or differs across reports")
    target_trait = next(iter(traits))

    collected: list[dict[str, Any]] = []
    trait_metric_names: set[str] = set()
    mmlu_primary_metrics: set[str] = set()
    for report, trait_key, mmlu_key, trait_hash, mmlu_hash in zip(
        reports, trait_keys, mmlu_keys, trait_hashes, mmlu_hashes
    ):
        requested = sorted({float(t) for t in report.get("requested_t_values", [])})
        path_method = str(report.get("path_method") or "")
        if not requested:
            return skipped(f"{path_method}: no requested t values")
        base_model, _ = _source_labels(report.get("source_models"))
        center_description = str(report.get("center_model") or "")
        center_model = base_model if center_description == "base" else center_description
        merge_settings = _merge_settings(report)
        for t in requested:
            trait_evaluation = _matching_evaluation(report, trait_key, t, trait_hash)
            mmlu_evaluation = _matching_evaluation(report, mmlu_key, t, mmlu_hash)
            if trait_evaluation is None:
                return skipped(f"{path_method}: missing current trait result for t={t}")
            if mmlu_evaluation is None:
                return skipped(f"{path_method}: missing current mmlu_pro result for t={t}")
            trait_metrics = trait_evaluation.get("metrics") or {}
            mmlu_metrics = mmlu_evaluation.get("metrics") or {}
            if not isinstance(trait_metrics, dict) or not isinstance(mmlu_metrics, dict):
                return skipped(f"{path_method}: invalid metrics for t={t}")
            trait_metric_names.update(
                str(name) for name, value in trait_metrics.items() if _number(value) is not None
            )
            mmlu_primary = str(mmlu_evaluation.get("primary_metric") or "")
            if mmlu_primary:
                mmlu_primary_metrics.add(mmlu_primary)
            collected.append(
                {
                    "path_method": path_method,
                    "t": t,
                    "run_name": str(report.get("run_name") or ""),
                    "center_model": center_model,
                    "merge_settings": merge_settings,
                    "sparsify": report.get("sparsify"),
                    "trait_metrics": trait_metrics,
                    "mmlu_metrics": mmlu_metrics,
                    "mmlu_primary": mmlu_primary,
                    "trait_result_path": str(trait_evaluation.get("result_path") or ""),
                    "mmlu_pro_result_path": str(mmlu_evaluation.get("result_path") or ""),
                }
            )

    trait_metric = _target_trait_metric(sorted(trait_metric_names), target_trait)
    if trait_metric is None:
        return skipped("target trait percentage metric is unavailable")
    if len(mmlu_primary_metrics) != 1:
        return skipped("MMLU-Pro primary_metric is missing or differs across reports or t values")
    mmlu_metric = next(iter(mmlu_primary_metrics))
    if "accuracy" not in mmlu_metric.casefold():
        return skipped(f"MMLU-Pro primary_metric is not Accuracy: {mmlu_metric!r}")
    if any(row["mmlu_primary"] != mmlu_metric for row in collected):
        return skipped("MMLU-Pro primary_metric is missing at one or more points")

    for row in collected:
        trait_score = _number(row["trait_metrics"].get(trait_metric))
        mmlu_score = _number(row["mmlu_metrics"].get(mmlu_metric))
        if trait_score is None:
            return skipped(
                f"{row['path_method']}: target trait metric is unavailable for t={row['t']}"
            )
        if mmlu_score is None:
            return skipped(
                f"{row['path_method']}: MMLU-Pro primary metric is unavailable for t={row['t']}"
            )
        row["trait_score"] = trait_score
        row["mmlu_pro_accuracy"] = mmlu_score

    csv_path = output_directory / "trait_mmlu_pro_pareto.csv"
    svg_path = output_directory / "trait_mmlu_pro_pareto.svg"
    png_path = output_directory / "trait_mmlu_pro_pareto.png"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "path_method",
            "t",
            "trait_metric",
            "trait_score",
            "mmlu_metric",
            "mmlu_pro_accuracy",
            "run_name",
            "center_model",
            "merge_settings",
            "sparsify",
            "trait_config_hash",
            "mmlu_pro_config_hash",
            "trait_result_path",
            "mmlu_pro_result_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in collected:
            writer.writerow(
                {
                    "path_method": row["path_method"],
                    "t": row["t"],
                    "trait_metric": trait_metric,
                    "trait_score": row["trait_score"],
                    "mmlu_metric": mmlu_metric,
                    "mmlu_pro_accuracy": row["mmlu_pro_accuracy"],
                    "run_name": row["run_name"],
                    "center_model": row["center_model"],
                    "merge_settings": row["merge_settings"],
                    "sparsify": row["sparsify"],
                    "trait_config_hash": trait_hashes[0],
                    "mmlu_pro_config_hash": mmlu_hashes[0],
                    "trait_result_path": row["trait_result_path"],
                    "mmlu_pro_result_path": row["mmlu_pro_result_path"],
                }
            )

    fig, ax = plt.subplots(figsize=(8, 6.2), layout="constrained")
    path_order = list(dict.fromkeys(row["path_method"] for row in collected))
    for path_method in path_order:
        path_rows = sorted(
            (row for row in collected if row["path_method"] == path_method),
            key=lambda row: row["t"],
        )
        line = ax.plot(
            [row["trait_score"] for row in path_rows],
            [row["mmlu_pro_accuracy"] for row in path_rows],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=path_method if shared_settings else f"{path_method} | {settings_by_path[path_method]}",
        )[0]
        for index, row in enumerate(path_rows):
            offset_y = 6 if index % 2 == 0 else -11
            ax.annotate(
                f"t={row['t']:g}",
                (row["trait_score"], row["mmlu_pro_accuracy"]),
                xytext=(5, offset_y),
                textcoords="offset points",
                fontsize=7.5,
                color=line.get_color(),
            )
    ax.set_xlabel(trait_metric)
    ax.set_ylabel(mmlu_metric)
    ax.set_title(f"{target_trait} strength vs. general capability\n{settings_caption}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(title="Path", loc="best", fontsize=8)
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    return {
        "status": "written",
        "output_directory": str(output_directory),
        "csv": str(csv_path),
        "svg": str(svg_path),
        "png": str(png_path),
        "trait_metric": trait_metric,
        "mmlu_pro_metric": mmlu_metric,
        "path_methods": path_order,
        "trait_config_hash": trait_hashes[0],
        "mmlu_pro_config_hash": mmlu_hashes[0],
        "merge_settings": settings_caption,
    }
