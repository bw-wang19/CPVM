"""Plot cumulative contribution, TRAIT score, and validation loss over Top-k.

This experiment uses global raw-contribution ordering, alpha=1, and samples
0, 10, ..., 100 percent. Each model is merged and evaluated in memory; no
model checkpoints are written. The score axis always spans 0..100. The loss
axis keeps the selected transformed units while placing the base value at 25%
of the axis height and the highest sampled value at 80%.
"""

from contextlib import ExitStack
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from safetensors import safe_open
import torch
from tqdm.auto import tqdm
import yaml

from CPVM.code.topk_merge import (
    CONTRIBUTION_CHANNELS,
    combined_contribution_chunks,
    discover_contribution_files,
    resolve_contribution_paths,
)
from CPVM.code.topk_curve_metrics import evaluate_curve_models
from CPVM.code.utils.arguments import parse_args_yaml


def cumulative_contributions(paths, chunk_numel=5_000_000, bins=8192):
    """Reproduce the notebook's full-data signed histogram accumulation."""
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(p), framework="pt", device="cpu")) for p in paths]
        names = sorted(handles[0].keys())
        if any(set(h.keys()) != set(names) for h in handles[1:]):
            raise ValueError("Contribution channels must contain the same parameters")
        minimum, maximum = float("inf"), float("-inf")
        for name in tqdm(names, desc="Contribution range"):
            for _, chunk in combined_contribution_chunks(handles, name, chunk_numel):
                if not torch.isfinite(chunk).all():
                    raise ValueError(f"Non-finite contribution: {name}")
                minimum = min(minimum, chunk.min().item())
                maximum = max(maximum, chunk.max().item())
        tiny = np.finfo(np.float32).tiny
        negative_edges = -np.geomspace(abs(minimum), tiny, bins // 2) if minimum < 0 else [-tiny]
        positive_edges = np.geomspace(tiny, maximum, bins // 2) if maximum > 0 else [tiny]
        edges = np.unique(np.r_[negative_edges, 0.0, positive_edges])
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
        sums = np.zeros(len(edges) - 1, dtype=np.float64)
        for name in tqdm(names, desc="Cumulative contributions"):
            for _, chunk in combined_contribution_chunks(handles, name, chunk_numel):
                values = chunk.numpy()
                indices = np.searchsorted(edges, values, side="right") - 1
                np.clip(indices, 0, len(counts) - 1, out=indices)
                counts += np.bincount(indices, minlength=len(counts))
                sums += np.bincount(indices, weights=values.astype(np.float64), minlength=len(sums))
    nonempty = counts[::-1] > 0
    descending_counts, descending_sums = counts[::-1][nonempty], sums[::-1][nonempty]
    total_signed = descending_sums.sum()
    if total_signed == 0 or counts.sum() == 0:
        raise ValueError("Cumulative contribution / net total is undefined for an empty or zero-total input")
    return pd.DataFrame({
        "k_percent": 100 * np.r_[0, np.cumsum(descending_counts)] / counts.sum(),
        "cumulative_contribution": np.r_[0, np.cumsum(descending_sums)] / total_signed,
    })


def plot_curves(contribution, metrics, config, output_dir):
    """Plot contributions, percentage scores, and the loss objective on independent axes."""
    channel = config["contribution_channel"]
    # Older CSVs used target-only NLL for every channel. They cannot be reused
    # for channel-aware plots without evaluating both matched completions.
    for column, expected in (
        ("contribution_channel", channel), ("target_level", config["target_level"]),
    ):
        if column not in metrics or not metrics[column].eq(expected).all():
            raise ValueError(
                f"Metrics must record {column}={expected!r}; "
                "set plot_only=false to regenerate metrics with the matching objective"
            )
    metrics = metrics.sort_values("k_percent").copy()
    metrics = metrics.set_index("k_percent")
    # Both endpoints are evaluated once in the 0..100% sweep. Their weights
    # equal the alpha=0 base and alpha=1 SFT endpoints of the parameter path.
    base_score = float(metrics.loc[0, "trait_score"])
    sft_score = float(metrics.loc[100, "trait_score"])
    print(
        f"Measured {config['main_trait']} score endpoints: "
        f"base (alpha=0, k=0%)={base_score:.4f}; "
        f"SFT (alpha=1, k=100%)={sft_score:.4f}"
    )
    base_loss = config.get("base_validation_loss")
    sft_loss = config.get("sft_validation_loss")
    base_loss = float(metrics.loc[0, "validation_loss"] if base_loss is None else base_loss)
    sft_loss = float(metrics.loc[100, "validation_loss"] if sft_loss is None else sft_loss)
    # Optional loss overrides affect the plotted endpoints and loss-axis mapping;
    # TRAIT points always use the measured scores.
    metrics.loc[0, "validation_loss"] = base_loss
    metrics.loc[100, "validation_loss"] = sft_loss
    transform = config.get("loss_axis", "negative")
    if transform == "negative":
        loss_values = -metrics["validation_loss"].to_numpy()
        objective_labels = {
            "L_plus": "-NLL(target)",
            "L_star": "+NLL(counter)",
            "L_contrast": "NLL(counter) - NLL(target)",
        }
        displayed_base_loss = -base_loss
        loss_label = f"-Validation loss = {objective_labels[channel]}"
    elif transform == "negative_log":
        if channel != "L_plus":
            raise ValueError("L_star/L_contrast use signed losses; set loss_axis=negative")
        if (metrics["validation_loss"] <= 0).any():
            raise ValueError("negative_log requires strictly positive loss")
        loss_values = -np.log(metrics["validation_loss"].to_numpy())
        displayed_base_loss = -np.log(base_loss)
        loss_label = "-log(Validation loss)"
    else:
        raise ValueError("loss_axis must be negative or negative_log")

    x = contribution["k_percent"].to_numpy()
    y = contribution["cumulative_contribution"].to_numpy()
    sample_x = metrics.index.to_numpy()
    scores = metrics["trait_score"].to_numpy()
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 100)).any():
        raise ValueError("TRAIT scores must be finite percentages in [0, 100]")
    if not np.isfinite(loss_values).all() or not np.isfinite(displayed_base_loss):
        raise ValueError("Displayed validation-loss values must be finite")
    displayed_max_loss = float(loss_values.max())
    loss_rise = displayed_max_loss - displayed_base_loss
    loss_tolerance = np.finfo(np.float64).eps * max(
        1.0, abs(displayed_base_loss), abs(displayed_max_loss)
    )
    if loss_rise <= loss_tolerance:
        raise ValueError(
            "Cannot place base validation loss at 25% and sampled maximum at "
            "80% because the base value is already the sampled maximum"
        )
    loss_span = loss_rise / (0.80 - 0.25)
    loss_limits = (
        displayed_base_loss - 0.25 * loss_span,
        displayed_base_loss + 0.75 * loss_span,
    )
    if float(loss_values.min()) < loss_limits[0] - loss_tolerance:
        raise ValueError(
            "The requested loss-axis mapping would clip a sampled value below "
            "the axis; base=25%, maximum=80%, and showing every point cannot "
            "all be satisfied on one linear axis"
        )
    width, dpi = config.get("figure_width", 30), config.get("dpi", 120)
    views = config.get("view_limits_pct", [100])
    fig, axes = plt.subplots(len(views), 1, figsize=(width, 6 * len(views)), dpi=dpi, squeeze=False)
    for ax, xmax in zip(axes[:, 0], views):
        if not 0 < xmax <= 100:
            raise ValueError("view_limits_pct must be in (0, 100]")
        visible = x < xmax
        view_x = np.r_[x[visible], xmax]
        view_y = np.r_[y[visible], np.interp(xmax, x, y)]
        score_ax, loss_ax = ax.twinx(), ax.twinx()
        loss_ax.spines["right"].set_position(("outward", 85))
        contribution_line, = ax.plot(view_x, view_y, color="#C44E52", lw=2.2, label="Cumulative signed contribution")
        score_line, = score_ax.plot(sample_x, scores, "o-", color="#2878B5", lw=2, label=f"{config['main_trait']} score")
        loss_line, = loss_ax.plot(sample_x, loss_values, "s-", color="#32965A", lw=2, label=loss_label)
        heights = np.r_[0.0, 1.0, view_y]
        low, high = float(heights.min()), float(heights.max())
        padding = 0.05 * max(high - low, 1e-12)
        ax.set_ylim(low - padding if low < 0 else 0, high + padding)
        score_ax.set_ylim(0, 100)
        score_ax.set_yticks(np.arange(0, 101, 10))
        loss_ax.set_ylim(loss_limits)
        loss_anchors = (displayed_base_loss, displayed_max_loss)
        minimum_tick_gap = 0.02 * (loss_limits[1] - loss_limits[0])
        loss_ticks = [
            float(value) for value in loss_ax.get_yticks()
            if loss_limits[0] <= value <= loss_limits[1]
            and all(abs(value - anchor) > minimum_tick_gap for anchor in loss_anchors)
        ]
        loss_ticks.extend(loss_anchors)
        loss_ax.set_yticks(sorted(set(loss_ticks)))
        loss_ax.set_ylim(loss_limits)
        ax.set_xlim(0, xmax)
        ax.axhline(1, color="gray", ls=":", lw=1)
        ax.set_xlabel("Top parameters (%)")
        ax.set_ylabel("Cumulative signed contribution / net total", color="#C44E52")
        score_ax.set_ylabel(f"{config['main_trait']} score (%)", color="#2878B5")
        loss_ax.set_ylabel(loss_label, color="#32965A")
        score_ax.tick_params(axis="y", colors="#2878B5")
        loss_ax.tick_params(axis="y", colors="#32965A")
        ax.grid(alpha=0.25)
        ax.set_title(f"{channel}: Top-k merge (0–{xmax:g}%; metric samples every 10%)\nBase model: qwen3-4b-instruct | SFT model: trait-level")
        ax.legend(handles=[contribution_line, score_line, loss_line], loc="best")
    fig.tight_layout(rect=(0, 0, 0.94, 1), h_pad=3)
    png = output_dir / "topk_curves.png"
    fig.savefig(png, dpi=dpi)
    fig.savefig(output_dir / "topk_curves.svg")
    plt.close(fig)
    (output_dir / "topk_curves.html").write_text(
        '<meta charset="utf-8"><div style="max-width:100%;overflow-x:auto">'
        f'<img src="{png.name}" style="width:{round(width*dpi)}px;max-width:none" '
        'alt="Top-k contribution, trait score and validation loss"></div>', encoding="utf-8",
    )
    print(f"Saved plots to {output_dir}")


def main():
    config = parse_args_yaml("Top-k contribution / TRAIT score / validation loss curves")
    merge_config = yaml.safe_load(Path(config["merge_config_path"]).read_text())
    channel = config.get("contribution_channel", merge_config.get("contribution_channel", "L_plus"))
    if channel not in CONTRIBUTION_CHANNELS:
        raise ValueError(f"contribution_channel must be one of {CONTRIBUTION_CHANNELS}")
    config["contribution_channel"] = channel
    if config["target_level"] not in {"high", "low"}:
        raise ValueError("target_level must be high or low")
    loss_axis = config.get("loss_axis", "negative")
    if loss_axis not in {"negative", "negative_log"}:
        raise ValueError("loss_axis must be negative or negative_log")
    if loss_axis == "negative_log" and channel != "L_plus":
        raise ValueError("L_star/L_contrast use signed losses; set loss_axis=negative")
    # Global raw-contribution order matches the original cumulative curve, including its
    # negative tail; alpha=1 makes 100% exactly the SFT checkpoint.
    merge_config.update(
        ranking_method="contribution", selection_mode="positive", alpha=1.0,
        layers=None, modules=None, contribution_channel=channel,
        use_weight_rescale=config.get("use_weight_rescale", False),
    )
    trait_name = config["main_trait"].strip().lower().replace(" ", "-")
    output_dir = Path(config["output_root"]) / f"{trait_name}-{config['target_level']}" / channel
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_path, metrics_path = output_dir / "contribution_curve.csv", output_dir / "model_metrics.csv"
    if config.get("plot_only", False):
        contribution, metrics = pd.read_csv(curve_path), pd.read_csv(metrics_path)
        print(metrics.to_string(index=False))
    else:
        required_paths = ["base_model_path", "sft_model_path", "contribution_dir"]
        for path_key in required_paths:
            if not config.get(path_key):
                raise ValueError(f"{path_key} must be set in topk_curve.yaml")
        contribution_files = discover_contribution_files(config["contribution_dir"])
        merge_config.update(
            base_model_path=config["base_model_path"],
            finetuned_model_path=config["sft_model_path"],
            contribution_dir=str(Path(config["contribution_dir"]).expanduser().resolve()),
            l_plus_contribution_path=str(contribution_files["L_plus"]),
            l_star_contribution_path=str(contribution_files["L_star"]),
        )
        print(
            "Curve inputs:\n"
            f"  base: {merge_config['base_model_path']}\n"
            f"  SFT: {merge_config['finetuned_model_path']}\n"
            f"  contribution directory: {merge_config['contribution_dir']}\n"
            f"  L_plus: {merge_config['l_plus_contribution_path']}\n"
            f"  L_star: {merge_config['l_star_contribution_path']}\n"
            f"  channel: {channel}",
            flush=True,
        )
        paths = resolve_contribution_paths(
            merge_config["l_plus_contribution_path"], merge_config.get("l_star_contribution_path"), channel,
        )
        contribution = cumulative_contributions(
            paths, merge_config.get("chunk_numel", 5_000_000), config.get("contribution_bins", 8192),
        )
        contribution.to_csv(curve_path, index=False)
        trait_config = yaml.safe_load(Path(config["trait_config_path"]).read_text())
        validation_config = yaml.safe_load(Path(config["validation_config_path"]).read_text())
        validation_config.update(config.get("validation_overrides") or {})
        validation_config["trait"] = config["main_trait"]
        validation_config["level"] = config["target_level"]
        validation_config["target_pole"] = config["target_level"]
        validation_config["contribution_channel"] = channel
        metrics = evaluate_curve_models(merge_config, validation_config, trait_config, str(metrics_path))
    plot_curves(contribution, metrics, config, output_dir)


if __name__ == "__main__":
    main()
