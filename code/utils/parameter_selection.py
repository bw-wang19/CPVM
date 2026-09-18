"""Layer/module candidate selection for Top-k model merging."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch


LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class StructuralSelection:
    """Size of the selected parameter blocks relative to the whole base model."""

    total_numel: int
    selected_numel: int
    selected_percent: float
    selected_tensor_count: int


def parameter_layer(name: str) -> int | None:
    """Extract the transformer block index from a parameter name."""

    match = LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def module_matches(parameter_name: str, selector: str) -> bool:
    """Match a dot-delimited module path without ambiguous substring matches."""

    module_path = parameter_name.rsplit(".", 1)[0]
    selector = selector.strip(".")
    return f".{selector}." in f".{module_path}."


def select_parameters(
    parameters: dict[str, torch.nn.Parameter],
    layers: list[int] | None,
    modules: list[str] | None,
) -> tuple[list[str], StructuralSelection]:
    """Intersect active selectors, or select all parameters when both are unset."""

    layer_set = set(layers or [])
    module_selectors = [module.strip(".") for module in modules or []]

    selected_names = []
    for name in parameters:
        if layer_set and parameter_layer(name) not in layer_set:
            continue
        if module_selectors and not any(
            module_matches(name, selector) for selector in module_selectors
        ):
            continue
        selected_names.append(name)

    if not selected_names:
        raise ValueError("No model parameters matched the configured layers/modules")

    total_numel = sum(parameter.numel() for parameter in parameters.values())
    selected_numel = sum(parameters[name].numel() for name in selected_names)
    selection = StructuralSelection(
        total_numel=total_numel,
        selected_numel=selected_numel,
        selected_percent=100 * selected_numel / total_numel,
        selected_tensor_count=len(selected_names),
    )
    return selected_names, selection


