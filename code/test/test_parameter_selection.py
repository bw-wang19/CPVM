"""Layer/module matching and candidate parameter counts for Top-k merging."""

import pytest
import torch

from CPVM.code.utils.parameter_selection import (
    module_matches,
    parameter_layer,
    select_parameters,
)


def toy_parameters():
    """Return Qwen-shaped names with 44 scalar parameters for selector tests."""

    shapes = {
        "model.embed_tokens.weight": (5, 2),
        "model.layers.0.self_attn.q_proj.weight": (2, 3),
        "model.layers.0.self_attn.o_proj.weight": (2, 2),
        "model.layers.0.mlp.gate_proj.weight": (2, 4),
        "model.layers.1.self_attn.q_proj.weight": (2, 3),
        "model.layers.1.mlp.gate_proj.weight": (2, 4),
        "model.norm.weight": (2,),
    }
    return {
        name: torch.nn.Parameter(torch.zeros(shape)) for name, shape in shapes.items()
    }


def test_layer_and_module_matching_uses_path_components():
    name = "model.layers.10.self_attn.q_proj.weight"
    assert parameter_layer(name) == 10
    assert module_matches(name, "q_proj")
    assert module_matches(name, "self_attn")
    assert module_matches(name, "self_attn.q_proj")
    assert not module_matches(name, "attn")
    assert not module_matches(name, "q")


def test_selection_counts_layers_modules_and_their_intersection():
    parameters = toy_parameters()

    layer_names, layer_selection = select_parameters(parameters, [1], None)
    module_names, module_selection = select_parameters(
        parameters, None, ["self_attn.q_proj"]
    )
    both_names, both_selection = select_parameters(
        parameters, [1], ["mlp.gate_proj"]
    )

    assert set(layer_names) == {
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
    }
    assert layer_selection.selected_numel == 14
    assert layer_selection.selected_percent == pytest.approx(14 / 44 * 100)
    assert module_selection.selected_numel == 12
    assert len(module_names) == 2
    assert both_names == ["model.layers.1.mlp.gate_proj.weight"]
    assert both_selection.selected_numel == 8
