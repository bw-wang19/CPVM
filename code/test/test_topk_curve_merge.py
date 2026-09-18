import math

from safetensors.torch import save_file
import torch

from CPVM.code.topk_curve_merge import open_curve_merge


class TinyModel(torch.nn.Module):
    def __init__(self, tensors):
        super().__init__()
        for name, value in tensors.items():
            self.register_parameter(name, torch.nn.Parameter(value.clone()))


def build_curve_case(tmp_path, use_weight_rescale=False):
    base_values = {
        "a": torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0, 50.0]),
        "b": torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0]),
    }
    sft_values = {
        "a": torch.tensor([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]),
        "b": torch.tensor([91.0, 92.0, 93.0, 94.0, 95.0]),
    }
    plus_values = {
        "a": torch.tensor([9.0, 8.0, 8.0, 4.0, 1.0, -1.0]),
        "b": torch.tensor([10.0, 8.0, 0.0, -3.0, -3.0]),
    }
    star_values = {
        "a": torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        "b": torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
    }
    base_path = tmp_path / "base"
    sft_path = tmp_path / "sft"
    base_path.mkdir()
    sft_path.mkdir()
    save_file(base_values, base_path / "model.safetensors")
    save_file(sft_values, sft_path / "model.safetensors")
    plus_path = tmp_path / "L_plus.safetensors"
    star_path = tmp_path / "L_star.safetensors"
    save_file(plus_values, plus_path)
    save_file(star_values, star_path)

    names = sorted(base_values)
    scores = torch.cat(
        [
            (plus_values[name] + star_values[name]).reshape(-1)
            for name in names
        ]
    )
    order = sorted(
        range(scores.numel()),
        key=lambda index: (-float(scores[index]), index),
    )
    config = {
        "base_model_path": str(base_path),
        "finetuned_model_path": str(sft_path),
        "l_plus_contribution_path": str(plus_path),
        "l_star_contribution_path": str(star_path),
        "contribution_channel": "L_contrast",
        "chunk_numel": 3,
        "use_weight_rescale": use_weight_rescale,
    }
    return base_values, sft_values, names, scores, order, config


def expected_weights(base_values, sft_values, names, order, k_percent, rescale):
    base = torch.cat([base_values[name].reshape(-1) for name in names])
    finetuned = torch.cat([sft_values[name].reshape(-1) for name in names])
    selected_numel = math.ceil(base.numel() * k_percent / 100)
    selected = torch.zeros(base.numel(), dtype=torch.bool)
    selected[order[:selected_numel]] = True
    alpha = 1 / (selected_numel / base.numel()) if rescale and selected_numel else 1
    expected = base.clone()
    expected[selected] = (
        base[selected] + alpha * (finetuned[selected] - base[selected])
    )
    return expected


def resident_weights(model, names):
    parameters = dict(model.named_parameters())
    return torch.cat(
        [parameters[name].detach().reshape(-1) for name in names]
    )


def test_curve_merge_incrementally_matches_independent_topk_models(tmp_path, capsys):
    base, sft, names, scores, order, config = build_curve_case(tmp_path)
    model = TinyModel(base)
    k_percents = [0, 10, 25, 50, 75, 100]

    with open_curve_merge(
        model, config, k_percents=k_percents
    ) as apply_merge:
        for k_percent in k_percents:
            metrics = apply_merge(k_percent)
            expected = expected_weights(
                base, sft, names, order, k_percent, rescale=False
            )
            torch.testing.assert_close(
                resident_weights(model, names), expected, rtol=0, atol=0
            )
            assert metrics["merged_numel"] == math.ceil(
                scores.numel() * k_percent / 100
            )

        apply_merge(25)
        torch.testing.assert_close(
            resident_weights(model, names),
            expected_weights(base, sft, names, order, 25, rescale=False),
            rtol=0,
            atol=0,
        )

    output = capsys.readouterr().out
    assert "mode=incremental" in output
    assert "mode=reconstructed" in output


def test_curve_merge_rescaling_reconstructs_each_point(tmp_path):
    base, sft, names, _, order, config = build_curve_case(
        tmp_path, use_weight_rescale=True
    )
    model = TinyModel(base)

    with open_curve_merge(model, config, k_percents=[0, 25, 50]) as apply_merge:
        for k_percent in [0, 25, 50]:
            apply_merge(k_percent)
            torch.testing.assert_close(
                resident_weights(model, names),
                expected_weights(
                    base, sft, names, order, k_percent, rescale=True
                ),
                rtol=0,
                atol=1e-6,
            )
