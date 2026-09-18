"""Apply a raw-contribution Top-k curve without saving checkpoints."""

from contextlib import ExitStack, contextmanager

import numpy as np
from safetensors import safe_open
import torch
from tqdm.auto import tqdm

from CPVM.code.topk_merge import (
    ModelTensorReader,
    chunk_indices,
    combined_contribution_chunks,
    contribution_score_bits,
    find_topk_selection,
    find_topk_selections,
    merge_selected_values,
    resolve_contribution_paths,
    selection_mask,
)


@contextmanager
def open_curve_merge(model, merge_config, k_percents=None):
    """Yield an in-memory apply(k_percent) function for a Top-k curve.

    Exact thresholds for the requested curve points are prepared together in
    four shared radix scans. With alpha=1 and rescaling disabled, increasing k
    values update only the newly selected weights; the resident model therefore
    moves monotonically from the already-loaded base checkpoint to the SFT
    checkpoint. Rescaled or decreasing sweeps are reconstructed from base.
    """

    chunk_numel = int(merge_config.get("chunk_numel", 5_000_000))
    if chunk_numel <= 0:
        raise ValueError("chunk_numel must be positive")
    parameters = dict(model.named_parameters())
    names = sorted(parameters)
    total_numel = sum(parameter.numel() for parameter in parameters.values())
    if not total_numel:
        raise ValueError("The resident model contains no parameters")
    for name, parameter in parameters.items():
        if parameter.device.type == "meta":
            raise ValueError(
                f"Cannot merge into a meta/offloaded parameter: {name}; "
                "keep every parameter resident on CPU or GPU"
            )

    channel = merge_config.get("contribution_channel", "L_plus")
    contribution_paths = resolve_contribution_paths(
        merge_config.get("l_plus_contribution_path"),
        merge_config.get("l_star_contribution_path"),
        channel,
    )
    use_weight_rescale = merge_config.get("use_weight_rescale", False)
    prepared_percents = (
        list(range(0, 101, 10)) if k_percents is None else list(k_percents)
    )
    prepared_percents = list(
        dict.fromkeys(float(k_percent) for k_percent in prepared_percents)
    )

    with ExitStack() as stack:
        base = stack.enter_context(ModelTensorReader(merge_config["base_model_path"]))
        finetuned = stack.enter_context(
            ModelTensorReader(merge_config["finetuned_model_path"])
        )
        contributions = [
            stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
            for path in contribution_paths
        ]
        if not set(names).issubset(base.keys) or not set(names).issubset(finetuned.keys):
            raise ValueError("Resident, base and SFT parameter names do not match")
        if any(set(handle.keys()) != set(names) for handle in contributions):
            raise ValueError(
                "Contribution parameters must match the resident model's unique parameters"
            )
        for name in names:
            shape = tuple(parameters[name].shape)
            if base.shape(name) != shape or finetuned.shape(name) != shape:
                raise ValueError(f"Base/SFT parameter shape mismatch: {name}")
            if any(
                tuple(handle.get_slice(name).get_shape()) != shape
                for handle in contributions
            ):
                raise ValueError(f"Contribution parameter shape mismatch: {name}")

        print(
            "Preparing exact raw-contribution Top-k thresholds for "
            f"{prepared_percents} in four shared radix scans",
            flush=True,
        )
        selections = find_topk_selections(
            merge_config.get("l_plus_contribution_path"),
            prepared_percents,
            chunk_numel=chunk_numel,
            selection_mode="positive",
            l_star_contribution_path=merge_config.get("l_star_contribution_path"),
            contribution_channel=channel,
            parameter_names=names,
        )
        if any(selection.total_numel != total_numel for selection in selections.values()):
            raise ValueError("Contribution and resident model parameter counts differ")

        def get_selection(k_percent):
            selection = selections.get(k_percent)
            if selection is None:
                selection = find_topk_selection(
                    merge_config.get("l_plus_contribution_path"),
                    k_percent,
                    chunk_numel=chunk_numel,
                    selection_mode="positive",
                    l_star_contribution_path=merge_config.get(
                        "l_star_contribution_path"
                    ),
                    contribution_channel=channel,
                    parameter_names=names,
                )
                if selection.total_numel != total_numel:
                    raise ValueError(
                        "Contribution and resident model parameter counts differ"
                    )
                selections[k_percent] = selection
            return selection

        def copy_endpoint(reader, description):
            with torch.no_grad():
                for name in tqdm(names, desc=description, unit="tensor"):
                    parameter = parameters[name]
                    if parameter.device.type == "meta":
                        raise ValueError(f"Parameter became meta/offloaded: {name}")
                    for index in chunk_indices(tuple(parameter.shape), chunk_numel):
                        target = parameter if index is None else parameter[index]
                        endpoint = reader.read(name, index).to(
                            device=target.device,
                            dtype=parameter.dtype,
                        )
                        target.copy_(endpoint)

        def reconstruct(k_percent, selection, effective_alpha):
            if selection.selected_numel == 0:
                copy_endpoint(base, f"In-memory Top-k {k_percent:g}%")
                return 0
            if selection.select_all:
                copy_endpoint(finetuned, f"In-memory Top-k {k_percent:g}%")
                return total_numel

            ties_remaining = selection.threshold_ties_to_keep
            merged_numel = 0
            with torch.no_grad():
                for name in tqdm(
                    names,
                    desc=f"In-memory Top-k {k_percent:g}%",
                    unit="tensor",
                ):
                    parameter = parameters[name]
                    if parameter.device.type == "meta":
                        raise ValueError(f"Parameter became meta/offloaded: {name}")
                    for index, scores in combined_contribution_chunks(
                        contributions, name, chunk_numel
                    ):
                        mask, ties_remaining = selection_mask(
                            contribution_score_bits(scores, "positive"),
                            selection,
                            ties_remaining,
                        )
                        selected_indices = torch.from_numpy(np.flatnonzero(mask))
                        merged_numel += selected_indices.numel()
                        base_chunk = base.read(name, index).to(
                            dtype=parameter.dtype
                        ).clone(memory_format=torch.contiguous_format)
                        if selected_indices.numel():
                            finetuned_chunk = finetuned.read(
                                name, index
                            ).reshape(-1)
                            merge_selected_values(
                                base_chunk.reshape(-1),
                                finetuned_chunk,
                                selected_indices,
                                effective_alpha,
                            )
                        target = parameter if index is None else parameter[index]
                        target.copy_(base_chunk.to(device=target.device))

            if merged_numel != selection.selected_numel or ties_remaining != 0:
                raise RuntimeError(
                    "In-memory Top-k merge did not match the exact selection count"
                )
            return merged_numel

        def advance(k_percent, previous, current):
            expected_new = current.selected_numel - previous.selected_numel
            if expected_new < 0:
                raise ValueError("Incremental Top-k merge requires nondecreasing k")
            if current.select_all:
                copy_endpoint(finetuned, f"In-memory Top-k {k_percent:g}%")
                return current.selected_numel, expected_new

            previous_ties = previous.threshold_ties_to_keep
            current_ties = current.threshold_ties_to_keep
            newly_merged = 0
            with torch.no_grad():
                for name in tqdm(
                    names,
                    desc=f"Incremental Top-k {k_percent:g}%",
                    unit="tensor",
                ):
                    parameter = parameters[name]
                    if parameter.device.type == "meta":
                        raise ValueError(f"Parameter became meta/offloaded: {name}")
                    for index, scores in combined_contribution_chunks(
                        contributions, name, chunk_numel
                    ):
                        score_bits = contribution_score_bits(scores, "positive")
                        previous_mask, previous_ties = selection_mask(
                            score_bits,
                            previous,
                            previous_ties,
                        )
                        current_mask, current_ties = selection_mask(
                            score_bits,
                            current,
                            current_ties,
                        )
                        if np.any(previous_mask & ~current_mask):
                            raise RuntimeError(
                                "Prepared Top-k selections are not nested"
                            )
                        selected_indices = torch.from_numpy(
                            np.flatnonzero(current_mask & ~previous_mask)
                        )
                        if not selected_indices.numel():
                            continue
                        newly_merged += selected_indices.numel()
                        target = parameter if index is None else parameter[index]
                        finetuned_chunk = finetuned.read(name, index).reshape(-1)
                        selected_values = finetuned_chunk.index_select(
                            0, selected_indices
                        ).to(device=target.device, dtype=parameter.dtype)
                        target.reshape(-1).index_copy_(
                            0,
                            selected_indices.to(device=target.device),
                            selected_values,
                        )

            if previous_ties != 0 or current_ties != 0:
                raise RuntimeError("Top-k threshold ties were not fully consumed")
            if newly_merged != expected_new:
                raise RuntimeError(
                    "Incremental Top-k merge did not match the exact selection delta"
                )
            return current.selected_numel, newly_merged

        current_percent = None
        current_selection = None

        def apply(k_percent):
            """Move the resident weights to k and return merged-parameter counts."""

            nonlocal current_percent, current_selection
            k_percent = float(k_percent)
            if not 0 <= k_percent <= 100:
                raise ValueError("k_percent must be between 0 and 100")
            selection = get_selection(k_percent)
            retain_fraction = selection.selected_numel / total_numel
            effective_alpha = (
                1.0 / retain_fraction
                if use_weight_rescale and selection.selected_numel
                else 1.0
            )

            if current_percent is None and selection.selected_numel == 0:
                # evaluate_curve_models has just loaded the base checkpoint.
                merged_numel = 0
                newly_merged = 0
                update_mode = "resident base"
            elif current_percent == k_percent:
                merged_numel = selection.selected_numel
                newly_merged = 0
                update_mode = "unchanged"
            elif (
                not use_weight_rescale
                and current_percent is not None
                and k_percent > current_percent
            ):
                merged_numel, newly_merged = advance(
                    k_percent, current_selection, selection
                )
                update_mode = "incremental"
            else:
                merged_numel = reconstruct(
                    k_percent, selection, effective_alpha
                )
                newly_merged = merged_numel
                update_mode = "reconstructed"

            current_percent = k_percent
            current_selection = selection
            if merged_numel != selection.selected_numel:
                raise RuntimeError(
                    "In-memory Top-k merge did not match the exact selection count"
                )
            merged_percent = 100 * merged_numel / total_numel
            print(
                f"Merged {merged_numel:,}/{total_numel:,} scalar parameters "
                f"({merged_percent:.8f}% of the model); "
                f"newly_selected={newly_merged:,}; mode={update_mode}; "
                f"channel={channel}; selection=positive; "
                f"effective_alpha={effective_alpha:g}",
                flush=True,
            )
            return {
                "merged_numel": merged_numel,
                "total_numel": total_numel,
                "merged_percent": merged_percent,
            }

        yield apply
