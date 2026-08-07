"""Calibrate a solved-puzzle margin floor on the exact late-recheck windows."""

import json
import os
import re
import time

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import (
    _load_balanced_sample,
    _load_model,
)
from looping.eval_update_decomposition import (
    _distribution,
    _minimum_target_margin,
)


DEFAULT_MODELS = (
    {
        "name": "control_step_39000",
        "path": (
            "/outputs/looping/"
            "loop_stay_control_50k_trial0_checkpoint_step39000.pt"
        ),
        "model_kwargs": {},
        "trusted_full_checkpoint": True,
    },
    {
        "name": "late_switch_consistency_best",
        "path": (
            "/outputs/looping/"
            "model_loop_stay_late_switch_consistency_from39k_best_probe.pt"
        ),
        "model_kwargs": {},
    },
)
DEFAULT_BURNIN_HORIZONS = (32, 64, 128, 256, 512)
DEFAULT_RECHECK_GAPS = (16, 64, 256)
DEFAULT_MARGIN_FLOORS = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0)
TRAINING_WINDOW = 16


def summarize_margin_floors(margins, floors):
    margins = margins.float()
    return {
        str(float(floor)): {
            "active_fraction": (margins < floor).float().mean().item(),
            "mean_hinge_loss": F.relu(floor - margins).mean().item(),
        }
        for floor in floors
    }


def _solved(logits, targets, empty_mask):
    predictions = logits.argmax(dim=-1)
    return ((predictions == targets) | ~empty_mask).all(dim=1)


def _required_horizons(burnin_horizons, recheck_gaps):
    required = set()
    for burnin_horizon in burnin_horizons:
        anchor_horizon = burnin_horizon + TRAINING_WINDOW
        required.add(anchor_horizon)
        for gap in recheck_gaps:
            future_start = anchor_horizon + gap + 1
            required.update(
                range(future_start, future_start + TRAINING_WINDOW)
            )
    return required


def evaluate_model(
    model,
    inputs,
    targets,
    empty_mask,
    *,
    burnin_horizons,
    recheck_gaps,
    margin_floors,
    batch_size,
):
    device = next(model.parameters()).device
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    required_horizons = _required_horizons(
        burnin_horizons,
        recheck_gaps,
    )
    maximum_horizon = max(required_horizons)
    stores = {
        (burnin_horizon, gap): {
            "anchor_margins": [],
            "future_margins": [],
            "worst_window_margins": [],
            "anchor_solved": 0,
            "retained_at_window_end": 0,
            "retained_through_window": 0,
        }
        for burnin_horizon in burnin_horizons
        for gap in recheck_gaps
    }

    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        for batch_start in range(0, inputs.size(0), batch_size):
            batch_end = min(batch_start + batch_size, inputs.size(0))
            batch_inputs = inputs[batch_start:batch_end].to(device)
            batch_targets = targets[batch_start:batch_end].to(device)
            batch_empty = empty_mask[batch_start:batch_end].to(device)
            hidden_state = model.initial_encoder(batch_inputs)
            predictions = torch.zeros(
                batch_inputs.size(0),
                81,
                9,
                device=device,
            )
            logits_by_horizon = {}

            for iteration in range(1, maximum_horizon + 1):
                hidden_state = model.recurrent_step(
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                )
                logits = model.output_head(hidden_state)
                predictions = F.softmax(logits, dim=-1)
                if iteration in required_horizons:
                    logits_by_horizon[iteration] = logits.float()

            for burnin_horizon in burnin_horizons:
                anchor_horizon = burnin_horizon + TRAINING_WINDOW
                anchor_logits = logits_by_horizon[anchor_horizon]
                anchor_solved = _solved(
                    anchor_logits,
                    batch_targets,
                    batch_empty,
                )
                anchor_margins = _minimum_target_margin(
                    anchor_logits,
                    batch_targets,
                    batch_empty,
                )
                for gap in recheck_gaps:
                    store = stores[(burnin_horizon, gap)]
                    future_start = anchor_horizon + gap + 1
                    future_logits = [
                        logits_by_horizon[horizon]
                        for horizon in range(
                            future_start,
                            future_start + TRAINING_WINDOW,
                        )
                    ]
                    future_margins = torch.stack([
                        _minimum_target_margin(
                            logits,
                            batch_targets,
                            batch_empty,
                        )
                        for logits in future_logits
                    ])
                    future_solved = torch.stack([
                        _solved(
                            logits,
                            batch_targets,
                            batch_empty,
                        )
                        for logits in future_logits
                    ])
                    store["anchor_solved"] += int(
                        anchor_solved.sum().item()
                    )
                    store["retained_at_window_end"] += int(
                        (
                            future_solved[-1] & anchor_solved
                        ).sum().item()
                    )
                    store["retained_through_window"] += int(
                        (
                            future_solved.all(dim=0) & anchor_solved
                        ).sum().item()
                    )
                    if anchor_solved.any():
                        store["anchor_margins"].append(
                            anchor_margins[anchor_solved].cpu()
                        )
                        selected_future = future_margins[
                            :, anchor_solved
                        ]
                        store["future_margins"].append(
                            selected_future.flatten().cpu()
                        )
                        store["worst_window_margins"].append(
                            selected_future.min(dim=0).values.cpu()
                        )

    result = {}
    for (burnin_horizon, gap), store in stores.items():
        anchor_count = store["anchor_solved"]
        anchor_margins = torch.cat(store["anchor_margins"])
        future_margins = torch.cat(store["future_margins"])
        worst_window_margins = torch.cat(
            store["worst_window_margins"]
        )
        result[f"burnin_{burnin_horizon}_gap_{gap}"] = {
            "burnin_horizon": burnin_horizon,
            "anchor_horizon": burnin_horizon + TRAINING_WINDOW,
            "recheck_gap": gap,
            "anchor_solved": anchor_count,
            "anchor_total": inputs.size(0),
            "retained_at_window_end": (
                store["retained_at_window_end"] / max(anchor_count, 1)
            ),
            "retained_through_window": (
                store["retained_through_window"] / max(anchor_count, 1)
            ),
            "anchor_minimum_margin": _distribution([anchor_margins]),
            "future_minimum_margin": _distribution([future_margins]),
            "worst_window_minimum_margin": _distribution(
                [worst_window_margins]
            ),
            "margin_floors": summarize_margin_floors(
                future_margins,
                margin_floors,
            ),
        }
    return result


def evaluate(
    model_configs=DEFAULT_MODELS,
    *,
    burnin_horizons=DEFAULT_BURNIN_HORIZONS,
    recheck_gaps=DEFAULT_RECHECK_GAPS,
    margin_floors=DEFAULT_MARGIN_FLOORS,
    examples_per_bucket=200,
    batch_size=100,
    seed=42,
    output_dir=".",
    output_prefix="margin_floor_calibration_n1000",
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{output_prefix}.log")
    json_path = os.path.join(output_dir, f"{output_prefix}.json")
    inputs, targets, empty_mask, _, _, _ = _load_balanced_sample(
        examples_per_bucket,
        seed,
    )
    device = torch.device("cuda")
    result = {
        "config": {
            "models": list(model_configs),
            "burnin_horizons": list(burnin_horizons),
            "recheck_gaps": list(recheck_gaps),
            "training_window": TRAINING_WINDOW,
            "margin_floors": list(margin_floors),
            "sample_size": inputs.size(0),
            "seed": seed,
        },
        "models": {},
    }

    with open(log_path, "w") as log_file:
        def log(message=""):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(
            f"Margin-floor calibration: {inputs.size(0)} puzzles, "
            f"models={len(model_configs)}"
        )
        for model_config in model_configs:
            model_name = model_config["name"]
            log(f"\nMODEL {model_name}: {model_config['path']}")
            model = _load_model(model_config, device)
            model.requires_grad_(False)
            started_at = time.time()
            model_result = evaluate_model(
                model,
                inputs,
                targets,
                empty_mask,
                burnin_horizons=burnin_horizons,
                recheck_gaps=recheck_gaps,
                margin_floors=margin_floors,
                batch_size=batch_size,
            )
            result["models"][model_name] = model_result
            for key, row in model_result.items():
                floor = row["margin_floors"]["1.0"]
                log(
                    f"  {key}: anchor={row['anchor_solved']}/"
                    f"{row['anchor_total']}, "
                    f"retained={100 * row['retained_at_window_end']:.1f}%, "
                    f"floor1 active={100 * floor['active_fraction']:.2f}%"
                )
            log(f"  done in {time.time() - started_at:.1f}s")
            del model
            torch.cuda.empty_cache()

        temporary_path = json_path + ".tmp"
        with open(temporary_path, "w") as result_file:
            json.dump(result, result_file, indent=2)
            result_file.write("\n")
        os.replace(temporary_path, json_path)
        log(f"\nStructured results: {json_path}")
    return result


if __name__ == "__main__":
    evaluate()
