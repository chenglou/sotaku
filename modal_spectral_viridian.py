"""
Modal wrapper: spectral radius comparison for the Viridian recreated-trainer checkpoint.

Compares the collapsed Viridian-trained checkpoint (step 50,000, 5.76% at 1024 test
iterations) against the canonical stable model and the two documented failure modes
(oscillatory divergence at LR=3e-3, stagnation at LR=1e-3), to classify which side of
the stability band the Viridian run landed on.

Usage:
    modal run --detach modal_spectral_viridian.py

Log: modal volume get sudoku-outputs viridian_diag/spectral_radius.log .
"""

import modal

from modal_config import PROJECT_IGNORE

app = modal.App("sudoku-spectral-viridian")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=PROJECT_IGNORE)
)


@app.function(
    image=image,
    gpu="H200",
    timeout=2 * 60 * 60,
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_analysis():
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    try:
        from iters.eval_spectral_radius import analyze_models

        os.makedirs("/outputs/viridian_diag", exist_ok=True)

        configs = [
            ('Viridian recreated (step 50k, collapse@1024)', '/outputs/model_viridian_recreated_step50000.pt', 'iters.exp_baseline_lr2e3'),
            ('LR=2e-3 canonical (stable)', '/outputs/model_baseline_lr2e3.pt', 'iters.exp_baseline_lr2e3'),
            ('LR=3e-3 (oscillatory collapse@64)', '/outputs/model_baseline_lr3e3.pt', 'iters.exp_baseline_lr3e3'),
            ('LR=1e-3 (stagnation)', '/outputs/model_baseline_lr1e3.pt', 'iters.exp_baseline_lr1e3'),
        ]

        analyze_models(configs, checkpoints=[16, 32, 64, 128, 256, 512, 1024],
                       device='cuda', output_dir='/outputs/viridian_diag')
    finally:
        outputs_volume.commit()
    print("Done. Download log with:")
    print("  modal volume get sudoku-outputs viridian_diag/spectral_radius.log .")


@app.local_entrypoint()
def main():
    call = run_analysis.spawn()
    print(f"Spawned function call: {call.object_id}")
