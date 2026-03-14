import modal

app = modal.App("sudoku-timing-probe")

outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "logs/", "runs/", "runs_modal/", "*.pt", "*.log"],
    )
)


@app.function(
    image=image,
    gpu="H200",
    timeout=30 * 60,
    volumes={"/outputs": outputs_volume},
)
def run_probe(
    exp_name: str = "iters.exp_baseline_lr2e3",
    model_name: str = "model_baseline_lr2e3.pt",
):
    import importlib
    import os
    import statistics
    import sys
    import time

    import torch

    sys.path.insert(0, "/root/project")
    mod = importlib.import_module(exp_name)

    device = torch.device("cuda")
    model_path = os.path.join("/outputs", model_name)

    load_start = time.perf_counter()
    model = mod.SudokuTransformer().to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start

    puzzle = "1....7.9..3..2...8..96..5....53..9...1..8...26....4...3......1..4......7..7...3.."
    cases = [
        (1, 32, 20),
        (1, 1024, 5),
        (64, 32, 10),
        (64, 1024, 3),
    ]

    results = []
    with torch.no_grad():
        for batch_size, n_iters, repeats in cases:
            mod.n_iterations = n_iters
            x = mod.encode_puzzles([puzzle] * batch_size).to(device)

            for _ in range(3):
                _ = model(x)
            torch.cuda.synchronize()

            times = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            mean_s = statistics.mean(times)
            stdev_s = statistics.pstdev(times)
            per_puzzle_ms = mean_s * 1000 / batch_size
            result = {
                "batch_size": batch_size,
                "n_iters": n_iters,
                "mean_s": mean_s,
                "stdev_s": stdev_s,
                "per_puzzle_ms": per_puzzle_ms,
            }
            results.append(result)
            print(
                f"batch={batch_size:>3} n_iters={n_iters:>4} "
                f"mean_s={mean_s:.4f} stdev_s={stdev_s:.4f} per_puzzle_ms={per_puzzle_ms:.2f}"
            )

    return {"load_s": load_s, "results": results}


@app.local_entrypoint()
def main(
    exp: str = "iters.exp_baseline_lr2e3",
    model: str = "model_baseline_lr2e3.pt",
):
    result = run_probe.remote(exp_name=exp, model_name=model)
    print(result)
