"""Print the torch/CUDA versions inside the same Modal image the training runs use.

Usage:
    modal run modal_print_versions.py
"""

import modal

app = modal.App("sudoku-print-versions")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
)


@app.function(image=image)
def print_versions():
    import torch
    import datasets
    import numpy

    print(f"torch: {torch.__version__}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"datasets: {datasets.__version__}")
    print(f"numpy: {numpy.__version__}")


@app.local_entrypoint()
def main():
    print_versions.remote()
