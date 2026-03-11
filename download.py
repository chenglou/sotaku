#!/usr/bin/env python3
# ARCHIVED SCRIPT
# Kaggle dataset download helper for the older local `data/sudoku-3m.csv` workflow.
# The current repo uses `sapientinc/sudoku-extreme` from Hugging Face instead.

import subprocess
import os

os.makedirs("data", exist_ok=True)
subprocess.run(["kaggle", "datasets", "download", "-d", "radcliffe/3-million-sudoku-puzzles-with-ratings", "-p", "data"])
subprocess.run(["unzip", "-o", "data/3-million-sudoku-puzzles-with-ratings.zip", "-d", "data"])
