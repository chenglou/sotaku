# Weight-Tying Results

All preregistered runs are included.

## Training

The validation floor and mean cover updates 12K-20K. The measured inference count is 16 for models trained only on iterations 1-16, and 1024 for models also trained on later iterations.

| Run | Status | Parameters | Training Hours | Total Recorded Hours | Validation Mean | Validation Floor |
|---|---|---:|---:|---:|---:|---:|
| tied_early_seed20260902 | complete | 796,937 | 1.05 | 1.19 | 77.49% | 75.60% |
| tied_early_seed20260903 | complete | 796,937 | 1.06 | 1.21 | 75.98% | 72.00% |
| tied_early_seed20260904 | complete | 796,937 | 1.05 | 1.21 | 77.33% | 75.20% |
| tied_late_seed20260902 | complete | 796,937 | 2.04 | 2.30 | 93.62% | 88.30% |
| tied_late_seed20260903 | complete | 796,937 | 2.03 | 2.39 | 90.60% | 82.10% |
| tied_late_seed20260904 | complete | 796,937 | 2.05 | 2.30 | 89.23% | 81.00% |
| untied_compute_early_seed20260902 | complete | 12,693,257 | 1.20 | 1.50 | 72.59% | 68.60% |
| untied_compute_early_seed20260903 | complete | 12,693,257 | 1.24 | 1.45 | 72.12% | 68.10% |
| untied_compute_early_seed20260904 | complete | 12,693,257 | 1.16 | 1.31 | 71.16% | 68.20% |
| untied_compute_late_seed20260902 | complete | 12,693,257 | 2.17 | 2.48 | 84.57% | 83.40% |
| untied_compute_late_seed20260903 | complete | 12,693,257 | 2.13 | 2.40 | 80.68% | 77.00% |
| untied_compute_late_seed20260904 | complete | 12,693,257 | 2.24 | 2.63 | 84.36% | 82.40% |
| untied_parameters_early_seed20260902 | complete | 797,385 | 0.74 | 0.85 | 38.08% | 31.90% |
| untied_parameters_early_seed20260903 | complete | 797,385 | 0.77 | 0.89 | 35.72% | 27.60% |
| untied_parameters_early_seed20260904 | complete | 797,385 | 0.75 | 0.88 | 39.87% | 31.60% |
| untied_parameters_late_seed20260902 | complete | 797,385 | 1.32 | 2.16 | 30.74% | 21.90% |
| untied_parameters_late_seed20260903 | complete | 797,385 | 1.30 | 2.27 | 32.28% | 26.80% |
| untied_parameters_late_seed20260904 | complete | 797,385 | 1.42 | 1.81 | 36.26% | 22.60% |

## Final Checkpoints (Primary)

### Reused 25K Sudoku-Extreme Benchmark

| Run | Selected Update | @16 | @128 | @1024 | @2048 | @4096 |
|---|---:|---:|---:|---:|---:|---:|
| tied_early_seed20260902 | 20000 | 79.40% | 12.01% | 0.13% | 0.04% | 0.00% |
| tied_early_seed20260903 | 20000 | 78.53% | 93.14% | 10.97% | 5.24% | 3.28% |
| tied_early_seed20260904 | 20000 | 79.48% | 94.00% | 97.58% | 98.03% | 98.30% |
| tied_late_seed20260902 | 20000 | 77.66% | 94.30% | 95.29% | 92.38% | 77.28% |
| tied_late_seed20260903 | 20000 | 78.27% | 94.17% | 92.13% | 91.00% | 89.84% |
| tied_late_seed20260904 | 20000 | 77.88% | 92.28% | 93.88% | 93.36% | 91.26% |
| untied_compute_early_seed20260902 | 20000 | 74.99% | 79.78% | 79.87% | 79.88% | 79.87% |
| untied_compute_early_seed20260903 | 20000 | 74.20% | 79.33% | 74.59% | 73.05% | 71.53% |
| untied_compute_early_seed20260904 | 20000 | 74.40% | 79.62% | 3.34% | 0.26% | 0.02% |
| untied_compute_late_seed20260902 | 20000 | 70.72% | 84.30% | 86.44% | 86.46% | 86.45% |
| untied_compute_late_seed20260903 | 20000 | 68.88% | 81.84% | 83.22% | 82.69% | 77.06% |
| untied_compute_late_seed20260904 | 20000 | 71.13% | 83.64% | 85.62% | 85.68% | 85.67% |
| untied_parameters_early_seed20260902 | 20000 | 42.50% | 49.73% | 3.20% | 0.53% | 0.12% |
| untied_parameters_early_seed20260903 | 20000 | 40.14% | 43.08% | 2.21% | 0.32% | 0.06% |
| untied_parameters_early_seed20260904 | 20000 | 43.46% | 51.87% | 45.00% | 38.31% | 32.13% |
| untied_parameters_late_seed20260902 | 20000 | 21.74% | 31.23% | 34.94% | 35.99% | 36.81% |
| untied_parameters_late_seed20260903 | 20000 | 24.48% | 31.08% | 35.49% | 36.91% | 38.01% |
| untied_parameters_late_seed20260904 | 20000 | 25.00% | 36.62% | 40.46% | 41.47% | 42.05% |

### New 10K QQWing Test Set

| Run | Selected Update | @16 | @128 | @1024 | @2048 | @4096 |
|---|---:|---:|---:|---:|---:|---:|
| tied_early_seed20260902 | 20000 | 98.16% | 9.81% | 0.00% | 0.00% | 0.00% |
| tied_early_seed20260903 | 20000 | 98.06% | 99.93% | 3.82% | 1.13% | 0.71% |
| tied_early_seed20260904 | 20000 | 98.04% | 99.92% | 99.96% | 99.97% | 99.97% |
| tied_late_seed20260902 | 20000 | 97.50% | 99.78% | 96.86% | 93.62% | 76.22% |
| tied_late_seed20260903 | 20000 | 97.76% | 99.77% | 95.76% | 94.46% | 93.53% |
| tied_late_seed20260904 | 20000 | 97.77% | 99.83% | 99.73% | 99.44% | 97.89% |
| untied_compute_early_seed20260902 | 20000 | 96.56% | 98.04% | 98.04% | 98.04% | 98.04% |
| untied_compute_early_seed20260903 | 20000 | 96.48% | 98.31% | 92.41% | 90.63% | 88.91% |
| untied_compute_early_seed20260904 | 20000 | 96.15% | 98.33% | 2.23% | 0.07% | 0.00% |
| untied_compute_late_seed20260902 | 20000 | 94.83% | 99.23% | 99.50% | 99.50% | 99.49% |
| untied_compute_late_seed20260903 | 20000 | 93.57% | 98.92% | 99.26% | 98.22% | 90.17% |
| untied_compute_late_seed20260904 | 20000 | 94.88% | 99.28% | 99.47% | 99.47% | 99.47% |
| untied_parameters_early_seed20260902 | 20000 | 79.93% | 86.26% | 5.41% | 0.95% | 0.32% |
| untied_parameters_early_seed20260903 | 20000 | 76.53% | 77.96% | 4.20% | 0.52% | 0.07% |
| untied_parameters_early_seed20260904 | 20000 | 80.65% | 88.63% | 76.74% | 64.91% | 54.16% |
| untied_parameters_late_seed20260902 | 20000 | 47.65% | 64.16% | 69.05% | 70.44% | 71.44% |
| untied_parameters_late_seed20260903 | 20000 | 53.11% | 64.23% | 71.08% | 72.83% | 74.37% |
| untied_parameters_late_seed20260904 | 20000 | 54.10% | 72.10% | 76.77% | 78.10% | 78.74% |

## Validation-Selected Checkpoints (Secondary)

### Reused 25K Sudoku-Extreme Benchmark

| Run | Selected Update | @16 | @128 | @1024 | @2048 | @4096 |
|---|---:|---:|---:|---:|---:|---:|
| tied_early_seed20260902 | 18000 | 79.15% | 56.16% | 0.25% | 0.04% | 0.00% |
| tied_early_seed20260903 | 19000 | 78.58% | 93.33% | 28.47% | 7.82% | 5.73% |
| tied_early_seed20260904 | 19000 | 79.24% | 94.04% | 97.74% | 98.19% | 98.50% |
| tied_late_seed20260902 | 17000 | 76.99% | 93.64% | 96.54% | 95.77% | 92.17% |
| tied_late_seed20260903 | 18000 | 78.04% | 94.00% | 94.66% | 94.23% | 93.95% |
| tied_late_seed20260904 | 20000 | 77.88% | 92.28% | 93.88% | 93.36% | 91.26% |
| untied_compute_early_seed20260902 | 19000 | 74.86% | 79.74% | 79.81% | 79.81% | 79.81% |
| untied_compute_early_seed20260903 | 19000 | 74.30% | 79.58% | 75.80% | 74.33% | 72.98% |
| untied_compute_early_seed20260904 | 20000 | 74.40% | 79.62% | 3.34% | 0.26% | 0.02% |
| untied_compute_late_seed20260902 | 20000 | 70.72% | 84.30% | 86.44% | 86.46% | 86.45% |
| untied_compute_late_seed20260903 | 20000 | 68.88% | 81.84% | 83.22% | 82.69% | 77.06% |
| untied_compute_late_seed20260904 | 16000 | 70.19% | 83.11% | 85.37% | 85.44% | 85.29% |
| untied_parameters_early_seed20260902 | 20000 | 42.50% | 49.73% | 3.20% | 0.53% | 0.12% |
| untied_parameters_early_seed20260903 | 20000 | 40.14% | 43.08% | 2.21% | 0.32% | 0.06% |
| untied_parameters_early_seed20260904 | 19000 | 43.58% | 51.87% | 44.77% | 38.06% | 31.77% |
| untied_parameters_late_seed20260902 | 17000 | 21.25% | 31.64% | 35.74% | 36.91% | 37.84% |
| untied_parameters_late_seed20260903 | 17000 | 23.44% | 31.25% | 36.02% | 37.38% | 38.54% |
| untied_parameters_late_seed20260904 | 18000 | 24.36% | 36.75% | 41.17% | 42.12% | 42.86% |

### New 10K QQWing Test Set

| Run | Selected Update | @16 | @128 | @1024 | @2048 | @4096 |
|---|---:|---:|---:|---:|---:|---:|
| tied_early_seed20260902 | 18000 | 97.95% | 65.06% | 0.00% | 0.00% | 0.00% |
| tied_early_seed20260903 | 19000 | 97.99% | 99.91% | 22.94% | 2.45% | 2.05% |
| tied_early_seed20260904 | 19000 | 98.03% | 99.91% | 99.96% | 99.96% | 99.96% |
| tied_late_seed20260902 | 17000 | 97.09% | 99.75% | 98.87% | 97.90% | 94.44% |
| tied_late_seed20260903 | 18000 | 97.78% | 99.77% | 98.46% | 98.16% | 97.87% |
| tied_late_seed20260904 | 20000 | 97.77% | 99.83% | 99.73% | 99.44% | 97.89% |
| untied_compute_early_seed20260902 | 19000 | 96.50% | 98.05% | 98.09% | 98.09% | 98.09% |
| untied_compute_early_seed20260903 | 19000 | 96.39% | 98.42% | 93.79% | 92.10% | 90.57% |
| untied_compute_early_seed20260904 | 20000 | 96.15% | 98.33% | 2.23% | 0.07% | 0.00% |
| untied_compute_late_seed20260902 | 20000 | 94.83% | 99.23% | 99.50% | 99.50% | 99.49% |
| untied_compute_late_seed20260903 | 20000 | 93.57% | 98.92% | 99.26% | 98.22% | 90.17% |
| untied_compute_late_seed20260904 | 16000 | 94.14% | 99.14% | 99.34% | 99.33% | 99.19% |
| untied_parameters_early_seed20260902 | 20000 | 79.93% | 86.26% | 5.41% | 0.95% | 0.32% |
| untied_parameters_early_seed20260903 | 20000 | 76.53% | 77.96% | 4.20% | 0.52% | 0.07% |
| untied_parameters_early_seed20260904 | 19000 | 80.92% | 88.68% | 76.46% | 64.14% | 53.45% |
| untied_parameters_late_seed20260902 | 17000 | 46.64% | 64.81% | 69.94% | 71.23% | 72.16% |
| untied_parameters_late_seed20260903 | 17000 | 50.90% | 64.44% | 71.23% | 72.86% | 74.21% |
| untied_parameters_late_seed20260904 | 18000 | 52.72% | 72.08% | 77.56% | 78.74% | 79.43% |

## Reliability

Healthy means at least 90% at 1024 iterations and no more than a 5-point drop by 4096. Numerical failures remain in the denominator. Pending runs are not failures.

| Architecture / Regime | Completed | Numerical Failures | Pending Training | Healthy Validation | Healthy Development | Healthy Holdout |
|---|---:|---:|---:|---:|---:|---:|
| tied/early | 3 | 0 | 0 | 1/3 (3 evaluated) | 1/3 (3 evaluated) | 1/3 (3 evaluated) |
| tied/late | 3 | 0 | 0 | 2/3 (3 evaluated) | 2/3 (3 evaluated) | 2/3 (3 evaluated) |
| untied_compute/early | 3 | 0 | 0 | 0/3 (3 evaluated) | 0/3 (3 evaluated) | 2/3 (3 evaluated) |
| untied_compute/late | 3 | 0 | 0 | 0/3 (3 evaluated) | 0/3 (3 evaluated) | 2/3 (3 evaluated) |
| untied_parameters/early | 3 | 0 | 0 | 0/3 (3 evaluated) | 0/3 (3 evaluated) | 0/3 (3 evaluated) |
| untied_parameters/late | 3 | 0 | 0 | 0/3 (3 evaluated) | 0/3 (3 evaluated) | 0/3 (3 evaluated) |

## Paired Comparisons

These are differences between three paired training seeds, not confidence intervals obtained by treating puzzles as independent training runs.

- early/untied_compute, iteration 16: +1.69 percentage points; 3 completed pairs, 0 missing or failed pairs.
- early/untied_parameters, iteration 16: +19.05 percentage points; 3 completed pairs, 0 missing or failed pairs.
- late/untied_compute, iteration 1024: -1.96 percentage points; 3 completed pairs, 0 missing or failed pairs.
- late/untied_parameters, iteration 1024: +25.15 percentage points; 3 completed pairs, 0 missing or failed pairs.

Early-trained untied stacks beyond iteration 16 are explicit stack-repetition diagnostics. Late-trained stacks already repeat during training. Neither result represents a fully untied network with thousands of independently trained stages.

Training time includes forward/backward passes, data transfer, optimizer work, and synchronization. Optimizer time is a subset, not an additional category. Total recorded hours add preparation, compilation, evaluation, and checkpoint time, but exclude queueing and container setup. New compiler signatures can add compilation time to the first training batches. The JSON records these categories and nominal FLOP estimates separately.

The new QQWing set and the reused sudoku-extreme benchmark are different distributions and must not be pooled. The paired comparisons above use the new set. Three training seeds do not establish a precise probability of a successful run.
