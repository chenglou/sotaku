# Release Validation Records

These are the 2026-09-02 release checks. [Numerical results](../PRECISION_RESULTS.md) explain the fixed-weight comparisons; the [dropout protocol](../../looping/BURNIN_DROPOUT.md) describes the separate training experiment. None of these runs changes the prepared reference checkpoint.

- `late_state_ce/` and `v1/`: full 25K evaluations and the fixed 1K precision matrix. Each condition retains its identity, dataset indices, runtime, source hashes, score, and per-puzzle predictions.
- `dropout/`: matched continuation results and final-checkpoint evaluations. Training summaries are named `training_result.json`; evaluator records are named `result.json`.
- `smoke_*_result.json`: CUDA training, save/resume, and module-mode checks. These use a tiny fixture, not the Sudoku benchmark.
- `jobs.json`: Modal app IDs and durable output locations.
- `source_snapshots.json`: checksums for the captured source archives. The original full BF16 evaluations retain source hashes but not a complete source archive.
- `verified_records.json`: independently recomputed counts and hashes for the downloaded records.

Large `per_puzzle.npz` files, repeated per-run benchmark index files, and source archives are excluded from Git and packaged separately for release. The canonical [25K indices](../benchmark_25k.json) remain in Git. Predictions use digits 0-8; scores compare originally empty cells against the dataset answers. Every-step solution-retention summaries are included only for conditions that explicitly enabled tracking.

The record archive contains `benchmark_25k.json` and a `validation/` directory at its root. Extract it into a new directory, then point the verifier at the extracted `validation/` directory. Verification does not need a GPU:

```sh
source venv/bin/activate
python -m release_tools.verify_records /path/to/extracted/validation --output /tmp/sotaku-verified.json
```

The command downloads the pinned test split. An already downloaded Arrow file can be supplied with `--cached-arrow`. The verifier checks the recorded row content, array checksums, predictions, bucket totals, and endpoint gains/losses. It cannot infer unrecorded intermediate predictions from endpoint-only runs.
