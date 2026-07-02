# Sotaku Serious Viridian Probe

This is the real-data B200 runner. It uses `sapientinc/sudoku-extreme`, the current `iters.exp_baseline_lr2e3` model, the same checkpoint shape as the repo, and R2 for durable status/report/checkpoint objects.

The full 50k-step training run completed through a guarded resume job:

```text
job: job_01KWEZ8FHSWP858CBAP5GNTGC7
prefix: viridian-runner-probes/runs/sotaku-full-b200-resume-guarded-20260701-135253/
final checkpoint: attempts/slot-000/checkpoints/checkpoint-00002.pt
step: 50,000
sha256: 40d85a10fe357cda62051430ff2625a976df39eceef3cbd2a16f3dad4a5134f4
```

The safe artifact layout writes outputs per eval attempt:

```text
attempts/slot-NNN/lease.json
attempts/slot-NNN/status.json
attempts/slot-NNN/report.jsonl
attempts/slot-NNN/latest.json
attempts/slot-NNN/checkpoints/checkpoint-00000.pt
attempts/slot-NNN/checkpoints/checkpoint-00000.pt.sha256
...
```

`status.json` is overwritten in place per slot. `report.jsonl` grows at one compact progress row per minute per slot. Checkpoints are step-specific ordinal files and are not overwritten.

## Full Run Result

The first full attempt was `job_01KWEQSQ66XJ1CBN1961Y9T2VA` under `viridian-runner-probes/runs/sotaku-full-b200-attempt-slots-20260701-114217/`. Slot 0 trained from scratch to step 46,753, then stopped updating. Its newest verified checkpoint was:

```text
attempts/slot-000/checkpoints/checkpoint-00023.pt
step: 46,297
sha256: 0eaa0a200e43b405d29ee52aa8c4e0c140d0975422a847e52a821bcaba781931
```

That job also spawned repeated duplicate evals; slots 1 through 4 were observed before cancellation. Attempt slots kept the artifacts separated, but they did not stop wasted duplicate compute.

The guarded resume job `job_01KWEZ8FHSWP858CBAP5GNTGC7` downloaded that checkpoint, verified the SHA sidecar, loaded the checkpoint before `torch.compile()`, restored the optimizer, and resumed from step 46,297. It finished at step 50,000 with:

```text
loss: 0.5634099841117859
train_acc: 0.8650548411663267
score: 50000.0
```

Local final-checkpoint verification passed:

```text
sha_match: yes
torch.load step: 50000
config experiment: exp_baseline_lr2e3
train_size: 2700000
batch_size: 2048
total_steps: 50000
checkpoint keys: config, model_state_dict, optimizer_state_dict, step
```

The guarded resume also proved the duplicate-attempt guard remotely. A second eval attempt claimed slot 1, saw slot 0 as `finished` at step 50,000, wrote `duplicate_attempt_skipped`, and exited without training. Private presigned manifest objects were deleted after verification.

## Sudoku Extreme Eval Result

The final checkpoint was evaluated on the canonical `sapientinc/sudoku-extreme` test path with deterministic bucket sampling: 5,000 puzzles per rating bucket, 25,000 puzzles total.

1024 test-time iterations:

```text
job: job_01KWFRB65Q8H33A983N5NTPJN3
prefix: viridian-runner-probes/runs/sotaku-eval-full-1024-20260701-211116/
checkpoint step: 50,000
checkpoint sha256: 40d85a10fe357cda62051430ff2625a976df39eceef3cbd2a16f3dad4a5134f4
result: 1,441 / 25,000 solved = 5.764%
B200 gpu_seconds: 375
```

The eval job used one attempt slot, verified the checkpoint SHA sidecar, loaded the 422,786-row test split, and selected 25,000 puzzles with the same bucket sampling as `iters/eval_more_iters.py`. This result is far below the repo's released-checkpoint reference of `24,728 / 25,000 = 98.9%` at 1024 test-time iterations.

Follow-up diagnostic at 16 and 128 iterations:

```text
job: job_01KWFRTBVFW9TSNAYG6AKX2DEB
prefix: viridian-runner-probes/runs/sotaku-eval-full-16-128-20260701-211933/
16 iters: 20,225 / 25,000 solved = 80.90%
128 iters: 21,665 / 25,000 solved = 86.66%
B200 gpu_seconds: 92
```

Control eval with the released checkpoint:

```text
job: job_01KWFSH9JER3XQ24MWJ7Y677BM
prefix: viridian-runner-probes/runs/sotaku-eval-released-1024-20260701-213203/
checkpoint sha256: 4f2ee45da4296fc2ce860dcf953c466df907d3878d39c8d4fa0afa9ba28df69e
1024 iters: 24,694 / 25,000 solved = 98.776%
B200 gpu_seconds: 172
```

That control means the Viridian eval harness is not the problem. The Viridian-trained checkpoint learned the 16-iteration task about as expected, improved at 128 iterations, but collapsed by 1024 iterations. The likely problem is the custom Viridian training wrapper, which reimplements the training loop instead of calling the blessed `iters.exp_baseline_lr2e3.train()` path that reproduces on Modal.

The pre-resume checkpoint already had the 1024-iteration collapse, so the guarded resume was not the cause:

```text
job: job_01KWFT6ADAMPY7D2VF7AH9BK90
prefix: viridian-runner-probes/runs/sotaku-eval-step46297-quick-20260701-214333/
checkpoint step: 46,297
sample: 500 puzzles per rating bucket, 2,500 total
16 iters: 1,984 / 2,500 solved = 79.36%
128 iters: 1,993 / 2,500 solved = 79.72%
1024 iters: 139 / 2,500 solved = 5.56%
```

The replacement path is `repo/train_canonical.py` plus `presign_canonical_train_r2.py`. That runner calls `iters.exp_baseline_lr2e3.train(output_dir=...)` directly and only handles Viridian/R2 concerns outside the training loop: slot claiming, optional checkpoint download for resume, background upload of the canonical log/checkpoints/final model, status, and duplicate-attempt skipping. This removes the custom training-loop fork as a variable. If canonical Viridian training still fails after that, the remaining suspect is a runtime/platform difference, e.g. the torch/CUDA/compiler stack.

**Resolution (2026-07-02, after the follow-up investigation):** the wrapper hypothesis above was wrong, and so was the runtime/platform suspicion. A line-by-line comparison plus adversarial review found the recreated loop mathematically identical in distribution to the canonical trainer, and the training data byte-identical (verified by digest). Fresh canonical-code runs then collapsed on Modal H200 with the original February image and settled once on Viridian B200 — both stacks appear in both outcome columns. Across all runs to date, the variable that separates success from failure is whether training ran uninterrupted: February's four clean runs all settled, while runs interrupted mid-anneal and resumed (by preemption, client cancellation, or the platform's 2-hour eval kill) almost all collapsed at long test-time iteration despite the resume restoring model and optimizer state exactly. See the repo README / iters/EXPERIMENTS_ITERS.md for the current account, and `viridian/train/` for the productized wrapper that replaced this probe's runner (single source of truth: it packages the live repo files at submit time).

## Eval Harness

The Viridian eval harness lives in `repo/eval_sudoku_extreme.py`, with presigned-R2 job generation in `presign_eval_r2.py`. It downloads the final checkpoint and SHA sidecar, verifies the SHA-256 before loading, runs the same bucketed test selection as `iters/eval_more_iters.py`, and writes compact `status.json`, `report.jsonl`, and `latest.json` objects under per-attempt R2 slots.

## Pilot Result

The first real-data B200 pilot used:

```text
job: job_01KWEK168F4RV08Q8EM5DX2603
prefix: viridian-runner-probes/runs/sotaku-serious-b200-20260701-101911/
baseline: sha256:5fc8c2746996d89defc6c79aba007db66e803c5dabdd31a45cc4691a848970c4
train_size: 2,700,000
batch_size: 2,048
compile: true
```

The real Hugging Face train split loaded on B200 after the wrapper passed explicit features for the Sudoku string columns. Without those features, the CSV loader inferred an 81-character Sudoku value as an integer and PyArrow failed with an overflow.

The useful first eval attempt loaded 3,831,994 available train rows in about 10.7s, encoded 2.7M rows in about 6.0s, compiled the model, reached phase 3, and uploaded a checkpoint at step 20,198. That checkpoint is archived here:

```text
r2:sotaku-viridian/viridian-runner-probes/runs/sotaku-serious-b200-20260701-101911/archive/pre-conflict-step-20198/
```

Local verification passed:

```text
checkpoint.pt sha256: 5a104561cee9e3e455fb03e18e4780bc1e0bc6ae17f88a870fe871b078fe57f2
torch.load step: 20198
keys: config, model_state_dict, optimizer_state_dict, step
```

The fixed-key artifact layout is unsafe for long runs. More than one eval attempt wrote to the same R2 keys under this job, so `latest.json` later regressed to a lower-step checkpoint. The attempt-slots layout writes step-specific checkpoints and attempt-specific reports, then treats `latest.json` only as a convenience pointer.

Trace evidence:

```text
first eval cwd: /tmp/vd-eval-tvpofe8u
first eval start: 2026-07-01T10:19:26Z
first eval observed through: step 20,624 at 2026-07-01T11:14:07Z

second eval cwd: /tmp/vd-eval-pcirlf3z
second eval start: 2026-07-01T10:47:21Z
second eval observed after cancel: step 14,912 at 2026-07-01T11:27:49Z
```

The Viridian `/logs` endpoint returned a platform-side redirect error for this job:

```text
gpu eval server: error following redirect for url (...modal.run/?__modal_function_call_id=...)
```

That points to Viridian losing contact with the GPU eval server and starting another eval attempt while the first attempt kept running. The API job status changed to `cancelled`, but the second attempt continued writing R2 objects for at least several minutes afterward.

## Duplicate-Writer Defense

The wrapper now supports an `attempt-slots` artifact layout. The presigner uploads a private manifest with pre-signed URLs for several attempt slots. At startup, each eval tries to claim a slot by writing `attempts/slot-NNN/lease.json` with `If-None-Match: *`. R2 returns `200` for the first claimant and `412` for later claimants, so overlapping eval attempts land in different slots.

Each slot has its own:

```text
attempts/slot-NNN/lease.json
attempts/slot-NNN/status.json
attempts/slot-NNN/report.jsonl
attempts/slot-NNN/latest.json
attempts/slot-NNN/checkpoints/checkpoint-00000.pt
attempts/slot-NNN/checkpoints/checkpoint-00000.pt.sha256
...
```

This removes cross-attempt overwrites. `latest.json` is only per-slot now. The source of truth is the set of step/ordinal checkpoint files under each claimed slot.

Future specs also include a duplicate-attempt guard. A nonzero slot checks slot 0's status; if slot 0 is finished or has updated recently, the duplicate records `duplicate_attempt_skipped`, uploads its short report, prints `METRIC: 0`, and exits. If slot 0 is stale or failed, the duplicate can continue as a fallback.

The presigner can also build a resume spec from a known R2 checkpoint:

```text
python3 viridian/sotaku_serious_probe/presign_r2.py \
  --mode resume \
  --resume-checkpoint-key viridian-runner-probes/runs/.../attempts/slot-000/checkpoints/checkpoint-00006.pt \
  --resume-checkpoint-sha-key viridian-runner-probes/runs/.../attempts/slot-000/checkpoints/checkpoint-00006.pt.sha256 \
  ...
```

The eval loads the checkpoint before `torch.compile()`, verifies the checkpoint config, restores the optimizer, and then writes fresh outputs under the new run's attempt slots.

Local verification:

```text
conditional R2 lease PUT: first claimant 200, second claimant 412
two local claimers: slot 0 then slot 1
dummy checkpoint upload: checkpoint-00000.pt + sha256 sidecar + per-slot latest.json
```

Remote B200 smoke:

```text
job: job_01KWEQH67R9GDA5BKV8YKG1HQD
prefix: viridian-runner-probes/runs/sotaku-attempt-slots-smoke-20260701-113749/
score: 1.0
```

That smoke reproduced the duplicate eval behavior. The two attempts claimed `slot-000` and `slot-001`, wrote separate reports/statuses/checkpoints, and both downloaded checkpoint hashes matched their sidecars. The private presigned manifest object was deleted after the smoke finished.
