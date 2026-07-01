# Viridian Pretraining Probe Runs

## Current Summary

Viridian runner mode works for small GPU jobs. Use `agents: 0`, `limits.max_gens: 1`, and `contract.eval_cmd`; the old `train.py` workaround and old required-looking compat fields are obsolete.

What is proven:

- CPU and L4 runner jobs can execute a one-line `eval.py`.
- The tier endpoint lists CPU, L4, A100, H100, and B200. The training-mechanics probes have tested L4, H100, and B200.
- L4 and H100 can import PyTorch and see CUDA. H100 ran a tiny CUDA matmul; L4 ran the R2 checkpoint/resume PyTorch probe.
- H100 can run a minutes-long PyTorch optimizer loop, write a local checkpoint and log, validate both, and finish with a metric.
- B200 can run a tiny real-Sotaku training/resume probe with `torch.compile`, checkpoint upload, SHA-256 sidecars, checkpoint download, `torch.load`, resume, and report upload.
- A six-minute B200 logging probe can update `status.json` and `report.jsonl` through R2 without log size blowup.
- B200 can load the real Hugging Face `sapientinc/sudoku-extreme` train split, encode 2.7M training rows, compile the current Sotaku model, train into phase 3, and upload a resumable checkpoint with model state, optimizer state, config, and step.
- A full B200 Sotaku training run can reach step 50,000 on Viridian by resuming from a verified R2 checkpoint.
- The `attempt-slots` artifact layout prevents overlapping eval attempts from overwriting each other's reports, status, and checkpoints. A B200 smoke reproduced the duplicate eval behavior and kept the two attempts separated.
- The duplicate-attempt guard works on a remote B200 job: a nonzero slot observed slot 0 as finished and exited with `duplicate_attempt_skipped`.
- `vd artifact` returns the repo snapshot, not runtime-created checkpoint/log files.
- GPU jobs can upload runtime-created checkpoint/log files over the network.
- A committed Viridian dataset can be reused by multiple jobs, including concurrent L4 jobs.
- L4 jobs can use controlled R2 object storage for checkpoint upload, hash sidecars, checkpoint download, `torch.load`, resume, and report upload.
- CPU jobs have no network, so storage upload/download probes must use a GPU tier.

What is not proven yet:

- Useful eval stdout/stderr through `/v1/jobs/{id}/logs`; all tested runner jobs returned `entries: []`.

## Controlled Object Storage

Created a dedicated Cloudflare R2 bucket for Viridian checkpoints/logs:

```text
sotaku-viridian
```

The local `r2:` remote now uses an admin key, has bucket checks enabled, and can list buckets at the account root:

```text
rclone lsd r2:
```

A tiny upload/read/delete probe under `viridian-runner-probes/` succeeded.

## Key Jobs

| Probe | Job | Result |
| --- | --- | --- |
| Current-doc minimal CPU runner | `job_01KWE6W85F72NH7SCB9F7J8W2Q` | `score 1.0`; omitted old compat fields |
| H100 CUDA/PyTorch smoke | `job_01KWE1J07N0GGTVRP6DSTBXSSD` | `score 1.0`, H100 CUDA visible |
| H100 checkpoint/log local write | `job_01KWE2547PDBM4K3EC7BCHK9G0` | `score 357331.0`; local checkpoint/log validated; artifact excluded runtime outputs |
| Viridian dataset visible | `job_01KWE407C714TGKGP5Y8MAQ2P2` | `score 1.0` |
| Shared dataset CPU check | `job_01KWE9F2BFTGSSJFGS0PBK7KE0` | `score 12004.0` |
| Shared dataset L4 check 1 | `job_01KWE9F2DAPTZ119Q6CSD51NHQ` | `score 12004.0` |
| Shared dataset L4 check 2 | `job_01KWE9F2DKNKADR936H5P33K5C` | `score 12004.0` |
| R2 fresh checkpoint upload | `job_01KWECTGRT2BJ5JWQP76CNG2YR` | `score 200.0`; uploaded checkpoint, sha256, and report |
| R2 checkpoint resume | `job_01KWECX497D4G7V9EZW25AZ2T9` | `score 400.0`; downloaded, verified, loaded, resumed, and uploaded a new checkpoint/report |
| B200 Sotaku tiny fresh | `job_01KWEENZVZJGB1VH31Z6007C6V` | `score 2.0`; B200 visible; compiled real Sotaku model; uploaded checkpoint, sha256, and report |
| B200 Sotaku tiny resume | `job_01KWEF5091J35YFCJA69EXBXEZ` | `score 4.0`; downloaded, verified, loaded, resumed, compiled, and uploaded a new checkpoint/report |
| B200 logging cadence | `job_01KWEG5M2PB9CV4SDBVDDC7X23` | `score 1.0`; 6-minute run; status/report uploads stayed small |
| B200 real-data dataset repro | `job_01KWEJYN97B684AKC34JNAXAZ7` | `score 1.0`; explicit HF `Features` fixed CSV integer overflow; checkpoint SHA verified |
| B200 one-hour real-data pilot | `job_01KWEK168F4RV08Q8EM5DX2603` | cancelled after artifact conflict; loaded full train split, encoded 2.7M rows, reached phase 3, archived checkpoint step 20,198 |
| B200 attempt-slots smoke | `job_01KWEQH67R9GDA5BKV8YKG1HQD` | `score 1.0`; duplicate eval attempts landed in separate slots; both checkpoint SHAs verified |
| B200 full attempt-slots run | `job_01KWEQSQ66XJ1CBN1961Y9T2VA` | cancelled after slot 0 stopped updating near step 46,753; checkpoint step 46,297 was verified and used for resume |
| B200 guarded resume to 50k | `job_01KWEZ8FHSWP858CBAP5GNTGC7` | `score 50000.0`; resumed from step 46,297, finished step 50,000, final checkpoint SHA verified |

## Shared Dataset Reuse

Created one 20-line committed dataset:

```text
ds_01KWE9EE4XGYF1M5H1V43FRF5R
bytes: 560
digest: sha256:418e6e8aa133be40d6d1a24a13761a42561a8df38e9fa741c52211c7f2cddc03
```

The probe reads `VD_DATA_TRAIN` and `VD_DATA_VAL`, verifies every mounted line has the expected token, verifies `data/test.txt` is absent, and prints `train_lines * 1000 + val_lines`. With `val: 0.2` and `test: 0.2` on 20 input lines, the expected score is `12004`: 12 train lines, 4 val lines, and 4 test lines not mounted.

One CPU job and two concurrent L4 jobs all returned `12004.0` against the same dataset id. This verifies the shared dataset-artifact path we want for parallel Sotaku runs. It does not verify a shared Hugging Face Arrow cache or shared runtime filesystem; each eval gets its own mounted train/val files.

## Controlled Storage Resume

R2 run prefix:

```text
s3://sotaku-viridian/
viridian-runner-probes/runs/resume-object-20260701-013041/
```

Packaged baseline digest:

```text
sha256:1f4672ad05e353cc90b30b2f87210d56e6d5e3ba3d0e702c7ba9a1a0fa5ce677
```

The fresh L4 job wrote a PyTorch checkpoint at step 200 and uploaded:

```text
checkpoint.pt
checkpoint.pt.sha256
fresh_report.jsonl
```

The resume L4 job downloaded `checkpoint.pt` and `checkpoint.pt.sha256`, verified the SHA-256, loaded the checkpoint with `torch.load`, resumed from step 200, trained to step 400, and uploaded:

```text
resume_checkpoint.pt
resume_checkpoint.pt.sha256
resume_report.jsonl
```

Downloaded artifacts were checked locally:

```text
checkpoint.pt sha256: b8ab2820851d297f64a2de1317624db35f656a86ff9045b1f3a6df8475da8b73
resume_checkpoint.pt sha256: 52b5fb6c35da9ba27f749777300a36bc0eed1ed52218a1a9a87b0005557c692c
```

Both hashes matched their sidecar files. A local `torch.load(..., map_location="cpu", weights_only=False)` check also confirmed `checkpoint.pt` has `step: 200` and `resume_checkpoint.pt` has `step: 400`.

## B200 Sotaku Tiny Run

R2 run prefix:

```text
s3://sotaku-viridian/
viridian-runner-probes/runs/sotaku-b200-20260701-090303/
```

Packaged baseline digest:

```text
sha256:bf9f08c76c384b8a0082a3b3f58bf67300a54c0c9f20185b3bcd228173c6fe92
```

This probe uses `iters.exp_baseline_lr2e3.SudokuTransformer`, the real encoding helpers, the real checkpoint shape, and the correct resume order: load model state before `torch.compile`, then create/load the optimizer. It uses a repo-packaged 128-puzzle CSV shard, so it does not test Hugging Face dataset download.

Settings:

```text
gpu_tier: b200
batch_size: 16
fresh steps: 2
resume steps: 2
compile: true
tiny data sha256: 5dfd069b53924f972cf45b95943287e337c2d44ab201fae8d72c09314ef68be9
```

Fresh B200 job `job_01KWEENZVZJGB1VH31Z6007C6V` converged with score `2.0`. The report shows `NVIDIA B200`, PyTorch `2.11.0+cu128`, CUDA available, `torch.compile` ran, and the checkpoint uploaded with SHA-256:

```text
fresh_checkpoint.pt sha256: ccaae8f54424a01b99c44538017f0f8a66d248ded21d1db1773e39ad3c15add5
```

Resume B200 job `job_01KWEF5091J35YFCJA69EXBXEZ` converged with score `4.0`. The report shows it downloaded the fresh checkpoint, verified the SHA-256, loaded step 2, compiled, trained to step 4, and uploaded:

```text
resume_checkpoint.pt sha256: fd75fe8b651661abc3f34bf8db6e5edbe17f8a3a042fdb9a9c9d21f5dd40a9f6
```

Downloaded artifacts were checked locally. Both hashes matched their sidecar files, and `torch.load(..., map_location="cpu", weights_only=False)` confirmed `fresh_checkpoint.pt` has `step: 2` and `resume_checkpoint.pt` has `step: 4`.

The two B200 jobs spent `409` and `242` B200 GPU-seconds. The JSONL reports show the Python work itself was short after allocation; most wall-clock waiting happened before eval output appeared.

## B200 Logging Cadence

R2 run prefix:

```text
s3://sotaku-viridian/
viridian-runner-probes/runs/sotaku-b200-logging-20260701-092857/
```

Packaged baseline digest:

```text
sha256:ec9fc80c70fed3ef4ecbcb4395cc0b30cc3835758d1703334e8e7041615528a3
```

Job `job_01KWEG5M2PB9CV4SDBVDDC7X23` ran on B200 for a six-minute training window with one compact progress row and one R2 upload per minute. `fresh_status.json` appeared before Viridian had useful logs and was overwritten in place throughout the run.

Viridian reported job score `1.0`, but the R2 report and checkpoint show the run reached step `6,926`. Treat R2 reports/checkpoints as the source of truth for training progress. After this run, the Sotaku probe was changed to print only event names to stdout and keep detailed numeric JSON in R2, so future runs are less likely to confuse metric parsing.

Final objects:

```text
fresh_checkpoint.pt          9,648,593 bytes
fresh_checkpoint.pt.sha256          65 bytes
fresh_report.jsonl                4,233 bytes
fresh_status.json                   239 bytes
```

Report shape:

```text
line_count: 16
max_line_bytes: 1,829
avg_line_bytes: 263.6
final_step: 6,926
```

The report had no presigned URL or obvious secret patterns. The checkpoint hash matched its sidecar:

```text
fresh_checkpoint.pt sha256: 4dda5a848501ee14c027d2dafb5dac9c06b920c7e0f00409e9f7c7f77ce18902
```

Local `torch.load(..., map_location="cpu", weights_only=False)` confirmed the checkpoint has `step: 6926`.

## B200 Real-Data Sotaku Pilot

R2 run prefix:

```text
s3://sotaku-viridian/
viridian-runner-probes/runs/sotaku-serious-b200-20260701-101911/
```

Packaged baseline digest:

```text
sha256:5fc8c2746996d89defc6c79aba007db66e803c5dabdd31a45cc4691a848970c4
```

The first real-data attempts found a dataset loader issue before training: `load_dataset("sapientinc/sudoku-extreme", split="train")` let Pandas infer the 81-character Sudoku strings as integers, then PyArrow failed with `OverflowError: Python int too large to convert to C long`. Passing explicit Hugging Face features fixed it:

```python
Features({
    "question": Value("string"),
    "answer": Value("string"),
    "rating": Value("int32"),
})
```

Tiny repro `job_01KWEJYN97B684AKC34JNAXAZ7` proved the fixed loader on B200. It prepared the train split, encoded 16 rows, ran one training step, uploaded `latest_checkpoint.pt`, and the SHA-256 sidecar matched locally.

The one-hour pilot `job_01KWEK168F4RV08Q8EM5DX2603` used the real train split and the current model:

```text
gpu_tier: b200
train_size: 2,700,000
available_train: 3,831,994
batch_size: 2,048
compile: true
duration_s: 3,600
first_checkpoint_s: 120
checkpoint_every_s: 300
```

Useful timing from the first eval attempt:

```text
dataset load: 10.682s
encoding 2.7M rows: 6.020s
first compiled step: about 190s
steady training: about 6 steps/s
phase 2 entered at step 10,000
phase 3 entered at step 20,000
```

The highest preserved checkpoint from the pilot is archived here:

```text
s3://sotaku-viridian/
viridian-runner-probes/runs/sotaku-serious-b200-20260701-101911/archive/pre-conflict-step-20198/
```

Archived checkpoint:

```text
checkpoint.pt sha256: 5a104561cee9e3e455fb03e18e4780bc1e0bc6ae17f88a870fe871b078fe57f2
step: 20,198
```

Local verification passed:

```text
sha_match: yes
torch.load step: 20198
config experiment: exp_baseline_lr2e3
train_size: 2700000
batch_size: 2048
compile: true
checkpoint keys: config, model_state_dict, optimizer_state_dict, step
```

Important artifact finding: the pilot used fixed R2 object names (`status.json`, `report.jsonl`, `latest_checkpoint.pt`, `latest_checkpoint.pt.sha256`, `latest.json`). During the run, those fixed keys were written by more than one eval attempt under the same Viridian job. We observed `latest.json` advance to step 20,198, then later regress to step 12,070 from another writer. That makes fixed keys unsafe for longer jobs unless Viridian can guarantee a single eval writer. The next serious run should write attempt-specific and step-specific objects, or use a separate storage credential inside the eval so the object key can include an attempt id and checkpoint step.

Trace evidence for the duplicate writer:

```text
first eval cwd: /tmp/vd-eval-tvpofe8u
first eval start: 2026-07-01T10:19:26Z
first eval observed through: step 20,624 at 2026-07-01T11:14:07Z

second eval cwd: /tmp/vd-eval-pcirlf3z
second eval start: 2026-07-01T10:47:21Z
second eval observed after cancel: step 14,912 at 2026-07-01T11:27:49Z
```

The two eval windows overlap for about 27 minutes. Both evals were `mode: fresh`, and both wrote to the same R2 prefix. The Viridian `/logs` endpoint for this job returned one platform-side line:

```text
gpu eval server: error following redirect for url (...modal.run/?__modal_function_call_id=...)
```

That suggests Viridian lost contact with a Modal eval server and started another eval attempt while the first one kept running. The API job status was later changed to `cancelled`, but the second eval continued updating R2 for at least several minutes afterward. So this is not just a reader-side stale-object issue.

Implemented defense in `viridian/sotaku_serious_probe/`: the presigner can generate an `attempt-slots` manifest. Each eval attempt claims a slot with an R2 conditional PUT (`If-None-Match: *`) before uploading outputs. Reports, status, per-slot latest pointers, and checkpoints are all written under `attempts/slot-NNN/`. Checkpoints use ordinal filenames, e.g. `checkpoints/checkpoint-00000.pt`, with sidecars. Cheap local/R2 verification passed:

```text
conditional lease PUT: first writer 200, second writer 412
two claimers against the same manifest: slot 0 then slot 1
dummy checkpoint upload: checkpoint, sha256 sidecar, and per-slot latest uploaded
```

Remote B200 smoke `job_01KWEQH67R9GDA5BKV8YKG1HQD` also passed, and it reproduced the duplicate-eval behavior in miniature. Two fresh attempts claimed different slots:

```text
prefix: viridian-runner-probes/runs/sotaku-attempt-slots-smoke-20260701-113749/
slot-000 latest: checkpoint-00001.pt, step 1, sha256 7938b1d5e2cb74d7825cc28fb005ff75604d4159f56e05380892961a901308ba
slot-001 latest: checkpoint-00001.pt, step 1, sha256 2641ed2aa23e3f4dba4639c801c553185b9b556a7a3d2ba30407c208ed365111
```

Both checkpoint hashes matched their sidecars when downloaded locally. The private presigned manifest object was deleted after the smoke job finished.

Future specs now also include a duplicate-attempt guard. A nonzero slot checks slot 0's status; if slot 0 is finished or has updated recently, the duplicate writes `duplicate_attempt_skipped`, uploads its short report, prints `METRIC: 0`, and exits. If slot 0 is stale or failed, the duplicate can continue as a fallback. The guard has a local test for fresh, stale, and finished primary statuses.

The same presigner can now generate a resume spec from a known R2 checkpoint object and SHA sidecar. That gives us an escape hatch if a long run needs to be restarted from the highest verified checkpoint.

## B200 Full Sotaku Run

Fresh run:

```text
job: job_01KWEQSQ66XJ1CBN1961Y9T2VA
prefix: viridian-runner-probes/runs/sotaku-full-b200-attempt-slots-20260701-114217/
baseline: sha256:c05c48ca32918166f4e438e4fe1b25ee18c887c906584d7ded9a1ebba002fea3
```

Slot 0 trained from scratch through all curriculum phases and reached step 46,753, then stopped updating `status.json` and `report.jsonl`. The newest verified checkpoint from that slot was:

```text
attempts/slot-000/checkpoints/checkpoint-00023.pt
step: 46,297
sha256: 0eaa0a200e43b405d29ee52aa8c4e0c140d0975422a847e52a821bcaba781931
```

The fresh job kept spawning duplicate evals; slots 1 through 4 were observed before the job was cancelled through the API. Attempt slots prevented checkpoint/report overwrites, but they do not stop wasted compute by themselves.

Guarded resume:

```text
job: job_01KWEZ8FHSWP858CBAP5GNTGC7
prefix: viridian-runner-probes/runs/sotaku-full-b200-resume-guarded-20260701-135253/
baseline: sha256:745d7c1a842598753fcd6a83d0cedfeb45674d8fdcb75c625e30766fa98e32f6
mode: resume
resume checkpoint: fresh job slot 0 checkpoint 00023
```

The resume job downloaded the checkpoint and sidecar, verified the SHA-256, loaded the checkpoint before `torch.compile()`, restored the optimizer, and resumed from step 46,297. It converged with score `50000.0`.

Final checkpoint:

```text
attempts/slot-000/checkpoints/checkpoint-00002.pt
step: 50,000
sha256: 40d85a10fe357cda62051430ff2625a976df39eceef3cbd2a16f3dad4a5134f4
loss: 0.5634099841117859
train_acc: 0.8650548411663267
```

Local verification passed:

```text
sha_match: yes
torch.load step: 50000
config experiment: exp_baseline_lr2e3
train_size: 2700000
batch_size: 2048
total_steps: 50000
checkpoint keys: config, model_state_dict, optimizer_state_dict, step
```

The guarded resume job also proved the duplicate skip behavior remotely. A second eval attempt claimed slot 1, saw slot 0's primary status as `finished` at step 50,000, wrote `duplicate_attempt_skipped`, and exited with metric 0. The private presigned manifest objects for both full-run prefixes were deleted after verification.

## Final Checkpoint Eval

The final Viridian checkpoint was evaluated on the canonical `sapientinc/sudoku-extreme` test path with deterministic bucket sampling: 5,000 puzzles per rating bucket, 25,000 puzzles total.

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

The Viridian eval harness now lives in `viridian/sotaku_serious_probe/repo/eval_sudoku_extreme.py`, with presigned-R2 job generation in `viridian/sotaku_serious_probe/presign_eval_r2.py`. It downloads the final checkpoint and SHA sidecar, verifies the SHA-256 before loading, runs the same bucketed test selection as `iters/eval_more_iters.py`, and writes compact `status.json`, `report.jsonl`, and `latest.json` objects under per-attempt R2 slots. Private presigned manifests and local job-spec temp dirs were deleted after the eval jobs finished.
