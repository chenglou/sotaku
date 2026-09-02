# Viridian Training Infrastructure Test

This was the cheap infrastructure experiment for deciding whether Viridian can be used as a training runner.

The goal is not to train `sotaku` yet. The goal is to test the cluster properties we would need for pretraining:

- Can a GPU job run for the requested wall time?
- Is CUDA visible from Python?
- Is PyTorch installed or installable in the runtime?
- Can the job write checkpoint and log files?
- Can the job upload those files to an external store during the run?
- Can a second job download a previous checkpoint and resume from it?
- Does `vd artifact` include runtime-created files, or only the optimized repo snapshot?

## Job Configuration

Use an eval-only runner job:

```json
"agents": 0,
"limits": { "max_gens": 1 }
```

This avoids the optimizer rollout path and directly runs the training command we provide. As of the latest runs, CPU and L4 runner jobs work, `eval.py` works directly, L4/H100 PyTorch sees CUDA, and a tiny CUDA matmul succeeds.

## Test Setup

Use zero agents and one generation. Lock the probe script so Viridian cannot improve the metric by editing the probe if this is later run with optimizer agents enabled.

Start with `l4` for the infrastructure test. A 10 minute GPU window costs about `$0.13` at the public tier price before token and overhead costs:

```text
600 seconds * $0.00022/sec = $0.132
```

If that passes, repeat once on `h100` to test the intended training hardware:

```text
600 seconds * $0.00110/sec = $0.660
```

## External Storage

Use short-lived presigned R2/S3 URLs for:

- checkpoint upload, e.g. `probe/checkpoint.pt`
- checkpoint SHA-256 upload, e.g. `probe/checkpoint.pt.sha256`
- report upload, e.g. `probe/report.jsonl`

For the resume test, create presigned GET URLs for the checkpoint and SHA-256 sidecar from the first run and pass them to the second run.

The probe uses plain HTTP PUT/GET through Python stdlib, so it does not require `awscli`, `boto3`, credentials, or provider-specific SDKs.

## Passing Result

This probe is good enough to justify a tiny Sotaku training run because all of these passed:

- A GPU job reaches the requested `--seconds` value and prints a final `METRIC: ...`.
- The log says `torch_cuda_available: true`.
- The log includes an NVIDIA GPU name.
- The R2 checkpoint upload URL receives a non-empty file during the run.
- The R2 report upload URL receives a readable JSONL file during the run.
- A second run downloads the first checkpoint, verifies the SHA-256, loads it with `torch.load`, resumes from a positive step, and uploads a new checkpoint/report.
- `vd artifact` does not contain runtime-created files, so R2 is mandatory for checkpoints and logs.

## What This Does Not Prove

This does not prove full `sotaku` training is stable on Viridian. It only proves that the runtime can support the mechanics of pretraining. If this passes, the next cheap experiment should train on a tiny fixed Sudoku subset for 10-20 minutes and upload resumable checkpoints the same way.
