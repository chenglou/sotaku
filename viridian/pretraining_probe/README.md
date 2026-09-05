# Viridian Training Infrastructure Test

This was the cheap infrastructure experiment for deciding whether Viridian can be used as a training runner.

This test checked the infrastructure needed before attempting Sotaku training:

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

This disables Viridian's code-editing agents and directly runs the supplied command. In these tests, CPU and L4 runner jobs worked, `eval.py` worked directly, L4/H100 PyTorch saw CUDA, and a tiny CUDA matrix multiplication succeeded.

## Test Setup

Use zero agents and one generation. Lock the probe script so Viridian cannot improve the metric by editing the probe if this is later run with optimizer agents enabled.

The initial `l4` test budget was about `$0.13` for 10 minutes at the published price at the time, before token and overhead costs:

```text
600 seconds * $0.00022/sec = $0.132
```

The corresponding `h100` budget was:

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

This established that the runtime supported training and checkpoint recovery, not that the trained model would solve Sudoku accurately. The follow-ups are recorded in the [small Sotaku test](../sotaku_probe/README.md) and [full training test](../sotaku_serious_probe/README.md).
