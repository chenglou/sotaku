# Sotaku Viridian Probe

This probe is the first production-shaped Viridian run. It uses the real `iters.exp_baseline_lr2e3` model and encoding helpers, but feeds a tiny repo-packaged Sudoku shard so the run can finish quickly.

The B200 path is:

1. upload this `repo/` as a Viridian baseline;
2. submit a B200 runner job that trains a few steps, saves a Sotaku-shaped checkpoint, writes a JSONL report, and uploads both to `r2:sotaku-viridian`;
3. submit a second B200 runner job that downloads the first checkpoint and SHA-256 sidecar, verifies the hash, loads the model and optimizer state before `torch.compile`, trains a few more steps, and uploads its own checkpoint/report.

This tests the real model code, checkpoint shape, compile order, B200 availability, and the same R2 durability path we plan to use for larger runs. It uses packaged data and stubs `datasets.load_dataset` during import, so it does not test Hugging Face dataset download or a long training schedule.

## Verified Run

Run prefix:

```text
r2:sotaku-viridian/viridian-runner-probes/runs/sotaku-b200-20260701-090303/
```

Jobs:

```text
fresh:  job_01KWEENZVZJGB1VH31Z6007C6V  score 2.0
resume: job_01KWEF5091J35YFCJA69EXBXEZ  score 4.0
```

Both jobs ran on `NVIDIA B200`, used `torch.compile`, uploaded checkpoint/report artifacts to R2, and verified/resumed through a SHA-256 sidecar. Local verification loaded the downloaded checkpoints and confirmed steps 2 and 4.

## Logging Probe

Run prefix:

```text
r2:sotaku-viridian/viridian-runner-probes/runs/sotaku-b200-logging-20260701-092857/
```

Job:

```text
job_01KWEG5M2PB9CV4SDBVDDC7X23
```

This ran a six-minute B200 training window with one compact progress row per minute, periodic `report.jsonl` upload, and an overwritten `status.json`. Final sizes were small: `fresh_report.jsonl` was 4,233 bytes and `fresh_status.json` was 239 bytes. The downloaded report had 16 lines, max line size 1,829 bytes, and no presigned URL or obvious secret patterns. The checkpoint hash matched its sidecar and loaded locally at step 6,926.

The Viridian job score stayed at `1.0` even though the R2 checkpoint reached step 6,926. For future probes, detailed JSON stays in R2 and stdout prints only event names plus the final `METRIC:` line.
