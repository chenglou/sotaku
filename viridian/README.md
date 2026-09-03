# Viridian Notes

This folder keeps the downloaded Viridian docs snapshot and records of the infrastructure tests for using Viridian to run Sotaku training.

The general, project-independent version of the runner machinery lives in the private template repo [tips-for-running-viridian](https://github.com/chenglou/tips-for-running-viridian), extracted from `viridian/train/`. When you learn a new general Viridian lesson, add it there; keep this folder to what is Sotaku-specific.

## Current End-to-End Path

Use Viridian runner jobs, not optimizer jobs:

```json
{
  "agents": 0,
  "limits": { "max_gens": 1 },
  "contract": { "eval_cmd": "python3 eval.py" }
}
```

Use R2 for durable outputs:

```text
r2:sotaku-viridian/viridian-runner-probes/
```

The tested workflow is:

1. package a tiny repo into a Viridian baseline;
2. run a GPU job that reads repo-packaged data, imports PyTorch, sees CUDA, writes a checkpoint and JSONL report, and uploads them to R2 with SHA-256 sidecars;
3. run a second GPU job that downloads the checkpoint and sidecar, verifies the SHA-256, loads the checkpoint with `torch.load`, resumes training, and uploads its own checkpoint and report.

That workflow passed with both a small generic PyTorch training test and a short run of the actual Sotaku model on B200.

The first longer B200 Sotaku pilot also proved the real Hugging Face path, but it exposed an artifact-layout problem: fixed R2 keys like `latest_checkpoint.pt`, `report.jsonl`, and `status.json` can be written by more than one eval attempt under one Viridian job. The highest preserved pilot checkpoint is archived at:

```text
r2:sotaku-viridian/viridian-runner-probes/runs/sotaku-serious-b200-20260701-101911/archive/pre-conflict-step-20198/
```

The duplicate-evaluator platform bug behind that conflict is fixed (confirmed by the Viridian author), so the attempt-slot defense from `sotaku_serious_probe/` is retired. For training runs, use `viridian/train/submit.py`: one output prefix per job, step-specific checkpoint files with SHA-256 sidecars, and automatic resume from the newest verified checkpoint on restart. Treat any `latest` key as a convenience pointer, not the source of truth.

## Setup

The `vd` CLI is not checked in (it is a ~2.5MB binary). Install it into the path `viridian/train/submit.py` expects:

```sh
VD_BIN="$(pwd)/viridian/bin" sh viridian/raw/install.sh
./viridian/bin/vd auth
```

## Files

- `docs.md` and `raw/` preserve the downloaded Viridian docs snapshot.
- `docs_issues_assessment.md` records which issues were resolved in that snapshot. The issues-and-features doc moved to the tips-for-running-viridian repo (issues_and_features.md).
- `min_repro/` keeps the smallest current-doc runner job.
- `gpu_repro/` keeps small GPU test job specs. The completed H100 test is listed in `pretraining_probe/RUNS.md`; L4 is covered by the R2 checkpoint/resume jobs.
- `logs_probe/` keeps the jobs used to verify that `/v1/jobs/{id}/logs` still returned empty entries.
- `shared_cache_probe/` keeps the shared Viridian dataset reuse probe.
- `pretraining_probe/` keeps the high-level run log and H100 local checkpoint/log probe.
- `resume_object_probe/` keeps the current R2 checkpoint/resume probe and presigned URL generator.
- `sotaku_probe/` keeps the B200 real-Sotaku tiny training/resume probe.
- `sotaku_serious_probe/` keeps the real-data B200 pilot wrapper and notes.
