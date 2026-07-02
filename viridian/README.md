# Viridian Notes

This folder keeps the current Viridian docs snapshot and the small probes that matter for using Viridian as a Sotaku training runner.

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

The tested shape is:

1. package a tiny repo into a Viridian baseline;
2. run a GPU job that reads repo-packaged data, imports PyTorch, sees CUDA, writes a checkpoint and JSONL report, and uploads them to R2 with SHA-256 sidecars;
3. run a second GPU job that downloads the checkpoint and sidecar, verifies the SHA-256, loads the checkpoint with `torch.load`, resumes training, and uploads its own checkpoint and report.

That path is proven for both a tiny generic PyTorch training probe and a tiny real-Sotaku B200 probe.

The first longer B200 Sotaku pilot also proved the real Hugging Face path, but it exposed an artifact-layout problem: fixed R2 keys like `latest_checkpoint.pt`, `report.jsonl`, and `status.json` can be written by more than one eval attempt under one Viridian job. The highest preserved pilot checkpoint is archived at:

```text
r2:sotaku-viridian/viridian-runner-probes/runs/sotaku-serious-b200-20260701-101911/archive/pre-conflict-step-20198/
```

The duplicate-evaluator platform bug behind that conflict is fixed (confirmed by the Viridian author), so the attempt-slot defense from `sotaku_serious_probe/` is retired. For training runs, use `viridian/train/submit.py`: one output prefix per job, step-specific checkpoint files with SHA-256 sidecars, and automatic resume from the newest verified checkpoint on restart. Treat any `latest` key as a convenience pointer, not the source of truth.

## Files

- `docs.md` and `raw/` are the current downloaded Viridian docs.
- `docs_issues_assessment.md` records which old docs issues were fixed by the current docs.
- `min_repro/` keeps the smallest current-doc runner job.
- `gpu_repro/` keeps small GPU smoke job specs. The live H100 smoke job is listed in `pretraining_probe/RUNS.md`; L4 is covered by the R2 checkpoint/resume jobs.
- `logs_probe/` keeps the jobs used to verify that `/v1/jobs/{id}/logs` still returned empty entries.
- `shared_cache_probe/` keeps the shared Viridian dataset reuse probe.
- `pretraining_probe/` keeps the high-level run log and H100 local checkpoint/log probe.
- `resume_object_probe/` keeps the current R2 checkpoint/resume probe and presigned URL generator.
- `sotaku_probe/` keeps the B200 real-Sotaku tiny training/resume probe.
- `sotaku_serious_probe/` keeps the real-data B200 pilot wrapper and notes.
