# Viridian Docs Issues Assessment

Compared `docs_issues.md` against the newly downloaded docs in `viridian/docs.md` and the raw files in `viridian/raw/`.

## Summary

The docs issues are mostly fixed. The important runner path is now documented: `agents: 0`, `eval_cmd` only, `max_gens: 1`, no runtime outputs in artifacts, dataset mount paths, GPU network access, billing basics, and the API-only console gap.

I also ran a tiny live API check with `viridian/min_repro/job.cpu-minimal-current-docs.json`. That job omitted `train_cmd`, `proxy_frac`, `squash_every`, `guards`, `env_locked`, and `locked_paths`. Viridian accepted it and completed:

```text
job_01KWE6W85F72NH7SCB9F7J8W2Q
gen 1
score 1.0
status maxed
spend $0.0000
```

The stored spec filled defaults back in server-side: `env_locked: {}`, `guards: { cmd: null, params_max: null }`, `locked_paths: []`, `proxy_frac: 0.0`, and `squash_every: 0`.

## Issue-by-Issue Status

| Issue from `docs_issues.md` | Status | Notes |
| --- | --- | --- |
| Eval-only runner mode is undocumented | Fixed | New docs have an explicit "Optimizer & runner modes" section. Runner mode is described as `agents: 0`, API/CLI-only, and suitable for training or benchmark scripts. |
| Minimal job JSON is underspecified | Fixed and live-verified | New docs include a minimal valid spec and say the old fields are optional or ignored. The live CPU runner job above passed without the old compat fields. |
| `eval_cmd` / `train_cmd` semantics are unclear | Fixed and live-verified | New docs say there is exactly one command field, `eval_cmd`, and `train_cmd` should not be sent. The live job passed with only `eval_cmd`. |
| Artifact behavior does not include runtime outputs | Fixed | New docs explicitly say checkpoints, logs, plots, `probe_output/`, and runtime-created files are ephemeral and not included in `vd artifact`. |
| Dataset mount path is not documented | Fixed | New docs specify `data/train.txt`, `data/val.txt`, `VD_DATA_TRAIN`, `VD_DATA_VAL`, CPU-only `VD_DATA_FILE`, contiguous split behavior, and dataset visibility in runner mode. |
| GPU timeout and billing semantics need clarification | Mostly fixed | New docs explain GPU-seconds, what counts, what does not count, retry/error billing, GPU rates, `budget` units, and `budget: 0`. They do not spell out a maximum allowed `gpu_timeout_s`, if any. |
| Console and API capabilities differ | Fixed | New docs say the console New Job form assumes `agents >= 1` and runner jobs must be submitted through CLI or API. |
| External upload works, artifact semantics remain unclear | Fixed | New docs recommend upload from inside `eval_cmd` over GPU-tier network and say there is no built-in artifact-upload hook. They also say CPU has no network. |
| Stuck job state is hard to interpret | Partly fixed | New docs explain `running` at gen 0 with `score: null` and `spend: 0`, the delayed `last_error`, local repro, and `GET /v1/jobs/{id}/logs`. The status explanation is useful, but live probes did not confirm useful `/logs` output. |

## Small Remaining Caveats

- The docs say `GET /v1/jobs/{id}/logs` returns captured eval transcripts. I tested four jobs:
  - CPU success with stdout, stderr, and `METRIC`: `job_01KWE7MEGW8232JHTYWG9JJCXJ`
  - L4 success with stdout, stderr, and `METRIC`: `job_01KWE7ND4CVGY6T5X4NYEYMPBB`
  - CPU failure with stdout, stderr, and exit 7: `job_01KWE7GMTJK4ACE0FK13B5BK45`
  - L4 failure with stdout, stderr, and exit 7: `job_01KWE7QAT05WYQD2YCJE1T6QQX`
  All four returned `entries: []` from `/v1/jobs/{id}/logs`. The failing jobs stayed `running` at gen 0 with `score: null`, `last_error: null`, and empty spend during the early retry window, then I cancelled them. So the endpoint exists, but I could not verify the docs' claim that it exposes useful eval stdout/stderr.
- The docs cover `gpu_timeout_s` as a per-eval timeout but do not state whether there is a hard maximum or recommended upper bound. For our use, the practical answer still appears to be: set a long enough timeout and cancel jobs ourselves.

## Refetch 2026-07-04

Re-downloaded the docs (console bundle `index-B8ozy92R.js`, previously `index-BxlIwdQY.js`) and diffed the embedded docs text. Changes since 2026-07-01: a new **Installed packages** section (GPU tiers: Python 3.11 + torch CUDA 12.8 + numpy + scipy, network available; CPU tier: torch CPU + numpy, no network), a new **`ultra`** job option (stronger frontier model for optimizer-mode agents; no effect on runner jobs), and a reworded Workbench section. Nothing touching the open items in docs_issues.md — the 2-hour clamp language, `/logs` behavior, and status semantics are unchanged.
