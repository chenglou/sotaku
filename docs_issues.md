# Viridian Docs Issues

Remaining issues after comparing the old notes with the newly downloaded docs on 2026-07-01. Resolved items were removed.

## `/v1/jobs/{id}/logs` Does Not Show Eval Output

The docs say `GET /v1/jobs/{id}/logs` returns captured eval stdout/stderr and should help debug failed evals. I could not reproduce that behavior for runner jobs.

Tested jobs:

- CPU success with stdout, stderr, and `METRIC`: `job_01KWE7MEGW8232JHTYWG9JJCXJ`
- L4 success with stdout, stderr, and `METRIC`: `job_01KWE7ND4CVGY6T5X4NYEYMPBB`
- CPU failure with stdout, stderr, and exit 7: `job_01KWE7GMTJK4ACE0FK13B5BK45`
- L4 failure with stdout, stderr, and exit 7: `job_01KWE7QAT05WYQD2YCJE1T6QQX`

All four returned:

```json
{ "entries": [] }
```

I rechecked this after the R2 resume probe on 2026-07-01. These current jobs also returned zero log entries:

- R2 fresh checkpoint upload: `job_01KWECTGRT2BJ5JWQP76CNG2YR`
- R2 checkpoint resume: `job_01KWECX497D4G7V9EZW25AZ2T9`
- L4 success with stdout/stderr: `job_01KWE7ND4CVGY6T5X4NYEYMPBB`
- B200 Sotaku tiny fresh: `job_01KWEENZVZJGB1VH31Z6007C6V`
- B200 Sotaku tiny resume: `job_01KWEF5091J35YFCJA69EXBXEZ`

So either logs are not wired up for runner evals, or the docs should say when logs are available.

## Failed Runner Jobs Still Have No Early Debug Signal

The failing CPU and L4 jobs stayed in this state during the early retry window:

```text
status: running
gen: 0
score: null
last_error: null
spend: {}
logs: []
```

The docs now explain this state, which is useful. But without `/logs` or an early error message, users still cannot tell whether the command crashed, failed to print `METRIC:`, referenced a missing file, or hit some other eval setup problem.

## `gpu_timeout_s` Bounds Are Still Unclear

The docs say `gpu_timeout_s` is the per-eval timeout. They do not say whether there is a hard maximum or recommended range.

For training-style jobs, users need to know whether setting `gpu_timeout_s` to hours or days is valid, or whether the intended pattern is to set a long timeout and cancel jobs manually.
