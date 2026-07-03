# Viridian Docs Issues

Remaining issues after rechecking against the platform on 2026-07-02 (post the duplicate-evaluator fix). Resolved items were removed. Recheck jobs: failing-eval probe `job_01KWGHY2GNWQ4SQQMRZ2J4GC7K` (exit 7, CPU), plus `/logs` queries against running, succeeded, and killed jobs from 2026-07-01/02.

Resolved since the 2026-07-01 pass:

- `/logs` now returns per-attempt stdout/stderr for failing evals, live during the retry window (entries appeared within 30s of the first attempt). The old "no early debug signal for failed runner jobs" issue is fixed.
- `gpu_timeout_s` bounds are now documented: hard cap, clamped to a 2-hour platform max on the GPU plane, 50 minutes on `rtx6000`.

## `/v1/jobs/{id}/logs` Is Empty for Long-Running and Committed Evals

Logs are captured per completed attempt. Two cases still return `{ "entries": [] }`:

- A long eval that is still executing (tested: a running ~2h B200 training job) — so you cannot tail a training run's stdout through `/logs`; exporting the log over the network from inside the eval remains the only live channel.
- Succeeded evals after their result is committed (tested on several finished jobs, including one killed at the 2h cap that ended `converged` with score 0 — arguably a failure whose logs should persist per the docs).

The docs say pruning applies to generations that *succeeded* and that "the logs that matter for debugging, the failing ones, persist". The 2h-killed job's logs did not persist. The docs should state precisely when logs appear (attempt completion) and when they are pruned.

## 2-Hour GPU-Plane Cap: Docs, Author, and Behavior Disagree

Three sources conflict:

- The docs (fetched 2026-07-02) say `gpu_timeout_s` clamps to a 2-hour platform max on the GPU plane.
- The platform author says the cap applies only to `rtx6000` (whose documented cap is 50 minutes, which is a further inconsistency).
- Measured behavior before the duplicate-evaluator fix: two B200 training evals were killed at +7,175s and +7,195s after eval start (`job_01KWEQSQ66XJ1CBN1961Y9T2VA`, `job_01KWG0GXAEX9KC6KK60XTX3631`) — consistent with a 2h limit, possibly as a side effect of the abandoned synchronous-call pathology the fix addressed.

Verified 2026-07-02 ~05:50 UTC: the cap is real and applies to `b200`, post-fix. All four B200 training jobs launched 03:38 UTC (after the duplicate-evaluator fix) died at ≈7,200s of eval time (last log lines at steps 45,700-46,400, ≈2h at ~6.3 steps/s; jobs `job_01KWGEGJH53BEEWRBPAGEMM1N0`, `job_01KWGEGMJMF6BYF9AV88QM173X`, `job_01KWGEGRBBPKAVVC2MNHDZ5ZVX`, `job_01KWGEGWP9SVJ0WMDBM6AMD3FN`). That makes six kills at 7,200±70s across two days. The docs' clamp language is accurate; the author's "rtx6000 only" statement is not. Likely home: a fixed timeout on the B200-plane Modal function. Re-tested 2026-07-03 after a reported fix: a pure sleep probe on b200 (`sh -c 'date; sleep 8100; date; echo METRIC: 1'`, gpu_timeout_s 10800, `job_01KWK3ZDTXEV9X15HH0BM8XAQY`) still died before printing its metric — `/logs` shows `gpu eval failed` — and the job again sat at `running` indefinitely instead of going terminal. The cap and the stuck-status behavior are both still present.

## Job Killed at the Cap Reports `converged` with Score 0 — or Never Flips at All

The B200 job killed at +7,195s on 2026-07-02 ended as `converged` with score `0.0` (no `METRIC:` line was ever printed — the eval was killed mid-training). A kill with no metric should surface as `failed` (or a distinct timeout status), not `converged`; `converged` at score 0 reads like a successful run of a bad model.

Worse, the four cap-killed jobs from later that day (ids above) never left `running`: 90+ minutes after their evals died, `GET /v1/jobs` still reported all four as `running` at gen 0. A customer watching job status has no signal that training is gone; the only tell is their own exported telemetry going stale.
