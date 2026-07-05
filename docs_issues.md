# Viridian Platform Issues

Findings from running ~40 training and fine-tuning jobs on the `b200` plane between 2026-07-01 and 2026-07-04. Job ids are included as evidence for each item.

## Open: 2-Hour Cap on GPU-Plane Evals

The docs, the author, and the measured behavior disagree:

- The docs (fetched 2026-07-02) say `gpu_timeout_s` clamps to a 2-hour platform max on the GPU plane.
- The author's stated understanding was that the cap applies only to `rtx6000` (whose documented cap is 50 minutes — a further inconsistency).
- Measured: seven-plus B200 evals killed at 7,200±70s across four days, before and after the duplicate-evaluator fix. Pre-fix: `job_01KWEQSQ66XJ1CBN1961Y9T2VA`, `job_01KWG0GXAEX9KC6KK60XTX3631`. Post-fix, four training jobs launched 2026-07-02 03:38 UTC all died at ≈7,200s of eval time: `job_01KWGEGJH53BEEWRBPAGEMM1N0`, `job_01KWGEGMJMF6BYF9AV88QM173X`, `job_01KWGEGRBBPKAVVC2MNHDZ5ZVX`, `job_01KWGEGWP9SVJ0WMDBM6AMD3FN`. A pure sleep probe on 2026-07-03 (`sh -c 'date; sleep 8100; date; echo METRIC: 1'`, `gpu_timeout_s` 10800, `job_01KWK3ZDTXEV9X15HH0BM8XAQY`) was killed before printing its metric. Latest: `job_01KWNZJYNP9SD92E3KADQBKRXF` killed at 7,200s on 2026-07-04.

The docs' clamp language matches the behavior; the "rtx6000 only" understanding does not. Likely home: a fixed timeout on the B200-plane Modal function. We work around it by chaining jobs from uploaded checkpoints, which is fine — but either lift the cap or make the docs and the author agree that it is a hard platform property.

## Open: `/v1/jobs/{id}/logs` Is Empty for Still-Executing and Committed Evals

Logs are captured per completed attempt. Two cases still return `{ "entries": [] }`:

- An eval that is still executing (tested against a running ~2h B200 training job) — so a training run's stdout cannot be tailed through `/logs`; exporting logs over the network from inside the eval is the only live channel.
- Succeeded evals after their result is committed (tested on several finished jobs).

The docs say pruning applies to generations that *succeeded* and that "the logs that matter for debugging, the failing ones, persist". At least one cap-killed job's logs did not persist. The docs should state precisely when logs appear (attempt completion) and when they are pruned.

## Open: `last_error` Embeds a Live Presigned URL

The `last_error` field of a failed job reproduces the killed eval command verbatim, including the full presigned GET URL for the job's artifact manifest (7-day expiry, `X-Amz-Signature` and all) — and that manifest in turn contains presigned upload URLs (observed on `job_01KWNZJYNP9SD92E3KADQBKRXF`, 2026-07-04). Only the job owner can read the job detail, so exposure is limited, but error messages would ideally truncate or redact signed query strings rather than store live credentials.
