# Viridian Docs Snapshot

## Getting started

Viridian runs your code on our compute, scores it with your own eval, and — if you let it — autonomously optimizes it against that score. You bring a repo and a command that prints a number; we give you back results, and (in optimizer mode) the best version we found.

**There are two ways to use it**, chosen by one field, `agents`:

- **Optimizer** (`agents` ≥ 1) — coding agents explore edits to your code each generation and keep only the ones that *measurably* improve your metric. This is what the console is built around.
- **Runner** (`agents: 0`) — no agents. Viridian just runs your `eval_cmd` and records the score, so you can use our GPUs as a plain training/eval runner. It's an API/CLI feature — the console's New-Job form assumes ≥ 1.

**The four steps** (same for both modes):

**1 · Package your repo.** Put your project under a top-level `repo/` directory and `tar czf` it. Upload it as a **baseline** (`vd baseline repo.tgz`, or `POST /v1/baselines`) → you get a content-addressed `sha256:…` digest naming that exact snapshot.

**2 · Write your eval.** One command that runs in `/repo` and prints one line:

```
METRIC: 1234.5
```

That number is the only thing Viridian reads. Name it, and say whether lower or higher is better.

**3 · Launch a job.** Point a job spec at your baseline and eval, pick a compute tier and a budget, optionally a dataset. Submit with `vd submit job.json` or `POST /v1/jobs`.

**4 · Get results.** Watch the score climb, then `vd artifact <id>` to download the optimized repo. (Training logs and checkpoints your eval writes are **not** in that artifact — see **Results & artifacts**.)

## Optimizer & runner modes

One field — `agents` — decides how Viridian treats your job.

## Optimizer mode — `agents` ≥ 1

Each **generation**, Viridian spawns `agents` coding-agent attempts in parallel. Each edits your code toward a goal; Viridian scores the result with your `eval_cmd`. Attempts that beat the current best by at least `margin` are **promoted**; the rest are discarded. It repeats until it plateaus (`converged`), hits `max_gens` (`maxed`), or spends its `budget` (`budget_out`). The console's charts, lineage, and journal all describe this loop.

## Runner mode — `agents: 0`

With `agents: 0` there are no coding agents. Viridian runs your `eval_cmd` once against the baseline and records the score — a clean way to run a training or benchmark script on our compute and read back a number. Your command can do anything (train a model, run a kernel benchmark) as long as it prints `METRIC: <number>`.

Two things to know:

- **Pin it to one shot with `limits.max_gens: 1`.** Generation 0 always runs the baseline eval with zero agents — but if a job continues to gen 1+ without ever promoting, plateau-escalation starts adding a few agents on its own. `max_gens: 1` stops cleanly right after the single baseline eval (terminal status `maxed`).
- **The console can't create these** (its form requires `agents` ≥ 1). Submit runner jobs with the CLI (`vd submit`) or `POST /v1/jobs`.

A runner spec is just a normal spec with `"agents": 0` and `"max_gens": 1` — see **Job spec reference**.

## The eval contract

Your eval **is** the contract. Viridian runs your `eval_cmd` with working directory `/repo` and reads the `METRIC: <number>` line it prints. There is exactly one command field — **`eval_cmd`**. (An older `train_cmd` was removed; do not send it. `eval_cmd` covers training too — your command does whatever it needs, then prints the metric.)

**What you define:**

- **`eval_cmd`** — the command to run (e.g. `python3 eval.py`). It must exit 0 and print `METRIC: <number>`.
- **`metric`** — a display name for the number.
- **`direction`** — `min` (loss, latency) or `max` (throughput, accuracy).
- **`locked_paths`** — files the optimizer may **not** edit; any attempt whose diff touches one is rejected before it is scored. This is how you protect your eval. (Irrelevant in runner mode — nothing edits your code.)

**Printing the metric — the one rule that matters.** Print **exactly one** line of the form `METRIC: <value>`, with a space after the colon:

```
METRIC: 0.8421
```

Scientific notation is fine (`METRIC: 1.2e-3`). Emit it once, last, on stdout. If your command prints no `METRIC:` line (or writes `METRIC:0.8` with no space), the eval fails — see **Lifecycle → debugging a stuck job**.

## Lock your scoring logic (optimizer mode)

Viridian optimizes *whatever your metric rewards*. If the metric can be moved without genuinely improving your code, a capable optimizer will find that shortcut. Defending against it reduces to one rule: **the code being optimized must not be able to influence how it is scored.**

- Put your eval harness, reference / ground-truth, and task spec in `locked_paths`.
- Generate eval inputs the optimized code can't precompute.
- Never let the edited code compute its own ground truth or its own timing.

Viridian adds black-box backstops — implausible scores are rejected, claimed speedups are corroborated against real wall-clock time, and you can declare the plausible range with `metric_max` / `metric_min`. These are a safety net. **A sealed eval is the real guarantee, and only you can write it.**

## The runtime environment

This is exactly what your `eval_cmd` sees when it runs.

## Working directory & files

Your command runs from the repo root — `/repo` on the CPU tier, the equivalent flattened repo directory on GPU tiers. Present at that root:

- **Your full repo** — every file from the baseline you uploaded: your code, your eval harness, `viridian.toml`, task files. In optimizer mode, promoted agent edits are already merged in (later wins).
- **`data/`** — the dataset splits, *if* the job references a dataset: `data/train.txt` and `data/val.txt` (see **Datasets**).

## Environment variables

Set for every eval:

- **`VD_SEED`** — an integer reproducibility seed (`0` for the baseline/full eval; non-zero for multi-seed runs). Seed your RNGs from it.
- **`VD_STEPS`** — a step-budget hint (defaults to `500` outside short proxy evals). Honor it if your training loop takes a step count; otherwise ignore it.

Set only when the job references a dataset:

- **`VD_DATA_TRAIN`** — the training split (`/data/train.txt` on CPU; `<repo>/data/train.txt` on GPU).
- **`VD_DATA_VAL`** — the validation split.
- **`VD_DATA_FILE`** (CPU tier only) — a single training-file path; `/opt/input.txt` when no dataset is attached.

The CPU tier also pins `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = `OPENBLAS_NUM_THREADS` = `1`.

## Installed packages

The GPU tiers run Python 3.11 with **`torch` (CUDA 12.8), `numpy`, and `scipy`** preinstalled — nothing else. Any other dependency must be vendored into your repo or installed by your `eval_cmd` (e.g. `pip install einops && python eval.py` — GPU tiers have network). An import of a missing package fails the eval with its traceback in the job's logs. The CPU tier ships `torch` (CPU) + `numpy`, and has no network to install more.

## Network access

- **GPU tiers** (`l4`, `a100`, `rtx6000`, `h100`, `b200`) — your eval **has outbound network**. It can `curl` / `PUT` to your own storage; this is how you export checkpoints and logs.
- **CPU tier** — the eval runs with **no network**. It can only read its inputs and print the metric.

**Downloading inputs?** The worker runs from a datacenter IP with a default `Python-urllib` / `curl` User-Agent, and some file hosts (e.g. Filebin) serve a small bot-protection HTML page to those instead of the file. Set a normal browser User-Agent, follow redirects, and **verify the download** — check the byte size / content-type / magic bytes before using it — so a wrong body fails loudly instead of silently. Presigned S3/GCS URLs are the reliable choice.

## What is not persisted

Anything your eval writes to disk (checkpoints, `probe_output/`, plots, logs) lives in an **ephemeral copy that is deleted after the eval**. It is *not* included in `vd artifact`. To keep training outputs, upload them from inside your eval over the network — see **Results & artifacts**.

## Datasets

If your eval needs held-out data, register a **dataset**; Viridian splits it and mounts train + val for the eval. Your code never sees the test split.

## Create, upload, commit

```
vd dataset create data.txt          # one shot: create + upload + commit
```

or over the API:

**1.** `POST /v1/datasets` → `{ id }`.

**2.** `PUT /v1/datasets/{id}/data` — raw bytes, any format, up to 1 GiB.

**3.** `POST /v1/datasets/{id}/commit` — makes it immutable. **A dataset must be committed before a job can use it.**

## Reference it from a job

```json
"data": { "split": { "dataset": "ds_…", "val": 0.05, "test": 0.05, "seed": 7 } }
```

- `val` and `test` are **fractions** of the file's lines; `seed` seeds the split.
- The split is **contiguous** to avoid leakage: train = the head, val = the next block, **test = the tail — and the test split is never placed in the sandbox.** Your eval sees only `data/train.txt` and `data/val.txt`.
- `split` derives train/val from one dataset; `explicit` takes three of your own dataset ids verbatim: `{"explicit": {"train": "ds_…", "val": "ds_…", "test": "ds_…"}}` — they land as `data/train.txt`, `data/val.txt`, `data/test.txt`. All three must be committed datasets you own.

If your repo is self-contained (your eval brings its own data), **omit `data` entirely.**

## In runner mode

An `agents: 0` job **does** get the dataset — the baseline eval mounts `data/train.txt` / `data/val.txt` and sets `VD_DATA_TRAIN` / `VD_DATA_VAL` exactly like any other eval. So you can use Viridian as a dataset-fed training runner.

## Job spec reference

The body of `POST /v1/jobs` (and the file `vd submit` sends). Every field:

## Top level

- **`baseline`** (required) — the starting code. Almost always a **stack** naming your uploaded digest:
```json
"baseline": { "stack": { "stack": [
  { "digest": "sha256:…", "parent": null, "job": "seed", "gen": 0, "bytes": 0 }
] } }
```
  (Also accepts `{"git":{"git_url","rev"}}`, and `{"job":{"job":"job_…"}}` to fork another job.)
- **`contract`** (required) — the eval contract (below).
- **`agents`** (required, integer) — rollouts per generation. `0` = runner mode.
- **`gpu_tier`** (required) — one of `cpu`, `l4`, `a100`, `rtx6000`, `h100`, `b200`. (`rtx6000` = Blackwell workstation silicon on our GCP plane — budget Blackwell; its evals cap at 50 min.)
- **`limits`** (required) — `{ turn_timeout_s, gpu_timeout_s, max_gens, budget }` (below).
- **`margin`** (required, number) — smallest metric improvement worth promoting. `0` = promote on any improvement.
- **`data`** (optional) — a dataset reference (see **Datasets**); omit for self-contained repos.
- **`goal`** (optional) — the objective handed to the coding agent (optimizer mode). Omit for runner jobs.
- **`patience`** (optional, default `5`) — consecutive generations with no promotion before `converged`. `0` = never converge on a plateau.
- **`name`**, **`campaign`** (optional) — labels; metadata only, no effect on execution.
- **`compute`** (optional) — the name of a BYO compute connection registered in your workspace (`POST /v1/compute`); the connection's own `type` decides the backend (`slurm-ssh` today, more types later). Omit to use Viridian's compute.
- **`ultra`** (optional, default `false`) — ultra mode: drives the coding agents with our strongest frontier model. More capable exploration per generation, billed at that model's (higher) per-token rate. No effect on runner jobs (`agents: 0`).
- **`customer`** — ignored (taken from your key); send any placeholder such as `"self"`.

## `contract` (the eval contract)

- **`eval_cmd`** (required, non-empty) — command run in `/repo`; must print `METRIC:`.
- **`metric`** (required) — the metric's display name.
- **`direction`** (required) — `min` or `max`.
- **`locked_paths`** (optional, default `[]`) — paths the optimizer may not edit.
- **`metric_max`** / **`metric_min`** (optional) — declared plausible bounds on the metric (reward-hack backstop).
- **`guards`** (optional) — `{ params_max, cmd }` extra acceptance checks.
- **`env_locked`** (optional) — env vars injected into the eval only.

There is no `train_cmd`. `proxy_frac` and `squash_every` are accepted for back-compat but ignored — omit them.

## `limits`

- **`turn_timeout_s`** — per-agent-turn timeout, seconds (floored to a backend minimum).
- **`gpu_timeout_s`** — the per-eval **hard cap**, in seconds: the eval is killed if it runs longer. Honored up to a platform max of **2 hours (7200s)** on the GPU plane (larger values clamp there). For training that needs longer, checkpoint to your own storage and resume across jobs, or use a BYO compute connection.
- **`max_gens`** — generation cap; hitting it → `maxed`. Use `1` for a one-shot runner job.
- **`budget`** — spend cap in **micro-dollars** (`1000000` = one dollar). `0` = **uncapped** (runs until cancelled or converged).

## Minimal valid spec

```json
{
  "customer": "self",
  "baseline": { "stack": { "stack": [
    { "digest": "sha256:<from vd baseline>", "parent": null, "job": "seed", "gen": 0, "bytes": 0 }
  ] } },
  "contract": { "eval_cmd": "python3 eval.py", "metric": "val_loss", "direction": "min" },
  "agents": 4,
  "gpu_tier": "b200",
  "limits": { "turn_timeout_s": 400, "gpu_timeout_s": 600, "max_gens": 20, "budget": 50000000 },
  "margin": 0.0001
}
```

**Runner variant:** set `"agents": 0` and `"max_gens": 1`. Everything not shown above (`data`, `goal`, `name`, `campaign`, `patience`, `compute`, `ultra`, and the contract's `locked_paths` / `metric_max` / `metric_min` / `guards`) is optional.

## Enum values

- **`gpu_tier`**: `cpu` · `l4` · `a100` · `rtx6000` · `h100` · `b200`
- **`direction`**: `min` · `max`
- **job `status`**: `pending` · `running` · `converged` · `maxed` · `budget_out` · `failed` · `cancelled`

## Lifecycle, statuses & debugging

A job runs in **generations**. Generation 0 always runs your `eval_cmd` once to establish the baseline score. In optimizer mode, each later generation fans out `agents` attempts, scores them, and promotes the best if it clears `margin`. The job stops when it converges, maxes out, spends its budget, is cancelled, or fails.

## Statuses

Non-terminal:

- **`pending`** — created, not yet placed on compute.
- **`running`** — actively evaluating / optimizing.

Terminal:

- **`converged`** — plateaued: no promotion for `patience` consecutive generations. The "done improving" outcome.
- **`maxed`** — hit `max_gens` (possibly still improving). A one-shot runner job (`max_gens: 1`) ends here.
- **`budget_out`** — spent its `budget`.
- **`cancelled`** — you stopped it.
- **`failed`** — an infrastructure or eval failure that persisted past the timeout window.

## Reading a run

- **Score** — your metric's best value so far; **×baseline** is how many times better than generation 0.
- **Lineage** — every attempt per generation; green = became the new best. A lineage that never goes green is a flat run.
- **Journal** — the optimizer's own notes: what it changed, the hypothesis, the measured verdict.
- **Rejected edits** — attempts thrown out for touching a `locked_path`.
- **Spend** — budget used (GPU-seconds + tokens). See **Billing**.

## Debugging a "stuck" job

A job `running` at **generation 0 with `score: null` and `spend: 0`** is stuck on the **baseline eval** — Viridian ran your `eval_cmd` but couldn't get a metric out of it. The usual causes:

- Your command **printed no `METRIC:` line** (or used `METRIC:1.0` with no space — use `METRIC: 1.0`).
- Your command **exited non-zero** — a crash, a missing dependency, an unreadable checkpoint.
- `eval_cmd` is **empty**.

What to check:

- **`GET /v1/jobs/{id}/logs`** → your eval's own stdout/stderr, per generation — where a stack trace, a missing `METRIC:` line, a failed download, or a non-zero exit shows up. It's populated **as soon as the eval runs**, during the retry window (before the job flips to `failed`), and works for runner (`agents: 0`) jobs too. This is your primary debugging tool. (Scratch from generations that *succeeded* may be pruned once their result is committed — the logs that matter for debugging, the failing ones, persist.)
- `GET /v1/jobs/{id}` → `current.last_error` carries a one-line reason, but only **once the job flips to `failed`**; while it's still retrying at gen 0 it sits at `running`, `score: null`, `spend: 0` with `last_error` empty — so use `/logs` above to see what's wrong during that window. A stale, unchanging gen-0 job is the live "stuck" signal.
- Fastest check of all: run your `eval_cmd` yourself inside `repo/` and confirm it prints exactly one `METRIC: <number>` line and exits 0.

## Billing & budgets

Everything is priced in **micro-dollars** (µ$): `1000000` µ$ = one dollar.

## What you pay for

- **GPU-seconds** — wall-clock time your `eval_cmd` runs on a GPU, per tier. This is the dominant cost for GPU jobs. It counts the eval subprocess's runtime (your `import torch`, model load, and compute) plus repo extraction. It does **not** count container cold-start, image pull, or teardown — those are on us. A transport-level retry (if a plane is unreachable) and evals that error after consuming GPU time both still count.
- **Tokens** — for the coding agents (optimizer mode). Viridian's self-hosted policy model bills at **GPU-time cost recovery**: ≈ $2 / Mtok in, ≈ $40 / Mtok out (the serving pool's hourly cost over its throughput). A fraction of rollouts use a Claude teacher model billed at standard Anthropic rates. Runner jobs (`agents: 0`) spend **no** tokens.
- **CPU tier** — cheap, not free: eval seconds bill at the `cpu` rate below (a sandbox slot on our fleet).

## Approximate compute rates

- `cpu` — ≈ $0.11 / hr
- `l4` — ≈ $0.80 / hr
- `a100` — ≈ $2.50 / hr
- `rtx6000` — ≈ $3.42 / hr
- `h100` — ≈ $3.95 / hr
- `b200` — ≈ $6.25 / hr

GPU-seconds are truncated to whole seconds (CPU eval seconds round up). The live per-tier menu is `GET /v1/tiers`.

## Budgets

`limits.budget` caps total spend (tokens + GPU) in µ$. When the ledger crosses it, the job ends as `budget_out`. **`budget: 0` means uncapped** — the job runs until it converges, maxes out, or is cancelled. Watch spend with `vd spend <id>` or `GET /v1/jobs/{id}/spend`.

## Credits

`balance = credits − spend`. Add credits under **Billing** (Stripe; $5 minimum top-up) or via `POST /v1/billing/checkout`; check with `vd balance`. Optional auto-recharge tops you up when the balance falls below a threshold. Billing enforcement can be off on a given deployment — `GET /v1/billing` reports `configured`.

## Results & artifacts

## What `vd artifact <id>` gives you

`GET /v1/jobs/{id}/artifact` (or `vd artifact <id> -o out.tgz`) returns a **gzipped tar of your repo** — the baseline with every promoted edit merged in (later wins), same `repo/` layout you uploaded, now holding the optimizer's best version of your code. In runner mode there are no promoted edits, so the artifact is your baseline repo unchanged.

## What it does NOT contain

**Files your `eval_cmd` writes at runtime are not in the artifact.** Checkpoints, `probe_output/report.jsonl`, `checkpoint.pt`, logs, plots — anything created *during* the eval — are written to an ephemeral working copy that is deleted after each eval. The artifact is assembled from your repo's committed layers, not the eval's scratch space. So a training run's weights and logs will **not** come back through `vd artifact`.

## Getting training outputs out: upload from inside the eval

Because eval scratch is discarded, export what you want to keep **from within `eval_cmd`, over the network**, before you print `METRIC:`:

```
# inside your eval, on a GPU tier (which has network access):
curl -fsS -T checkpoint.pt "https://<your-presigned-url>"
# or, in Python:  requests.put(url, data=open("report.jsonl","rb"))
```

- GPU tiers (`l4` / `a100` / `rtx6000` / `h100` / `b200`) have outbound network — use S3/GCS presigned URLs or your own endpoint.
- The **CPU tier has no network**, so it cannot upload; use a GPU tier for jobs that must export checkpoints.
- There is no built-in artifact-upload hook — this pattern is the supported way to capture training outputs.

## Command-line (vd)

The `vd` CLI does everything the API does, against the same key. Install it with one line:

```
curl -fsSL https://console.viridianresear.ch/install.sh | sh
```

Then log in — this opens your browser, mints a key, and saves it to `~/.config/viridian/config`:

```
vd auth
```

(Or skip the browser: `vd login --key vd_… --api <url>`.)

## A full run

```
tar czf repo.tgz repo/                 # package your project
vd baseline repo.tgz                   # → prints a sha256:… digest
vd submit job.json --watch             # create the job, follow it to the end
vd artifact <job-id> -o result.tgz     # download the optimized repo
```

The `job.json` is the spec from **Job spec reference**, with your baseline digest dropped in.

## Every command

```
vd auth [--api URL]              browser login → mints + saves a key
vd login [--api URL] --key vd_…  save config from an existing key
vd baseline <repo.tgz>           upload a repo snapshot → prints the digest
vd submit <spec.json> [--watch]  create a job → prints its id
vd watch <id>                    follow a job until it reaches a terminal state
vd jobs                          list your jobs (id / status / gen / score / spend)
vd job <id>                      one job, full JSON
vd series <id> [key]             a metric curve (default key: score) as x<tab>y lines
vd spend <id>                    this job's spend breakdown
vd artifact <id> [-o file]       download the result tar.gz (default <id>.tgz)
vd balance                       your credit balance
vd dataset create <file>         create + upload + commit a dataset → prints id
vd dataset get <id>              dataset info (JSON)
vd dataset rm <id>               delete a dataset
```

## Config & environment

- Config file: `~/.config/viridian/config` — `KEY=VALUE` lines (`VD_API`, `VD_KEY`), written mode `0600`.
- **`VD_API`** — the deployment base URL (default `https://console.viridianresear.ch`); overrides the config file.
- **`VD_KEY`** — your API key; overrides the config file.
- The `--api` and `--key` flags apply to `vd auth` and `vd login` only. Other commands read `VD_API` / `VD_KEY` or the saved config.
- There is no `--help` flag — running `vd` with no command (or an unknown one) prints the command list.

Job actions that live only in the console/API (not the CLI): `cancel`, `logs`, `lineage`, `journal`, and API-key management. Use the console or the REST endpoints for those.

## Viridian Workbench

**Viridian Workbench** is a local-first environment for optimizing things: an interactive multi-agent session that runs on **your** machine (macOS/Linux). You bring something you want made better — an architecture, a kernel, a solver, a pipeline — and the coordinator agent turns it into a fast, sealed **proxy eval** (a miniature of your setup that scores in minutes), baselines it, and points Viridian's optimizer at it. It delegates bounded work to specialist agents and passes every contract, claim, and report through an **adversarial reviewer** — including whether the proxy is faithful enough for improvements to transfer — before you're asked to rely on it.

## Install & log in

```
curl -fsSL https://console.viridianresear.ch/install.sh | sh -s -- workbench
vd-workbench auth        # browser login — shared with the vd CLI
vd-workbench             # start a session
```

The Workbench comes in two faces sharing one session format: the `vd-workbench` terminal REPL and the **desktop app** (sessions sidebar, chat with live tool activity, journal and artifact panels, command-approval dialogs). A session started in the terminal opens in the app and vice versa — the directory on disk is the truth.

Everything a session produces lives in a directory on your machine (`~/.viridian/workbench/sessions/<id>/`): the transcript, a typed research **journal** (hypothesis → result → verdict, append-only), provenance-tracked **artifacts** (figures, tables, reports — every file hashed, every claim citing journal entries and job ids), and one folder per **experiment** holding the exact spec submitted plus everything Viridian said back (series, lineage, logs, the engine's own journal, the optimized repo).

## How it uses Viridian

The Workbench is a normal customer of this API — same key, same `/v1` endpoints, same billing. Its agents run on models behind the **inference proxy** (`POST /v1/inference/messages` for `claude-*`, `POST /v1/inference/chat/completions` for the self-hosted `policy-v*` coding policy), metered per token against your credit balance and itemized under Billing. When a hypothesis needs a number you could credibly cite, the session turns it into a job:

- **Runner mode** (`agents: 0`) — Viridian as a trusted scorer: baselines, scoring locally-built candidates, measuring eval noise, scoring forecasts against a dataset committed before the outcome was knowable.
- **Optimizer mode** (`agents` ≥ 1) — "how far can this metric be pushed" questions, with the engine's reward-hacking defenses and k-seed promotion doing the verifying.

## Review gates

Money and claims only move over reviewed material — the host enforces it, not just the prompt:

- **Contract gate** — `vd_submit_job` refuses any spec without a passing reviewer audit of that exact json (metric validity, locked-path seal, data leakage, plausibility bounds, margin vs noise, cost sanity).
- **Claim gate** — a journal VERDICT must cite both a RESULT (the raw numbers) and a REVIEW (the audit of the interpretation).
- **Report gate** — a report artifact requires a passing review of the byte-identical draft. Edit one byte and the gate re-opens.

A failed review can be rebutted once with new evidence; after that the dispute goes to you verbatim, and only your recorded decision overrides it.

## One contract, many domains

The Workbench has **no built-in domains** — a research area is just a way of filling in the same four contract fields (`eval_cmd`, `direction`, `locked_paths`, bounds):

- **ML benchmark** — eval trains and prints the held-out loss; harness locked; data via a platform split; `metric_min: 0`; `guards.params_max` against "just make it bigger".
- **Solver / heuristic** — eval runs the solver on held-out instances under a time cap and prints the objective; `metric_min` = known lower bound; `guards.cmd` runs a feasibility checker from the baseline copy.
- **Forecasting** — commit the ground truth as an immutable dataset *before* the horizon closes; the eval scores a forecast file against the platform-held split. Runner mode as a notary.
- **Formal proofs** — eval runs the proof checker and prints the fraction of targets closed; theorem statements and checker config locked; `metric_max: 1.0`.

Same engine, same seal — a new domain is a new contract, not a new feature.

## MCP server

Drive Viridian from Claude Code, Claude Desktop, or any MCP client. The server wraps this same REST API as tools, so an agent can upload a baseline, launch a job, watch the score, read eval logs, and download the artifact — end to end.

## Setup

Create an API key (**API Keys** in the console), then register the server (it ships in the platform repo under `mcp/`):

```
claude mcp add viridian -e VD_API_KEY=vd_... -- uvx --from <viridian>/mcp viridian-mcp
```

`VD_API_URL` overrides the API base (defaults to this console's origin).

## Tools

- `list_tiers` — the compute tier catalog with rates.
- `list_jobs` / `get_job` / `submit_job` / `cancel_job` / `delete_job` — the job lifecycle. `submit_job` takes a full job spec (see **Job spec reference**).
- `job_logs` / `job_spend` / `job_series` — eval stdout/stderr, billing breakdown, metric curves.
- `upload_baseline` — tars a local directory (or takes a ready `.tgz`) and uploads it → the content-addressed digest job specs reference.
- `list_datasets` / `create_dataset` — dataset management; `create_dataset` does create + upload + commit in one call.
- `download_artifact` — pull a finished job's optimized repo to a local file.

## API reference

Everything in the console is a REST API. Create a key under **API Keys**, then send it on every request:

```
x-api-key: vd_…
# or
Authorization: Bearer vd_…
```

Errors come back as `{ "error": "<message>" }`. Amounts are micro-dollars. "Auth" below means an API key (or, from the console, a session cookie plus an `X-Workspace: <workspace_id>` header).

## Jobs

- **POST `/v1/jobs`** — create a job. Body: a job spec (see **Job spec reference**). → `{ "id": "job_…" }`. `400` if `eval_cmd` is empty; `402` if out of credits (when billing is enforced).
- **GET `/v1/jobs`** — your jobs → array of `{ id, status, gen, score, spend, tier, metric, budget, max_gens, name, campaign }`.
- **GET `/v1/jobs/{id}`** — one job → `{ gen, status, current, spec, spend }`; `current` = `{ stack, score }`.
- **DELETE `/v1/jobs/{id}`** — delete a **terminal** job. `409` if still running (cancel first).
- **POST `/v1/jobs/{id}/cancel`** — stop a running/pending job → `{ id, status: "cancelled" }`. `409` if already finished.
- **GET `/v1/jobs/{id}/series?key=<key>`** — a metric curve (default `score`) → `{ key, points }`.
- **GET `/v1/jobs/{id}/keys`** — the series keys available for this job.
- **GET `/v1/jobs/{id}/spend`** — → `{ total, gpu_seconds, agent_seconds, agent_calls, tok_in, tok_out }`.
- **GET `/v1/jobs/{id}/artifact`** — the result tar.gz (`application/gzip`). See **Results & artifacts**.
- **GET `/v1/jobs/{id}/logs`** — your eval's captured stdout/stderr, newest generations first → `{ job, entries: [{ gen, name, text }] }` (`name` is the log file, e.g. `<vm>.stdout.log`). Works for runner and optimizer jobs.
- **GET `/v1/jobs/{id}/lineage`** — the per-generation attempt tree.
- **GET `/v1/jobs/{id}/journal`** — the optimizer's markdown journal.
- **GET `/v1/jobs/{id}/journal/summary`** — a model-written TL;DR of the journal.

A job id you don't own returns `404`, identical to a nonexistent one.

## Baselines & datasets

- **POST `/v1/baselines`** — body: raw gzipped tar with `repo/` at top level (≤ 256 MiB) → `{ "digest": "sha256:…" }`. Rejects absolute paths, `..`, symlinks, and archives with no top-level `repo/`.
- **POST `/v1/datasets`** — → `{ id }`.
- **PUT `/v1/datasets/{id}/data`** — raw bytes (≤ 1 GiB) → `{ digest, bytes }`. Re-uploadable until committed.
- **POST `/v1/datasets/{id}/commit`** — → `{ status: "committed" }`. Required before use.
- **GET `/v1/datasets`** — → `[{ id, committed, created }]`.
- **GET `/v1/datasets/{id}`** — → `{ digest, bytes, status }`.
- **DELETE `/v1/datasets/{id}`** — → `{ deleted: true }`.

## Keys, tiers, profile

- **GET `/v1/keys`** — → `[{ id, name, created, revoked }]`.
- **POST `/v1/keys`** — body `{ name? }` → `{ key: "vd_…", name }` (the plaintext key is shown once).
- **DELETE `/v1/keys/{id}`** — revoke → `{ revoked: true }`.
- **GET `/v1/me`** — your workspace → `{ id, name, created }`.
- **GET `/v1/tiers`** — the compute menu (public) → `{ tiers: [{ tier, backing, gpu, note, usd_per_sec }] }`.

## Billing

- **GET `/v1/billing`** — → `{ balance, credits, spend, auto_recharge, recharge_threshold, recharge_amount, has_card, configured, events }`.
- **POST `/v1/billing/checkout`** — body `{ amount_micros }` (min $5) → `{ url }` (Stripe).
- **POST `/v1/billing/setup-card`** — → `{ url }`.
- **POST `/v1/billing/auto-recharge`** — body `{ enabled, threshold_micros, amount_micros }`.

## Accounts & workspaces (console session)

- **POST `/auth/signup`**, **POST `/auth/login`** — body `{ email, password, … }` → sets a session cookie, returns `{ user, workspaces }`.
- **POST `/auth/logout`**, **GET `/auth/me`**, **POST `/auth/accept-invite`**.
- **GET/POST `/v1/workspaces`**, **GET `/v1/members`**, **POST `/v1/invites`**.
- These use the session cookie. Every other `/v1/*` endpoint accepts an API key; when you call one with a session cookie instead, send the active workspace as `X-Workspace: <workspace_id>`.

## BYO compute (Slurm over SSH)

- **POST `/v1/compute`** — register a cluster → `{ id, pubkey }` (authorize the returned key on your cluster).
- **GET `/v1/compute`**, **DELETE `/v1/compute/{id}`**, **POST `/v1/compute/{id}/test`** (runs `sinfo`).
- Reference the connection by name via `spec.compute`.

## GitHub app

- **POST `/v1/github/connect`** → `{ url }` to install the app (GitHub bounces back and links the install to this workspace automatically); **GET `/v1/github/installations`** lists the installs linked here. A push to a configured branch triggers a run from the repo's `.viridian.toml`.
