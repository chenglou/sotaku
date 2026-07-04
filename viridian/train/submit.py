"""Submit a canonical training run to Viridian. The peer of `modal run modal_run.py`.

Usage (from the repo root, venv active):
    python viridian/train/submit.py                          # train iters.exp_baseline_lr2e3 on B200
    python viridian/train/submit.py --exp iters.exp_bs2048_mixed
    python viridian/train/submit.py --dry-run                # package + presign + write job.json, no submit
    python viridian/train/submit.py --resume-checkpoint-key <r2 key> --resume-checkpoint-sha-key <r2 key>

What it does, in order:
1. Packages the job tarball FROM THE REAL REPO FILES (the experiment module,
   checkpoint_utils.py) plus this directory's wrapper (viridian_train.py, r2_io.py).
   There are no vendored copies checked in anywhere, so nothing can drift.
2. Uploads the tarball as a Viridian baseline (vd baseline) to get its digest.
3. Presigns the job's R2 manifest and writes the job spec.
4. Submits with vd, prints the job id and the R2 prefix to watch.

The expected output filenames (log, per-step checkpoints, final model) are derived by
importing the experiment module, so a renamed experiment (e.g. a rerun variant) gets
correct presigned upload URLs automatically.
"""

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from presign import build_train_manifest, presign_url, read_r2_config, shell_quote
from r2_io import upload_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRA_MODULE_NAME = ""
SEED_MODEL_PATH = ""
WRAPPER_DIR = Path(__file__).resolve().parent
VD_BIN = REPO_ROOT / "viridian" / "bin" / "vd"


def expected_output_files(exp_module):
    """Filenames train() writes into output_dir, derived from the experiment module.

    The final model filename follows the repo convention: checkpoint prefix
    "foo_checkpoint_step" pairs with final model "model_foo.pt" (e.g.
    baseline_lr2e3_checkpoint_step -> model_baseline_lr2e3.pt).
    """
    prefix = exp_module.CHECKPOINT_PREFIX
    suffix = "_checkpoint_step"
    if not prefix.endswith(suffix):
        raise SystemExit(
            f"CHECKPOINT_PREFIX {prefix!r} does not end with {suffix!r}; "
            "cannot derive the final model filename — check the experiment module"
        )
    final_model = f"model_{prefix[: -len(suffix)]}.pt"
    checkpoint_steps = list(range(0, exp_module.total_steps, exp_module.eval_every))
    checkpoint_steps.append(exp_module.total_steps - 1)
    return [
        exp_module.log_name,
        final_model,
        *[f"{prefix}{step}.pt" for step in checkpoint_steps],
    ]


def copy_module_into(staging, module_name, module_file):
    parts = module_name.split(".")[:-1]
    package_dir = staging
    source_dir = module_file.parent
    source_roots = [source_dir]
    for _ in parts[1:]:
        source_roots.insert(0, source_roots[0].parent)
    for part, source_root in zip(parts, source_roots):
        package_dir = package_dir / part
        package_dir.mkdir(exist_ok=True)
        init_file = source_root / "__init__.py"
        if init_file.is_file() and not (package_dir / "__init__.py").exists():
            shutil.copy2(init_file, package_dir / "__init__.py")
    shutil.copy2(module_file, package_dir / module_file.name)


def package_baseline(exp_name, exp_module, out_dir):
    """Assemble repo/{viridian_train.py, r2_io.py, checkpoint_utils.py, <exp package>}
    from the live repo files and return the .tgz path."""
    staging = Path(tempfile.mkdtemp(prefix="viridian-train-")) / "repo"
    staging.mkdir(parents=True)

    shutil.copy2(WRAPPER_DIR / "viridian_train.py", staging / "viridian_train.py")
    shutil.copy2(WRAPPER_DIR / "r2_io.py", staging / "r2_io.py")
    shutil.copy2(REPO_ROOT / "checkpoint_utils.py", staging / "checkpoint_utils.py")

    exp_file = Path(exp_module.__file__).resolve()
    if "." in exp_name:
        copy_module_into(staging, exp_name, exp_file)
    else:
        shutil.copy2(exp_file, staging / exp_file.name)

    if EXTRA_MODULE_NAME:
        extra_module = importlib.import_module(EXTRA_MODULE_NAME)
        copy_module_into(staging, EXTRA_MODULE_NAME, Path(extra_module.__file__).resolve())
    if SEED_MODEL_PATH:
        shutil.copy2(SEED_MODEL_PATH, staging / "seed_model.pt")

    tgz_path = out_dir / "baseline.tgz"
    with tarfile.open(tgz_path, "w:gz") as tar:
        tar.add(staging, arcname="repo")
    return tgz_path


def upload_baseline(tgz_path):
    completed = subprocess.run([str(VD_BIN), "baseline", str(tgz_path)], text=True, capture_output=True, check=False)
    lines = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not lines or not lines[-1].startswith("sha256:"):
        raise SystemExit(
            f"vd baseline failed (exit {completed.returncode}): stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    return lines[-1]


def build_spec(args, digest, exp_name, eval_cmd):
    return {
        "customer": "self",
        "baseline": {
            "stack": {
                "stack": [
                    {
                        "bytes": 0,
                        "digest": digest,
                        "gen": 0,
                        "job": "seed",
                        "parent": None,
                    }
                ]
            }
        },
        "contract": {
            "eval_cmd": eval_cmd,
            "locked_paths": [
                "viridian_train.py",
                "r2_io.py",
                "checkpoint_utils.py",
                exp_name.replace(".", "/") + ".py",
            ],
            "metric": "training_step",
            "direction": "max",
        },
        "gpu_tier": args.gpu_tier,
        "agents": 0,
        "margin": 0.0001,
        "patience": 1,
        "limits": {
            "budget": args.budget,
            "gpu_timeout_s": args.timeout,
            "max_gens": 1,
            "turn_timeout_s": args.timeout,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="iters.exp_baseline_lr2e3")
    parser.add_argument("--seed", type=int, default=None, help="passed through to viridian_train.py --seed")
    parser.add_argument("--seed-model", default="", help="checkpoint file packaged into the job as seed_model.pt (for fine-tuning experiments)")
    parser.add_argument("--extra-module", default="", help="additional module packaged alongside the experiment, e.g. iters.exp_baseline_lr2e3 when the experiment imports it")
    parser.add_argument("--gpu-tier", default="b200")
    parser.add_argument("--bucket", default="sotaku-viridian")
    parser.add_argument("--remote", default="r2")
    parser.add_argument("--prefix", default="", help="R2 prefix; default derives from --exp + UTC timestamp")
    # 7 days, the SigV4 presigning maximum. Deliberately much longer than the GPU
    # timeout: queue wait plus preemption restarts must not outlive the upload URLs.
    parser.add_argument("--expires", type=int, default=7 * 24 * 60 * 60)
    # The GPU plane hard-caps each eval at 2 hours (larger values silently clamp), so a
    # full 50k-step B200 run (~2.2h) gets killed near step 45k. Finish long trainings by
    # chaining jobs: resubmit with --resume-checkpoint-key pointing at the last uploaded
    # checkpoint from the killed job's prefix.
    parser.add_argument("--timeout", type=int, default=7200)
    # Viridian budgets are in micro-dollars: 1_000_000 = $1, 0 = uncapped. A full 50k-step
    # B200 training run costs ~$1.5, so the $50 default is generous headroom while still
    # bounding a hung-but-spinning job (the 20h GPU timeout alone would allow ~$130).
    parser.add_argument("--budget", type=int, default=50_000_000,
                        help="spend cap in micro-dollars (1000000 = $1); 0 = uncapped")
    parser.add_argument("--upload-every-s", type=int, default=60)
    parser.add_argument("--resume-checkpoint-key", default="")
    parser.add_argument("--resume-checkpoint-sha-key", default="")
    parser.add_argument("--resume-checkpoint-sha256", default="")
    parser.add_argument("--dry-run", action="store_true", help="package + presign + write job.json, skip vd submit")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    global EXTRA_MODULE_NAME, SEED_MODEL_PATH
    EXTRA_MODULE_NAME = args.extra_module
    SEED_MODEL_PATH = args.seed_model
    exp_module = importlib.import_module(args.exp)
    uploads = expected_output_files(exp_module)

    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    exp_slug = args.exp.rsplit(".", 1)[-1].replace("_", "-")
    seed_slug = f"-seed{args.seed}" if args.seed is not None else ""
    prefix = args.prefix or f"viridian-runner-probes/runs/sotaku-train-{exp_slug}{seed_slug}-{args.gpu_tier}-{timestamp}"

    out_dir = WRAPPER_DIR / "jobs" / prefix.rstrip("/").rsplit("/", 1)[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = package_baseline(args.exp, exp_module, out_dir)
    digest = upload_baseline(tgz_path)
    print(f"baseline: {digest}")

    config = read_r2_config(args.remote)
    manifest = build_train_manifest(config, args.bucket, prefix, args.expires, uploads)
    manifest_key = f"{prefix.rstrip('/')}/private/presigned_artifact_manifest.json"
    upload_bytes(
        json.dumps(manifest, sort_keys=True).encode(),
        presign_url(config, "PUT", args.bucket, manifest_key, args.expires),
        content_type="application/json",
    )
    manifest_get_url = presign_url(config, "GET", args.bucket, manifest_key, args.expires)

    eval_parts = [
        "python3 viridian_train.py",
        "--exp",
        args.exp,
        "--artifact-manifest-get-url",
        shell_quote(manifest_get_url),
        "--upload-every-s",
        str(args.upload_every_s),
        "--metric-step",
        str(exp_module.total_steps),
    ]
    if args.seed is not None:
        eval_parts.extend(["--seed", str(args.seed)])
    if args.resume_checkpoint_key:
        eval_parts.extend(
            [
                "--resume-checkpoint-get-url",
                shell_quote(presign_url(config, "GET", args.bucket, args.resume_checkpoint_key, args.expires)),
            ]
        )
        if args.resume_checkpoint_sha_key:
            eval_parts.extend(
                [
                    "--resume-checkpoint-sha-get-url",
                    shell_quote(presign_url(config, "GET", args.bucket, args.resume_checkpoint_sha_key, args.expires)),
                ]
            )
        if args.resume_checkpoint_sha256:
            eval_parts.extend(["--resume-checkpoint-sha256", args.resume_checkpoint_sha256])
    eval_cmd = " ".join(eval_parts)

    spec = build_spec(args, digest, args.exp, eval_cmd)
    spec_path = out_dir / "job.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    (out_dir / "manifest.safe.json").write_text(
        json.dumps(
            {
                "bucket": args.bucket,
                "exp": args.exp,
                "gpu_tier": args.gpu_tier,
                "prefix": prefix,
                "private_manifest_key": manifest_key,
                "resume_checkpoint_key": args.resume_checkpoint_key,
                "spec": str(spec_path),
                "timeout": args.timeout,
                "uploads": uploads,
            },
            indent=2,
        )
    )
    print(f"job spec: {spec_path}")
    print(f"r2 prefix: {prefix}")

    if args.dry_run:
        print("dry run: not submitting")
        return

    completed = subprocess.run([str(VD_BIN), "submit", str(spec_path)], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            f"vd submit failed (exit {completed.returncode}): stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    job_id = completed.stdout.strip()
    print(f"job: {job_id}")
    print(f"watch: rclone cat 'r2:{args.bucket}/{prefix}/outputs/{exp_module.log_name}'")


if __name__ == "__main__":
    main()
