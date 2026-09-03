"""Run research tests in separate processes to isolate study-local imports."""

import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = ROOT / "looping" / "trajectory_viz"


def test_directories():
    return sorted({path.parent for path in RESEARCH_ROOT.rglob("test_*.py")})


def run_directory(directory):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(directory))
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for path in sorted(directory.glob("test_*.py")):
        tests = loader.loadTestsFromModule(importlib.import_module(path.stem))
        if tests.countTestCases() == 0:
            raise RuntimeError(f"No tests discovered in {path}")
        suite.addTests(tests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.directory is not None:
        directory = args.directory.resolve()
        if directory not in test_directories():
            parser.error("Directory must contain tests within looping/trajectory_viz")
        return 0 if run_directory(directory) else 1

    directories = test_directories()
    if not directories:
        parser.error("No research tests found")
    failures = []
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    for directory in directories:
        print(f"\nResearch tests: {directory.relative_to(ROOT)}", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, "--directory", str(directory)],
            cwd=ROOT, env=environment,
        )
        if result.returncode:
            failures.append(str(directory.relative_to(ROOT)))
    if failures:
        print("\nFailed study directories:\n" + "\n".join(failures))
    print(f"\n{len(directories) - len(failures)}/{len(directories)} research test directories passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
