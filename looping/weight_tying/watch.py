"""Wait for already-submitted study inputs; never launch or replace a worker."""

import argparse
import json
import time
from pathlib import Path

import modal

from checkpoint_utils import atomic_json_save
from looping.weight_tying.common import protocol_sha256, run_name


def wait_for_results(registry_path, output_path, timeout):
    registry = json.loads(Path(registry_path).read_text())
    if registry["protocol_sha256"] != protocol_sha256():
        raise ValueError("Job registry belongs to another protocol")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads(output_path.read_text()) if output_path.exists() else {"results": {}, "errors": {}}
    expected = {run_name(job["architecture"], job["regime"], job["seed"]): job for job in registry["jobs"]}
    if set(snapshot["results"]) - set(expected) or snapshot["errors"]:
        raise ValueError("Unexpected cached results or unresolved failures")
    for name, result in snapshot["results"].items():
        config = result["config"]
        if config["protocol_sha256"] != protocol_sha256() or run_name(config["architecture"], config["regime"], config["seed"]) != name:
            raise ValueError("Cached result identity mismatch")
    deadline = time.monotonic() + timeout
    connection_failures = {}
    while len(snapshot["results"]) < len(expected):
        if time.monotonic() >= deadline:
            raise TimeoutError("Local result wait expired; detached workers are unaffected")
        for name, job in expected.items():
            if name in snapshot["results"]:
                continue
            try:
                result = modal.FunctionCall.from_id(job["call_id"]).get(timeout=1)
            except TimeoutError:
                continue
            except modal.exception.ConnectionError as error:
                connection_failures[name] = connection_failures.get(name, 0) + 1
                if connection_failures[name] == 1 or connection_failures[name] % 10 == 0:
                    print(json.dumps({"run": name, "read_retry": connection_failures[name],
                                      "error": str(error), "workers_unchanged": True}), flush=True)
                continue
            except Exception as error:
                snapshot["errors"][name] = repr(error)
                atomic_json_save(snapshot, output_path)
                raise
            config = result["config"]
            if config["protocol_sha256"] != protocol_sha256() or run_name(config["architecture"], config["regime"], config["seed"]) != name:
                raise ValueError("Remote result identity mismatch")
            snapshot["results"][name] = result
            atomic_json_save(snapshot, output_path)
            print(json.dumps({"run": name, "status": result["status"], "updates": result["updates"],
                              "finished": len(snapshot["results"]), "planned": len(expected)}), flush=True)
        if len(snapshot["results"]) < len(expected):
            time.sleep(60)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path(__file__).with_name("jobs.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=21600)
    arguments = parser.parse_args()
    wait_for_results(arguments.registry, arguments.output, arguments.timeout)


if __name__ == "__main__":
    main()
