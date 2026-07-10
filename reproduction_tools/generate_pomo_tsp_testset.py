"""Generate an auditable fixed TSP test set with POMO's official generator."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_RELATIVE_PATH = Path("NEW_py_ver/TSP/TSProblemDef.py")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(pomo_root):
    rev = subprocess.run(
        ["git", "-C", str(pomo_root), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    status = subprocess.run(
        ["git", "-C", str(pomo_root), "status", "--porcelain"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    return {
        "git_commit": rev.stdout.strip() if rev.returncode == 0 else "unavailable",
        "git_dirty": status.returncode != 0 or bool(status.stdout.strip()),
    }


def validate_pomo_root(pomo_root):
    generator_source = pomo_root / GENERATOR_RELATIVE_PATH
    if not generator_source.is_file():
        raise FileNotFoundError(
            "POMO root does not contain the official TSP generator: {}".format(
                generator_source
            )
        )
    return generator_source


def add_import_paths(pomo_root):
    tsp_root = pomo_root / "NEW_py_ver" / "TSP"
    text = str(tsp_root)
    if text not in sys.path:
        sys.path.insert(0, text)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed TSP data with POMO NEW_py_ver's official generator."
    )
    parser.add_argument(
        "--pomo-root",
        default=str(DEFAULT_POMO_ROOT),
        help="POMO repository root (default: repository containing this script).",
    )
    parser.add_argument(
        "--problem-size", type=int, required=True, choices=[20, 50, 100]
    )
    parser.add_argument("--num-instances", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly replace an existing test set and its metadata sidecar.",
    )
    args = parser.parse_args()

    if args.num_instances <= 0:
        raise ValueError("--num-instances must be positive.")

    pomo_root = Path(args.pomo_root).expanduser().resolve()
    generator_source = validate_pomo_root(pomo_root)
    add_import_paths(pomo_root)
    from TSProblemDef import get_random_problems

    output = Path(args.output).expanduser().resolve()
    sidecar = output.with_name(output.name + ".metadata.json")
    existing = [path for path in (output, sidecar) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing file(s): {}. Pass --overwrite explicitly.".format(
                ", ".join(str(path) for path in existing)
            )
        )

    # The official generator calls torch.rand without accepting a Generator.
    # Keep generation on CPU and seed the global CPU RNG immediately beforehand.
    torch.set_default_tensor_type("torch.FloatTensor")
    torch.manual_seed(args.seed)
    problems = get_random_problems(args.num_instances, args.problem_size)

    expected_shape = (args.num_instances, args.problem_size, 2)
    if tuple(problems.shape) != expected_shape:
        raise RuntimeError(
            "Official generator returned {}, expected {}.".format(
                tuple(problems.shape), expected_shape
            )
        )
    if not torch.is_floating_point(problems):
        raise TypeError("TSP coordinates must be floating-point.")
    if not torch.isfinite(problems).all().item():
        raise ValueError("TSP coordinates contain NaN or Inf.")
    coordinate_min = problems.min().item()
    coordinate_max = problems.max().item()
    if coordinate_min < 0.0 or coordinate_max > 1.0:
        raise ValueError(
            "TSP coordinates are outside [0, 1]: min={}, max={}.".format(
                coordinate_min, coordinate_max
            )
        )

    metadata = {
        "format_version": 2,
        "problems": problems,
        "problem": "TSP",
        "problem_size": args.problem_size,
        "num_instances": args.num_instances,
        "seed": args.seed,
        "generator": (
            "POMO NEW_py_ver/TSP/TSProblemDef.get_random_problems"
        ),
        "generator_source": str(GENERATOR_RELATIVE_PATH),
        "generator_source_sha256": sha256_file(generator_source),
        "distribution": "independent uniform coordinates in [0,1]^2",
        "provenance": (
            "Reproduction test set generated with the official POMO generator; "
            "not the paper authors' original test instances."
        ),
        "paper_scale_subset": (
            "first 10000 instances" if args.num_instances >= 10_000 else "not available"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    metadata.update(git_metadata(pomo_root))

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(metadata, str(output))
    testset_sha256 = sha256_file(output)

    sidecar_payload = {
        key: value for key, value in metadata.items() if key != "problems"
    }
    sidecar_payload.update(
        {
            "testset_path": str(output),
            "testset_sha256": testset_sha256,
            "shape": list(problems.shape),
            "dtype": str(problems.dtype),
            "coordinate_min": coordinate_min,
            "coordinate_max": coordinate_max,
        }
    )
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(sidecar_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(sidecar_payload, indent=2, sort_keys=True), flush=True)
    print("Metadata sidecar: {}".format(sidecar), flush=True)


if __name__ == "__main__":
    main()
