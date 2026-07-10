"""Generate fixed TSP test sets using POMO's official problem generator."""

import argparse
import sys
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(r"D:\学习\MRTA\POMO")


def add_import_paths(pomo_root: Path):
    tsp_root = pomo_root / "NEW_py_ver" / "TSP"
    sys.path.insert(0, str(tsp_root))


def main():
    parser = argparse.ArgumentParser(description="Generate fixed TSP test data for POMO-style evaluation.")
    parser.add_argument("--pomo-root", default=str(DEFAULT_POMO_ROOT))
    parser.add_argument("--problem-size", type=int, required=True, choices=[20, 50, 100])
    parser.add_argument("--num-instances", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    add_import_paths(Path(args.pomo_root))
    from TSProblemDef import get_random_problems

    torch.manual_seed(args.seed)
    problems = get_random_problems(args.num_instances, args.problem_size)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "problems": problems,
            "problem_size": args.problem_size,
            "num_instances": args.num_instances,
            "seed": args.seed,
            "generator": "POMO NEW_py_ver/TSP/TSProblemDef.get_random_problems",
        },
        str(output),
    )
    print(f"Wrote {output} with shape {tuple(problems.shape)}")


if __name__ == "__main__":
    main()
