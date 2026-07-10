"""Run POMO checkpoint reproduction tests for TSP and CVRP.

The script reads the official POMO repository from D:\学习\MRTA\POMO and writes
only the CSV requested by --output.
"""

import argparse
import csv
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(r"D:\学习\MRTA\POMO")


@dataclass(frozen=True)
class CaseConfig:
    name: str
    problem: str
    size: int
    checkpoint_dir: Path
    checkpoint_epoch: int
    default_episodes: int
    no_aug_batch_size: int
    aug_batch_size: int
    saved_test_data: Path = None
    note: str = ""


MODEL_PARAMS = {
    "embedding_dim": 128,
    "sqrt_embedding_dim": 128 ** (1 / 2),
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
    "eval_type": "argmax",
}


CSV_FIELDS = [
    "case", "problem", "n", "checkpoint", "test_data", "episodes",
    "augmentation", "aug_factor", "batch_size", "no_aug_score", "aug_score",
    "elapsed_seconds", "device", "seed", "torch", "python", "note",
]


def build_cases(pomo_root: Path):
    new_py = pomo_root / "NEW_py_ver"
    return {
        "tsp20": CaseConfig(
            "tsp20", "TSP", 20,
            new_py / "TSP" / "POMO" / "result" / "saved_tsp20_model",
            510, 100_000, 10_000, 1_000,
            note="Matches NEW_py_ver/TSP/POMO/test_n20.py.",
        ),
        "tsp50": CaseConfig(
            "tsp50", "TSP", 50,
            new_py / "TSP" / "POMO" / "result" / "saved_tsp50_model",
            1000, 100_000, 10_000, 400,
            note="Checkpoint exists, but NEW_py_ver has no official test_n50.py.",
        ),
        "tsp100": CaseConfig(
            "tsp100", "TSP", 100,
            new_py / "TSP" / "POMO" / "result" / "saved_tsp100_model2_longTrain",
            3100, 100_000, 10_000, 100,
            note="Matches NEW_py_ver/TSP/POMO/test_n100.py.",
        ),
        "cvrp100": CaseConfig(
            "cvrp100", "CVRP", 100,
            new_py / "CVRP" / "POMO" / "result" / "saved_CVRP100_model",
            30500, 10_000, 1_000, 400,
            saved_test_data=new_py / "CVRP" / "vrp100_test_seed1234.pt",
            note="Matches NEW_py_ver/CVRP/POMO/test_n100.py.",
        ),
    }


def add_import_paths(pomo_root: Path):
    new_py = pomo_root / "NEW_py_ver"
    paths = [
        new_py,
        new_py / "utils",
        new_py / "TSP",
        new_py / "TSP" / "POMO",
        new_py / "CVRP",
        new_py / "CVRP" / "POMO",
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def validate_case(case: CaseConfig):
    checkpoint = case.checkpoint_dir / f"checkpoint-{case.checkpoint_epoch}.pt"
    missing = []
    if not checkpoint.exists():
        missing.append(str(checkpoint))
    if case.saved_test_data is not None and not case.saved_test_data.exists():
        missing.append(str(case.saved_test_data))
    if missing:
        raise FileNotFoundError("Missing required file(s):\n  " + "\n  ".join(missing))


def configure_device(device_name: str):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device_name == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return torch.device("cuda", 0)
    torch.set_default_tensor_type("torch.FloatTensor")
    return torch.device("cpu")


def make_env_and_model(case: CaseConfig, pomo_root: Path):
    add_import_paths(pomo_root)
    if case.problem == "TSP":
        from TSPEnv import TSPEnv as Env
        from TSPModel import TSPModel as Model
    elif case.problem == "CVRP":
        from CVRPEnv import CVRPEnv as Env
        from CVRPModel import CVRPModel as Model
    else:
        raise ValueError(f"Unsupported problem type: {case.problem}")
    return Env(problem_size=case.size, pomo_size=case.size), Model(**MODEL_PARAMS)


def run_one_batch(env, model, batch_size: int, aug_factor: int):
    env.load_problems(batch_size, aug_factor)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        state, reward, done = env.step(selected)

    aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
    max_pomo_reward = aug_reward.max(dim=2).values
    no_aug_score = -max_pomo_reward[0, :].float().mean().item()
    aug_score = -max_pomo_reward.max(dim=0).values.float().mean().item()
    return no_aug_score, aug_score


def run_case(case: CaseConfig, pomo_root: Path, episodes: int, augmentation: bool, device_name: str, seed: int = None):
    validate_case(case)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = configure_device(device_name)
    env, model = make_env_and_model(case, pomo_root)
    checkpoint = case.checkpoint_dir / f"checkpoint-{case.checkpoint_epoch}.pt"
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    if case.problem == "CVRP" and case.saved_test_data is not None:
        env.use_saved_problems(str(case.saved_test_data), device)

    aug_factor = 8 if augmentation else 1
    batch_size = case.aug_batch_size if augmentation else case.no_aug_batch_size
    total_no_aug = 0.0
    total_aug = 0.0
    seen = 0
    started = time.perf_counter()

    with torch.no_grad():
        while seen < episodes:
            current_batch = min(batch_size, episodes - seen)
            no_aug_score, aug_score = run_one_batch(env, model, current_batch, aug_factor)
            total_no_aug += no_aug_score * current_batch
            total_aug += aug_score * current_batch
            seen += current_batch
            print(
                f"{case.name}: {seen}/{episodes} "
                f"no_aug={no_aug_score:.4f} aug={aug_score:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    return {
        "case": case.name,
        "problem": case.problem,
        "n": case.size,
        "checkpoint": str(checkpoint),
        "test_data": str(case.saved_test_data) if case.saved_test_data else "",
        "episodes": episodes,
        "augmentation": augmentation,
        "aug_factor": aug_factor,
        "batch_size": batch_size,
        "no_aug_score": total_no_aug / episodes,
        "aug_score": total_aug / episodes,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "seed": "" if seed is None else seed,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "note": case.note,
    }


def write_csv(rows, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Run POMO TSP/CVRP checkpoint reproduction tests.")
    parser.add_argument("--pomo-root", default=str(DEFAULT_POMO_ROOT))
    parser.add_argument("--cases", nargs="+", default=["tsp20", "tsp50", "tsp100", "cvrp100"])
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count for every selected case.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="outputs/pomo_reproduction_results.csv")
    args = parser.parse_args()

    pomo_root = Path(args.pomo_root)
    cases = build_cases(pomo_root)
    rows = []
    for name in args.cases:
        if name not in cases:
            raise KeyError(f"Unknown case '{name}'. Available: {', '.join(cases)}")
        case = cases[name]
        episodes = args.episodes if args.episodes is not None else (10 if args.mode == "smoke" else case.default_episodes)
        rows.append(run_case(case, pomo_root, episodes, not args.no_augmentation, args.device, args.seed))

    output = Path(args.output)
    write_csv(rows, output)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()