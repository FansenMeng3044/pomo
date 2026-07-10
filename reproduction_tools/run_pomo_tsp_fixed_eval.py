"""Evaluate POMO TSP checkpoints on a saved fixed TSP test set."""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(r"D:\学习\MRTA\POMO")


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


def add_import_paths(pomo_root: Path):
    new_py = pomo_root / "NEW_py_ver"
    for path in reversed([new_py, new_py / "utils", new_py / "TSP", new_py / "TSP" / "POMO"]):
        sys.path.insert(0, str(path))


def checkpoint_for(pomo_root: Path, problem_size: int):
    root = pomo_root / "NEW_py_ver" / "TSP" / "POMO" / "result"
    if problem_size == 20:
        return root / "saved_tsp20_model" / "checkpoint-510.pt"
    if problem_size == 50:
        return root / "saved_tsp50_model" / "checkpoint-1000.pt"
    if problem_size == 100:
        return root / "saved_tsp100_model2_longTrain" / "checkpoint-3100.pt"
    raise ValueError("problem_size must be 20, 50, or 100")


def load_test_data(path: Path, device):
    data = torch.load(str(path), map_location=device)
    problems = data["problems"] if isinstance(data, dict) else data
    if problems.dim() != 3 or problems.size(2) != 2:
        raise ValueError(f"Bad TSP data shape: {tuple(problems.shape)}")
    return problems


def load_saved_batch(env, problems, start: int, batch_size: int, aug_factor: int):
    from TSProblemDef import augment_xy_data_by_8_fold

    env.batch_size = batch_size
    env.problems = problems[start:start + batch_size]
    if aug_factor > 1:
        if aug_factor != 8:
            raise NotImplementedError
        env.batch_size *= 8
        env.problems = augment_xy_data_by_8_fold(env.problems)

    env.BATCH_IDX = torch.arange(env.batch_size)[:, None].expand(env.batch_size, env.pomo_size)
    env.POMO_IDX = torch.arange(env.pomo_size)[None, :].expand(env.batch_size, env.pomo_size)


def run_one_batch(env, model, problems, start: int, batch_size: int, aug_factor: int):
    load_saved_batch(env, problems, start, batch_size, aug_factor)
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
    single_traj_score = -aug_reward[0, :, 0].float().mean().item()
    return single_traj_score, no_aug_score, aug_score


def main():
    parser = argparse.ArgumentParser(description="Evaluate POMO TSP on a fixed saved test set.")
    parser.add_argument("--pomo-root", default=str(DEFAULT_POMO_ROOT))
    parser.add_argument("--problem-size", type=int, required=True, choices=[20, 50, 100])
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        device = torch.device("cuda", 0)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    pomo_root = Path(args.pomo_root)
    add_import_paths(pomo_root)
    from TSPEnv import TSPEnv
    from TSPModel import TSPModel

    problems = load_test_data(Path(args.test_data), device)
    if problems.size(1) != args.problem_size:
        raise ValueError(f"Test data has N={problems.size(1)}, expected {args.problem_size}")
    episodes = args.episodes or problems.size(0)
    if episodes > problems.size(0):
        raise ValueError(f"Requested {episodes} episodes, but test data has {problems.size(0)}")

    env = TSPEnv(problem_size=args.problem_size, pomo_size=args.problem_size)
    model = TSPModel(**MODEL_PARAMS)
    checkpoint = checkpoint_for(pomo_root, args.problem_size)
    state = torch.load(str(checkpoint), map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    aug_factor = 1 if args.no_augmentation else 8
    default_batch = {20: 1000, 50: 400, 100: 100}[args.problem_size]
    batch_size = args.batch_size or default_batch

    totals = {"single_traj_score": 0.0, "no_aug_score": 0.0, "aug_score": 0.0}
    seen = 0
    started = time.perf_counter()
    with torch.no_grad():
        while seen < episodes:
            current_batch = min(batch_size, episodes - seen)
            single, no_aug, aug = run_one_batch(env, model, problems, seen, current_batch, aug_factor)
            totals["single_traj_score"] += single * current_batch
            totals["no_aug_score"] += no_aug * current_batch
            totals["aug_score"] += aug * current_batch
            seen += current_batch
            print(f"{seen}/{episodes} single={single:.4f} no_aug={no_aug:.4f} aug={aug:.4f}", flush=True)

    row = {
        "problem": "TSP",
        "n": args.problem_size,
        "test_data": str(Path(args.test_data)),
        "checkpoint": str(checkpoint),
        "episodes": episodes,
        "batch_size": batch_size,
        "aug_factor": aug_factor,
        "single_traj_score": totals["single_traj_score"] / episodes,
        "no_aug_score": totals["no_aug_score"] / episodes,
        "aug_score": totals["aug_score"] / episodes,
        "elapsed_seconds": time.perf_counter() - started,
        "device": str(device),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
