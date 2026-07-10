"""Evaluate POMO TSP checkpoints on a fixed test set with x8 augmentation."""

import argparse
import csv
import hashlib
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(__file__).resolve().parents[1]
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
TSP_CONFIGS = {
    20: {
        "checkpoint_dir": "saved_tsp20_model",
        "checkpoint_epoch": 510,
        "default_aug_batch_size": 1000,
        "batch_size_basis": "official NEW_py_ver/TSP/POMO/test_n20.py",
    },
    50: {
        "checkpoint_dir": "saved_tsp50_model",
        "checkpoint_epoch": 1000,
        "default_aug_batch_size": 400,
        "batch_size_basis": (
            "engineering default; no official NEW_py_ver TSP50 test batch exists"
        ),
    },
    100: {
        "checkpoint_dir": "saved_tsp100_model2_longTrain",
        "checkpoint_epoch": 3100,
        "default_aug_batch_size": 100,
        "batch_size_basis": "official NEW_py_ver/TSP/POMO/test_n100.py",
    },
}
CSV_FIELDS = [
    "problem",
    "n",
    "formal_target",
    "primary_metric",
    "eval_type",
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "checkpoint_load",
    "test_data",
    "test_data_sha256",
    "test_data_total_instances",
    "test_data_seed",
    "test_data_generator",
    "test_subset",
    "episodes",
    "batch_size",
    "batch_size_basis",
    "aug_factor",
    "no_aug_diagnostic_score",
    "x8_aug_score",
    "elapsed_seconds",
    "device",
    "cuda_device",
    "evaluation_seed",
    "git_commit",
    "git_dirty",
    "python",
    "torch",
    "torch_cuda",
    "cuda_available",
    "cudnn",
    "gpu",
    "platform",
    "note",
]


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


def environment_metadata(device):
    gpu = ""
    cuda_device = ""
    if device.type == "cuda":
        cuda_device = device.index
        gpu = torch.cuda.get_device_name(device.index)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "",
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version() or "",
        "gpu": gpu,
        "platform": platform.platform(),
        "cuda_device": cuda_device,
    }


def validate_pomo_root(pomo_root):
    required = [
        pomo_root / "NEW_py_ver" / "TSP" / "TSProblemDef.py",
        pomo_root / "NEW_py_ver" / "TSP" / "POMO" / "TSPEnv.py",
        pomo_root / "NEW_py_ver" / "TSP" / "POMO" / "TSPModel.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Invalid POMO root; missing required source file(s):\n  "
            + "\n  ".join(missing)
        )


def add_import_paths(pomo_root):
    new_py = pomo_root / "NEW_py_ver"
    paths = [new_py, new_py / "utils", new_py / "TSP", new_py / "TSP" / "POMO"]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def checkpoint_for(pomo_root, problem_size):
    config = TSP_CONFIGS[problem_size]
    checkpoint = (
        pomo_root
        / "NEW_py_ver"
        / "TSP"
        / "POMO"
        / "result"
        / config["checkpoint_dir"]
        / "checkpoint-{}.pt".format(config["checkpoint_epoch"])
    )
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing checkpoint: {}".format(checkpoint))
    return checkpoint, config


def load_test_data(path, problem_size):
    if not path.is_file():
        raise FileNotFoundError("Missing fixed TSP test set: {}".format(path))
    data = torch.load(str(path), map_location="cpu")
    metadata = data if isinstance(data, dict) else {}
    problems = data.get("problems") if isinstance(data, dict) else data
    if not isinstance(problems, torch.Tensor):
        raise TypeError("Fixed TSP data must be a tensor or a dict containing 'problems'.")
    if problems.dim() != 3 or problems.size(2) != 2:
        raise ValueError("Bad TSP data shape: {}".format(tuple(problems.shape)))
    if problems.size(1) != problem_size:
        raise ValueError(
            "Test data has N={}, expected N={}.".format(problems.size(1), problem_size)
        )
    if not torch.is_floating_point(problems):
        raise TypeError("TSP coordinates must be floating-point.")
    if not torch.isfinite(problems).all().item():
        raise ValueError("TSP test data contains NaN or Inf.")
    coordinate_min = problems.min().item()
    coordinate_max = problems.max().item()
    if coordinate_min < 0.0 or coordinate_max > 1.0:
        raise ValueError(
            "TSP coordinates are outside [0, 1]: min={}, max={}.".format(
                coordinate_min, coordinate_max
            )
        )
    if "problem_size" in metadata and int(metadata["problem_size"]) != problem_size:
        raise ValueError("Test-set problem_size metadata disagrees with tensor shape.")
    if "num_instances" in metadata and int(metadata["num_instances"]) != problems.size(0):
        raise ValueError("Test-set num_instances metadata disagrees with tensor shape.")
    return problems.contiguous(), metadata


def configure_device(device_name, cuda_device):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        if cuda_device < 0 or cuda_device >= torch.cuda.device_count():
            raise ValueError(
                "CUDA device {} is outside available range [0, {}).".format(
                    cuda_device, torch.cuda.device_count()
                )
            )
        torch.cuda.set_device(cuda_device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        return torch.device("cuda", cuda_device)
    torch.set_default_tensor_type("torch.FloatTensor")
    return torch.device("cpu")


def synchronize_cuda(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_saved_batch(env, problems, start, batch_size, aug_factor, device):
    from TSProblemDef import augment_xy_data_by_8_fold

    batch = problems[start : start + batch_size]
    if batch.size(0) != batch_size:
        raise RuntimeError("Fixed test-set slice returned fewer instances than requested.")
    env.batch_size = batch_size
    env.problems = batch.to(device=device, non_blocking=False)
    if aug_factor != 8:
        raise ValueError("Formal evaluator requires aug_factor=8.")
    env.batch_size *= aug_factor
    env.problems = augment_xy_data_by_8_fold(env.problems)

    env.BATCH_IDX = (
        torch.arange(env.batch_size, device=device)[:, None]
        .expand(env.batch_size, env.pomo_size)
    )
    env.POMO_IDX = (
        torch.arange(env.pomo_size, device=device)[None, :]
        .expand(env.batch_size, env.pomo_size)
    )


def run_one_batch(env, model, problems, start, batch_size, device):
    aug_factor = 8
    load_saved_batch(env, problems, start, batch_size, aug_factor, device)
    reset_state, _, _ = env.reset()
    model.pre_forward(reset_state)

    state, reward, done = env.pre_step()
    while not done:
        selected, _ = model(state)
        state, reward, done = env.step(selected)

    expected_reward_shape = (aug_factor * batch_size, env.pomo_size)
    if tuple(reward.shape) != expected_reward_shape:
        raise RuntimeError(
            "Unexpected reward shape {}, expected {}.".format(
                tuple(reward.shape), expected_reward_shape
            )
        )
    if not torch.isfinite(reward).all().item():
        raise RuntimeError("Model reward contains NaN or Inf.")

    # Official augment_xy_data_by_8_fold concatenates full batches in fold order,
    # so reshape(8, batch, pomo) exactly matches the official TSPTester layout.
    aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
    best_pomo_reward = aug_reward.max(dim=2).values
    no_aug_costs = -best_pomo_reward[0].float()
    x8_costs = -best_pomo_reward.max(dim=0).values.float()
    if torch.any(x8_costs > no_aug_costs + 1e-6).item():
        raise RuntimeError(
            "x8 cost exceeded no-augmentation cost although x8 contains fold 0."
        )
    return no_aug_costs.double().sum().item(), x8_costs.double().sum().item()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the formal POMO TSP x8 target on a fixed test set."
    )
    parser.add_argument(
        "--pomo-root",
        default=str(DEFAULT_POMO_ROOT),
        help="POMO repository root (default: repository containing this script).",
    )
    parser.add_argument(
        "--problem-size", type=int, required=True, choices=sorted(TSP_CONFIGS)
    )
    parser.add_argument("--test-data", required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Evaluate the first N fixed instances; default evaluates the full file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Augmented inference batch before the x8 expansion. "
            "TSP50's default 400 is an engineering choice, not an official parameter."
        ),
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--evaluation-seed",
        type=int,
        default=1234,
        help="Recorded process seed; argmax x8 inference itself is deterministic.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pomo_root = Path(args.pomo_root).expanduser().resolve()
    validate_pomo_root(pomo_root)
    add_import_paths(pomo_root)
    from TSPEnv import TSPEnv
    from TSPModel import TSPModel

    test_data_path = Path(args.test_data).expanduser().resolve()
    problems, test_metadata = load_test_data(test_data_path, args.problem_size)
    episodes = problems.size(0) if args.episodes is None else args.episodes
    if episodes <= 0:
        raise ValueError("--episodes must be positive.")
    if episodes > problems.size(0):
        raise ValueError(
            "Requested {} episodes, but test data has {}.".format(
                episodes, problems.size(0)
            )
        )

    checkpoint, case_config = checkpoint_for(pomo_root, args.problem_size)
    batch_size = (
        case_config["default_aug_batch_size"]
        if args.batch_size is None
        else args.batch_size
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    batch_size_basis = (
        case_config["batch_size_basis"]
        if args.batch_size is None
        else "explicit command-line value"
    )

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite {}. Pass --overwrite explicitly.".format(output)
        )

    torch.manual_seed(args.evaluation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.evaluation_seed)
    device = configure_device(args.device, args.cuda_device)

    env = TSPEnv(problem_size=args.problem_size, pomo_size=args.problem_size)
    model = TSPModel(**MODEL_PARAMS)
    state = torch.load(str(checkpoint), map_location=device)
    checkpoint_epoch = state.get("epoch")
    if checkpoint_epoch is not None and int(checkpoint_epoch) != int(
        case_config["checkpoint_epoch"]
    ):
        raise ValueError(
            "Checkpoint metadata epoch {} does not match expected {}.".format(
                checkpoint_epoch, case_config["checkpoint_epoch"]
            )
        )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    totals = {"no_aug": 0.0, "x8": 0.0}
    seen = 0
    synchronize_cuda(device)
    started = time.perf_counter()
    with torch.no_grad():
        while seen < episodes:
            current_batch = min(batch_size, episodes - seen)
            no_aug_sum, x8_sum = run_one_batch(
                env, model, problems, seen, current_batch, device
            )
            totals["no_aug"] += no_aug_sum
            totals["x8"] += x8_sum
            seen += current_batch
            print(
                "{}/{} no_aug_diagnostic={:.4f} x8={:.4f}".format(
                    seen,
                    episodes,
                    no_aug_sum / current_batch,
                    x8_sum / current_batch,
                ),
                flush=True,
            )
    synchronize_cuda(device)
    elapsed_seconds = time.perf_counter() - started

    row = {
        "problem": "TSP",
        "n": args.problem_size,
        "formal_target": "POMO x8 augmentation",
        "primary_metric": "x8_aug_score",
        "eval_type": MODEL_PARAMS["eval_type"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": case_config["checkpoint_epoch"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_load": "strict_ok",
        "test_data": str(test_data_path),
        "test_data_sha256": sha256_file(test_data_path),
        "test_data_total_instances": problems.size(0),
        "test_data_seed": test_metadata.get("seed", ""),
        "test_data_generator": test_metadata.get("generator", "unrecorded"),
        "test_subset": "prefix[0:{}]".format(episodes),
        "episodes": episodes,
        "batch_size": batch_size,
        "batch_size_basis": batch_size_basis,
        "aug_factor": 8,
        "no_aug_diagnostic_score": totals["no_aug"] / episodes,
        "x8_aug_score": totals["x8"] / episodes,
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "evaluation_seed": args.evaluation_seed,
        "note": (
            "No single-trajectory metric is emitted. No-augmentation is diagnostic "
            "only; x8_aug_score is the formal target."
        ),
    }
    row.update(git_metadata(pomo_root))
    row.update(environment_metadata(device))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print("Wrote {}".format(output), flush=True)


if __name__ == "__main__":
    main()
