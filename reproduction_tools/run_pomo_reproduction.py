"""Run short POMO checkpoint checks without modifying official NEW_py_ver code.

TSP cases in this runner intentionally generate data on the fly and are smoke-only.
Formal TSP evaluation must use run_pomo_tsp_fixed_eval.py with a saved test set.
CVRP100 uses the official fixed vrp100_test_seed1234.pt file.
"""

import argparse
import csv
import hashlib
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch


DEFAULT_POMO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CaseConfig:
    name: str
    problem: str
    size: int
    checkpoint_dir: Path
    checkpoint_epoch: int
    default_episodes: int
    aug_batch_size: int
    batch_size_basis: str
    no_aug_batch_size: int = None
    no_aug_batch_size_basis: str = ""
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
    "case",
    "problem",
    "n",
    "run_scope",
    "formal_target",
    "evaluation_mode",
    "primary_metric",
    "primary_score",
    "eval_type",
    "checkpoint",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "checkpoint_load",
    "test_data",
    "test_data_kind",
    "test_data_sha256",
    "test_data_total_instances",
    "episodes",
    "batch_size",
    "batch_size_basis",
    "aug_factor",
    "no_aug_diagnostic_score",
    "x8_aug_score",
    "elapsed_seconds",
    "device",
    "cuda_device",
    "seed",
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
        pomo_root / "NEW_py_ver" / "TSP" / "POMO" / "TSPEnv.py",
        pomo_root / "NEW_py_ver" / "CVRP" / "POMO" / "CVRPEnv.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Invalid POMO root; missing required source file(s):\n  "
            + "\n  ".join(missing)
        )


def build_cases(pomo_root):
    new_py = pomo_root / "NEW_py_ver"
    return {
        "tsp20": CaseConfig(
            "tsp20",
            "TSP",
            20,
            new_py / "TSP" / "POMO" / "result" / "saved_tsp20_model",
            510,
            100_000,
            1_000,
            "official NEW_py_ver/TSP/POMO/test_n20.py",
            note="Random-generator smoke only; formal TSP requires a fixed test set.",
        ),
        "tsp50": CaseConfig(
            "tsp50",
            "TSP",
            50,
            new_py / "TSP" / "POMO" / "result" / "saved_tsp50_model",
            1000,
            100_000,
            400,
            "engineering default; no official NEW_py_ver TSP50 test batch exists",
            note=(
                "Checkpoint epoch 1000, argmax, x8. Random-generator smoke only; "
                "formal TSP50 requires a fixed test set."
            ),
        ),
        "tsp100": CaseConfig(
            "tsp100",
            "TSP",
            100,
            new_py
            / "TSP"
            / "POMO"
            / "result"
            / "saved_tsp100_model2_longTrain",
            3100,
            100_000,
            100,
            "official NEW_py_ver/TSP/POMO/test_n100.py",
            note="Random-generator smoke only; formal TSP requires a fixed test set.",
        ),
        "cvrp100": CaseConfig(
            "cvrp100",
            "CVRP",
            100,
            new_py / "CVRP" / "POMO" / "result" / "saved_CVRP100_model",
            30500,
            10_000,
            400,
            "official NEW_py_ver/CVRP/POMO/test_n100.py x8 aug_batch_size",
            no_aug_batch_size=1_000,
            no_aug_batch_size_basis=(
                "official NEW_py_ver/CVRP/POMO/test_n100.py "
                "no-augmentation test_batch_size"
            ),
            saved_test_data=new_py / "CVRP" / "vrp100_test_seed1234.pt",
            note="Uses the official fixed CVRP100 test set.",
        ),
    }


def add_import_paths(pomo_root):
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


def validate_case(case, episodes):
    checkpoint = case.checkpoint_dir / "checkpoint-{}.pt".format(
        case.checkpoint_epoch
    )
    missing = []
    if not checkpoint.is_file():
        missing.append(str(checkpoint))
    if case.saved_test_data is not None and not case.saved_test_data.is_file():
        missing.append(str(case.saved_test_data))
    if missing:
        raise FileNotFoundError(
            "Missing required file(s):\n  " + "\n  ".join(missing)
        )

    test_data_total_instances = ""
    if case.problem == "CVRP":
        loaded = torch.load(str(case.saved_test_data), map_location="cpu")
        required_keys = ("depot_xy", "node_xy", "node_demand")
        if not isinstance(loaded, dict) or any(key not in loaded for key in required_keys):
            raise ValueError("CVRP fixed data is missing required tensors.")
        depot_xy = loaded["depot_xy"]
        node_xy = loaded["node_xy"]
        node_demand = loaded["node_demand"]
        expected_tail_shapes = ((1, 2), (case.size, 2), (case.size,))
        tensors = (depot_xy, node_xy, node_demand)
        for tensor, tail_shape, key in zip(tensors, expected_tail_shapes, required_keys):
            if tuple(tensor.shape[1:]) != tail_shape:
                raise ValueError(
                    "Bad {} shape {}, expected (*, {}).".format(
                        key, tuple(tensor.shape), tail_shape
                    )
                )
            if not torch.isfinite(tensor).all().item():
                raise ValueError("{} contains NaN or Inf.".format(key))
        counts = {tensor.size(0) for tensor in tensors}
        if len(counts) != 1:
            raise ValueError("CVRP fixed-data tensors have different instance counts.")
        test_data_total_instances = counts.pop()
        if episodes > test_data_total_instances:
            raise ValueError(
                "Requested {} episodes, but CVRP fixed data has {}.".format(
                    episodes, test_data_total_instances
                )
            )
    return checkpoint, test_data_total_instances


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


def make_env_and_model(case, pomo_root):
    add_import_paths(pomo_root)
    if case.problem == "TSP":
        from TSPEnv import TSPEnv as Env
        from TSPModel import TSPModel as Model
    elif case.problem == "CVRP":
        from CVRPEnv import CVRPEnv as Env
        from CVRPModel import CVRPModel as Model
    else:
        raise ValueError("Unsupported problem type: {}".format(case.problem))
    return Env(problem_size=case.size, pomo_size=case.size), Model(**MODEL_PARAMS)


def run_one_batch(env, model, batch_size, aug_factor):
    if aug_factor not in (1, 8):
        raise ValueError("aug_factor must be 1 or 8.")

    env.load_problems(batch_size, aug_factor)
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

    aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
    best_pomo_reward = aug_reward.max(dim=2).values
    no_aug_costs = -best_pomo_reward[0].float()
    no_aug_sum = no_aug_costs.double().sum().item()

    if aug_factor == 1:
        return no_aug_sum, None

    x8_costs = -best_pomo_reward.max(dim=0).values.float()
    if torch.any(x8_costs > no_aug_costs + 1e-6).item():
        raise RuntimeError(
            "x8 cost exceeded no-augmentation cost although x8 contains fold 0."
        )
    return no_aug_sum, x8_costs.double().sum().item()


def run_case(
    case,
    pomo_root,
    episodes,
    device_name,
    cuda_device,
    seed,
    mode,
    aug_factor,
    batch_size_override,
):
    if episodes <= 0:
        raise ValueError("Episode count must be positive.")
    if aug_factor not in (1, 8):
        raise ValueError("aug_factor must be 1 or 8.")
    if aug_factor == 1 and case.no_aug_batch_size is None:
        raise ValueError(
            "No-augmentation mode is not configured for case {}.".format(case.name)
        )
    checkpoint, test_data_total_instances = validate_case(case, episodes)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = configure_device(device_name, cuda_device)
    env, model = make_env_and_model(case, pomo_root)

    state = torch.load(str(checkpoint), map_location=device)
    checkpoint_epoch = state.get("epoch")
    if checkpoint_epoch is not None and int(checkpoint_epoch) != case.checkpoint_epoch:
        raise ValueError(
            "Checkpoint metadata epoch {} does not match expected {}.".format(
                checkpoint_epoch, case.checkpoint_epoch
            )
        )
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()

    if case.problem == "CVRP":
        env.use_saved_problems(str(case.saved_test_data), device)

    if batch_size_override is not None:
        configured_batch_size = batch_size_override
        batch_size_basis = "explicit command-line value"
    elif aug_factor == 1:
        configured_batch_size = case.no_aug_batch_size
        batch_size_basis = case.no_aug_batch_size_basis
    else:
        configured_batch_size = case.aug_batch_size
        batch_size_basis = case.batch_size_basis
    if configured_batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    actual_batch_size = min(configured_batch_size, episodes)
    totals = {"no_aug": 0.0, "x8": 0.0}
    seen = 0
    synchronize_cuda(device)
    started = time.perf_counter()
    with torch.no_grad():
        while seen < episodes:
            current_batch = min(actual_batch_size, episodes - seen)
            no_aug_sum, x8_sum = run_one_batch(
                env, model, current_batch, aug_factor
            )
            totals["no_aug"] += no_aug_sum
            if x8_sum is not None:
                totals["x8"] += x8_sum
            seen += current_batch
            if aug_factor == 1:
                message = "{}: {}/{} no_aug={:.4f}".format(
                    case.name,
                    seen,
                    episodes,
                    no_aug_sum / current_batch,
                )
            else:
                message = "{}: {}/{} no_aug_diagnostic={:.4f} x8={:.4f}".format(
                    case.name,
                    seen,
                    episodes,
                    no_aug_sum / current_batch,
                    x8_sum / current_batch,
                )
            print(message, flush=True)
    synchronize_cuda(device)
    elapsed_seconds = time.perf_counter() - started

    is_formal = mode == "full" and case.problem == "CVRP"
    test_data_kind = (
        "official fixed CVRP100 test set"
        if case.saved_test_data is not None
        else "on-the-fly official generator; smoke only"
    )
    row = {
        "case": case.name,
        "problem": case.problem,
        "n": case.size,
        "run_scope": mode,
        "formal_target": is_formal,
        "evaluation_mode": (
            "no_augmentation" if aug_factor == 1 else "x8_augmentation"
        ),
        "primary_metric": (
            "no_aug_diagnostic_score" if aug_factor == 1 else "x8_aug_score"
        ),
        "primary_score": (
            totals["no_aug"] / episodes
            if aug_factor == 1
            else totals["x8"] / episodes
        ),
        "eval_type": MODEL_PARAMS["eval_type"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_epoch": case.checkpoint_epoch,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_load": "strict_ok",
        "test_data": (
            str(case.saved_test_data.resolve()) if case.saved_test_data else ""
        ),
        "test_data_kind": test_data_kind,
        "test_data_sha256": (
            sha256_file(case.saved_test_data) if case.saved_test_data else ""
        ),
        "test_data_total_instances": test_data_total_instances,
        "episodes": episodes,
        "batch_size": actual_batch_size,
        "batch_size_basis": batch_size_basis,
        "aug_factor": aug_factor,
        "no_aug_diagnostic_score": totals["no_aug"] / episodes,
        "x8_aug_score": (
            "" if aug_factor == 1 else totals["x8"] / episodes
        ),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "seed": seed,
        "note": case.note,
    }
    row.update(git_metadata(pomo_root))
    row.update(environment_metadata(device))
    return row


def write_csv(rows, output, overwrite):
    if output.exists() and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite {}. Pass --overwrite explicitly.".format(output)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Run POMO checkpoint smoke checks and CVRP100 fixed evaluation."
    )
    parser.add_argument(
        "--pomo-root",
        default=str(DEFAULT_POMO_ROOT),
        help="POMO repository root (default: repository containing this script).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["tsp20", "tsp50", "tsp100", "cvrp100"],
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episode count for every selected case.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the augmented pre-expansion batch size for every selected case.",
    )
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help=(
            "Run the configured no-augmentation POMO mode (CVRP100 only). "
            "Without this flag, use x8 augmentation."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_POMO_ROOT
            / "reproduction_outputs"
            / "pomo_reproduction_results.csv"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pomo_root = Path(args.pomo_root).expanduser().resolve()
    validate_pomo_root(pomo_root)
    cases = build_cases(pomo_root)
    unknown = [name for name in args.cases if name not in cases]
    if unknown:
        raise KeyError(
            "Unknown case(s) {}. Available: {}.".format(
                ", ".join(unknown), ", ".join(sorted(cases))
            )
        )
    if args.mode == "full" and any(cases[name].problem == "TSP" for name in args.cases):
        raise ValueError(
            "Full TSP evaluation is forbidden in this random-data runner. "
            "Use run_pomo_tsp_fixed_eval.py with --test-data."
        )

    aug_factor = 1 if args.no_augmentation else 8

    rows = []
    for name in args.cases:
        case = cases[name]
        episodes = (
            args.episodes
            if args.episodes is not None
            else (10 if args.mode == "smoke" else case.default_episodes)
        )
        rows.append(
            run_case(
                case,
                pomo_root,
                episodes,
                args.device,
                args.cuda_device,
                args.seed,
                args.mode,
                aug_factor,
                args.batch_size,
            )
        )

    output = Path(args.output).expanduser().resolve()
    write_csv(rows, output, args.overwrite)
    print("Wrote {}".format(output), flush=True)


if __name__ == "__main__":
    main()
