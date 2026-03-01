import argparse
import logging
import math
from pathlib import Path

import torch
from tensordict import TensorDict
from tqdm import tqdm

from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator
from or_tools import darp_solver


log = logging.getLogger(__name__)


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_instance_data(state: TensorDict) -> TensorDict:
    """Build a problem-instance TensorDict from the environment state."""
    batch_size = state.batch_size
    instance = TensorDict({}, batch_size=batch_size)
    for key in ("h3_indices", "travel_time_matrix", "time_windows", "demand"):
        if key in state.keys():
            instance.set(key, state.get(key).clone())
    if "vehicle_capacity" in state.keys():
        instance.set("capacity", state.get("vehicle_capacity").squeeze(-1).clone())
    for key in ("locs", "flexibility", "user_id"):
        if key in state.keys():
            instance.set(key, state.get(key).clone())
    return instance


def _solve_actions(
    state: TensorDict,
    max_vehicles: int,
    time_limit_seconds: int,
    run_until_solution: bool,
) -> torch.Tensor | None:
    result = darp_solver(
        state,
        max_vehicles=max_vehicles,
        time_limit_seconds=time_limit_seconds,
        run_until_solution=run_until_solution,
    )
    total_time = result.get("total_time", float("inf"))
    if not math.isfinite(total_time) or not result.get("routes"):
        return None
    actions = result.get("actions_tensor")
    if actions is None:
        actions = torch.tensor(result.get("actions", []), dtype=torch.long)
    return actions.to(dtype=torch.long)


def _pad_action_sequences(sequences: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(seq.numel() for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.numel()] = seq
    return padded


def generate_instance_action_dataset(
    env: PDPTWEnv,
    dataset_size: int,
    batch_size: int,
    max_vehicles: int,
    time_limit_seconds: int,
    run_until_solution: bool,
) -> TensorDict:
    """Generate a dataset of problem instances and desired action sequences."""
    instance_batches = []
    action_sequences: list[torch.Tensor] = []
    progress = tqdm(total=dataset_size, desc="Generating instances", unit="inst")

    try:
        while len(action_sequences) < dataset_size:
            remaining = dataset_size - len(action_sequences)
            current_batch = min(batch_size, remaining)
            state = env.reset(batch_size=[current_batch]).to("cpu")

            for batch_idx in range(current_batch):
                instance_state = state[batch_idx: batch_idx + 1]
                actions = _solve_actions(
                    instance_state,
                    max_vehicles=max_vehicles,
                    time_limit_seconds=time_limit_seconds,
                    run_until_solution=run_until_solution,
                )
                if actions is None:
                    continue
                instance_batches.append(_extract_instance_data(instance_state))
                action_sequences.append(actions)
                progress.update(1)
    finally:
        progress.close()

    instances = TensorDict.cat(instance_batches, dim=0)
    dataset = TensorDict({}, batch_size=instances.batch_size)
    for key in instances.keys():
        dataset.set(key, instances.get(key))
    dataset.set("expert_actions", _pad_action_sequences(action_sequences))
    return dataset


def quick_test(
    env: PDPTWEnv,
    max_vehicles: int,
    time_limit_seconds: int,
    run_until_solution: bool,
) -> None:
    """Quick sanity check: solve one instance and print the action length."""
    state = env.reset(batch_size=[1]).to("cpu")
    actions = _solve_actions(
        state,
        max_vehicles=max_vehicles,
        time_limit_seconds=time_limit_seconds,
        run_until_solution=run_until_solution,
    )
    if actions is None:
        raise RuntimeError("OR-Tools failed to solve the test instance.")
    log.info("Test actions length: %d", actions.numel())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instance-action dataset for PDPTW.")
    parser.add_argument("--dataset-size", type=int, default=50000, help="Number of instances to generate.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for generation.")
    parser.add_argument("--max-vehicles", type=int, default=5, help="Max vehicles for OR-Tools solver.")
    parser.add_argument("--time-limit-seconds", type=int, default=1, help="OR-Tools time limit.")
    parser.add_argument(
        "--run-until-solution",
        action="store_true",
        help="Run OR-Tools until a feasible solution is found.",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to save the dataset (torch.save).")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    device = get_device(args.device)
    log.info("Using device: %s", device)

    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        perturbation=30,
        pickup_earliest_min=7 * 60,   # 7:00 AM
        dropoff_latest_min=17 * 60,   # 5:00 PM
    )
    env = PDPTWEnv(generator=generator)

    quick_test(
        env,
        max_vehicles=args.max_vehicles,
        time_limit_seconds=args.time_limit_seconds,
        run_until_solution=args.run_until_solution,
    )

    if args.output is None:
        log.info("No output path provided; skipping dataset save.")
        return

    dataset = generate_instance_action_dataset(
        env=env,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        max_vehicles=args.max_vehicles,
        time_limit_seconds=args.time_limit_seconds,
        run_until_solution=args.run_until_solution,
    )
    torch.save(dataset, args.output)
    log.info("Saved dataset to %s", args.output)


if __name__ == "__main__":
    main()
