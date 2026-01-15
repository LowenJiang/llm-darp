import argparse
import logging
from pathlib import Path

import torch
from tensordict import TensorDict

from env import PDPTWEnv
from generator import SFGenerator
from policy import PDPTWAttentionPolicy


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


@torch.no_grad()
def _rollout_actions(
    policy: PDPTWAttentionPolicy,
    env: PDPTWEnv,
    state: TensorDict,
    decode_type: str,
    max_steps: int,
) -> torch.Tensor:
    policy.eval()
    outputs = policy(
        state,
        env,
        phase="test",
        decode_type=decode_type,
        max_steps=max_steps,
    )
    if outputs.get("truncated"):
        raise RuntimeError("Rollout truncated before completion; increase max_steps.")
    return outputs["actions"]


def generate_instance_action_dataset(
    env: PDPTWEnv,
    policy: PDPTWAttentionPolicy,
    dataset_size: int,
    batch_size: int,
    decode_type: str,
    max_steps: int,
    device: torch.device,
) -> TensorDict:
    """Generate a dataset of problem instances and desired action sequences."""
    num_batches = (dataset_size + batch_size - 1) // batch_size
    instance_batches = []
    action_batches = []

    for batch_idx in range(num_batches):
        current_batch = min(batch_size, dataset_size - batch_idx * batch_size)
        state = env.reset(batch_size=[current_batch]).to(device)
        actions = _rollout_actions(policy, env, state, decode_type, max_steps).cpu()
        instance_batches.append(_extract_instance_data(state).cpu())
        action_batches.append(actions)

    instances = TensorDict.cat(instance_batches, dim=0)
    dataset = TensorDict({}, batch_size=instances.batch_size)
    for key in instances.keys():
        dataset.set(key, instances.get(key))
    dataset.set("expert_actions", torch.cat(action_batches, dim=0))
    return dataset


def quick_test(env: PDPTWEnv, policy: PDPTWAttentionPolicy, device: torch.device) -> None:
    """Quick sanity check: generate one batch and print shapes."""
    state = env.reset(batch_size=[2]).to(device)
    actions = _rollout_actions(policy, env, state, decode_type="greedy", max_steps=300)
    log.info("Test batch size: %s", state.batch_size)
    log.info("Actions shape: %s", tuple(actions.shape))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate instance-action dataset for PDPTW.")
    parser.add_argument("--dataset-size", type=int, default=128, help="Number of instances to generate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation.")
    parser.add_argument("--max-steps", type=int, default=300, help="Max rollout steps.")
    parser.add_argument("--decode-type", type=str, default="greedy", help="Decode type for desired actions.")
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
    generator = SFGenerator(csv_path=csv_path, travel_time_matrix_path=ttm_path)
    env = PDPTWEnv(generator=generator)
    policy = PDPTWAttentionPolicy()
    policy.to(device)

    quick_test(env, policy, device)

    if args.output is None:
        log.info("No output path provided; skipping dataset save.")
        return

    dataset = generate_instance_action_dataset(
        env=env,
        policy=policy,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        decode_type=args.decode_type,
        max_steps=args.max_steps,
        device=device,
    )
    torch.save(dataset, args.output)
    log.info("Saved dataset to %s", args.output)


if __name__ == "__main__":
    main()
