import argparse
import logging
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter

from env import PDPTWEnv
from generator import SFGenerator
from policy import PDPTWAttentionPolicy
from reinforce import REINFORCE, _clone_state


log = logging.getLogger(__name__)


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    log.info("Using device: %s", device)

    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
    )

    env = PDPTWEnv(generator=generator)
    policy = PDPTWAttentionPolicy()
    policy.to(device)
    policy.train()

    trainer = REINFORCE(env, policy, baseline="mean")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    writer = SummaryWriter(log_dir=args.log_dir)

    for step in range(args.steps):
        batch = generator(batch_size=[args.batch_size]).to(device)
        state = env.reset(batch)

        outputs = policy(
            state,
            env,
            phase="train",
            decode_type=args.decode_type,
            max_steps=args.max_steps,
        )
        outputs = trainer.calculate_loss(_clone_state(state), batch, outputs)

        loss = outputs["loss"]
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
        optimizer.step()

        reward_mean = outputs["reward"].detach().mean().item()
        writer.add_scalar("train/reward_mean", reward_mean, step)
        vehicles_used = outputs.get("vehicles_used")
        if vehicles_used is not None:
            writer.add_scalar(
                "train/vehicles_used_mean",
                vehicles_used.detach().float().mean().item(),
                step,
            )

        if (step + 1) % args.log_every == 0:
            log.info(
                "Step %d/%d | loss=%.4f | reward_mean=%.4f",
                step + 1,
                args.steps,
                loss.detach().item(),
                reward_mean,
            )

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PDPTW policy with REINFORCE.")
    parser.add_argument("--steps", type=int, default=10_000, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=512, help="Parallel envs per step.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max-steps", type=int, default=300, help="Max decode steps.")
    parser.add_argument("--decode-type", type=str, default="sampling", help="Decode strategy.")
    parser.add_argument("--log-dir", type=str, default="runs/test_run", help="TensorBoard log dir.")
    parser.add_argument("--log-every", type=int, default=10, help="Log interval.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
