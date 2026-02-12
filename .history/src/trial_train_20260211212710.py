# =========================================================================
# Main training entry point
# =========================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from tensordict import TensorDict
from tqdm import tqdm
from trial_pomo_reinforce import POMOReinforce

log = logging.getLogger(__name__)

# =========================================================================
# Data generation helper
# =========================================================================

def generate_dataset(generator, size: int, device: torch.device) -> TensorDict:
    """Generate a dataset of problem instances."""
    return generator(batch_size=[size]).to(device)


# =========================================================================
# Device selection
# =========================================================================

def get_device(device_str: str | None) -> torch.device:
    """Auto-detect the best available device."""
    if device_str:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")





def train(args: argparse.Namespace) -> None:
    """Full training loop."""
    device = get_device(args.device)
    log.info("Using device: %s", device)

    # --- Setup ---
    from oracle_env import PDPTWEnv
    from oracle_generator import SFGenerator
    from trial_gnn_policy import DARPPolicy

    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(csv_path=csv_path, travel_time_matrix_path=ttm_path)
    env = PDPTWEnv(generator=generator)

    policy = DARPPolicy(
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        ff_hidden=args.ff_hidden,
        tanh_clip=args.tanh_clip,
        temperature=args.temperature,
    )

    # Optionally load checkpoint
    checkpoint_epoch = 0
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint.get("policy_state_dict", checkpoint))
        checkpoint_epoch = checkpoint.get("epoch", 0)
        log.info("Loaded checkpoint from %s (epoch %d)", args.init_checkpoint, checkpoint_epoch)

    trainer = POMOReinforce(
        env=env,
        policy=policy,
        pomo_size=args.pomo_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        warmup_epochs=args.warmup_epochs,
        max_steps=args.max_steps,
        temperature=args.temperature,
        device=device,
    )

    # --- Checkpoint & logging dirs ---
    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.log_dir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=args.log_dir)

    # --- Generate fixed validation set ---
    log.info("Generating validation dataset (%d instances)...", args.val_data_size)
    val_data = generate_dataset(generator, args.val_data_size, device)

    best_val_reward = float("-inf")

    # --- Training loop ---
    for epoch in range(checkpoint_epoch, checkpoint_epoch + args.epochs):
        t0 = time.time()
        log.info("=" * 60)
        log.info("Epoch %d / %d", epoch + 1, checkpoint_epoch + args.epochs)

        # Generate fresh training data each epoch
        train_data = generate_dataset(generator, args.train_data_size, device)

        # Train
        train_metrics = trainer.train_epoch(
            train_data, batch_size=args.batch_size, epoch=epoch,
        )
        train_time = time.time() - t0

        log.info(
            "  Train - loss: %.4f  reward: %.4f +/- %.4f  advantage: %.4f  lr: %.6f  (%.1fs)",
            train_metrics["loss"],
            train_metrics["reward_mean"],
            train_metrics["reward_std"],
            train_metrics["advantage_mean"],
            train_metrics["lr"],
            train_time,
        )

        # Validate
        t1 = time.time()
        val_metrics = trainer.validate(
            val_data,
            batch_size=args.batch_size,
            pomo_size=args.pomo_size if args.val_pomo else None,
        )
        val_time = time.time() - t1

        val_str = f"  Val   - reward: {val_metrics['reward_mean']:.4f} +/- {val_metrics['reward_std']:.4f}"
        if "reward_best_mean" in val_metrics:
            val_str += f"  best: {val_metrics['reward_best_mean']:.4f}"
        val_str += f"  ({val_time:.1f}s)"
        log.info(val_str)

        # TensorBoard
        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # Checkpointing
        trainer.save(ckpt_dir / "latest.pt", epoch=epoch, extra={"val_metrics": val_metrics})

        val_reward = val_metrics.get("reward_best_mean", val_metrics["reward_mean"])
        if val_reward > best_val_reward:
            best_val_reward = val_reward
            trainer.save(ckpt_dir / "best.pt", epoch=epoch, extra={"val_metrics": val_metrics})
            log.info("  >> New best validation reward: %.4f", best_val_reward)

    if writer:
        writer.close()
    log.info("Training complete. Best val reward: %.4f", best_val_reward)


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="POMO REINFORCE trainer for DARP.")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--train-data-size", type=int, default=10_000)
    p.add_argument("--val-data-size", type=int, default=1_000)
    p.add_argument("--pomo-size", type=int, default=20)
    p.add_argument("--val-pomo", action="store_true", default=True,
                   help="Use POMO during validation for best-of selection.")

    # Model
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--num-encoder-layers", type=int, default=5)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--ff-hidden", type=int, default=512)
    p.add_argument("--tanh-clip", type=float, default=10.0)

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-epochs", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=300)

    # Logging & checkpoints
    p.add_argument("--log-dir", type=str, default="runs/pomo_reinforce")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/pomo_reinforce")
    p.add_argument("--init-checkpoint", type=str, default=None)

    # Device
    p.add_argument("--device", type=str, default=None)

    return p.parse_args()


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
