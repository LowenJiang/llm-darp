"""
POMO REINFORCE Trainer for DARP.

Implements REINFORCE with POMO (Policy Optimisation with Multiple Optima)
baseline for training the GNN-based DARP policy.

Key features
------------
- **POMO baseline**: runs ``pomo_size`` rollouts per instance and uses the
  mean reward across rollouts as the baseline, which greatly reduces
  variance without requiring a separate baseline network.
- **Shared baseline**: advantage = reward_i - mean(rewards across POMO rollouts)
- **Gradient clipping**: configurable norm clipping for stability.
- **Warm-up**: optional linear learning-rate warm-up over the first few
  epochs.

Usage::

    python src/trial_pomo_reinforce.py --epochs 50 --pomo-size 20

The trainer is compatible with ``PDPTWEnv`` and ``DARPPolicy`` from the
trial_* modules.
"""

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

log = logging.getLogger(__name__)


# =========================================================================
# POMO REINFORCE Trainer
# =========================================================================

class POMOReinforce:
    """REINFORCE trainer using POMO shared baseline.

    Parameters
    ----------
    env : PDPTWEnv
        The DARP environment.
    policy : DARPPolicy
        The GNN policy to train.
    pomo_size : int
        Number of rollouts per instance (default 20).
    lr : float
        Learning rate.
    weight_decay : float
        L2 regularisation.
    grad_clip : float
        Maximum gradient norm (0 = no clipping).
    warmup_epochs : int
        Number of epochs for linear LR warm-up.
    max_steps : int
        Maximum decoding steps per episode.
    temperature : float
        Softmax temperature during training.
    device : torch.device or str
        Compute device.
    """

    def __init__(
        self,
        env,
        policy: nn.Module,
        pomo_size: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
        warmup_epochs: int = 1,
        max_steps: int = 300,
        temperature: float = 1.0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.env = env
        self.policy = policy.to(device)
        self.pomo_size = pomo_size
        self.lr = lr
        self.grad_clip = grad_clip
        self.warmup_epochs = warmup_epochs
        self.max_steps = max_steps
        self.temperature = temperature
        self.device = torch.device(device) if isinstance(device, str) else device

        self.policy.temperature = temperature

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # LR scheduler: linear warm-up then cosine
        self._base_lr = lr

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train_epoch(
        self,
        train_data: TensorDict,
        batch_size: int = 64,
        epoch: int = 0,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Parameters
        ----------
        train_data : TensorDict [num_instances, ...]
        batch_size : int
        epoch : int
            Current epoch number (for LR scheduling).

        Returns
        -------
        dict with keys: loss, reward_mean, reward_std, advantage_mean
        """
        self.policy.train()
        self._adjust_lr(epoch)

        num_instances = train_data.batch_size[0]
        num_batches = (num_instances + batch_size - 1) // batch_size

        epoch_loss = 0.0
        epoch_reward = 0.0
        epoch_reward_sq = 0.0
        epoch_advantage = 0.0
        total_instances = 0

        for batch_idx in tqdm(
            range(num_batches), desc=f"Epoch {epoch}", unit="batch", leave=False,
        ):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_instances)
            batch = train_data[start:end].to(self.device)
            actual_B = end - start

            # Reset environment
            state = self.env.reset(batch)

            # Forward pass with POMO
            outputs = self.policy(
                state,
                self.env,
                phase="train",
                pomo_size=self.pomo_size,
                decode_type="sampling",
                max_steps=self.max_steps,
                calc_reward=True,
                select_best=False,
            )

            reward = outputs["reward"]                    # [B*P]
            log_likelihood = outputs["log_likelihood"]    # [B*P]

            # Reshape for POMO baseline: [B, P]
            reward_2d = reward.view(actual_B, self.pomo_size)
            ll_2d     = log_likelihood.view(actual_B, self.pomo_size)

            # Shared baseline: mean across POMO rollouts
            baseline = reward_2d.mean(dim=1, keepdim=True)   # [B, 1]
            advantage = reward_2d - baseline                  # [B, P]

            # REINFORCE loss: -E[advantage * log_likelihood]
            reinforce_loss = -(advantage.detach() * ll_2d).mean()

            # Backward
            self.optimizer.zero_grad()
            reinforce_loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.grad_clip
                )
            self.optimizer.step()

            # Metrics
            epoch_loss += reinforce_loss.detach().item() * actual_B
            mean_r = reward_2d.mean().item()
            epoch_reward += mean_r * actual_B
            epoch_reward_sq += (reward_2d ** 2).mean().item() * actual_B
            epoch_advantage += advantage.abs().mean().item() * actual_B
            total_instances += actual_B

        avg_reward = epoch_reward / total_instances
        avg_reward_sq = epoch_reward_sq / total_instances
        return {
            "loss": epoch_loss / total_instances,
            "reward_mean": avg_reward,
            "reward_std": (avg_reward_sq - avg_reward ** 2) ** 0.5,
            "advantage_mean": epoch_advantage / total_instances,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def validate(
        self,
        val_data: TensorDict,
        batch_size: int = 64,
        pomo_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Validate with greedy decoding (optionally with POMO best-of).

        Parameters
        ----------
        val_data : TensorDict [num_instances, ...]
        batch_size : int
        pomo_size : int or None
            If given, use POMO and select the best rollout per instance.

        Returns
        -------
        dict with keys: reward_mean, reward_std, reward_best_mean
        """
        self.policy.eval()
        use_pomo = pomo_size is not None and pomo_size > 1

        num_instances = val_data.batch_size[0]
        num_batches = (num_instances + batch_size - 1) // batch_size

        all_rewards = []
        all_best_rewards = []

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_instances)
            batch = val_data[start:end].to(self.device)
            actual_B = end - start

            state = self.env.reset(batch)

            if use_pomo:
                # POMO greedy with best selection
                outputs = self.policy(
                    state,
                    self.env,
                    phase="val",
                    pomo_size=pomo_size,
                    decode_type="greedy",
                    max_steps=self.max_steps,
                    calc_reward=True,
                    select_best=False,
                )
                reward = outputs["reward"].view(actual_B, pomo_size)
                all_rewards.append(reward.mean(dim=1))
                all_best_rewards.append(reward.max(dim=1).values)
            else:
                outputs = self.policy(
                    state,
                    self.env,
                    phase="val",
                    decode_type="greedy",
                    max_steps=self.max_steps,
                    calc_reward=True,
                )
                all_rewards.append(outputs["reward"])

        rewards_cat = torch.cat(all_rewards, dim=0)
        result = {
            "reward_mean": rewards_cat.mean().item(),
            "reward_std": rewards_cat.std().item(),
        }
        if all_best_rewards:
            best_cat = torch.cat(all_best_rewards, dim=0)
            result["reward_best_mean"] = best_cat.mean().item()
        return result

    # -----------------------------------------------------------------
    # Save / Load
    # -----------------------------------------------------------------

    def save(self, path: str | Path, epoch: int = 0, extra: dict = None) -> None:
        """Save a training checkpoint.

        Parameters
        ----------
        path : str or Path
        epoch : int
        extra : dict
            Additional metadata to store.
        """
        payload = {
            "epoch": epoch,
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "pomo_size": self.pomo_size,
        }
        if extra:
            payload.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        log.info("Checkpoint saved to %s", path)

    def load(self, path: str | Path, map_location: str | torch.device = None) -> dict:
        """Load a training checkpoint.

        Parameters
        ----------
        path : str or Path
        map_location : device

        Returns
        -------
        dict  -- the full checkpoint payload
        """
        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        log.info("Loaded checkpoint from %s (epoch %d)", path, checkpoint.get("epoch", -1))
        return checkpoint

    # -----------------------------------------------------------------
    # LR scheduling
    # -----------------------------------------------------------------

    def _adjust_lr(self, epoch: int) -> None:
        """Apply linear warm-up for the first ``warmup_epochs``."""
        if epoch < self.warmup_epochs and self.warmup_epochs > 0:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            factor = 1.0
        for pg in self.optimizer.param_groups:
            pg["lr"] = self._base_lr * factor


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


# =========================================================================
# Main training entry point
# =========================================================================

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
