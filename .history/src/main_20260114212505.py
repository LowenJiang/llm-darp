import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm
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


def generate_dataset(generator, size: int, device: torch.device):
    """Generate a dataset of problem instances."""
    log.info(f"Generating dataset with {size} instances...")
    return generator(batch_size=[size]).to(device)


def train_epoch(policy, env, trainer, optimizer, train_data, batch_size, max_steps, decode_type, grad_clip, device):
    """Train for one epoch on the training dataset."""
    policy.train()
    num_instances = train_data.batch_size[0] if hasattr(train_data, 'batch_size') else train_data.shape[0]
    num_batches = (num_instances + batch_size - 1) // batch_size

    epoch_loss = 0.0
    epoch_reward = 0.0
    epoch_vehicles = 0.0

    for batch_idx in tqdm(range(num_batches), desc="Train batches", unit="batch", leave=False):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_instances)

        # Get batch slice
        batch = train_data[start_idx:end_idx]

        # IMPORTANT: batch now contains "extra" field with precomputed baseline values
        # The REINFORCE.calculate_loss() will use these cached values instead of
        # running expensive greedy rollouts for each batch

        state = env.reset(batch)

        # Forward pass
        outputs = policy(
            state,
            env,
            phase="train",
            decode_type=decode_type,
            max_steps=max_steps,
        )

        # Calculate loss (will use batch["extra"] for baseline if available)
        outputs = trainer.calculate_loss(_clone_state(state), batch, outputs)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        # Track metrics
        epoch_loss += loss.detach().item()
        epoch_reward += outputs["reward"].detach().mean().item()
        if "vehicles_used" in outputs:
            epoch_vehicles += outputs["vehicles_used"].detach().float().mean().item()

    # Average over batches
    return {
        "loss": epoch_loss / num_batches,
        "reward": epoch_reward / num_batches,
        "vehicles_used": epoch_vehicles / num_batches if epoch_vehicles > 0 else 0,
    }


def validate(policy, env, val_data, batch_size, max_steps, device):
    """Validate on validation dataset using greedy decoding."""
    policy.eval()
    num_instances = val_data.batch_size[0] if hasattr(val_data, 'batch_size') else val_data.shape[0]
    num_batches = (num_instances + batch_size - 1) // batch_size

    val_reward = 0.0
    val_vehicles = 0.0

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_instances)

            batch = val_data[start_idx:end_idx]
            state = env.reset(batch)

            # Greedy rollout for validation
            outputs = policy(
                state,
                env,
                phase="val",
                decode_type="greedy",
                max_steps=max_steps,
            )

            val_reward += outputs["reward"].detach().mean().item()
            if "vehicles_used" in outputs:
                val_vehicles += outputs["vehicles_used"].detach().float().mean().item()

    return {
        "reward": val_reward / num_batches,
        "vehicles_used": val_vehicles / num_batches if val_vehicles > 0 else 0,
    }


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    log.info("Using device: %s", device)

    # Setup checkpoint directory
    checkpoints_dir = Path(args.checkpoint_dir).expanduser()
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generator and environment
    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(csv_path=csv_path, travel_time_matrix_path=ttm_path)
    env = PDPTWEnv(generator=generator)

    # Initialize policy
    policy = PDPTWAttentionPolicy(
        embed_dim=128,
        num_encoder_layers=3,
        num_heads=8,
        feedforward_hidden=512
    )
    policy.to(device)

    # Initialize REINFORCE trainer with rollout baseline
    trainer = REINFORCE(env, policy, baseline="rollout")

    # Setup baseline (creates initial baseline policy and evaluation dataset)
    log.info("Setting up baseline...")
    trainer.post_setup_hook(
        batch_size=args.batch_size,
        device=device,
        dataset_size=args.val_data_size,
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Generate initial validation dataset (reused across epochs)
    val_data = generate_dataset(generator, args.val_data_size, device)

    best_val_reward = float("-inf")

    # Training loop: epoch-based like pdptw_testing.ipynb
    for epoch in range(args.epochs):
        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch + 1}/{args.epochs}")
        log.info(f"{'='*60}")

        # Generate NEW training dataset for this epoch (prevents overfitting)
        train_data = generate_dataset(generator, args.train_data_size, device)

        # Precompute baseline values for training data
        # This runs greedy rollouts ONCE per epoch and caches results in dataset["extra"]
        log.info("Precomputing baseline values for training data...")
        train_data = trainer.wrap_dataset(
            train_data,
            batch_size=args.batch_size,
            device=device
        )

        # Train for one epoch
        train_metrics = train_epoch(
            policy, env, trainer, optimizer, train_data,
            args.batch_size, args.max_steps, args.decode_type,
            args.grad_clip, device
        )

        # Log training metrics
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", value, epoch)

        log.info(
            f"Train - loss: {train_metrics['loss']:.4f}, "
            f"reward: {train_metrics['reward']:.4f}, "
            f"vehicles: {train_metrics['vehicles_used']:.2f}"
        )

        # Validation
        log.info("Running validation...")
        val_metrics = validate(policy, env, val_data, args.batch_size, args.max_steps, device)

        # Log validation metrics
        for key, value in val_metrics.items():
            writer.add_scalar(f"val/{key}", value, epoch)

        log.info(
            f"Val   - reward: {val_metrics['reward']:.4f}, "
            f"vehicles: {val_metrics['vehicles_used']:.2f}"
        )

        # Update baseline at end of epoch (with t-test comparison)
        log.info("Updating rollout baseline...")
        trainer.baseline.epoch_callback(
            policy,
            env,
            batch_size=args.batch_size,
            device=device,
            epoch=epoch,
            dataset_size=args.val_data_size,
        )

        # Save checkpoints
        checkpoint_payload = {
            "epoch": epoch,
            "train_reward": train_metrics["reward"],
            "val_reward": val_metrics["reward"],
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        # Always save latest
        torch.save(checkpoint_payload, checkpoints_dir / "latest.pt")

        # Save best based on validation reward
        if val_metrics["reward"] > best_val_reward:
            best_val_reward = val_metrics["reward"]
            torch.save(checkpoint_payload, checkpoints_dir / "best.pt")
            log.info(f"✓ New best validation reward: {best_val_reward:.4f}")

    writer.close()
    log.info("\nTraining complete!")
    log.info(f"Best validation reward: {best_val_reward:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PDPTW policy with REINFORCE (epoch-based).")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for training/validation.")
    parser.add_argument("--train-data-size", type=int, default=100_000, help="Training instances per epoch.")
    parser.add_argument("--val-data-size", type=int, default=10_000, help="Validation dataset size.")

    # Optimization
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm.")

    # Decoding
    parser.add_argument("--max-steps", type=int, default=300, help="Max decode steps.")
    parser.add_argument("--decode-type", type=str, default="sampling", help="Training decode strategy.")

    # Logging and checkpoints
    parser.add_argument("--log-dir", type=str, default="runs/pdptw_train", help="TensorBoard log dir.")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/pdptw", help="Checkpoint dir.")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()