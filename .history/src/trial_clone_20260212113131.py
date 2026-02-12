"""
Behavior Cloning for DARP using GNN Policy (DARPPolicy).

Trains the GREAT-encoder + attention-decoder policy via supervised learning
on expert action sequences. Adapted from policy_clone.py which uses the
simpler PDPTWAttentionPolicy.

Usage::

    python src/trial_clone.py --dataset-path dataset/size_40000.pt --epochs 50
"""

import argparse
import logging
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator
from trial_gnn_policy import DARPPolicy


log = logging.getLogger(__name__)


def get_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        dataset = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        dataset = torch.load(path, map_location="cpu")
    return dataset


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def train(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    log.info("Using device: %s", device)

    dataset_path = Path(args.dataset_path).expanduser()
    dataset = load_dataset(dataset_path)
    if "expert_actions" not in dataset.keys():
        raise KeyError("Dataset is missing required key: expert_actions")

    num_instances = dataset.batch_size[0] if hasattr(dataset, "batch_size") else dataset["expert_actions"].shape[0]
    log.info("Loaded dataset with %d instances from %s", num_instances, dataset_path)

    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(csv_path=csv_path, travel_time_matrix_path=ttm_path)
    env = PDPTWEnv(generator=generator, vehicle_penalty=0)

    policy = DARPPolicy(
        embed_dim=args.embed_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_heads=args.num_heads,
        ff_hidden=args.ff_hidden,
        tanh_clip=args.tanh_clip,
        temperature=args.temperature,
    ).to(device)

    log.info("Policy parameters: %d", sum(p.numel() for p in policy.parameters()))

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    num_batches = (num_instances + args.batch_size - 1) // args.batch_size
    best_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    start_epoch = 0

    if args.resume:
        checkpoint_path = Path(args.resume_path).expanduser() if args.resume_path else checkpoint_dir / "latest.pt"
        checkpoint = load_checkpoint(checkpoint_path, device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", best_loss)))
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        log.info("Resuming from %s (epoch %d)", checkpoint_path, start_epoch + 1)

    for epoch in range(start_epoch, args.epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_valid = 0
        epoch_invalid = 0

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}", unit="batch"):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, num_instances)

            batch = dataset[start_idx:end_idx].to(device)
            actions = batch["expert_actions"]

            state = env.reset(batch)
            outputs = policy(
                state,
                env,
                phase="train",
                actions=actions,
                max_steps=actions.size(-1),
                return_actions=False,
            )

            log_likelihood = outputs["log_likelihood"]
            valid_mask = torch.isfinite(log_likelihood)
            valid_count = int(valid_mask.sum().item())
            invalid_count = int(valid_mask.numel() - valid_count)
            epoch_valid += valid_count
            epoch_invalid += invalid_count
            if valid_count == 0:
                continue

            loss = -log_likelihood[valid_mask].mean()

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, num_batches)
        train_losses.append(avg_loss)
        log.info(
            "Epoch %d/%d - NLL: %.4f (valid=%d, invalid=%d)",
            epoch + 1,
            args.epochs,
            avg_loss,
            epoch_valid,
            epoch_invalid,
        )

        is_new_best = avg_loss < best_loss
        if is_new_best:
            best_loss = avg_loss

        checkpoint_payload = {
            "epoch": epoch,
            "loss": avg_loss,
            "best_loss": best_loss,
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint_payload, checkpoint_dir / "latest.pt")
        if is_new_best:
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")

    log.info("Training complete. Best NLL: %.4f", best_loss)
    if args.plot_path:
        plot_path = Path(args.plot_path).expanduser()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(8, 4.5))
        plt.plot(range(1 + start_epoch, args.epochs + 1), train_losses, marker="o", linewidth=1.5)
        plt.title("GNN Policy Cloning Training Curve")
        plt.xlabel("Epoch")
        plt.ylabel("NLL (lower is better)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        log.info("Saved training curve to %s", plot_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior cloning for DARP using GNN policy (DARPPolicy).")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="dataset/size_40000.pt",
        help="Path to the instance-action dataset.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm.")

    # GNN Policy architecture
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--num-encoder-layers", type=int, default=5, help="Number of GREAT encoder layers.")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--ff-hidden", type=int, default=512, help="Feed-forward hidden dimension.")
    parser.add_argument("--tanh-clip", type=float, default=10.0, help="Logit clipping parameter.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Softmax temperature.")

    parser.add_argument(
        "--plot-path",
        type=str,
        default="runs/gnn_clone/train_curve.png",
        help="Path to save training curve (PNG).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/gnn_clone",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from a checkpoint.",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from (defaults to checkpoint-dir/latest.pt).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/mps).")
    return parser.parse_args()


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
