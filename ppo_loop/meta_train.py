"""
Main training script for DVRP-TW with PPO

Usage:
    python train.py --episodes 1000 --save-dir ./checkpoints
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from ppo_agent import PPOAgent
from dvrp_env import DVRPEnv
from vectorized_env import VectorizedDVRPEnv
from embedding import EmbeddingFFN, TravelerDataset, OnlineTravelerDataset, likelihood_loss, flexibility_personalities, n_flexibilities
from collections import deque

import wandb
wandb.login()

"""PERSONALITY_RULES = {
    # Key: The personality name (must match model output indices 0-3)
    # Value: Maximum minutes they are willing to shift
    0: {"name": "flexible for both early pickup and late dropoff",
        "max_early": 30, "max_late": 30},

    1: {"name": "flexible for early pickup, but inflexible for late dropoff",
        "max_early": 30, "max_late": 0},

    2: {"name": "flexible for late dropoff, but inflexible for early pickup",
        "max_early": 0, "max_late": 30},

    3: {"name": "inflexible for any schedule changes",
        "max_early": 0, "max_late": 0}
}
"""

# Action space mapping for mask computation
ACTION_SPACE_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (0, 0), (0, 10), (0, 20), (0, 30),
]


def compute_masks_from_flexibility(predicted_flexibility: torch.Tensor, action_dim: int = 16) -> torch.Tensor:
    """
    Compute action masks based on predicted flexibility types (VECTORIZED).

    Args:
        predicted_flexibility: Tensor of shape (num_customers,) with predicted flexibility type indices
        action_dim: Number of actions (16)

    Returns:
        masks: Tensor of shape (num_customers, action_dim) where 1 = allowed, 0 = masked

    Flexibility types (must match embedding.py order):
        0: flexible for late dropoff, but inflexible for early pickup
        1: flexible for early pickup, but inflexible for late dropoff
        2: inflexible for any schedule changes
        3: flexible for both early pickup and late dropoff
    """
    num_customers = predicted_flexibility.shape[0]

    # Pre-compute action properties once (16 actions, 2 properties each)
    action_props = torch.tensor([
        [abs(pickup), dropoff] for pickup, dropoff in ACTION_SPACE_MAP
    ], dtype=torch.float32)  # (16, 2)

    early_shifts = action_props[:, 0]  # (16,) - how much earlier pickup is
    late_shifts = action_props[:, 1]   # (16,) - how much later dropoff is

    # Initialize all actions as allowed
    masks = torch.ones(num_customers, action_dim, dtype=torch.float32)

    # Broadcast shapes for vectorized operations
    # early_shifts: (1, 16), late_shifts: (1, 16)
    # predicted_flexibility: (num_customers,) -> (num_customers, 1) for broadcasting
    early_shifts_2d = early_shifts.unsqueeze(0)  # (1, 16)
    late_shifts_2d = late_shifts.unsqueeze(0)    # (1, 16)
    flex_types = predicted_flexibility.unsqueeze(1)  # (num_customers, 1)

    # Type 0: flexible for late dropoff, inflexible for early pickup
    # Mask actions where early_shift > 0
    type0_mask = (flex_types == 0) & (early_shifts_2d > 0)  # (num_customers, 16)
    masks = torch.where(type0_mask, torch.zeros_like(masks), masks)

    # Type 1: flexible for early pickup, inflexible for late dropoff
    # Mask actions where late_shift > 0
    type1_mask = (flex_types == 1) & (late_shifts_2d > 0)
    masks = torch.where(type1_mask, torch.zeros_like(masks), masks)

    # Type 2: inflexible for any schedule changes
    # Only allow action with early_shift == 0 AND late_shift == 0
    type2_mask = (flex_types == 2) & ((early_shifts_2d > 0) | (late_shifts_2d > 0))
    masks = torch.where(type2_mask, torch.zeros_like(masks), masks)

    # Type 3: flexible for both (all actions allowed - no masking needed)

    return masks

def update_embedding_model(embedding_model, online_data, num_epochs=50, batch_size=64, lr=1e-3):
    """
    Update embedding model using online data to learn customer flexibility preferences.

    Args:
        embedding_model: EmbeddingFFN model to update
        online_data: List of dicts with keys: customer_id, action, accepted
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate

    Returns:
        Updated embedding model
    """

    # Data quality checks
    if len(online_data) < batch_size:
        print(f"  [Embedding Update] Insufficient data: {len(online_data)} < {batch_size}. Skipping update.")
        return embedding_model

    df_online = pd.DataFrame(online_data)

    # Check per-customer data availability
    customer_counts = df_online['customer_id'].value_counts()
    min_samples_per_customer = 3
    valid_customers = customer_counts[customer_counts >= min_samples_per_customer]

    if len(valid_customers) < 2:
        print(f"  [Embedding Update] Too few customers with sufficient data: {len(valid_customers)}. Skipping update.")
        return embedding_model

    # Filter to only include customers with enough samples
    df_online = df_online[df_online['customer_id'].isin(valid_customers.index)]

    # Check action diversity
    action_counts = df_online['action'].value_counts()
    if len(action_counts) < 3:
        print(f"  [Embedding Update] Low action diversity: {len(action_counts)} unique actions. Skipping update.")
        return embedding_model

    # Map customer_ids to 0-indexed embedding slots
    # customer_id from environment (1-indexed) -> embedding index (0-indexed)
    # Use customer_id - 1 as the embedding index to maintain consistent mapping
    df_online['traveler_id'] = df_online['customer_id'] - 1
    unique_customers = sorted(df_online['customer_id'].unique())

    print(f"  [Embedding Update] Training on {len(df_online)} samples from {len(unique_customers)} customers")
    print(f"  [Embedding Update] Action distribution: {action_counts.head(5).to_dict()}")

    try:
        # Use OnlineTravelerDataset which calculates consistency matrix correctly
        dataset = OnlineTravelerDataset(df_online, flexibility_personalities, ACTION_SPACE_MAP)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(embedding_model.parameters(), lr=lr)

        embedding_model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for entity_ids, _, ind_matrix, _ in dataloader:
                optimizer.zero_grad()
                pred_proba = embedding_model(entity_ids)
                beta_matrix = embedding_model.get_embed()
                loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1

            if epoch == 0 or epoch == num_epochs - 1:
                print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

        embedding_model.eval()

        # Log predictions for tracked customers
        with torch.no_grad():
            tracked_ids = torch.LongTensor([cid - 1 for cid in unique_customers[:5]])
            pred_proba = embedding_model(tracked_ids)
            predicted_types = torch.argmax(pred_proba, dim=1)
            print(f"  [Embedding Update] Sample predictions for customers {unique_customers[:5]}: {predicted_types.tolist()}")

    except Exception as e:
        print(f"  [Embedding Update] Error: {e}")
        import traceback
        traceback.print_exc()

    return embedding_model


def train(
    num_episodes: int = 30,
    num_customers: int = 30,
    num_envs: int = 64,
    save_dir: str = "./checkpoints",
    save_interval: int = 100,
    log_interval: int = 10,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Train PPO agent on DVRP environment with parallel rollouts and per-user learning.

    Uses vectorized environments for parallel data collection with batched neural oracle.
    The embedding model learns flexibility preferences for individual users based on
    their user_id from the environment.

    IMPORTANT CHANGES FROM SEQUENTIAL VERSION:
        - Runs num_envs environments in parallel (default: 64)
        - Batches neural oracle calls for massive speedup
        - Uses fixed-size buffer (64000) for embedding training data
        - Updates embedding every K total environment steps (not episodes)

    Args:
        num_episodes: Total number of episodes to run (will be divided across parallel envs)
        num_customers: Number of unique users (also requests per episode)
        num_envs: Number of parallel environments (default: 64)
        save_dir: Directory to save checkpoints
        save_interval: Save model every N epochs (1 epoch = num_envs episodes)
        log_interval: Log statistics every N epochs
        device: Device to run on ('cpu' or 'cuda')
        seed: Random seed
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Calculate number of epochs (1 epoch = num_envs parallel episodes)
    num_epochs = max(1, num_episodes // num_envs)

    # Create vectorized environment (num_envs parallel environments)
    vec_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=seed,
    )

    # Create PPO agent
    action_dim = 16
    agent = PPOAgent(
        state_dim=num_customers * 6,  # 30 * 6 = 180
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,
        device=device,
    )

    # Initialize action masks (num_customers x action_dim)
    # Mask = 1 means action is allowed, 0 means masked
    # Start with all ones (all actions allowed for all users initially)
    action_masks = torch.ones(num_customers, action_dim)

    # Initialize embedding model for learning customer preferences
    embedding_model = EmbeddingFFN(
        num_entities=num_customers,
        embed_dim=64,
        hidden_dim=128,
        output_dim=n_flexibilities
    )
    embedding_model.eval()

    # Fixed-size online data collection buffer (prevents memory growth)
    online_data = deque(maxlen=12800)

    # Epsilon for epsilon-greedy masking (starts at 0.2, decays to 0)
    initial_epsilon = 0.2
    final_epsilon = 0.0

    # Update embedding every K total environment steps
    # 10 epochs * num_envs = e.g., 10 * 64 = 640 total episodes
    steps_per_embedding_update = 10 * num_envs * num_customers  # 10 epochs worth of steps
    total_steps = 0

    # Initialize wandb
    wandb.init(
        project="rl4co",
        config={
            "num_episodes": num_episodes,
            "num_epochs": num_epochs,
            "num_envs": num_envs,
            "num_customers": num_customers,
            "device": device,
            "seed": seed,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "initial_epsilon": initial_epsilon,
            "final_epsilon": final_epsilon,
            "embedding_update_freq": steps_per_embedding_update,
        }
    )

    print("=" * 80)
    print("Training DVRP-TW with PPO (PARALLELIZED)")
    print("=" * 80)
    print(f"Total episodes: {num_episodes}")
    print(f"Parallel environments: {num_envs}")
    print(f"Training epochs: {num_epochs} (1 epoch = {num_envs} episodes)")
    print(f"Customers per episode: {num_customers}")
    print(f"Device: {device}")
    print(f"Save directory: {save_path}")
    print(f"Embedding update frequency: every {steps_per_embedding_update} steps")
    print("=" * 80)

    # Training statistics (track per epoch, aggregated over num_envs episodes)
    epoch_rewards = []
    epoch_costs = []
    epoch_accepted_rates = []
    epoch_failures = []

    # Training loop (parallel episodes)
    for epoch in range(1, num_epochs + 1):
        # Reset all environments
        states, infos = vec_env.reset()  # (num_envs, num_customers, 6)

        # Track statistics for this epoch (num_envs episodes)
        epoch_episode_rewards = []
        epoch_episode_costs = []
        epoch_episode_accepted = []
        epoch_episode_failed = []

        # Compute current epsilon (linear decay based on epoch)
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)

        # Episode loop (each step processes all num_envs environments)
        for step in range(num_customers):
            # Get user IDs for current request across all environments
            user_ids = vec_env.get_current_user_ids()  # (num_envs,)

            # Get masks for all users (batch indexing)
            # user_ids are 1-indexed, so subtract 1 for 0-indexed action_masks
            batch_masks = action_masks[user_ids - 1]  # (num_envs, action_dim)

            # Select actions for all environments in parallel
            actions = agent.select_action_batch(
                states,
                masks=batch_masks,
                epsilon=epsilon
            )  # (num_envs,)

            # Step all environments in parallel (BATCHED NEURAL ORACLE!)
            next_states, rewards, dones, truncs, step_infos = vec_env.step(actions)

            # Collect online data from all environments
            for i in range(num_envs):
                online_data.append({
                    'customer_id': user_ids[i],
                    'action': actions[i],
                    'accepted': step_infos[i].get('accepted', False)
                })

            # Store rewards for all environments
            agent.store_rewards_batch(rewards, dones | truncs)

            # Update states
            states = next_states
            total_steps += num_envs

        # Collect episode statistics from all environments
        for i in range(num_envs):
            info = step_infos[i]
            # Calculate episode reward (sum of all rewards for this env)
            # Note: we don't track per-env rewards separately in this implementation
            # Instead we use the final cost as a proxy
            cost = info.get('current_cost', float('inf'))
            failed = info.get('solver_failed', False)

            epoch_episode_costs.append(cost if not failed else float('inf'))
            epoch_episode_failed.append(1 if failed else 0)
            # Note: accepted rate is averaged across the episode in the info

        # Perform PPO update after each epoch (with data from num_envs episodes)
        train_stats = agent.update(num_value_epochs=50, num_policy_epochs=10, batch_size=64)

        # Update embedding model every K steps
        if total_steps % steps_per_embedding_update == 0 and len(online_data) > 0:
            print(f"\n[Epoch {epoch}, Step {total_steps}] Updating embedding model with {len(online_data)} samples...")

            # Convert deque to list for update
            online_data_list = list(online_data)

            # Update embedding model
            embedding_model = update_embedding_model(
                embedding_model,
                online_data_list,
                num_epochs=50,
                batch_size=min(64, len(online_data_list)),
                lr=1e-3
            )

            # Predict flexibility types for all customers
            with torch.no_grad():
                customer_ids = torch.arange(num_customers)
                pred_proba = embedding_model(customer_ids)
                predicted_flexibility = torch.argmax(pred_proba, dim=1)

            # Recompute action masks (VECTORIZED!)
            action_masks = compute_masks_from_flexibility(predicted_flexibility, action_dim)

            print(f"  Updated masks based on predicted flexibility types")
            print(f"  Flexibility distribution: {torch.bincount(predicted_flexibility, minlength=n_flexibilities).tolist()}")

        # Aggregate epoch statistics
        avg_cost = np.mean([c for c in epoch_episode_costs if not np.isinf(c)]) if any(not np.isinf(c) for c in epoch_episode_costs) else float('inf')
        failure_rate = np.mean(epoch_episode_failed)

        # Store epoch statistics
        epoch_costs.append(avg_cost)
        epoch_failures.append(failure_rate)

        # Logging
        if epoch % log_interval == 0:
            recent_costs = [c for c in epoch_costs[-log_interval:] if not np.isinf(c)]
            recent_failures = epoch_failures[-log_interval:]

            avg_recent_cost = np.mean(recent_costs) if recent_costs else float("inf")
            avg_failure_rate = np.mean(recent_failures)

            print(f"\n[Epoch {epoch}/{num_epochs}] (Episodes {epoch * num_envs}/{num_episodes})")
            print(f"  Avg Cost: {avg_recent_cost:.2f} km")
            print(f"  Failure Rate: {avg_failure_rate * 100:.1f}%")
            print(f"  Policy Loss: {train_stats.get('policy_loss', 0):.4f}")
            print(f"  Value Loss: {train_stats.get('value_loss', 0):.4f}")
            print(f"  Entropy: {train_stats.get('entropy', 0):.4f}")
            print(f"  Total Steps: {total_steps}")
            print(f"  Epsilon: {epsilon:.3f}")

            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "total_episodes": epoch * num_envs,
                "total_steps": total_steps,
                "avg_cost": avg_recent_cost,
                "failure_rate": avg_failure_rate,
                "epsilon": epsilon,
                "policy_loss": train_stats.get('policy_loss', 0),
                "value_loss": train_stats.get('value_loss', 0),
                "entropy": train_stats.get('entropy', 0),
            })

        # Save checkpoint
        if epoch % save_interval == 0:
            checkpoint_path = save_path / f"ppo_agent_ep{epoch * num_envs}.pt"
            agent.save(str(checkpoint_path))
            print(f"\n[Checkpoint saved: {checkpoint_path}]")

    # Save final model
    final_path = save_path / "ppo_agent_final.pt"
    agent.save(str(final_path))

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Final model saved: {final_path}")
    print("=" * 80)

    # Print final statistics
    valid_costs = [c for c in epoch_costs if not np.isinf(c)]
    print(f"\nFinal Statistics:")
    print(f"  Total Epochs: {num_epochs}")
    print(f"  Total Episodes: {num_epochs * num_envs}")
    print(f"  Total Steps: {total_steps}")
    print(f"  Avg Cost (final epoch): {epoch_costs[-1] if epoch_costs else 0:.2f} km")
    print(f"  Avg Cost (all epochs): {np.mean(valid_costs):.2f} km" if valid_costs else "  No valid costs")
    print(f"  Solver Failure Rate: {np.mean(epoch_failures) * 100:.1f}%")

    # Clean up
    vec_env.close()

    return agent, epoch_costs, epoch_failures


def evaluate(
    agent: PPOAgent,
    num_episodes: int = 10,
    num_customers: int = 30,
    seed: int = 999,
):
    """
    Evaluate trained agent and compare with baseline (no time shift).

    Args:
        agent: Trained PPO agent
        num_episodes: Number of evaluation episodes
        num_customers: Number of requests per episode
        seed: Random seed
    """
    print("\n" + "=" * 80)
    print("Evaluating Agent vs Baseline (No Time Shift)")
    print("=" * 80)

    # Results storage
    trained_rewards = []
    trained_costs = []
    baseline_costs = []

    for episode in range(1, num_episodes + 1):
        episode_seed = seed + episode  # Unique seed for each episode

        # --- Run with trained agent ---
        env = DVRPEnv(
            num_customers=num_customers,
            max_vehicles=5,
            solver_time_limit=1,
            seed=episode_seed,
        )
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(num_customers):
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        trained_cost = info.get("current_cost", float("inf"))
        trained_rewards.append(episode_reward)
        trained_costs.append(trained_cost)

        # --- Run with baseline (action 12 = no shift) ---
        env_baseline = DVRPEnv(
            num_customers=num_customers,
            max_vehicles=5,
            solver_time_limit=1,
            seed=episode_seed,  # Same seed for fair comparison
        )
        state_baseline, _ = env_baseline.reset()

        for step in range(num_customers):
            # Action 12 = (0, 0) = no time shift
            action_baseline = 12
            state_baseline, _, terminated, truncated, info_baseline = env_baseline.step(action_baseline)

            if terminated or truncated:
                break

        baseline_cost = info_baseline.get("current_cost", float("inf"))
        baseline_costs.append(baseline_cost)

        # Calculate improvement
        improvement = baseline_cost - trained_cost
        improvement_pct = (improvement / baseline_cost * 100) if baseline_cost > 0 else 0

        print(
            f"Episode {episode}: "
            f"Trained={trained_cost:.2f} km, "
            f"Baseline={baseline_cost:.2f} km, "
            f"Improvement={improvement:.2f} km ({improvement_pct:.1f}%)"
        )

    # Summary statistics
    avg_trained = np.mean(trained_costs)
    avg_baseline = np.mean(baseline_costs)
    avg_improvement = avg_baseline - avg_trained
    avg_improvement_pct = (avg_improvement / avg_baseline * 100) if avg_baseline > 0 else 0

    print("\n" + "=" * 80)
    print("Evaluation Summary:")
    print(f"  Trained Agent Avg Cost:  {avg_trained:.2f} km")
    print(f"  Baseline Avg Cost:       {avg_baseline:.2f} km")
    print(f"  Average Improvement:     {avg_improvement:.2f} km ({avg_improvement_pct:.1f}%)")
    print(f"  Avg Reward (trained):    {np.mean(trained_rewards):.2f}")
    print("=" * 80)

    # Log evaluation metrics to wandb
    wandb.log({
        "eval/trained_avg_cost": avg_trained,
        "eval/baseline_avg_cost": avg_baseline,
        "eval/improvement_km": avg_improvement,
        "eval/improvement_pct": avg_improvement_pct,
        "eval/avg_reward": np.mean(trained_rewards),
    })

    # Log evaluation summary table
    eval_data = [[i+1, trained_costs[i], baseline_costs[i], baseline_costs[i] - trained_costs[i]]
                 for i in range(len(trained_costs))]
    eval_table = wandb.Table(
        data=eval_data,
        columns=["Episode", "Trained Cost", "Baseline Cost", "Improvement"]
    )
    wandb.log({"eval/episode_comparison": eval_table})

    # Finish wandb run
    wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train PPO agent for DVRP-TW (Parallelized)")

    parser.add_argument(
        "--episodes", type=int, default=1000, help="Total number of training episodes"
    )
    parser.add_argument(
        "--customers", type=int, default=30, help="Number of customers per episode"
    )
    parser.add_argument(
        "--num-envs", type=int, default=64, help="Number of parallel environments"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save model every N epochs (1 epoch = num_envs episodes)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1, help="Log stats every N epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--eval", action="store_true", help="Run evaluation after training"
    )

    args = parser.parse_args()

    # Train
    agent, costs, failures = train(
        num_episodes=args.episodes,
        num_customers=args.customers,
        num_envs=args.num_envs,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
    )

    # Evaluate
    if args.eval:
        evaluate(agent, num_episodes=10, num_customers=args.customers)
    else:
        # Finish wandb if no evaluation
        wandb.finish()


if __name__ == "__main__":
    main()
