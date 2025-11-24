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
from embedding import EmbeddingFFN, TravelerDataset, OnlineTravelerDataset, likelihood_loss, flexibility_personalities, n_flexibilities

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
    Compute action masks based on predicted flexibility types.

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
    masks = torch.ones(num_customers, action_dim)

    for i in range(num_customers):
        flex_type = predicted_flexibility[i].item()

        for action_idx, (pickup_shift, dropoff_shift) in enumerate(ACTION_SPACE_MAP):
            # Check if action should be masked based on flexibility type
            early_shift = abs(pickup_shift)  # How much earlier pickup is (pickup_shift is negative)
            late_shift = dropoff_shift  # How much later dropoff is

            if flex_type == 0:  # Flexible for late dropoff, but inflexible for early pickup
                # Mask actions with early pickup > 0
                if early_shift > 0:
                    masks[i, action_idx] = 0
            elif flex_type == 1:  # Flexible for early pickup, but inflexible for late dropoff
                # Mask actions with late dropoff > 0
                if late_shift > 0:
                    masks[i, action_idx] = 0
            elif flex_type == 2:  # Inflexible for any schedule changes
                # Only allow no-shift action (action 12: (0, 0))
                if early_shift > 0 or late_shift > 0:
                    masks[i, action_idx] = 0
            elif flex_type == 3:  # Flexible for both early pickup and late dropoff
                # All actions allowed
                pass

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
    save_dir: str = "./checkpoints",
    save_interval: int = 100,
    log_interval: int = 10,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Train PPO agent on DVRP environment with per-user learning.

    The embedding model learns flexibility preferences for individual users based on
    their user_id from the environment. Users with the same user_id are treated as
    the same person across episodes.

    IMPORTANT ASSUMPTION:
        - Environment must provide user_id in range [0, num_customers-1]
        - Same user_id = same person with consistent flexibility preferences
        - No modulo mapping is applied - user_ids are used directly

    Args:
        num_episodes: Number of training episodes
        num_customers: Number of unique users (also requests per episode)
        save_dir: Directory to save checkpoints
        save_interval: Save model every N episodes
        log_interval: Log statistics every N episodes
        device: Device to run on ('cpu' or 'cuda')
        seed: Random seed
    """
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env = DVRPEnv(
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

    # Online data collection for embedding model
    online_data = []

    # Epsilon for epsilon-greedy masking (starts at 0.2, decays to 0)
    initial_epsilon = 0.2
    final_epsilon = 0.0

    # Initialize wandb
    wandb.init(
        project="rl4co",
        config={
            "num_episodes": num_episodes,
            "num_customers": num_customers,
            "device": device,
            "seed": seed,
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "initial_epsilon": initial_epsilon,
            "final_epsilon": final_epsilon,
        }
    )

    print("=" * 80)
    print("Training DVRP-TW with PPO")
    print("=" * 80)
    print(f"Episodes: {num_episodes}")
    print(f"Customers per episode: {num_customers}")
    print(f"Device: {device}")
    print(f"Save directory: {save_path}")
    print("=" * 80)

    # Training statistics
    episode_rewards = []
    episode_costs = []
    episode_lengths = []
    solver_failures = []
    episode_accepted_rates = []

    # Training loop
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        failed = False
        accepted_count = 0

        # Compute current epsilon (linear decay from initial to final)
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (episode - 1) / max(num_episodes - 1, 1)

        # Episode loop
        for step in range(num_customers):
            # Get actual user_id from environment
            current_customer_id = env.get_current_user_id()

            # Get mask for current customer using actual user_id
            # Assumes user_id is in valid range [0, num_customers-1]
            customer_mask = action_masks[current_customer_id-1].to(device)

            # Select action with mask and epsilon
            action = agent.select_action(state, mask=customer_mask, epsilon=epsilon)

            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)

            # Collect online data for embedding model update
            accepted = step_info.get("accepted", False)
            if accepted:
                accepted_count += 1

            # Use actual user_id from environment if available, otherwise use step index
            user_id = step_info.get("user_id")
            if user_id is None:
                user_id = current_customer_id

            online_data.append({
                'customer_id': user_id,
                'action': action,
                'accepted': accepted
            })

            # Store reward
            agent.store_reward(reward, terminated or truncated)

            # Update statistics
            episode_reward += reward
            episode_length += 1

            # Check for solver failure
            if step_info.get("solver_failed", False):
                failed = True
                break

            # Update state
            state = next_state

            if terminated or truncated:
                break

        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        solver_failures.append(1 if failed else 0)
        accepted_rate = accepted_count / episode_length if episode_length > 0 else 0.0
        episode_accepted_rates.append(accepted_rate)

        # Store final cost (if not failed)
        if not failed and episode_length > 0:
            episode_costs.append(step_info.get("current_cost", 0.0))
        else:
            episode_costs.append(float("inf"))

        # Perform PPO update
        if episode % 1 == 0:  # Update every episode
            train_stats = agent.update(num_epochs=10, batch_size=64)
        else:
            train_stats = {}

        # Every 10 policy iterations, update embedding model and recompute masks
        if episode % 10 == 0 and len(online_data) > 0:
            print(f"\n[Episode {episode}] Updating embedding model with {len(online_data)} samples...")

            # Update embedding model with collected online data
            embedding_model = update_embedding_model(
                embedding_model,
                online_data,
                num_epochs=50,
                batch_size=min(64, len(online_data)),
                lr=1e-3
            )

            # Predict flexibility types for all customers (user_ids 0 to num_customers-1)
            # Note: This assumes user_ids are in range [0, num_customers-1]
            with torch.no_grad():
                # Predict for all possible user_ids
                customer_ids = torch.arange(num_customers)
                pred_proba = embedding_model(customer_ids)
                predicted_flexibility = torch.argmax(pred_proba, dim=1)

            # Recompute action masks based on predicted flexibility
            # Each row corresponds to user_id (0-indexed)
            action_masks = compute_masks_from_flexibility(predicted_flexibility, action_dim)

            print(f"  Updated masks based on predicted flexibility types")
            print(f"  Flexibility distribution: {torch.bincount(predicted_flexibility, minlength=n_flexibilities).tolist()}")

        # Logging
        if episode % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_costs = [c for c in episode_costs[-log_interval:] if not np.isinf(c)]
            recent_failures = solver_failures[-log_interval:]
            recent_accepted_rates = episode_accepted_rates[-log_interval:]

            avg_reward = np.mean(recent_rewards)
            avg_cost = np.mean(recent_costs) if recent_costs else float("inf")
            failure_rate = np.mean(recent_failures)
            avg_accepted_rate = np.mean(recent_accepted_rates)

            print(f"\n[Episode {episode}/{num_episodes}]")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Cost: {avg_cost:.2f} km")
            print(f"  Accepted Rate: {avg_accepted_rate * 100:.1f}%")
            print(f"  Failure Rate: {failure_rate * 100:.1f}%")

            if train_stats:
                print(f"  Policy Loss: {train_stats.get('policy_loss', 0):.4f}")
                print(f"  Value Loss: {train_stats.get('value_loss', 0):.4f}")
                print(f"  Entropy: {train_stats.get('entropy', 0):.4f}")

            # Log to wandb
            wandb.log({
                "episode": episode,
                "avg_reward": avg_reward,
                "avg_cost": avg_cost,
                "accepted_rate": avg_accepted_rate,
                "failure_rate": failure_rate,
                "epsilon": epsilon,
                "policy_loss": train_stats.get('policy_loss', 0) if train_stats else 0,
                "value_loss": train_stats.get('value_loss', 0) if train_stats else 0,
                "entropy": train_stats.get('entropy', 0) if train_stats else 0,
            })

        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = save_path / f"ppo_agent_ep{episode}.pt"
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
    valid_costs = [c for c in episode_costs if not np.isinf(c)]
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Avg Cost: {np.mean(valid_costs):.2f} km")
    print(f"  Avg Accepted Rate: {np.mean(episode_accepted_rates) * 100:.1f}%")
    print(f"  Solver Failure Rate: {np.mean(solver_failures) * 100:.1f}%")

    return agent, episode_rewards, episode_costs


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
    parser = argparse.ArgumentParser(description="Train PPO agent for DVRP-TW")

    parser.add_argument(
        "--episodes", type=int, default=1000, help="Number of training episodes"
    )
    parser.add_argument(
        "--customers", type=int, default=30, help="Number of customers per episode"
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
        default=100,
        help="Save model every N episodes",
    )
    parser.add_argument(
        "--log-interval", type=int, default=10, help="Log stats every N episodes"
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
    agent, rewards, costs = train(
        num_episodes=args.episodes,
        num_customers=args.customers,
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
