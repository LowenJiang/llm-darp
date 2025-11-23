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

from agents.ppo_agent import PPOAgent
from env.dvrp_env import DVRPEnv


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
    Train PPO agent on DVRP environment.

    Args:
        num_episodes: Number of training episodes
        num_customers: Number of requests per episode
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
        vehicle_capacity=4,
        max_vehicles=5,
        solver_time_limit=1,
        acceptance_rate=0.5,
        seed=seed,
    )

    # Create PPO agent
    agent = PPOAgent(
        state_dim=num_customers * 8,  # 30 * 8 = 240
        action_dim=16,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device=device,
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
    
    ## TODO: Initialize a matrix of mask (dimension is num_customers x action_dim), where each row corresponds to the action mask of a customer. The entry of the mask is 1 if LLM predicts that the customer will accept the time shift, otherwise it is 0

    # Training loop
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        failed = False

        # Episode loop
        for step in range(num_customers):
            # Select action
            action = agent.select_action(state)

            ## TODO: Make sure that the state records which customer ID is currently making request
            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)

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

        # Store final cost (if not failed)
        if not failed and episode_length > 0:
            episode_costs.append(step_info.get("current_cost", 0.0))
        else:
            episode_costs.append(float("inf"))
        
        ## TODO: Every 10 policy iterations of PPO, call the embedding.py to recompute the mask

        # Perform PPO update
        if episode % 1 == 0:  # Update every episode
            train_stats = agent.update(num_epochs=10, batch_size=64)
        else:
            train_stats = {}

        # Logging
        if episode % log_interval == 0:
            recent_rewards = episode_rewards[-log_interval:]
            recent_costs = [c for c in episode_costs[-log_interval:] if not np.isinf(c)]
            recent_failures = solver_failures[-log_interval:]

            avg_reward = np.mean(recent_rewards)
            avg_cost = np.mean(recent_costs) if recent_costs else float("inf")
            failure_rate = np.mean(recent_failures)

            print(f"\n[Episode {episode}/{num_episodes}]")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Cost: {avg_cost:.2f} km")
            print(f"  Failure Rate: {failure_rate * 100:.1f}%")

            if train_stats:
                print(f"  Policy Loss: {train_stats.get('policy_loss', 0):.4f}")
                print(f"  Value Loss: {train_stats.get('value_loss', 0):.4f}")
                print(f"  Entropy: {train_stats.get('entropy', 0):.4f}")

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
    print(f"  Solver Failure Rate: {np.mean(solver_failures) * 100:.1f}%")

    return agent, episode_rewards, episode_costs


def evaluate(
    agent: PPOAgent,
    num_episodes: int = 10,
    num_customers: int = 30,
    seed: int = 999,
):
    """
    Evaluate trained agent.

    Args:
        agent: Trained PPO agent
        num_episodes: Number of evaluation episodes
        num_customers: Number of requests per episode
        seed: Random seed
    """
    print("\n" + "=" * 80)
    print("Evaluating Agent")
    print("=" * 80)

    env = DVRPEnv(
        num_customers=num_customers,
        vehicle_capacity=4,
        max_vehicles=5,
        solver_time_limit=1,
        acceptance_rate=0.5,
        seed=seed,
    )

    episode_rewards = []
    episode_costs = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for step in range(num_customers):
            action = agent.select_action(state)
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_costs.append(info.get("current_cost", float("inf")))

        print(
            f"Episode {episode}: Reward={episode_reward:.2f}, Cost={info.get('current_cost', 'N/A')}"
        )

    print("\n" + "=" * 80)
    print(f"Evaluation Results:")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Avg Cost: {np.mean(episode_costs):.2f} km")
    print("=" * 80)


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


if __name__ == "__main__":
    main()
