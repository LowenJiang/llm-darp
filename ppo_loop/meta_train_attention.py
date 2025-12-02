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

from ppo_agent_v2 import PPOAgent
from dvrp_env import DVRPEnv
from vectorized_env import VectorizedDVRPEnv
from embedding import (
    EmbeddingFFN, flexibility_personalities, n_flexibilities,
    update_embedding_model
)
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


def create_eval_envs(
    num_eval_envs: int = 8,
    num_customers: int = 30,
    eval_seed: int = 9999,
    traveler_decisions_path: Path = None,
    device: str = "cpu"
):
    """
    Create fixed evaluation environment for consistent testing across epochs.

    Generates problem instances once and stores them for reuse across all evaluations.
    Both agent and baseline will be evaluated on the same environment sequentially.

    Args:
        num_eval_envs: Number of parallel evaluation environments
        num_customers: Number of customers perc episode
        eval_seed: Fixed seed for reproducible evaluation
        traveler_decisions_path: Path to traveler decisions CSV
        device: Device to run on

    Returns:
        eval_env: VectorizedDVRPEnv for evaluations
        eval_initial_states: Initial states to reset to
    """
    eval_env = VectorizedDVRPEnv(
        num_envs=num_eval_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=eval_seed,
        traveler_decisions_path=traveler_decisions_path,
        device=device,
    )

    # Generate fixed problem instances once
    eval_initial_states, _ = eval_env.reset()

    return eval_env, eval_initial_states


def evaluate_epoch(
    agent: PPOAgent,
    embedding_model: EmbeddingFFN,
    eval_env: VectorizedDVRPEnv,
    eval_initial_states: np.ndarray,
    epoch: int,
    num_customers: int = 30,
    flexibility_personalities: list = None,
    device: str = "cpu"
) -> dict:
    """
    Evaluate agent on fixed test environments using greedy rollout.

    Runs both agent and baseline on the same environment sequentially.
    Reuses the exact same problem instances generated at startup for consistent evaluation.

    Args:
        agent: Trained PPO agent
        embedding_model: Embedding model for predicting flexibility
        eval_env: Fixed evaluation environment
        eval_initial_states: Fixed initial states
        epoch: Current training epoch
        num_customers: Number of customers per episode
        flexibility_personalities: List of flexibility type names
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    num_eval_envs = eval_env.num_envs

    # ========== AGENT ROLLOUT ==========
    # Restore to the SAME initial states (no reset, just restore the saved states)
    agent_states = eval_initial_states.copy()

    # Manually reset environment step counters and internal state
    for i in range(num_eval_envs):
        eval_env.envs[i].current_request_index = 0

    # Track statistics
    agent_accepted_counts = np.zeros(num_eval_envs)

    # Greedy rollout with masking (epsilon=0 means always apply masks)
    for step in range(num_customers):
        # Get user IDs
        user_ids = eval_env.get_current_user_ids()

        # Predict flexibility and compute masks
        with torch.no_grad():
            user_embedding_ids = torch.LongTensor(user_ids) - 1
            pred_proba = embedding_model(user_embedding_ids)
            dist = torch.distributions.Categorical(probs=pred_proba)
            predicted_flexibilities = dist.sample()
            masks = eval_env.get_masks(user_ids, predicted_flexibilities)
            # epsilon=0 for evaluation, so always apply the predicted masks

        # Agent actions selected according to policy probabilities (with masking)
        agent_actions = agent.select_action_batch(
            agent_states,
            masks=masks
        )

        # Step agent environment (no baseline marginal costs needed for eval)
        agent_next_states, _, _, _, agent_infos = eval_env.step(agent_actions, baseline_marginal_costs=None)

        # Track acceptance statistics
        for i in range(num_eval_envs):
            if agent_infos[i].get('accepted', False):
                agent_accepted_counts[i] += 1

        # Update states
        agent_states = agent_next_states

    # Collect agent costs from all environments
    agent_episode_costs = []
    for i in range(num_eval_envs):
        agent_cost = agent_infos[i].get('current_cost', float('inf'))
        agent_episode_costs.append(agent_cost)

    # ========== BASELINE ROLLOUT ==========
    # Restore to the SAME initial states again
    baseline_states = eval_initial_states.copy()

    # Reset environment step counters
    for i in range(num_eval_envs):
        eval_env.envs[i].current_request_index = 0

    # Baseline rollout (always action 12 = no time shift)
    for step in range(num_customers):
        baseline_actions = np.full(num_eval_envs, 12, dtype=np.int64)
        baseline_next_states, _, _, _, baseline_infos = eval_env.step(baseline_actions)
        baseline_states = baseline_next_states

    # Collect baseline costs from all environments
    baseline_episode_costs = []
    for i in range(num_eval_envs):
        baseline_cost = baseline_infos[i].get('current_cost', float('inf'))
        baseline_episode_costs.append(baseline_cost)

    # Compute metrics (filter out inf costs)
    valid_indices = [i for i, c in enumerate(agent_episode_costs) if not np.isinf(c)]

    if len(valid_indices) > 0:
        valid_agent_costs = [agent_episode_costs[i] for i in valid_indices]
        valid_baseline_costs = [baseline_episode_costs[i] for i in valid_indices]

        avg_agent_cost = np.mean(valid_agent_costs)
        avg_baseline_cost = np.mean(valid_baseline_costs)

        improvements_pct = [(b - a) / b * 100 for a, b in zip(valid_agent_costs, valid_baseline_costs) if b > 0]
        avg_improvement_pct = np.mean(improvements_pct) if len(improvements_pct) > 0 else 0.0
        max_improvement_pct = np.max(improvements_pct) if len(improvements_pct) > 0 else 0.0
        min_improvement_pct = np.min(improvements_pct) if len(improvements_pct) > 0 else 0.0
        avg_improvement_km = avg_baseline_cost - avg_agent_cost

    avg_accepted_rate = np.mean(agent_accepted_counts / num_customers)

    # Create metrics dictionary
    metrics = {
        "eval/avg_cost": avg_agent_cost,
        "eval/avg_baseline_cost": avg_baseline_cost,
        "eval/avg_improvement_pct": avg_improvement_pct,
        "eval/max_improvement_pct": max_improvement_pct,
        "eval/min_improvement_pct": min_improvement_pct,
        "eval/avg_improvement_km": avg_improvement_km,
        "eval/avg_accepted_rate": avg_accepted_rate,
    }

    # Print summary
    print(f"\n[Eval Epoch {epoch}] Greedy rollout on {num_eval_envs} fixed envs:")
    print(f"  Avg Cost: {avg_agent_cost:.2f} km (Baseline: {avg_baseline_cost:.2f} km)")
    print(f"  Avg Improvement: {avg_improvement_pct:.2f}% ({avg_improvement_km:.2f} km)")
    print(f"  Max/Min Improvement: {max_improvement_pct:.2f}% / {min_improvement_pct:.2f}%")
    print(f"  Acceptance Rate: {avg_accepted_rate * 100:.1f}%")

    return metrics


def train(
    num_episodes: int = 30,
    num_customers: int = 30,
    num_envs: int = 64,
    save_dir: str = "./checkpoints",
    save_interval: int = 100,
    log_interval: int = 10,
    policy_update_interval: int = 1,
    device: str = "cpu",
    seed: int = 42,
    resume_path: str = None,
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
        policy_update_interval: Update policy every N epochs (default: 1, i.e., every epoch)
        device: Device to run on ('cpu' or 'cuda')
        seed: Random seed
        resume_path: Path to checkpoint to resume training from (optional)
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
    # Use relative path to traveler_decisions_augmented.csv
    traveler_decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"
    vec_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=seed,
        traveler_decisions_path=traveler_decisions_path,
        device=device,
    )

    # Create baseline environment for comparison (no negotiation = always action 12)
    baseline_vec_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=seed,  # Same seed for fair comparison
        traveler_decisions_path=traveler_decisions_path,
        device=device,
    )

    # Create fixed evaluation environment (8 envs with fixed seed for reproducibility)
    print("\nCreating fixed evaluation environment...")
    eval_env, eval_initial_states = create_eval_envs(
        num_eval_envs=8,
        num_customers=num_customers,
        eval_seed=9999,  # Different seed from training
        traveler_decisions_path=traveler_decisions_path,
        device=device
    )
    print("Fixed evaluation instances generated and stored for reuse.")

    # Calculate state_dim based on environment's observation space
    # New state space: (state_rows, 2) where state_rows = 2*num_locations + 192
    # For SF dataset with 207 locations: (606, 2) -> flattened to 1212
    obs_shape = vec_env.envs[0].observation_space.shape  # e.g., (606, 2)
    state_dim = obs_shape[0] * obs_shape[1]  # 606 * 2 = 1212
    raw_travel_time_matrix = vec_env.envs[0].data_generator.generator.travel_time_matrix
    if isinstance(raw_travel_time_matrix, torch.Tensor):
        travel_time_matrix = raw_travel_time_matrix.to(device)
    else:
        travel_time_matrix = torch.tensor(raw_travel_time_matrix, dtype=torch.float32, device=device)
    # Create PPO agent
    action_dim = 16
    agent = PPOAgent(
        travel_time_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=1500,  # Set to 1500 for safety margin (time range [0, 1440] minutes)
        transformer_embed_dim=64,
        action_dim=action_dim,
        hidden_dim=512,   # MLP Hidden Layer Size
        num_heads=4,
        num_layers=3,
        lr=3e-4,
        gamma=1,
        gae_lambda=0.95,
        clip_epsilon=0.1,
        value_coef=0.5,
        entropy_coef=0.05,
        max_grad_norm=0.5,
        device=device
    )

    # Load checkpoint if resuming
    start_epoch = 1
    if resume_path is not None:
        print(f"Loading checkpoint from {resume_path}...")
        agent.load(resume_path)

        # Extract epoch number from checkpoint filename
        # Expected format: ppo_agent_ep<number>.pt
        import re
        match = re.search(r'ep(\d+)', Path(resume_path).stem)
        if match:
            total_episodes_done = int(match.group(1))
            start_epoch = (total_episodes_done // num_envs) + 1
            print(f"Resuming from episode {total_episodes_done} (epoch {start_epoch})")
        else:
            print("Could not extract epoch from filename, starting from epoch 1")

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
    steps_per_embedding_update = 25 * num_envs * num_customers  # 10 epochs worth of steps
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
            "gamma": 1,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "initial_epsilon": initial_epsilon,
            "final_epsilon": final_epsilon,
            "embedding_update_freq": steps_per_embedding_update,
            "policy_update_interval": policy_update_interval,
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
    print(f"Policy update interval: every {policy_update_interval} epoch(s)")
    print(f"Embedding update frequency: every {steps_per_embedding_update} steps")
    print(f"Reward scheme: Option 3 (Step-wise marginal cost comparison)")
    print("=" * 80)

    # Training statistics (track per epoch, aggregated over num_envs episodes)
    epoch_rewards = []
    epoch_costs = []
    epoch_accepted_rates = []
    epoch_failures = []
    epoch_improvements = []  # Track percentage improvement vs baseline

    # Training loop (parallel episodes)
    for epoch in range(start_epoch, num_epochs + 1):
        # Reset all environments (both agent and baseline)
        states, infos = vec_env.reset()  # (num_envs, num_customers, 6)
        baseline_states, _ = baseline_vec_env.reset()  # Reset baseline with same seed

        # Track statistics for this epoch (num_envs episodes)
        epoch_episode_rewards = []
        epoch_episode_costs = []
        epoch_episode_accepted = []
        epoch_episode_failed = []
        epoch_episode_improvements = []  # Per-episode average improvement vs baseline

        # Initialize per-environment accumulators
        env_rewards = np.zeros(num_envs)  # Accumulate rewards per environment
        env_accepted_count = np.zeros(num_envs)  # Count accepted requests per environment

        # Track baseline costs for Option 3 reward scheme (marginal cost comparison)
        baseline_previous_costs = np.zeros(num_envs)  # Previous costs in baseline trajectory

        # Compute current epsilon (linear decay based on epoch)
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)

        # Episode loop (each step processes all num_envs environments)
        for step in range(num_customers):
            # Get user IDs for current request across all environments
            user_ids = vec_env.get_current_user_ids()  # (num_envs,)

            # Predict flexibility and compute masks based on embeddings
            with torch.no_grad():
                user_embedding_ids = torch.LongTensor(user_ids) - 1
                pred_proba = embedding_model(user_embedding_ids)

                # Create a categorical distribution and sample from it
                dist = torch.distributions.Categorical(probs=pred_proba)
                predicted_flexibilities = dist.sample()

                masks = vec_env.get_masks(user_ids, predicted_flexibilities)

                # Epsilon-greedy masking: with probability epsilon, don't apply masks
                # With probability (1-epsilon), use the predicted masks
                if epsilon > 0:
                    random_vals = np.random.random(num_envs)
                    for i in range(num_envs):
                        if random_vals[i] < epsilon:
                            # Don't apply masking - all actions are valid
                            if isinstance(masks[i], torch.Tensor):
                                masks[i] = torch.ones_like(masks[i])
                            else:
                                # If masks[i] is a list or numpy array, convert to tensor
                                masks[i] = torch.ones(len(masks[i]), dtype=torch.float32)

            # Select actions for all environments in parallel with masks
            # PPO always selects according to policy probabilities
            actions = agent.select_action_batch(
                states,
                masks=masks
            )  # (num_envs,)

            # Step baseline environments with no-shift action (action 12 = (0, 0))
            baseline_actions = np.full(num_envs, 12, dtype=np.int64)  # All take action 12
            baseline_next_states, _, baseline_dones, baseline_truncs, baseline_step_infos = baseline_vec_env.step(baseline_actions)

            # Compute baseline marginal costs for Option 3 reward scheme
            baseline_current_costs = np.array([info.get('current_cost', 0.0) for info in baseline_step_infos])
            baseline_marginal_costs = baseline_current_costs - baseline_previous_costs  # Δ cost in baseline
            baseline_previous_costs = baseline_current_costs  # Update for next step

            # Step agent environments with Option 3 reward (marginal cost comparison)
            next_states, rewards, dones, truncs, step_infos = vec_env.step(actions, baseline_marginal_costs=baseline_marginal_costs)

            # Collect online data from all environments and accumulate statistics
            for i in range(num_envs):
                accepted = step_infos[i].get('accepted', False)
                # Store trip context for CSV lookup during embedding learning
                env = vec_env.envs[i]
                user_id = user_ids[i]

                # Get trip metadata using the helper method from env
                try:
                    metadata = env._get_meta_data_single_traveler(user_id)
                    online_data.append({
                        'customer_id': user_ids[i],
                        'action': actions[i],
                        'accepted': accepted,
                        'trip_purpose': metadata.get('trip_purpose'),
                        'departure_location': metadata.get('departure_location'),
                        'arrival_location': metadata.get('arrival_location'),
                        'departure_time_window': metadata.get('departure_time_window'),
                        'arrival_time_window': metadata.get('arrival_time_window'),
                    })
                except (KeyError, IndexError, TypeError):
                    # Fallback without metadata if extraction fails
                    online_data.append({
                        'customer_id': user_ids[i],
                        'action': actions[i],
                        'accepted': accepted
                    })
                # Accumulate statistics
                env_rewards[i] += rewards[i]
                if accepted:
                    env_accepted_count[i] += 1

            # Store rewards for all environments
            agent.store_rewards_batch(rewards, dones | truncs)

            # Update states
            states = next_states
            total_steps += num_envs

        # Collect episode statistics from all environments
        for i in range(num_envs):
            info = step_infos[i]
            baseline_info = baseline_step_infos[i]

            agent_cost = info.get('current_cost', float('inf'))
            baseline_cost = baseline_info.get('current_cost', float('inf'))
            failed = info.get('solver_failed', False)

            # Store episode statistics
            epoch_episode_rewards.append(env_rewards[i])
            epoch_episode_costs.append(agent_cost if not failed else float('inf'))
            epoch_episode_failed.append(1 if failed else 0)
            epoch_episode_accepted.append(env_accepted_count[i] / num_customers)  # Acceptance rate

            # Calculate percentage improvement based on FINAL routing costs
            if (not np.isinf(agent_cost) and not np.isinf(baseline_cost) and
                baseline_cost > 0 and not failed):
                improvement_pct = ((baseline_cost - agent_cost) / baseline_cost) * 100
            else:
                improvement_pct = 0.0
            epoch_episode_improvements.append(improvement_pct)

        # Perform PPO update every N epochs (accumulates data across epochs)
        train_stats = {}
        if epoch % policy_update_interval == 0:
            train_stats = agent.update(
                num_value_epochs=40,
                num_policy_epochs=10,
                batch_size=64,
                num_envs=num_envs,
                num_steps=num_customers
            )
            print(f"\n[Policy Updated at Epoch {epoch}] Buffer had {num_envs * num_customers * policy_update_interval} samples")

        # Update embedding model every K steps
        if total_steps % steps_per_embedding_update == 0 and len(online_data) > 0:
            print(f"\n[Epoch {epoch}, Step {total_steps}] Updating embedding model with {len(online_data)} samples...")

            # Convert deque to list for update
            online_data_list = list(online_data)

            # Update embedding model
            embedding_model = update_embedding_model(
                embedding_model,
                online_data_list,
                flexibility_personalities,
                ACTION_SPACE_MAP,
                num_epochs=50,
                batch_size=min(64, len(online_data_list)),
                lr=1e-3
            )

            
            with torch.no_grad():
                customer_ids = torch.arange(num_customers)
                pred_proba = embedding_model(customer_ids)
                predicted_flexibility = torch.argmax(pred_proba, dim=1)

            print(f"  Embedding model updated (masking enabled)")
            print(f"  Flexibility distribution: {torch.bincount(predicted_flexibility, minlength=n_flexibilities).tolist()}")

        # Aggregate epoch statistics
        avg_reward = np.mean(epoch_episode_rewards)
        avg_cost = np.mean([c for c in epoch_episode_costs if not np.isinf(c)]) if any(not np.isinf(c) for c in epoch_episode_costs) else float('inf')
        avg_accepted_rate = np.mean(epoch_episode_accepted)
        failure_rate = np.mean(epoch_episode_failed)
        avg_improvement = np.mean(epoch_episode_improvements)

        # Store epoch statistics
        epoch_rewards.append(avg_reward)
        epoch_costs.append(avg_cost)
        epoch_accepted_rates.append(avg_accepted_rate)
        epoch_failures.append(failure_rate)
        epoch_improvements.append(avg_improvement)

        # Logging
        if epoch % log_interval == 0:
            recent_rewards = epoch_rewards[-log_interval:]
            recent_costs = [c for c in epoch_costs[-log_interval:] if not np.isinf(c)]
            recent_accepted_rates = epoch_accepted_rates[-log_interval:]
            recent_failures = epoch_failures[-log_interval:]
            recent_improvements = epoch_improvements[-log_interval:]

            avg_recent_reward = np.mean(recent_rewards)
            avg_recent_cost = np.mean(recent_costs) if recent_costs else float("inf")
            avg_recent_accepted_rate = np.mean(recent_accepted_rates)
            avg_failure_rate = np.mean(recent_failures)
            avg_recent_improvement = np.mean(recent_improvements)

            print(f"\n[Epoch {epoch}/{num_epochs}] (Episodes {epoch * num_envs}/{num_episodes})")
            print(f"  Avg Reward: {avg_recent_reward:.2f}")
            print(f"  Avg Cost: {avg_recent_cost:.2f} km")
            print(f"  Avg Accepted Rate: {avg_recent_accepted_rate * 100:.1f}%")
            print(f"  Avg Improvement vs No Negotiation: {avg_recent_improvement:.2f}%")
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
                "avg_reward": avg_recent_reward,
                "avg_cost": avg_recent_cost,
                "avg_accepted_rate": avg_recent_accepted_rate,
                "avg_improvement_pct": avg_recent_improvement,
                "failure_rate": avg_failure_rate,
                "policy_loss": train_stats.get('policy_loss', 0),
                "value_loss": train_stats.get('value_loss', 0),
                "entropy": train_stats.get('entropy', 0),
            })

        # Evaluate every 5 epochs on fixed test environments
        if epoch % 5 == 0:
            eval_metrics = evaluate_epoch(
                agent=agent,
                embedding_model=embedding_model,
                eval_env=eval_env,
                eval_initial_states=eval_initial_states,
                epoch=epoch,
                num_customers=num_customers,
                flexibility_personalities=flexibility_personalities,
                device=device
            )

            # Log eval metrics to wandb
            wandb.log({
                "epoch": epoch,
                **eval_metrics
            })

            # Clear agent buffer to avoid contamination from eval data
            agent.clear_buffer()

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
    print(f"  Avg Reward (final epoch): {epoch_rewards[-1] if epoch_rewards else 0:.2f}")
    print(f"  Avg Reward (all epochs): {np.mean(epoch_rewards):.2f}")
    print(f"  Avg Cost (final epoch): {epoch_costs[-1] if epoch_costs else 0:.2f} km")
    print(f"  Avg Cost (all epochs): {np.mean(valid_costs):.2f} km" if valid_costs else "  No valid costs")
    print(f"  Avg Accepted Rate (all epochs): {np.mean(epoch_accepted_rates) * 100:.1f}%")
    print(f"  Avg Improvement vs No Negotiation (all epochs): {np.mean(epoch_improvements):.2f}%")
    print(f"  Solver Failure Rate: {np.mean(epoch_failures) * 100:.1f}%")

    # Clean up
    vec_env.close()
    baseline_vec_env.close()
    eval_env.close()

    return agent, epoch_rewards, epoch_costs, epoch_accepted_rates, epoch_failures


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
    improvement_percentages = []  # Track percentage improvement vs baseline

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
        improvement_percentages.append(improvement_pct)

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
    best_improvement_pct = np.max(improvement_percentages)
    worst_improvement_pct = np.min(improvement_percentages)

    print("\n" + "=" * 80)
    print("Evaluation Summary:")
    print(f"  Trained Agent Avg Cost:  {avg_trained:.2f} km")
    print(f"  Baseline Avg Cost:       {avg_baseline:.2f} km")
    print(f"  Average Improvement:     {avg_improvement:.2f} km ({avg_improvement_pct:.1f}%)")
    print(f"  Best Episode Improvement: {best_improvement_pct:.1f}%")
    print(f"  Worst Episode Improvement: {worst_improvement_pct:.1f}%")
    print(f"  Avg Reward (trained):    {np.mean(trained_rewards):.2f}")
    print("=" * 80)

    # Log evaluation metrics to wandb
    wandb.log({
        "eval/trained_avg_cost": avg_trained,
        "eval/baseline_avg_cost": avg_baseline,
        "eval/improvement_km": avg_improvement,
        "eval/improvement_pct": avg_improvement_pct,
        "eval/best_improvement_pct": best_improvement_pct,
        "eval/worst_improvement_pct": worst_improvement_pct,
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
        default=100,
        help="Save model every N epochs (1 epoch = num_envs episodes)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1, help="Log stats every N epochs"
    )
    parser.add_argument(
        "--policy-update-interval",
        type=int,
        default=1,
        help="Update policy every N epochs (default: 1, i.e., every epoch)",
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from (e.g., ./checkpoints/ppo_agent_ep6400.pt)"
    )

    args = parser.parse_args()

    # Train
    agent, rewards, costs, accepted_rates, failures = train(
        num_episodes=args.episodes,
        num_customers=args.customers,
        num_envs=args.num_envs,
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        policy_update_interval=args.policy_update_interval,
        device=args.device,
        seed=args.seed,
        resume_path=args.resume,
    )

    # Evaluate
    if args.eval:
        evaluate(agent, num_episodes=10, num_customers=args.customers)
    else:
        # Finish wandb if no evaluation
        wandb.finish()


if __name__ == "__main__":
    main()
