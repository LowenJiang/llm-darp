"""
Main training script for DVRP-TW with PPO (Batched Version)

This script trains a PPO agent to optimize time window perturbations in a 
Dynamic Vehicle Routing Problem. It utilizes a batched environment to 
collect experience in parallel, significantly speeding up training compared 
to sequential execution.

Usage:
    python train.py --episodes 1000 --num-envs 64 --device cuda
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
import wandb
from tensordict import TensorDict

# Import specific modules
# Assuming these files are in the same directory or properly installed
from dvrp_env import DVRPEnv
from ppo_agent_v2 import PPOAgent
from embedding import (
    EmbeddingFFN, flexibility_personalities, n_flexibilities,
    update_embedding_model
)

# Action space mapping for mask computation: (pickup_shift, dropoff_shift)
ACTION_SPACE_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (0, 0),   (0, 10),   (0, 20),   (0, 30),
]

def prepare_agent_observation(obs: TensorDict) -> np.ndarray:
    """
    Converts the Environment's TensorDict observation into the flattened 
    tensor format expected by the PPOAgent's embedding layer.
    
    Args:
        obs: TensorDict containing:
             - 'fixed': [Batch, N, 6] (History)
             - 'new':   [Batch, 1, 6] (Current Request)
    
    Returns:
        np.ndarray: [Batch, N+1, 6] concatenated state
    """
    # Concatenate history and new request along the sequence dimension
    full_state = torch.cat([obs['fixed'], obs['new']], dim=1)
    return full_state.cpu().numpy()

def get_batch_masks(
    env: DVRPEnv, 
    user_ids: np.ndarray, 
    predicted_flexibilities: torch.Tensor, 
    device: str
) -> torch.Tensor:
    """
    Compute action masks for the entire batch based on traveler constraints
    and predicted flexibility personalities.
    """
    batch_size = len(user_ids)
    # Default: Allow all actions (1s)
    masks = torch.ones((batch_size, 16), dtype=torch.float32, device=device)
    
    trip_metadata_col = env.pending_requests.get("trip_metadata")

    # If no metadata or mask dataframe is loaded, return all-ones
    if env.df_mask is None or trip_metadata_col is None:
        return masks

    # Iterate batch to retrieve specific traveler constraints
    # (Note: This part is sequential due to dictionary lookups in metadata, 
    # but fast enough compared to solver time)
    for b in range(batch_size):
        u_id = user_ids[b]
        flex_idx = predicted_flexibilities[b].item()
        
        try:
            # metadata is a list of dicts, one per batch
            md_batch = trip_metadata_col[b]
            if md_batch is not None and u_id in md_batch:
                user_meta = md_batch[u_id]

                mask_list = env._get_mask_from_flex(
                    traveler_id=u_id,
                    trip_purpose=user_meta["trip_purpose"],
                    departure_location=user_meta["departure_location"],
                    arrival_location=user_meta["arrival_location"],
                    departure_time_window=user_meta["departure_time_window"],
                    arrival_time_window=user_meta["arrival_time_window"],
                    predicted_flex_index=flex_idx
                )
                masks[b] = torch.tensor(mask_list, device=device, dtype=torch.float32)
        except (KeyError, IndexError, TypeError, RuntimeError):
            # Keep mask as ones if lookup fails
            pass
            
    return masks

def create_eval_env(
    batch_size: int,
    num_customers: int,
    seed: int,
    traveler_decisions_path: str,
    device: str
) -> tuple[DVRPEnv, TensorDict]:
    """Helper to create a fixed-seed evaluation environment."""
    eval_env = DVRPEnv(
        num_customers=num_customers,
        batch_size=batch_size,
        seed=seed,
        traveler_decisions_path=traveler_decisions_path,
        device=device,
    )
    initial_state, _ = eval_env.reset()
    return eval_env, initial_state

def evaluate_epoch(
    agent: PPOAgent,
    embedding_model: EmbeddingFFN,
    eval_env: DVRPEnv,
    epoch: int,
    num_customers: int,
    device: str
) -> dict:
    """
    Run a validation epoch on the evaluation environment.
    Compares Agent performance against a "Do Nothing" baseline.
    """
    batch_size = eval_env.batch_size
    
    # --- Agent Rollout ---
    # Reset to fixed seed for consistent comparison
    eval_env.seed_val = 9999 
    obs, _ = eval_env.reset()
    
    agent_accepted = np.zeros(batch_size)
    
    for _ in range(num_customers):
        # Prepare inputs
        pickup_idx = 2 * eval_env.current_step + 1
        user_ids = eval_env.pending_requests["user_id"][:, pickup_idx].cpu().numpy()
        
        # Flexibility Prediction
        with torch.no_grad():
            user_embedding_ids = torch.LongTensor(user_ids) - 1
            user_embedding_ids = torch.clamp(user_embedding_ids, 0, 999).to(device)
            pred_proba = embedding_model(user_embedding_ids)
            predicted_flexs = torch.argmax(pred_proba, dim=1) # Greedy for eval
            
            masks = get_batch_masks(eval_env, user_ids, predicted_flexs, device)

        # Agent Action
        full_state = prepare_agent_observation(obs)
        # Use select_action_batch for vectorized inference
        # actions returned as numpy array
        action = agent.select_action_batch(full_state, masks=masks)
        
        obs, _, _, _, info = eval_env.step(action)
        agent_accepted += info['accepted'].cpu().numpy()

    agent_costs = info['current_cost'].cpu().numpy()

    # --- Baseline Rollout (Do Nothing / Action 12) ---
    eval_env.seed_val = 9999
    obs, _ = eval_env.reset()
    
    for _ in range(num_customers):
        # Action 12 corresponds to (0, 0) shift
        action = torch.full((batch_size,), 12, dtype=torch.long, device=device)
        obs, _, _, _, info = eval_env.step(action)

    baseline_costs = info['current_cost'].cpu().numpy()

    # --- Metrics Calculation ---
    # Filter out solver failures (extreme costs)
    valid_mask = (agent_costs < 4000) & (baseline_costs < 4000)
    
    if valid_mask.sum() > 0:
        valid_agent = agent_costs[valid_mask]
        valid_base = baseline_costs[valid_mask]
        
        avg_agent_cost = np.mean(valid_agent)
        avg_base_cost = np.mean(valid_base)
        
        improvement_km = valid_base - valid_agent
        improvement_pct = (improvement_km / valid_base) * 100
        
        avg_imp_pct = np.mean(improvement_pct)
        avg_imp_km = np.mean(improvement_km)
    else:
        avg_agent_cost = avg_base_cost = avg_imp_pct = avg_imp_km = 0.0

    avg_acc_rate = np.mean(agent_accepted / num_customers)

    print(f"\n[Eval Epoch {epoch}] Batch {batch_size}:")
    print(f"  Avg Cost:     {avg_agent_cost:.2f} (Base: {avg_base_cost:.2f})")
    print(f"  Improvement:  {avg_imp_pct:.2f}% ({avg_imp_km:.2f} cost units)")
    print(f"  Acceptance:   {avg_acc_rate*100:.1f}%")

    return {
        "eval/avg_cost": avg_agent_cost,
        "eval/avg_baseline_cost": avg_base_cost,
        "eval/avg_improvement_pct": avg_imp_pct,
        "eval/avg_improvement_km": avg_imp_km,
        "eval/avg_accepted_rate": avg_acc_rate
    }

def train(
    num_episodes: int,
    num_customers: int,
    num_envs: int,
    save_dir: str,
    save_interval: int,
    log_interval: int,
    device: str,
    seed: int,
    resume_path: str,
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Determine epochs based on total episodes desired vs batch size
    num_epochs = max(1, num_episodes // num_envs)

    # 1. Initialize Environments
    traveler_decisions_path = Path("traveler_decisions_augmented.csv")
    if not traveler_decisions_path.exists():
        print(f"Warning: {traveler_decisions_path} not found. Running without traveler decisions.")
        traveler_decisions_path = None
    else:
        traveler_decisions_path = str(traveler_decisions_path)

    print(f"Initializing {num_envs} parallel environments (Batched execution)...")
    
    # Main Training Environment
    import copy

    env = DVRPEnv(
        num_customers=num_customers,
        batch_size=num_envs,
        seed=seed,
        traveler_decisions_path=traveler_decisions_path,
        device=device
    )

    # Parallel Baseline Environment (runs "Do Nothing" to calculate advantage)
    baseline_env = copy.deepcopy(env)

    # Evaluation Environment
    eval_env, _ = create_eval_env(
        batch_size=32, # Larger batch for stable eval stats
        num_customers=num_customers,
        seed=9999,
        traveler_decisions_path=traveler_decisions_path,
        device=device
    )

    # 2. Initialize Agent
    # Extract travel time matrix from env for Graph Embedding
    tt_matrix = env.data_generator.generator.travel_time_matrix.to(device)
    
    agent = PPOAgent(
        travel_time_matrix=tt_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=1500,  # Set to 1500 for safety margin (time range [0, 1440] minutes)
        transformer_embed_dim=64,
        action_dim=16,
        hidden_dim=512,
        lr=3e-4,
        device=device
    )

    if resume_path:
        print(f"Loading checkpoint {resume_path}...")
        try:
             agent.load(resume_path)
        except Exception as e:
             print(f"Error loading checkpoint: {e}")

    # 3. Initialize Embedding Model (Traveler Personality Predictor)
    embedding_model = EmbeddingFFN(
        num_entities=1000,
        embed_dim=64,
        hidden_dim=128,
        output_dim=n_flexibilities
    ).to(device)
    embedding_model.eval()

    # Buffer for Online Embedding Learning
    online_data = deque(maxlen=12800)
    
    # Epsilon-Greedy Scheduling
    initial_epsilon = 0.2
    final_epsilon = 0.0
    
    wandb.init(project="dvrp-ppo-batched", config={
        "num_envs": num_envs, 
        "total_episodes": num_episodes, 
        "customers_per_episode": num_customers,
        "device": device
    })

    total_steps = 0
    steps_per_embedding_update = 25 * num_envs * num_customers

    print("Starting training loop...")

    # --- Training Loop ---
    for epoch in range(1, num_epochs + 1):
        # Reset Environments
        obs, info = env.reset()
        base_obs, _ = baseline_env.reset()
        
        prev_base_cost = torch.zeros(num_envs, device=device)
        epoch_rewards = torch.zeros(num_envs, device=device)
        epoch_accepted = torch.zeros(num_envs, device=device)
        epoch_improvement = torch.zeros(num_envs, device=device)
        # Decay Epsilon
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)
        epoch_unperturbed = torch.zeros(num_envs, device=device)
        epoch_perturbed = torch.zeros(num_envs, device=device)
        # Rollout (One full episode per env)
        for step in range(num_customers):
            # A. Get User IDs for current step
            pickup_idx = 2 * step + 1
            user_ids_tensor = env.pending_requests["user_id"][:, pickup_idx]
            user_ids_np = user_ids_tensor.cpu().numpy()
            
            # B. Embedding Prediction & Masking
            with torch.no_grad():
                u_embed_ids = (user_ids_tensor - 1).long()
                u_embed_ids = torch.clamp(u_embed_ids, 0, 999) 
                
                pred_proba = embedding_model(u_embed_ids)
                dist = torch.distributions.Categorical(probs=pred_proba)
                predicted_flexs = dist.sample()
                
                masks = get_batch_masks(env, user_ids_np, predicted_flexs, device)
                
                # Epsilon-Greedy Mask Exploration
                if epsilon > 0:
                    rand_vals = torch.rand(num_envs, device=device)
                    # Create a mask that allows everything
                    no_mask = torch.ones_like(masks)
                    # Apply random choice
                    apply_random = (rand_vals < epsilon).unsqueeze(1)
                    masks = torch.where(apply_random, no_mask, masks)

            # C. Agent Action
            # Convert TensorDict obs to numpy array format for Agent
            current_state_np = prepare_agent_observation(obs)
            # select_action_batch returns numpy array of actions
            action_np = agent.select_action_batch(current_state_np, masks=masks)
            action_tensor = torch.tensor(action_np, device=device, dtype=torch.long)
            
            # D. Baseline Action (Do Nothing = Index 12)
            base_action = torch.full((num_envs,), 12, dtype=torch.long, device=device)
            _, baseline_reward, _, _, base_info = baseline_env.step(base_action)
            


            # Calculate Baseline Marginal Improvement (usually negative cost increase)

            # E. Environment Step
            next_obs, env_reward, term, trunc, info = env.step(action_tensor) # env_reward negative

            # F. Calculate Advantage-Based Reward
            # env_reward is (old_cost - new_cost - penalty).
            # base_marginal is (base_new_cost - base_old_cost).
            # We want Reward = Agent_Improvement - Baseline_Improvement
            # agent_imp = - (new_cost - old_cost) = env_reward + penalty (ignoring penalty for moment)
            # This heuristic incentivizes beating the baseline.
            adjusted_reward = env_reward - baseline_reward

            epoch_improvement += adjusted_reward
            epoch_unperturbed += baseline_reward

            epoch_perturbed += env_reward
            # Store in Agent Buffer
            dones = (term | trunc).cpu().numpy()
            agent.store_rewards_batch(adjusted_reward.cpu().numpy(), dones)

            #print(f"env_reward: {adjusted_reward}")

            # Accumulate stats for logging (using raw MDP reward)
            epoch_rewards += adjusted_reward 
            epoch_accepted += info['accepted'].float()

            # G. Collect Data for Embedding Model
            trip_metadata_col = env.pending_requests.get("trip_metadata")
            accepted_np = info['accepted'].cpu().numpy()
            
            if trip_metadata_col is not None:
                for b in range(num_envs):
                    u_id = user_ids_np[b]
                    # We only learn from valid interactions
                    try:
                        md = trip_metadata_col[b][u_id]
                        online_data.append({
                            'customer_id': u_id,
                            'action': action_np[b],
                            'accepted': accepted_np[b],
                            'trip_purpose': md.get('trip_purpose'),
                            'departure_location': md.get('departure_location'),
                            'arrival_location': md.get('arrival_location'),
                            'departure_time_window': md.get('departure_time_window'),
                            'arrival_time_window': md.get('arrival_time_window'),
                        })
                    except (KeyError, IndexError, TypeError, RuntimeError):
                        pass

            obs = next_obs
            total_steps += num_envs

        # Sanity Checking
        print(epoch_rewards.mean())
        print(epoch_improvement.mean())

        # --- End of Episode: Updates ---

        # 1. Update PPO Agent
        train_stats = agent.update(
            num_value_epochs=40, 
            num_policy_epochs=10, 
            batch_size=64,
            # Parallel GAE args
            num_envs=num_envs,
            num_steps=num_customers 
        )
        
        # 2. Update Embedding Model (Periodically)
        if len(online_data) > 2000 and total_steps % steps_per_embedding_update < num_envs * num_customers:
             print(f"\n[Embedding] Updating model with {len(online_data)} samples...")
             embedding_model = update_embedding_model(
                 embedding_model, list(online_data), flexibility_personalities,
                 ACTION_SPACE_MAP, num_epochs=5, batch_size=256, lr=1e-3, device=device
             )
             online_data.clear()

        # 3. Logging
        if epoch % log_interval == 0:
            avg_rew = epoch_rewards.mean().item()
            avg_imp = epoch_improvement.mean().item()
            avg_acc = (epoch_accepted / num_customers).mean().item()
            avg_pct_imp = (epoch_perturbed.mean() - epoch_unperturbed.mean())/epoch_unperturbed * 100
            # Filter valid costs (remove solver failure penalties)
            final_costs = info['current_cost']
            valid_costs = final_costs[final_costs < 4000]
            avg_cost = valid_costs.mean().item() if len(valid_costs) > 0 else 0
            
            print(f"[Epoch {epoch}/{num_epochs}] Rewards: {avg_rew:.2f} | Cost: {avg_cost:.2f} | Acc: {avg_acc:.2%} | Loss: {train_stats.get('policy_loss', 0):.4f}")
            
            wandb.log({
                "epoch": epoch,
                "avg_reward": avg_rew,
                "avg_imp": avg_imp,
                "avg_cost": avg_cost,
                "avg_acceptance": avg_acc,
                "actor_loss": train_stats.get("policy_loss", 0),
                "value_loss": train_stats.get("value_loss", 0),
                "entropy": train_stats.get("entropy", 0),
                "epsilon": epsilon,
                "avg_pct_tmp": avg_pct_imp
            })

        # 4. Evaluation
        if epoch % 10 == 0:
            eval_metrics = evaluate_epoch(agent, embedding_model, eval_env, epoch, num_customers, device)
            wandb.log(eval_metrics)
            # Clear buffer to prevent stale data mixing with eval states if any leaked (safety)
            agent.clear_buffer()

        # 5. Save Checkpoint
        if epoch % save_interval == 0:
            agent.save(str(save_path / f"policy_ep{epoch}.pt"))

    print("Training Complete.")
    agent.save(str(save_path / "policy_final.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for DVRP-TW")
    parser.add_argument("--episodes", type=int, default=2000, help="Total number of training episodes (across all envs)")
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments")
    parser.add_argument("--customers", type=int, default=30, help="Number of customers per episode")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu/cuda/mps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Handle WandB login if necessary
    # wandb.login() 
    
    train(
        num_episodes=args.episodes,
        num_customers=args.customers,
        num_envs=args.num_envs,
        save_dir=args.save_dir,
        save_interval=20,
        log_interval=1,
        device=args.device,
        seed=args.seed,
        resume_path=args.resume
    )

if __name__ == "__main__":
    main()