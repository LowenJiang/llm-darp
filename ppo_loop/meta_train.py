"""
Main training script for DVRP-TW with PPO (Batched Version)

This script trains a PPO agent to optimize time window perturbations in a 
Dynamic Vehicle Routing Problem. It utilizes a batched environment to 
collect experience in parallel.

Updates:
- Corrected Epsilon-Greedy logic: Randomizes flexibility personality instead of ignoring masks.
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path
import copy

import numpy as np
import torch
import wandb
from tensordict import TensorDict
from tqdm import tqdm

# Import specific modules
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
    """
    # Concatenate history and new request along the sequence dimension
    full_state = torch.cat([obs['fixed'], obs['new']], dim=1)
    return full_state.cpu().numpy()

def get_batch_masks(
    env: DVRPEnv, 
    user_ids: np.ndarray, 
    target_flexibilities: torch.Tensor, 
    device: str
) -> torch.Tensor:
    """
    Compute action masks for the entire batch based on traveler constraints
    and the TARGET flexibility personalities (Predicted or Random).
    """
    batch_size = len(user_ids)
    # Default: Allow all actions (1s)
    masks = torch.ones((batch_size, 16), dtype=torch.float32, device=device)
    
    trip_metadata_col = env.pending_requests.get("trip_metadata")

    # If no metadata or mask dataframe is loaded, return all-ones
    if env.df_mask is None or trip_metadata_col is None:
        return masks

    # Iterate batch to retrieve specific traveler constraints
    for b in range(batch_size):
        u_id = user_ids[b]
        # Use the flexibility index passed in (either predicted or random)
        flex_idx = target_flexibilities[b].item()
        
        try:
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
            pass
            
    return masks

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
    
    num_epochs = max(1, num_episodes // num_envs)

    # 1. Initialize Environments
    traveler_decisions_path = Path("/home/jiangwolin/ppo_loop/traveler_decisions_augmented.csv")
    if not traveler_decisions_path.exists():
        # Fallback for local testing if needed, or raise error
        # Assuming file exists based on context
        print("path doens't exist")
        assert False

    print(f"Initializing {num_envs} parallel environments (Batched execution)...")
    
    env = DVRPEnv(
        num_customers=num_customers,
        batch_size=num_envs,
        seed=seed,
        traveler_decisions_path=str(traveler_decisions_path) if traveler_decisions_path.exists() else None,
        device=device
    )

    baseline_env = copy.deepcopy(env)

    # 2. Initialize Agent
    tt_matrix = env.data_generator.generator.travel_time_matrix.to(device)
    
    agent = PPOAgent(
        travel_time_matrix=tt_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=50,
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

    # 3. Initialize Embedding Model
    embedding_model = EmbeddingFFN(
        num_entities=30,
        embed_dim=64,
        hidden_dim=128,
        output_dim=n_flexibilities
    ).to(device)
    embedding_model.eval()

    online_data = deque(maxlen=12800)
    
    initial_epsilon = 0.2
    final_epsilon = 0.0
    
    # Initialize WandB
    wandb.init(
        project="ppo_time",
        name=f"first major run",
        config={
            "num_envs": 300,
            "episodes": 33000,
            "device": "cpu"
        }
    )



    total_steps = 0
    embedding_update_epoch = 5

    print("Starting training loop...")

    # --- Training Loop ---
    for epoch in range(1, num_epochs + 1):
        obs, info = env.reset()
        base_obs, _ = baseline_env.reset()
        
        epoch_improvement = torch.zeros(num_envs, device=device)
        epoch_accepted = torch.zeros(num_envs, device=device)
        
        # Decay Epsilon
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)

        base_info = None 
        
        with tqdm(range(num_customers), desc=f"Epoch {epoch}/{num_epochs}", unit="step") as pbar:
            for step in pbar:
                # -----------------------------------------------------------
                # A. Get User IDs
                # -----------------------------------------------------------
                pickup_idx = 2 * step + 1
                user_ids_tensor = env.pending_requests["user_id"][:, pickup_idx]
                user_ids_np = user_ids_tensor.cpu().numpy()
                
                # -----------------------------------------------------------
                # B. Embedding Prediction & Epsilon Greedy Logic
                # -----------------------------------------------------------
                with torch.no_grad():
                    # 1. Predict Probabilities
                    u_embed_ids = (user_ids_tensor - 1).long()
                    u_embed_ids = torch.clamp(u_embed_ids, 0, 999) 
                    pred_proba = embedding_model(u_embed_ids)
                    
                    # 2. Sample Predicted Flexibility
                    dist = torch.distributions.Categorical(probs=pred_proba)
                    predicted_flexs = dist.sample() # Shape: [Batch]

                    # 3. Apply Epsilon Logic
                    # With prob epsilon: Pick Random Flexibility (0-3)
                    # With prob 1-eps:   Use Predicted Flexibility
                    target_flexs = predicted_flexs
                    
                    if epsilon > 0:
                        # Generate random flexibilities [0, n_flexibilities)
                        random_flexs = torch.randint(0, n_flexibilities, (num_envs,), device=device)
                        
                        # Determine which envs will explore
                        rand_vals = torch.rand(num_envs, device=device)
                        explore_mask = (rand_vals < epsilon)
                        
                        # Combine: where explore is True use random, else use predicted
                        target_flexs = torch.where(explore_mask, random_flexs, predicted_flexs)

                    # 4. Generate Masks based on the TARGET flexibility
                    masks = get_batch_masks(env, user_ids_np, target_flexs, device)

                # -----------------------------------------------------------
                # C. Agent Action
                # -----------------------------------------------------------
                current_state_np = prepare_agent_observation(obs)
                # Pass the computed masks. The agent will sample from allowed actions.
                action_np = agent.select_action_batch(current_state_np, masks=masks)
                action_tensor = torch.tensor(action_np, device=device, dtype=torch.long)
                
                # -----------------------------------------------------------
                # D. Baseline Action (Do Nothing)
                # -----------------------------------------------------------
                base_action = torch.full((num_envs,), 12, dtype=torch.long, device=device)
                _, baseline_reward, _, _, base_info = baseline_env.step(base_action)

                # -----------------------------------------------------------
                # E. Environment Step
                # -----------------------------------------------------------
                next_obs, env_reward, term, trunc, info = env.step(action_tensor)

                # -----------------------------------------------------------
                # F. Calculate Advantage-Based Reward
                # -----------------------------------------------------------
                adjusted_reward = env_reward - baseline_reward
                epoch_improvement += adjusted_reward
                epoch_accepted += info['accepted'].float()

                is_last_step = (step == num_customers - 1)
                dones = (term | trunc).cpu().numpy()
                if is_last_step:
                    dones[:] = True

                agent.store_rewards_batch(adjusted_reward.cpu().numpy(), dones)

                # -----------------------------------------------------------
                # G. Collect Data for Embedding Model
                # -----------------------------------------------------------
                trip_metadata_col = env.pending_requests.get("trip_metadata")
                accepted_np = info['accepted'].cpu().numpy()
                
                if trip_metadata_col is not None:
                    for b in range(num_envs):
                        u_id = int(user_ids_np[b])
                        try:
                            md = trip_metadata_col[b].get(u_id)
                            if md is None: continue 

                            online_data.append({
                                'customer_id': u_id,
                                'action': int(action_np[b]),
                                'accepted': int(accepted_np[b]),
                                'trip_purpose': md.get('trip_purpose'),
                                'departure_location': md.get('departure_location'),
                                'arrival_location': md.get('arrival_location'),
                                'departure_time_window': md.get('departure_time_window'),
                                'arrival_time_window': md.get('arrival_time_window'),
                            })
                        except Exception:
                            pass

                obs = next_obs
                total_steps += num_envs

                curr_acc = (epoch_accepted.mean().item() / (step + 1))
                curr_cum_rew = epoch_improvement.mean().item()
                pbar.set_postfix({
                    "CumRew": f"{curr_cum_rew:.1f}",
                    "Acc": f"{curr_acc:.1%}",
                    "Eps": f"{epsilon:.2f}" 
                })

        # --- End of Episode: Updates ---

        # 1. Update PPO Agent
        train_stats = agent.update(
            num_value_epochs=40, 
            num_policy_epochs=10, 
            batch_size=64,
            num_envs=num_envs,
            num_steps=num_customers 
        )
        
        # 2. Update Embedding Model
        if len(online_data) > 100 and epoch % embedding_update_epoch == 0: 
             print(f"\n[Embedding] Updating model with {len(online_data)} samples...")
             embedding_model = update_embedding_model(
                 embedding_model, list(online_data), flexibility_personalities,
                 ACTION_SPACE_MAP, num_epochs=5, batch_size=256, lr=1e-3, device=device
             )
             online_data.clear()

        # 3. Logging
        if epoch % log_interval == 0:
            final_agent_cost = info['current_cost']
            final_base_cost = base_info['current_cost']
            valid_mask = (final_agent_cost < 4000) & (final_base_cost < 4000)
            
            if valid_mask.sum() > 0:
                valid_agent_c = final_agent_cost[valid_mask]
                valid_base_c = final_base_cost[valid_mask]
                avg_cost_agent = valid_agent_c.mean().item()
                avg_cost_base = valid_base_c.mean().item()
                avg_pct_imp = ((avg_cost_base - avg_cost_agent) / avg_cost_base) * 100
            else:
                avg_cost_agent = 0.0
                avg_cost_base = 0.0
                avg_pct_imp = 0.0

            avg_reward = epoch_improvement.mean().item()
            avg_acc = (epoch_accepted / num_customers).mean().item()

            print(f"[Epoch {epoch}/{num_epochs}] Rew: {avg_reward:.2f} | BaseCost: {avg_cost_base:.1f} | AgentCost: {avg_cost_agent:.1f} | Imp: {avg_pct_imp:.2f}% | Acc: {avg_acc:.2%}")
            
            wandb.log({
                "epoch": epoch,
                "avg_reward": avg_reward,
                "avg_cost_base": avg_cost_base,
                "avg_cost_agent": avg_cost_agent,
                "avg_percent_improvement": avg_pct_imp,
                "avg_acceptance": avg_acc,
            })

        # 4. Save Checkpoint
        if epoch % save_interval == 0:
            agent.save(str(save_path / f"policy_ep{epoch}.pt"))

    print("Training Complete.")
    agent.save(str(save_path / "policy_final.pt"))


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for DVRP-TW")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--customers", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    train(
        num_episodes=args.episodes,
        num_customers=args.customers,
        num_envs=args.num_envs,
        save_dir=args.save_dir,
        save_interval=4,
        log_interval=1,
        device=args.device,
        seed=args.seed,
        resume_path=args.resume
    )

if __name__ == "__main__":
    main()