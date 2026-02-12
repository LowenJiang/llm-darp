"""
Main training script for DVRP-TW with PPO (Batched Version)

This script trains a PPO agent to optimize time window perturbations in a 
Dynamic Vehicle Routing Problem. It utilizes a batched environment to 
collect experience in parallel.

Updates:
- Corrected Epsilon-Greedy logic: Randomizes flexibility personality instead of ignoring masks.
"""
#. python src/meta_train.py --episodes 100_000 --num-envs 512 --ckpt-dir ./checkpoints/Feb_01_1 --log-dir ./runs/Feb_01_1 --solver-ckpt ./checkpoints/Jan_2501/best.pt

from __future__ import annotations
import importlib
import argparse
import os
import math
from collections import deque
from pathlib import Path
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensordict import TensorDict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import specific modules
from dvrp_env import DVRPEnv
from dvrp_ppo_agent_gcn import PPOAgent
from dvrp_embedding import EmbeddingFFN, OnlineTravelerDataset, likelihood_loss

# Action space mapping for mask computation: (pickup_shift, dropoff_shift)
ACTION_SPACE_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (0, 0),   (0, 10),   (0, 20),   (0, 30),
]

FLEXIBILITY_PERSONALITIES = [
    "flexible for late dropoff, but inflexible for early pickup",
    "flexible for early pickup, but inflexible for late dropoff",
    "inflexible for any schedule changes",
    "flexible for both early pickup and late dropoff",
]



def prepare_agent_observation(obs: TensorDict, as_numpy: bool = False) -> torch.Tensor | np.ndarray:
    """
    PREPARE the state for PPO agent: 
    transforming the obs state from the dvrpenv to the ppo state
    This part can be flexible to adapt to different ppoagent.
    This is correct. Feedable to GNN agent. Checked
    """
    # Concatenate history and new request along the sequence dimension
    full_state = torch.cat([obs['fixed'], obs['new']], dim=1)
    if as_numpy:
        return full_state.cpu().numpy()
    return full_state

def get_masks(
    env: DVRPEnv, # batched
    user_ids: np.ndarray, # 1d array of users at "this step"
    target_flexibilities: torch.Tensor, # [batch_size, num_users] or [batch_size]
    device: str
) -> torch.Tensor:
    """
    Compute action masks for the batched environment based on traveler constraints
    and the PREDICTED FLEXIBILITY
    This mask is looked up from the augmented csv. # we take this to be from LLM: guessing a mask
    Information to lookup: traveler_id, flexibility, trip_purpose, departure_location, arrival_location -> Mask over 16 actions
    """
    batch_size = target_flexibilities.shape[0]  # Should be number of environments (300)
    num_actions = 16
    
    # Default: Allow all actions (1s)
    masks = torch.ones((batch_size, num_actions), dtype=torch.float32, device=device)
    
    flexibility_columns = FLEXIBILITY_PERSONALITIES

    # For each environment in the batch
    for env_i in range(batch_size):
        u_id = user_ids[env_i] if len(user_ids) == batch_size else user_ids[0]  # Handle if same user across all envs
        #print(u_id)
        # Get metadata for this specific user in this specific environment
        user_meta = env.pending_requests['trip_metadata'][env_i][u_id]
        #print(user_meta)
        # Get predicted flexibility for this environment
        flex_idx = target_flexibilities[env_i].item()
        flex_col = flexibility_columns[flex_idx]
        
        # Build mask for all 16 actions for this environment
        action_mask = []
        if getattr(env, "decision_lookup", None) is not None:
            base_key = (
                int(u_id),
                str(user_meta["trip_purpose"]).strip(),
                str(user_meta["departure_location"]).strip(),
                str(user_meta["arrival_location"]).strip(),
            )
            for pickup_shift, dropoff_shift in ACTION_SPACE_MAP:
                p_abs = abs(pickup_shift)
                d_abs = abs(dropoff_shift)
                key = base_key + (int(p_abs), int(d_abs))
                decisions = env.decision_lookup.get(key)
                if decisions is None:
                    action_mask.append(0.0)
                    continue
                decision = decisions.get(flex_col)
                action_mask.append(1.0 if decision else 0.0)
        else:
            for pickup_shift, dropoff_shift in ACTION_SPACE_MAP:
                p_abs = abs(pickup_shift)
                d_abs = abs(dropoff_shift)

                # Query the decisions dataframe
                df_mask = (
                    (env.traveler_decisions_df["traveler_id"] == u_id) &
                    (env.traveler_decisions_df["trip_purpose"] == user_meta["trip_purpose"]) &
                    (env.traveler_decisions_df["departure_location"] == user_meta["departure_location"]) &
                    (env.traveler_decisions_df["arrival_location"] == user_meta["arrival_location"]) &
                    (env.traveler_decisions_df["pickup_shift_min"] == p_abs) &
                    (env.traveler_decisions_df["dropoff_shift_min"] == d_abs)
                )

                # Get decision for this action under predicted flexibility
                matched_rows = env.traveler_decisions_df[df_mask]
                if len(matched_rows) > 0:
                    decision = matched_rows.iloc[0][flex_col]
                    action_mask.append(1.0 if decision == "accept" else 0.0)
                else:
                    print("WARNING: NOT FOUND ERROR")
                    action_mask.append(0.0)
        #print(action_mask)
        masks[env_i] = torch.tensor(action_mask, device=device, dtype=torch.float32)
    
    return masks  # [batch_size, 16]

def train(
    num_episodes: int,
    num_customers: int,
    num_envs: int,
    ckpt_dir: str,
    log_dir: str,
    save_interval: int,
    log_interval: int,
    device: str,
    seed: int,
    resume_path: str,
    solver_ckpt: str | None,
    no_mask: bool = False,
    force_accept: bool = False,
):
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    num_epochs = max(1, num_episodes // num_envs)
    print("="*50)
    print(f"[Config] no_mask={no_mask}, force_accept={force_accept}")
    print(f"Initializing {num_envs} parallel environments (Batched execution)...")
    print("="*50)
    env = DVRPEnv(
        num_customers=num_customers,
        batch_size=num_envs,
        seed=seed,
        device=device,
        model_path=solver_ckpt,
        force_accept=force_accept,
    )

    baseline_env = copy.deepcopy(env)

    # 2. Initialize Agent
    tt_matrix = env.data_generator.generator.travel_time_matrix.to(device)
    # print(f"{tt_matrix=}")
    agent = PPOAgent(
        travel_time_matrix=tt_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=1500,
        transformer_embed_dim=64,
        action_dim=16,
        hidden_dim=512,
        policy_lr=1e-4,
        value_lr=1e-4,
        device=device
    )

    if resume_path:
        print(f"Loading checkpoint {resume_path}...")
        agent.load(resume_path)

    # 3. Initialize Embedding Model
    print("="*50)
    print("Initialize Embedding Model")
    print("="*50)
    embedding_model = EmbeddingFFN(
        num_entities=30,
        embed_dim=64,
        hidden_dim=128,
        output_dim=4
    ).to(device)
    embedding_model.eval()

    online_data = deque(maxlen=512000)

    initial_epsilon = 0.2
    final_epsilon = 0.0
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=str(log_path))
    writer.add_text("run/config", f"num_envs={num_envs}, episodes={num_episodes}, device={device}")

    total_steps = 0
    embedding_update_epoch = 5

    print("Starting training loop...")

    # --- Training Loop ---
    for epoch in range(1, num_epochs + 1):
        obs, info = env.reset()
        base_obs, _ = baseline_env.reset()

        epoch_improvement = torch.zeros(num_envs, device=device)
        epoch_accepted = torch.zeros(num_envs, device=device)
        epoch_actions = []  # track all actions for histogram

        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)
        if epoch < 5:
            epsilon = 1
        action_epsilon = 0.0

        base_info = None

        with tqdm(range(num_customers), desc=f"Epoch {epoch}/{num_epochs}", unit="step") as pbar:
            for step in pbar: # 30 steps in total; user_id is 1,2,...30
                user_ids_np = np.full(num_envs, step + 1)
                with torch.no_grad():
                    u_embed_ids = torch.full((num_envs,), step, dtype=torch.long, device=device) #contains: 1 1 1 1 ...
                    pred_proba = embedding_model(u_embed_ids) # embed user id to a vector: vec vec vec ... 

                    dist = torch.distributions.Categorical(probs=pred_proba)
                    predicted_flexs = dist.sample() 

                    target_flexs = predicted_flexs

                    random_flexs = torch.randint(0, 4, (num_envs,), device=device)

                    rand_vals = torch.rand(num_envs, device=device) # with (starting epsilon) probability set random category
                    explore_mask = (rand_vals < epsilon)

                    # apply the explore_mask as a filter for random / prediction
                    target_flexs = torch.where(explore_mask, random_flexs, predicted_flexs)

                # predicted mask (skip if --no-mask)
                if no_mask:
                    masks = None
                else:
                    masks = get_masks(env, user_ids_np, target_flexs, device) # [batch_size, 16(actions)]
                current_state = prepare_agent_observation(obs)
                action_np = agent.select_action_batch(current_state, masks=masks, epsilon=action_epsilon)

                #print("actions:")
                #print(action_np)

                epoch_actions.append(action_np)
                action_tensor = torch.tensor(action_np, device=device, dtype=torch.long)
                next_obs, env_reward, term, trunc, info = env.step(action_tensor)

                #print(info)

                base_action = torch.full((num_envs,), 12, dtype=torch.long, device=device) # action 0
                _, baseline_reward, _, _, base_info = baseline_env.step(base_action)

                adjusted_reward = env_reward - baseline_reward

                epoch_improvement += adjusted_reward
                epoch_accepted += info['accepted'].float() # this needs to be checked : The acceptance rate is not normal

                is_last_step = (step == num_customers - 1) 
                dones = (term | trunc).cpu().numpy()
                if is_last_step:
                    dones[:] = True 

                agent.store_rewards_batch(adjusted_reward.cpu().numpy(), dones)

                accepted_np = info['accepted'].cpu().numpy()

                for b in range(num_envs):
                    trip_metadata_list = env.pending_requests.get("trip_metadata", None)
                    user_meta = None
                    if trip_metadata_list is not None:
                        md = trip_metadata_list
                        if not isinstance(md, dict) and hasattr(md, "__len__"):
                            md = md[b] if b < len(md) else {}
                        if hasattr(md, "data"):
                            md = md.data
                        if isinstance(md, dict):
                            user_meta = md.get(int(user_ids_np[b]))

                    online_data.append({
                        'traveler_id': int(user_ids_np[b]),
                        'action': int(action_np[b]),
                        'accepted': int(accepted_np[b]),
                        'trip_purpose': user_meta.get("trip_purpose") if user_meta else None,
                        'departure_location': user_meta.get("departure_location") if user_meta else None,
                        'arrival_location': user_meta.get("arrival_location") if user_meta else None,
                        'departure_time_window': user_meta.get("departure_time_window") if user_meta else None,
                        'arrival_time_window': user_meta.get("arrival_time_window") if user_meta else None,
                    })

                obs = next_obs
                total_steps += num_envs

                # ?
                curr_acc = (epoch_accepted.mean().item() / (step + 1))
                curr_cum_rew = epoch_improvement.mean().item()
                best_improve = epoch_improvement.max().item()
                pbar.set_postfix({
                    "CumRew": f"{curr_cum_rew:.1f}",
                    "BestImp": f"{best_improve:.1f}",
                    "Acc": f"{curr_acc:.1%}",
                    "Eps": f"{epsilon:.2f}"
                })

        # --- End of Episode: Updates ---

        stats = agent.update(
            num_value_epochs=5,
            num_policy_epochs=3,
            batch_size=64,
            num_envs=num_envs,
            num_steps=num_customers
        )
        if stats:
            writer.add_scalar("loss/policy", stats.get("policy_loss", 0.0), epoch)
            writer.add_scalar("loss/value", stats.get("value_loss", 0.0), epoch)
            writer.add_scalar("loss/entropy", stats.get("entropy", 0.0), epoch)
            # PPO diagnostics
            for key in ("clip_fraction", "approx_kl", "ratio_mean", "ratio_std",
                        "advantage_mean", "advantage_std", "value_pred_mean",
                        "returns_mean", "returns_std", "policy_grad_norm",
                        "value_grad_norm"):
                if key in stats:
                    writer.add_scalar(f"ppo/{key}", stats[key], epoch)

        # this part is embedding update logic
        if len(online_data) > 0 and epoch % embedding_update_epoch == 0:
            print(f"\n[Embedding] Updating model with {len(online_data)} samples...")
            df_online = pd.DataFrame(list(online_data))
            df_online = df_online.sample(frac=1).reset_index(drop=True)
            required_cols = [
                "traveler_id",
                "action",
                "accepted",
                "trip_purpose",
                "departure_location",
                "arrival_location",
                "departure_time_window",
                "arrival_time_window",
            ]
            print("online dataframe:")
            print(df_online)
            df_online = df_online.dropna(subset=required_cols)
            if len(df_online) < 10:
                print("  [Embedding] Skipping update (insufficient rows with full metadata).")
            else:
                split_idx = int(0.8 * len(df_online))
                df_train = df_online.iloc[:split_idx]
                df_eval = df_online.iloc[split_idx:]

                dataset_train = OnlineTravelerDataset(
                    df_train,
                    FLEXIBILITY_PERSONALITIES,
                    ACTION_SPACE_MAP,
                )
                dataset_eval = OnlineTravelerDataset(
                    df_eval,
                    FLEXIBILITY_PERSONALITIES,
                    ACTION_SPACE_MAP,
                )

                optimizer = optim.Adam(embedding_model.parameters(), lr=1e-3)

                sub_epochs = 30
                batch_size = 128

                dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)

                for sub_epoch in range(sub_epochs):
                    embedding_model.train()
                    total_loss = 0.0
                    for entity_ids, _, ind_matrix, _ in dataloader_train:
                        optimizer.zero_grad()
                        pred_proba = embedding_model(entity_ids)
                        beta_matrix = embedding_model.get_embed()
                        loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix, alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(entity_ids)

                    if sub_epoch % 10 == 0:
                        avg_loss_train = total_loss / len(dataset_train)
                        embedding_model.eval()
                        eval_loss = 0.0
                        for entity_ids, _, ind_matrix, _ in dataloader_eval:
                            with torch.no_grad():
                                pred_proba = embedding_model(entity_ids)
                                beta_matrix = embedding_model.get_embed()
                                loss = likelihood_loss(beta_matrix, pred_proba, ind_matrix, alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False)
                                eval_loss += loss.item() * len(entity_ids)
                        avg_loss_eval = eval_loss / len(dataset_eval)
                        print(f"  SubEpoch {sub_epoch}: Train Loss={avg_loss_train:.4f}, Eval Loss={avg_loss_eval:.4f}")

                embedding_model.eval()

        if epoch % log_interval == 0:
            final_agent_cost = info['current_cost']
            final_base_cost = base_info['current_cost']

            avg_cost_agent = final_agent_cost.mean().item()
            avg_cost_base = final_base_cost.mean().item()
            avg_pct_imp = ((avg_cost_base - avg_cost_agent) / avg_cost_base) * 100

            avg_reward = epoch_improvement.mean().item()
            avg_acc = (epoch_accepted / num_customers).mean().item()

            print(f"[Epoch {epoch}/{num_epochs}] Rew: {avg_reward:.2f} | BaseCost: {avg_cost_base:.1f} | AgentCost: {avg_cost_agent:.1f} | Imp: {avg_pct_imp:.2f}% | Acc: {avg_acc:.2%}")

            writer.add_scalar("metrics/avg_reward", avg_reward, epoch)
            writer.add_scalar("metrics/avg_cost_base", avg_cost_base, epoch)
            writer.add_scalar("metrics/avg_cost_agent", avg_cost_agent, epoch)
            writer.add_scalar("metrics/avg_percent_improvement", avg_pct_imp, epoch)
            writer.add_scalar("metrics/avg_acceptance", avg_acc, epoch)

            # Extended reward/improvement metrics
            writer.add_scalar("metrics/best_improvement", epoch_improvement.max().item(), epoch)
            writer.add_scalar("metrics/worst_improvement", epoch_improvement.min().item(), epoch)
            writer.add_scalar("metrics/median_improvement", epoch_improvement.median().item(), epoch)
            writer.add_scalar("metrics/std_improvement", epoch_improvement.std().item(), epoch)
            writer.add_scalar("metrics/avg_cost_diff_per_step", avg_reward / num_customers, epoch)

            # Action histogram
            all_actions = np.concatenate(epoch_actions)
            writer.add_histogram("actions/distribution", all_actions, epoch)

        if epoch % save_interval == 0:
            agent.save(str(ckpt_path / f"policy_ep{epoch}.pt"))

    print("Training Complete.")
    agent.save(str(ckpt_path / "policy_final.pt"))
    writer.close()


def test_online_dataset(
    num_envs: int = 32,
    num_customers: int = 30,
    device: str = "cpu",
    seed: int = 42,
    solver_ckpt ="checkpoints/Feb_01_6_finetune/latest.pt",
) -> None:
    """
    Only for Testing
    Docstring for test_online_dataset
    
    :param num_envs: Description
    :type num_envs: int
    :param num_customers: Description
    :type num_customers: int
    :param device: Description
    :type device: str
    :param seed: Description
    :type seed: int
    :param solver_ckpt: Description
    """
    if solver_ckpt is None:
        raise ValueError("solver_ckpt is required to use the saved neural solver.")
    env = DVRPEnv(
        num_customers=num_customers,
        batch_size=num_envs,
        seed=seed,
        device=device,
        model_path=solver_ckpt,
    )

    tt_matrix = env.data_generator.generator.travel_time_matrix.to(device)
    agent = PPOAgent(
        travel_time_matrix=tt_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=1500,
        transformer_embed_dim=64,
        action_dim=16,
        hidden_dim=512,
        policy_lr=1e-4,
        value_lr=1e-4,
        device=device,
        entropy_coef=0.1
    )

    embedding_model = EmbeddingFFN(
        num_entities=30,
        embed_dim=64,
        hidden_dim=128,
        output_dim=4,
    ).to(device)
    embedding_model.eval()

    online_data = deque(maxlen=512000)

    obs, _ = env.reset()
    epsilon = 0.2

    for step in range(num_customers):
        action_epsilon = 0.0
        user_ids_np = np.full(num_envs, step + 1)
        with torch.no_grad():
            u_embed_ids = torch.full((num_envs,), step, dtype=torch.long, device=device)
            pred_proba = embedding_model(u_embed_ids)
            dist = torch.distributions.Categorical(probs=pred_proba)
            predicted_flexs = dist.sample()

            random_flexs = torch.randint(0, 4, (num_envs,), device=device)
            rand_vals = torch.rand(num_envs, device=device)
            explore_mask = rand_vals < epsilon
            target_flexs = torch.where(explore_mask, random_flexs, predicted_flexs)

        masks = get_masks(env, user_ids_np, target_flexs, device)
        current_state = prepare_agent_observation(obs)
        action_np = agent.select_action_batch(current_state, masks=masks, epsilon=action_epsilon)
        action_tensor = torch.tensor(action_np, device=device, dtype=torch.long)
        next_obs, _, _, _, info = env.step(action_tensor)

        accepted_np = info["accepted"].cpu().numpy()
        for b in range(num_envs):
            trip_metadata_list = env.pending_requests.get("trip_metadata", None)
            user_meta = None
            if trip_metadata_list is not None:
                md = trip_metadata_list
                if not isinstance(md, dict) and hasattr(md, "__len__"):
                    md = md[b] if b < len(md) else {}
                if hasattr(md, "data"):
                    md = md.data
                if isinstance(md, dict):
                    user_meta = md.get(int(user_ids_np[b]))

            online_data.append({
                "traveler_id": int(user_ids_np[b]),
                "action": int(action_np[b]),
                "accepted": int(accepted_np[b]),
                "trip_purpose": user_meta.get("trip_purpose") if user_meta else None,
                "departure_location": user_meta.get("departure_location") if user_meta else None,
                "arrival_location": user_meta.get("arrival_location") if user_meta else None,
                "departure_time_window": user_meta.get("departure_time_window") if user_meta else None,
                "arrival_time_window": user_meta.get("arrival_time_window") if user_meta else None,
            })

        obs = next_obs

    df_online = pd.DataFrame(list(online_data))
    df_online.to_csv("test_2.csv", index=False)
    print(df_online)


def main():
    parser = argparse.ArgumentParser(description="Train PPO Agent for DVRP-TW")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--customers", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--log-dir", type=str, default="./runs")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--solver-ckpt", type=str, default=None)
    parser.add_argument("--no-mask", action="store_true", default=False,
                        help="Disable action masking (allow all 16 actions)")
    parser.add_argument("--force-accept", action="store_true", default=False,
                        help="Force all proposed shifts to be accepted (no rejection)")

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        num_customers=args.customers,
        num_envs=args.num_envs,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        save_interval=10,
        log_interval=1,
        device=args.device,
        seed=args.seed,
        resume_path=args.resume,
        solver_ckpt=args.solver_ckpt,
        no_mask=args.no_mask,
        force_accept=args.force_accept,
    )

if __name__ == "__main__":
    main()
