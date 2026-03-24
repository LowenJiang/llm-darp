"""
Training script for DVRP-TW with PPO — 49-action extended version.

Adapts meta_train.py for:
- 49 discrete actions: all (pickup_shift, dropoff_shift) pairs from a 7×7 grid of
  shifts in {-30, -20, -10, 0, 10, 20, 30} minutes.
- A frozen oracle (attention) encoder with learned FFN heads (TestPPOAgent).
- TestDVRPEnv, which exposes get_encoder_input() and get_feasibility_mask().
- Combined masking: physics feasibility AND user-preference acceptance.

The no-op action (0, 0) maps to index 24 because 0 is at position 3 in
_SHIFT_STEPS, giving row 3 × 7 + col 3 = 24.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from tqdm import tqdm

# Local imports — test variants of the env and agent
from test_dvrp_env import TestDVRPEnv
from test_ppo_agent import TestPPOAgent
from dvrp_embedding import EmbeddingFFN, OnlineTravelerDataset, likelihood_loss


# ---------------------------------------------------------------------------
# Action space: 7 pickup shifts × 7 dropoff shifts = 49 actions
# ---------------------------------------------------------------------------

_SHIFT_STEPS = [-30, -20, -10, 0, 10, 20, 30]

# ACTION_SPACE_MAP[i] = (pickup_shift_minutes, dropoff_shift_minutes)
# Row-major: iterate over pickup shifts in the outer loop.
ACTION_SPACE_MAP = [(ps, ds) for ps in _SHIFT_STEPS for ds in _SHIFT_STEPS]

# Index of the no-op (0, 0) action.
# 0 is at position 3 in _SHIFT_STEPS, so index = 3 * 7 + 3 = 24.
NOOP_ACTION: int = 24

assert ACTION_SPACE_MAP[NOOP_ACTION] == (0, 0), (
    f"Invariant failed: NOOP_ACTION={NOOP_ACTION} should map to (0,0), "
    f"got {ACTION_SPACE_MAP[NOOP_ACTION]}"
)

# ---------------------------------------------------------------------------
# Flexibility personality labels — order must match dvrp_embedding.py
# ---------------------------------------------------------------------------

FLEXIBILITY_PERSONALITIES = [
    "flexible for late dropoff, but inflexible for early pickup",
    "flexible for early pickup, but inflexible for late dropoff",
    "inflexible for any schedule changes",
    "flexible for both early pickup and late dropoff",
]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def get_user_preference_masks(
    env: TestDVRPEnv,
    user_ids_np: np.ndarray,
    target_flexibilities: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute user-preference masks over the 49-action space.

    For each environment in the batch, look up the traveler decisions CSV
    (cached in env.decision_lookup) to determine which of the 49 actions the
    traveler's predicted flexibility type would accept.

    Mirrors get_masks() from meta_train.py but iterates over all 49 entries
    of ACTION_SPACE_MAP instead of 16.

    Args:
        env: The batched TestDVRPEnv instance (must have decision_lookup or
             traveler_decisions_df attribute, matching DVRPEnv's interface).
        user_ids_np: Array of shape (num_envs,) with 1-indexed traveler IDs
            for the current step.
        target_flexibilities: Long tensor of shape (num_envs,) with predicted
            flexibility category indices (0–3).
        device: Target device string (e.g. "cpu", "cuda:0").

    Returns:
        masks: Float tensor of shape (num_envs, 49) where 1.0 = action is
            acceptable to the traveler and 0.0 = action is not acceptable.
    """
    num_envs = target_flexibilities.shape[0]
    num_actions = len(ACTION_SPACE_MAP)  # 49

    # Start permissive — restrict below
    masks = torch.ones((num_envs, num_actions), dtype=torch.float32, device=device)

    flexibility_columns = FLEXIBILITY_PERSONALITIES

    for env_i in range(num_envs):
        # user_ids_np may be a scalar broadcast or per-env array
        u_id = user_ids_np[env_i] if len(user_ids_np) == num_envs else user_ids_np[0]

        # Retrieve trip metadata for this traveler in this environment
        user_meta = env.pending_requests["trip_metadata"][env_i][u_id]

        # Predicted flexibility category for this environment
        flex_idx = target_flexibilities[env_i].item()
        flex_col = flexibility_columns[flex_idx]

        action_mask = []

        if getattr(env, "decision_lookup", None) is not None:
            # Fast O(1) hash-table lookup path
            base_key = (
                int(u_id),
                str(user_meta["trip_purpose"]).strip(),
                str(user_meta["departure_location"]).strip(),
                str(user_meta["arrival_location"]).strip(),
            )
            for pickup_shift, dropoff_shift in ACTION_SPACE_MAP:
                key = base_key + (int(abs(pickup_shift)), int(abs(dropoff_shift)))
                decisions = env.decision_lookup.get(key)
                if decisions is None:
                    # Unknown key → conservatively reject
                    action_mask.append(0.0)
                    continue
                decision = decisions.get(flex_col)
                action_mask.append(1.0 if decision else 0.0)
        else:
            # Slow DataFrame scan fallback (no cached lookup available)
            for pickup_shift, dropoff_shift in ACTION_SPACE_MAP:
                p_abs = abs(pickup_shift)
                d_abs = abs(dropoff_shift)
                df_mask = (
                    (env.traveler_decisions_df["traveler_id"] == u_id)
                    & (env.traveler_decisions_df["trip_purpose"] == user_meta["trip_purpose"])
                    & (env.traveler_decisions_df["departure_location"] == user_meta["departure_location"])
                    & (env.traveler_decisions_df["arrival_location"] == user_meta["arrival_location"])
                    & (env.traveler_decisions_df["pickup_shift_min"] == p_abs)
                    & (env.traveler_decisions_df["dropoff_shift_min"] == d_abs)
                )
                matched = env.traveler_decisions_df[df_mask]
                if len(matched) > 0:
                    decision = matched.iloc[0][flex_col]
                    action_mask.append(1.0 if decision == "accept" else 0.0)
                else:
                    print(
                        f"WARNING: No decision found for traveler={u_id}, "
                        f"purpose={user_meta['trip_purpose']}, "
                        f"pickup_shift={p_abs}, dropoff_shift={d_abs}"
                    )
                    action_mask.append(0.0)

        masks[env_i] = torch.tensor(action_mask, device=device, dtype=torch.float32)

    return masks  # [num_envs, 49]


def combine_masks(
    feasibility_mask: torch.Tensor,
    user_mask: torch.Tensor,
) -> torch.Tensor:
    """Combine physics feasibility and user-preference masks.

    An action is only valid if it is both physically feasible AND acceptable to
    the traveler. We achieve this with element-wise multiplication (logical AND
    for binary {0, 1} tensors).

    If the combined mask for any environment is all-zero (no jointly valid action),
    fall back to the feasibility mask alone to keep the episode moving.

    Args:
        feasibility_mask: Float tensor [B, 49] from env.get_feasibility_mask().
        user_mask: Float tensor [B, 49] from get_user_preference_masks().

    Returns:
        combined: Float tensor [B, 49] — 1.0 where action is fully valid.
    """
    combined = feasibility_mask * user_mask  # logical AND for 0/1 tensors

    # Detect rows where the intersection is empty (all zeros) and fall back
    # to the feasibility mask alone so the agent is never completely stuck.
    row_sums = combined.sum(dim=1, keepdim=True)  # [B, 1]
    all_blocked = row_sums < 1e-8  # [B, 1] bool

    if all_blocked.any():
        combined = torch.where(all_blocked, feasibility_mask, combined)

    return combined


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

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
    resume_path: Optional[str],
    solver_ckpt: Optional[str],
    no_mask: bool = False,
    force_accept: bool = False,
) -> None:
    """Train the TestPPOAgent on TestDVRPEnv with 49 actions.

    Structure mirrors meta_train.py:
    - Outer loop: epochs (num_episodes // num_envs)
    - Inner loop: num_customers steps (one traveler request per step)
    - Per step: predict flexibility → compute masks → select action →
                step env → store experience
    - End of epoch: PPO update + optional embedding model update

    Args:
        num_episodes: Total training episodes (epochs = num_episodes // num_envs).
        num_customers: Number of customer requests per episode.
        num_envs: Number of parallel environments in the batch.
        ckpt_dir: Directory to save agent checkpoints.
        log_dir: Directory for TensorBoard logs.
        save_interval: Save a checkpoint every this many epochs.
        log_interval: Log metrics to TensorBoard every this many epochs.
        device: PyTorch device string ("cpu", "cuda", "mps", …).
        seed: Random seed for reproducibility.
        resume_path: If not None, load agent checkpoint from this path before
            training starts.
        solver_ckpt: Path to the oracle attention policy checkpoint used to
            initialise the frozen encoder inside TestPPOAgent.
        no_mask: If True, disable all masking (all 49 actions allowed).
        force_accept: If True, all proposed perturbations are accepted by
            travellers regardless of their flexibility type.
    """
    # ------------------------------------------------------------------
    # 1. Directory setup (wipe and recreate for clean runs)
    # ------------------------------------------------------------------
    ckpt_path = Path(ckpt_dir)
    if ckpt_path.exists():
        shutil.rmtree(ckpt_path)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    log_path = Path(log_dir)
    if log_path.exists():
        shutil.rmtree(log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    num_epochs = max(1, num_episodes // num_envs)

    print("=" * 60)
    print(f"[Config] no_mask={no_mask}, force_accept={force_accept}")
    print(f"[Config] action_space=49, num_envs={num_envs}, epochs={num_epochs}")
    print(f"[Config] solver_ckpt={solver_ckpt}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Environments
    # ------------------------------------------------------------------
    env = TestDVRPEnv(
        num_customers=num_customers,
        batch_size=num_envs,
        seed=seed,
        device=device,
        model_path=solver_ckpt,
        force_accept=force_accept,
    )

    # A deep-copy baseline environment runs the no-op action every step.
    # We compare agent reward against this to compute the improvement signal.
    baseline_env = copy.deepcopy(env)

    # ------------------------------------------------------------------
    # 3. Agent — frozen oracle encoder + FFN heads
    # ------------------------------------------------------------------
    agent = TestPPOAgent(
        encoder_checkpoint=solver_ckpt,
        action_dim=49,
        embed_dim=128,
        policy_lr=1e-4,
        value_lr=1e-4,
        device=device,
    )

    if resume_path:
        print(f"[Init] Resuming from checkpoint: {resume_path}")
        agent.load(resume_path)

    # ------------------------------------------------------------------
    # 4. Embedding model for traveller flexibility prediction
    # ------------------------------------------------------------------
    print("=" * 60)
    print("[Init] Initialising EmbeddingFFN for flexibility prediction")
    print("=" * 60)

    embedding_model = EmbeddingFFN(
        num_entities=30,   # 30 distinct travellers (one per step)
        embed_dim=64,
        hidden_dim=128,
        output_dim=4,      # 4 flexibility categories
    ).to(device)
    embedding_model.eval()

    # Circular buffer for online learning data
    online_data: deque = deque(maxlen=512_000)

    # Epsilon schedule: linear decay from initial_epsilon to 0
    initial_epsilon = 0.2
    final_epsilon = 0.0

    # Update the embedding model every this many epochs
    embedding_update_epoch = 100

    # ------------------------------------------------------------------
    # 5. Logging
    # ------------------------------------------------------------------
    writer = SummaryWriter(log_dir=str(log_path))
    writer.add_text(
        "run/config",
        f"num_envs={num_envs}, episodes={num_episodes}, device={device}, "
        f"action_dim=49, no_mask={no_mask}, force_accept={force_accept}",
    )

    total_steps = 0

    # Welford's online algorithm for running reward normalisation
    reward_running_mean = 0.0
    reward_running_var = 1.0
    reward_count = 0

    print("[Train] Starting training loop…")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):

        obs, info = env.reset()
        base_obs, _ = baseline_env.reset()

        # Per-epoch accumulators
        epoch_improvement = torch.zeros(num_envs, device=device)
        epoch_accepted = torch.zeros(num_envs, device=device)
        epoch_actions: list[int] = []
        epoch_step_costs: list[torch.Tensor] = []
        epoch_raw_rewards: list[torch.Tensor] = []

        # Epsilon for this epoch: starts high, decays to 0
        # Force full exploration for the first 5 warm-up epochs
        epsilon = initial_epsilon - (initial_epsilon - final_epsilon) * (epoch - 1) / max(num_epochs - 1, 1)
        if epoch < 5:
            epsilon = 1.0

        # Action-level epsilon (used inside select_action_batch for mixing)
        action_epsilon = 0.0

        base_info = None  # Will be set on first baseline step

        with tqdm(range(num_customers), desc=f"Epoch {epoch}/{num_epochs}", unit="step") as pbar:
            for step in pbar:
                # user IDs are 1-indexed (step 0 → traveler 1)
                user_ids_np = np.full(num_envs, step + 1)

                # ----------------------------------------------------------
                # 6a. Predict flexibility category via embedding model
                # ----------------------------------------------------------
                with torch.no_grad():
                    # step is 0-indexed; embedding uses same 0-indexed IDs
                    u_embed_ids = torch.full(
                        (num_envs,), step, dtype=torch.long, device=device
                    )
                    pred_proba = embedding_model(u_embed_ids)  # [B, 4]

                    dist = torch.distributions.Categorical(probs=pred_proba)
                    predicted_flexs = dist.sample()  # [B]

                    # Epsilon-greedy: with probability epsilon, use a random
                    # flexibility category instead of the model prediction.
                    random_flexs = torch.randint(0, 4, (num_envs,), device=device)
                    rand_vals = torch.rand(num_envs, device=device)
                    explore_mask = rand_vals < epsilon
                    target_flexs = torch.where(explore_mask, random_flexs, predicted_flexs)

                # ----------------------------------------------------------
                # 6b. Compute combined mask [B, 49]
                # ----------------------------------------------------------
                if no_mask:
                    # Training with masking disabled: all 49 actions permitted
                    combined_mask = torch.ones(
                        (num_envs, 49), dtype=torch.float32, device=device
                    )
                else:
                    # Physics feasibility mask from the environment
                    feasibility_mask = env.get_feasibility_mask(step)  # [B, 49]

                    # User-preference mask from embedding model prediction
                    user_mask = get_user_preference_masks(
                        env, user_ids_np, target_flexs, device
                    )  # [B, 49]

                    combined_mask = combine_masks(feasibility_mask, user_mask)

                # ----------------------------------------------------------
                # 6c. Agent selects actions
                # ----------------------------------------------------------
                # Pass the oracle encoder input TensorDict directly to the
                # agent; TestPPOAgent handles pooling internally.
                encoder_input = env.get_encoder_input()  # TensorDict

                action_np = agent.select_action_batch(
                    encoder_input, masks=combined_mask, epsilon=action_epsilon
                )  # numpy [B]

                epoch_actions.extend(action_np.tolist())
                action_tensor = torch.tensor(
                    action_np, device=device, dtype=torch.long
                )

                # ----------------------------------------------------------
                # 6d. Step both environments
                # ----------------------------------------------------------
                next_obs, env_reward, term, trunc, info = env.step(action_tensor)

                # Baseline: always apply no-op (0, 0) → index 24
                base_action = torch.full(
                    (num_envs,), NOOP_ACTION, dtype=torch.long, device=device
                )
                _, baseline_reward, _, _, base_info = baseline_env.step(base_action)

                # Improvement over no-op baseline
                adjusted_reward = env_reward - baseline_reward

                # Track raw diagnostics before normalisation
                epoch_step_costs.append(info["current_cost"].clone())
                epoch_raw_rewards.append(adjusted_reward.clone())

                # ----------------------------------------------------------
                # 6e. Welford's online reward normalisation
                # ----------------------------------------------------------
                batch_mean = adjusted_reward.mean().item()
                batch_var = adjusted_reward.var().item() if num_envs > 1 else 0.0
                batch_n = num_envs
                new_count = reward_count + batch_n
                delta = batch_mean - reward_running_mean
                reward_running_mean += delta * batch_n / max(new_count, 1)
                reward_running_var += (
                    batch_var * batch_n
                    + delta ** 2 * reward_count * batch_n / max(new_count, 1)
                )
                reward_count = new_count
                reward_std = max((reward_running_var / max(reward_count, 1)) ** 0.5, 1e-8)
                normalized_reward = adjusted_reward / reward_std

                epoch_improvement += adjusted_reward
                epoch_accepted += info["accepted"].float()

                # Mark all environments as done on the last step of the episode
                is_last_step = step == num_customers - 1
                dones = (term | trunc).cpu().numpy()
                if is_last_step:
                    dones[:] = True

                agent.store_rewards_batch(normalized_reward.cpu().numpy(), dones)

                # ----------------------------------------------------------
                # 6f. Collect online data for embedding model update
                # ----------------------------------------------------------
                accepted_np = info["accepted"].cpu().numpy()

                for b in range(num_envs):
                    # Safely extract trip metadata for this environment slot
                    trip_metadata_list = env.pending_requests.get("trip_metadata", None)
                    user_meta = None
                    if trip_metadata_list is not None:
                        md = trip_metadata_list
                        # Handle various container types that may wrap the dict
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
                total_steps += num_envs

                # Live progress bar metrics
                curr_acc = epoch_accepted.mean().item() / (step + 1)
                curr_cum_rew = epoch_improvement.mean().item()
                best_improve = epoch_improvement.max().item()
                pbar.set_postfix({
                    "CumRew": f"{curr_cum_rew:.1f}",
                    "BestImp": f"{best_improve:.1f}",
                    "Acc": f"{curr_acc:.1%}",
                    "Eps": f"{epsilon:.2f}",
                })

        # ------------------------------------------------------------------
        # 7. PPO update
        # ------------------------------------------------------------------
        stats = agent.update(
            num_value_epochs=6,
            num_policy_epochs=3,
            batch_size=64,
            num_envs=num_envs,
            num_steps=num_customers,
        )

        if stats:
            writer.add_scalar("loss/policy", stats.get("policy_loss", 0.0), epoch)
            writer.add_scalar("loss/value", stats.get("value_loss", 0.0), epoch)
            writer.add_scalar("loss/entropy", stats.get("entropy", 0.0), epoch)
            for key in [
                "clip_fraction", "approx_kl", "ratio_mean", "ratio_std",
                "advantage_mean", "advantage_std", "value_pred_mean",
                "returns_mean", "returns_std", "policy_grad_norm", "value_grad_norm",
            ]:
                if key in stats:
                    writer.add_scalar(f"ppo/{key}", stats[key], epoch)

        # Action histogram
        if epoch_actions:
            writer.add_histogram(
                "actions/distribution", torch.tensor(epoch_actions), epoch
            )

        # Solver cost and reward diagnostics
        if epoch_step_costs:
            all_costs = torch.stack(epoch_step_costs)       # (num_steps, num_envs)
            all_raw_rewards = torch.stack(epoch_raw_rewards)
            writer.add_scalar("diagnostics/cost_max", all_costs.max().item(), epoch)
            writer.add_scalar("diagnostics/cost_min", all_costs.min().item(), epoch)
            writer.add_scalar("diagnostics/cost_mean", all_costs.mean().item(), epoch)
            writer.add_scalar("diagnostics/cost_std", all_costs.std().item(), epoch)
            writer.add_scalar("diagnostics/reward_max", all_raw_rewards.max().item(), epoch)
            writer.add_scalar("diagnostics/reward_min", all_raw_rewards.min().item(), epoch)
            writer.add_scalar("diagnostics/reward_std", all_raw_rewards.std().item(), epoch)
            writer.add_scalar("diagnostics/reward_running_std", reward_std, epoch)

        # ------------------------------------------------------------------
        # 8. Embedding model update (periodic)
        # ------------------------------------------------------------------
        if len(online_data) > 0 and epoch % embedding_update_epoch == 0:
            print(f"\n[Embedding] Updating model with {len(online_data)} samples…")

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

            print("[Embedding] Online dataframe sample:")
            print(df_online.head())
            df_online = df_online.dropna(subset=required_cols)

            if len(df_online) < 10:
                print("[Embedding] Skipping update — insufficient rows with full metadata.")
            else:
                split_idx = int(0.8 * len(df_online))
                df_train = df_online.iloc[:split_idx]
                df_eval = df_online.iloc[split_idx:]

                # Pass the 49-action ACTION_SPACE_MAP to the dataset so that
                # indicator matrix computation uses the extended action space.
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
                sub_batch_size = 128

                dataloader_train = DataLoader(
                    dataset_train, batch_size=sub_batch_size, shuffle=True
                )
                dataloader_eval = DataLoader(
                    dataset_eval, batch_size=sub_batch_size, shuffle=False
                )

                for sub_epoch in range(sub_epochs):
                    embedding_model.train()
                    total_loss = 0.0

                    for entity_ids, _, ind_matrix, _ in dataloader_train:
                        optimizer.zero_grad()
                        pred_proba = embedding_model(entity_ids)
                        beta_matrix = embedding_model.get_embed()
                        loss = likelihood_loss(
                            beta_matrix, pred_proba, ind_matrix,
                            alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False,
                        )
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
                                loss = likelihood_loss(
                                    beta_matrix, pred_proba, ind_matrix,
                                    alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False,
                                )
                                eval_loss += loss.item() * len(entity_ids)
                        avg_loss_eval = eval_loss / len(dataset_eval)
                        print(
                            f"  SubEpoch {sub_epoch}: "
                            f"Train Loss={avg_loss_train:.4f}, Eval Loss={avg_loss_eval:.4f}"
                        )

                embedding_model.eval()

        # ------------------------------------------------------------------
        # 9. Logging
        # ------------------------------------------------------------------
        if epoch % log_interval == 0:
            # base_info may still be None if num_customers == 0 (should not
            # occur in practice, but guard defensively).
            if base_info is not None:
                final_agent_cost = info["current_cost"]
                final_base_cost = base_info["current_cost"]
                avg_cost_agent = final_agent_cost.mean().item()
                avg_cost_base = final_base_cost.mean().item()
                # Avoid divide-by-zero if baseline cost is 0
                avg_pct_imp = (
                    (avg_cost_base - avg_cost_agent) / avg_cost_base * 100
                    if avg_cost_base != 0.0 else 0.0
                )
                writer.add_scalar("metrics/avg_cost_base", avg_cost_base, epoch)
                writer.add_scalar("metrics/avg_cost_agent", avg_cost_agent, epoch)
                writer.add_scalar("metrics/avg_percent_improvement", avg_pct_imp, epoch)
            else:
                avg_pct_imp = 0.0

            avg_reward = epoch_improvement.mean().item()
            avg_acc = (epoch_accepted / num_customers).mean().item()

            print(
                f"[Epoch {epoch}/{num_epochs}] "
                f"Rew: {avg_reward:.2f} | "
                f"Imp: {avg_pct_imp:.2f}% | "
                f"Acc: {avg_acc:.2%}"
            )

            writer.add_scalar("metrics/avg_reward", avg_reward, epoch)
            writer.add_scalar("metrics/avg_acceptance", avg_acc, epoch)
            writer.add_scalar("metrics/best_improvement", epoch_improvement.max().item(), epoch)
            writer.add_scalar("metrics/worst_improvement", epoch_improvement.min().item(), epoch)
            writer.add_scalar("metrics/median_improvement", epoch_improvement.median().item(), epoch)
            writer.add_scalar("metrics/std_improvement", epoch_improvement.std().item(), epoch)
            writer.add_scalar(
                "metrics/avg_cost_diff_per_step", avg_reward / num_customers, epoch
            )

        # ------------------------------------------------------------------
        # 10. Checkpoint
        # ------------------------------------------------------------------
        if epoch % save_interval == 0:
            agent.save(str(ckpt_path / f"policy_ep{epoch}.pt"))

    print("[Train] Training complete.")
    agent.save(str(ckpt_path / "policy_final.pt"))
    writer.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse command-line arguments and launch training."""
    parser = argparse.ArgumentParser(
        description="Train TestPPOAgent (49 actions, frozen oracle encoder) on TestDVRPEnv"
    )
    parser.add_argument(
        "--episodes", type=int, default=2000,
        help="Total number of training episodes (epochs = episodes // num-envs)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=64,
        help="Number of parallel environments in the batch",
    )
    parser.add_argument(
        "--customers", type=int, default=30,
        help="Number of customer requests per episode",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="PyTorch device string (cpu / cuda / mps)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ckpt-dir", type=str, default="./checkpoints/test_ppo",
        help="Directory to write agent checkpoints",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./runs/test_ppo",
        help="Directory for TensorBoard SummaryWriter logs",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to an existing checkpoint to resume training from",
    )
    parser.add_argument(
        "--solver-ckpt", type=str, default=None,
        help="Path to the oracle attention policy checkpoint (encoder weights)",
    )
    parser.add_argument(
        "--save-interval", type=int, default=10,
        help="Save a checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-interval", type=int, default=1,
        help="Log metrics every N epochs",
    )
    parser.add_argument(
        "--no-mask", action="store_true", default=False,
        help="Disable action masking (all 49 actions always permitted)",
    )
    parser.add_argument(
        "--force-accept", action="store_true", default=False,
        help="Force all proposed perturbations to be accepted by travellers",
    )

    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        num_customers=args.customers,
        num_envs=args.num_envs,
        ckpt_dir=args.ckpt_dir,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
        resume_path=args.resume,
        solver_ckpt=args.solver_ckpt,
        no_mask=args.no_mask,
        force_accept=args.force_accept,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick smoke test with minimal config — verifies imports and training loop
    # structure without requiring a GPU or a full dataset.
    train(
        num_episodes=64,
        num_customers=5,
        num_envs=2,
        ckpt_dir="/tmp/test_ckpt",
        log_dir="/tmp/test_log",
        save_interval=10,
        log_interval=1,
        device="cpu",
        seed=42,
        resume_path=None,
        solver_ckpt="/Users/jiangwolin/Desktop/Research/DARPSolver/checkpoints/refined/best.pt",
        force_accept=True,
    )
    print("Training smoke test passed!")
