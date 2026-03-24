"""PPO training loop for NDVRP perturbation optimisation.

Generates DARP instances via SFGenerator, runs the PPO agent through
sequential customer arrivals, and trains via clipped PPO with per-step
rewards  r_t = baseline_cost_t − ppo_cost_t − penalty.

Includes traveler personality modelling: an embedding model predicts each
traveler's flexibility type, and perturbation actions are filtered through
an accept/reject lookup so that only traveler-accepted perturbations are
applied.

Training structure:
    epoch  →  20 episodes (each episode = one parallel env rollout)
    PPO update happens once per epoch over the buffered trajectories.
    Embedding model is updated every 5 epochs.
"""

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from oracle_generator import SFGenerator
from oracle_policy import PDPTWAttentionPolicy
from ndvrp_env import NDVRPEnv, NUM_ACTIONS, P_SHIFTS, D_SHIFTS, NO_PERTURBATION_ACTION
from ndvrp_ppo import PPOAgent
from dvrp_embedding import EmbeddingFFN, OnlineTravelerDataset, likelihood_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EPISODES_PER_EPOCH = 10

ACTION_SPACE_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (  0, 0), (  0, 10), (  0, 20), (  0, 30),
]

FLEXIBILITY_PERSONALITIES = [
    "flexible for late dropoff, but inflexible for early pickup",
    "flexible for early pickup, but inflexible for late dropoff",
    "inflexible for any schedule changes",
    "flexible for both early pickup and late dropoff",
]


# --------------------------------------------------------------------------- #
# Epsilon schedule for personality exploration
# --------------------------------------------------------------------------- #
def get_personality_epsilon(epoch: int) -> float:
    """Epsilon for personality exploration.

    epoch <= 5:  1.0  (fully random personality)
    epoch > 5:   starts at 0.2, decreases 0.02 per epoch until 0
    """
    if epoch <= 5:
        return 1.0
    return max(0.0, 0.2 - 0.02 * (epoch - 6))


# --------------------------------------------------------------------------- #
# Decision lookup
# --------------------------------------------------------------------------- #
def build_decision_lookup(csv_path: str) -> dict:
    """Load traveler decisions CSV and build a hash-map for O(1) lookups.

    Key: (traveler_id, trip_purpose, departure_loc, arrival_loc,
          pickup_shift_abs, dropoff_shift_abs)
    Value: {flexibility_column: bool}
    """
    df = pd.read_csv(csv_path)
    lookup = {}
    for _, row in df.iterrows():
        key = (
            int(row["traveler_id"]),
            str(row["trip_purpose"]).strip(),
            str(row["departure_location"]).strip(),
            str(row["arrival_location"]).strip(),
            int(row["pickup_shift_min"]),
            int(row["dropoff_shift_min"]),
        )
        decisions = {}
        for col in FLEXIBILITY_PERSONALITIES:
            if col in df.columns:
                decisions[col] = str(row[col]).strip().lower() == "accept"
        lookup[key] = decisions
    log.info("Decision lookup: %d entries from %s", len(lookup), csv_path)
    return lookup


# --------------------------------------------------------------------------- #
# Acceptance masking
# --------------------------------------------------------------------------- #
def get_acceptance_masks(
    decision_lookup: dict,
    trip_metadata: list,
    user_ids: torch.Tensor,
    target_flexs: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute per-env action masks from predicted flexibility types.

    Returns:
        masks: [B, 16] binary mask (1 = traveler would accept)
    """
    B = user_ids.shape[0]
    masks = torch.ones(B, NUM_ACTIONS, dtype=torch.float32, device=device)

    for i in range(B):
        uid = int(user_ids[i].item())
        meta = trip_metadata[i].get(uid)
        if meta is None:
            continue

        flex_col = FLEXIBILITY_PERSONALITIES[target_flexs[i].item()]
        base_key = (
            uid,
            str(meta["trip_purpose"]).strip(),
            str(meta.get("departure_location", "")).strip(),
            str(meta.get("arrival_location", "")).strip(),
        )

        for j, (ps, ds) in enumerate(ACTION_SPACE_MAP):
            key = base_key + (abs(ps), abs(ds))
            decisions = decision_lookup.get(key)
            if decisions:
                masks[i, j] = float(decisions.get(flex_col, False))
            else:
                masks[i, j] = 0.0

    return masks


def check_acceptance(
    decision_lookup: dict,
    trip_metadata: list,
    user_ids: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Check whether each traveler accepts the chosen action using their
    **ground truth** flexibility type from trip_metadata.

    Returns:
        accepted: [B] bool tensor
    """
    B = user_ids.shape[0]
    accepted = torch.ones(B, dtype=torch.bool, device=actions.device)

    for i in range(B):
        a = int(actions[i].item())
        if a == NO_PERTURBATION_ACTION:
            continue

        uid = int(user_ids[i].item())
        meta = trip_metadata[i].get(uid)
        if meta is None:
            accepted[i] = False
            continue

        # Use ground truth flexibility, not predicted
        true_flex = str(meta.get("flexibility", "")).strip()
        if true_flex not in FLEXIBILITY_PERSONALITIES:
            accepted[i] = False
            continue

        ps, ds = ACTION_SPACE_MAP[a]
        key = (
            uid,
            str(meta["trip_purpose"]).strip(),
            str(meta.get("departure_location", "")).strip(),
            str(meta.get("arrival_location", "")).strip(),
            abs(ps), abs(ds),
        )
        decisions = decision_lookup.get(key)
        if decisions:
            accepted[i] = decisions.get(true_flex, False)
        else:
            accepted[i] = False

    return accepted


# --------------------------------------------------------------------------- #
# Online data collection
# --------------------------------------------------------------------------- #
def collect_online_samples(
    trip_metadata: list,
    user_ids: torch.Tensor,
    actions: torch.Tensor,
    accepted: torch.Tensor,
) -> list:
    """Build records for embedding dataset update."""
    records = []
    B = user_ids.shape[0]
    for i in range(B):
        uid = int(user_ids[i].item())
        meta = trip_metadata[i].get(uid, {})
        records.append({
            "traveler_id": uid,
            "action": int(actions[i].item()),
            "accepted": int(accepted[i].item()),
            "trip_purpose": meta.get("trip_purpose"),
            "departure_location": meta.get("departure_location"),
            "arrival_location": meta.get("arrival_location"),
            "departure_time_window": str(meta.get("departure_time_window")),
            "arrival_time_window": str(meta.get("arrival_time_window")),
        })
    return records


# --------------------------------------------------------------------------- #
# Embedding update
# --------------------------------------------------------------------------- #
def update_embedding(embedding_model, online_data, device):
    """Retrain the embedding model on accumulated online data."""
    df = pd.DataFrame(list(online_data)).dropna(
        subset=["traveler_id", "action", "accepted", "trip_purpose",
                "departure_location", "arrival_location",
                "departure_time_window", "arrival_time_window"]
    ).sample(frac=1).reset_index(drop=True)

    if len(df) < 64:
        log.info("  [Embedding] Skipping update (insufficient rows: %d)", len(df))
        return

    split = int(0.8 * len(df))
    ds_train = OnlineTravelerDataset(df.iloc[:split], FLEXIBILITY_PERSONALITIES, ACTION_SPACE_MAP)
    ds_eval = OnlineTravelerDataset(df.iloc[split:], FLEXIBILITY_PERSONALITIES, ACTION_SPACE_MAP)
    dl_train = DataLoader(ds_train, batch_size=128, shuffle=True)
    dl_eval = DataLoader(ds_eval, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(embedding_model.parameters(), lr=1e-3)
    loss_kwargs = dict(alpha_e=0.5, alpha=0.4, eps=1e-9, unbiased_var=False)

    embedding_model.train()
    for sub_epoch in range(30):
        total_loss = 0.0
        for entity_ids, _, ind_matrix, _ in dl_train:
            entity_ids, ind_matrix = entity_ids.to(device), ind_matrix.to(device)
            optimizer.zero_grad()
            loss = likelihood_loss(
                embedding_model.get_embed(),
                embedding_model(entity_ids),
                ind_matrix, **loss_kwargs,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(entity_ids)

        if sub_epoch % 10 == 0:
            embedding_model.eval()
            eval_loss = sum(
                likelihood_loss(
                    embedding_model.get_embed(),
                    embedding_model(eid.to(device)),
                    im.to(device), **loss_kwargs,
                ).item() * len(eid)
                for eid, _, im, _ in dl_eval
            )
            log.info("  [Embedding] sub-epoch %d: train=%.4f eval=%.4f",
                      sub_epoch, total_loss / len(ds_train), eval_loss / len(ds_eval))
            embedding_model.train()

    embedding_model.eval()


# --------------------------------------------------------------------------- #
# Solver policy loader
# --------------------------------------------------------------------------- #
def load_solver_policy(ckpt_path: str, device: str) -> PDPTWAttentionPolicy:
    policy = PDPTWAttentionPolicy(embed_dim=128, num_encoder_layers=3, num_heads=8)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy.to(device)


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def train(args):
    device = args.device
    log.info("Device: %s", device)

    # Generator
    generator = SFGenerator(
        num_customers=args.num_customers,
        perturbation=0,
        device=device,
    )

    # Solver policy (for routing cost evaluation)
    solver_policy = load_solver_policy(args.solver_ckpt, device)

    # Environment
    env = NDVRPEnv(
        generator=generator,
        solver_policy=solver_policy,
        num_envs=args.num_envs,
        num_customers=args.num_customers,
        free_vehicles=args.free_vehicles,
        vehicle_penalty=args.vehicle_penalty,
        device=device,
    )

    # PPO agent (GCN-based)
    agent = PPOAgent(
        travel_time_matrix=generator.travel_time_matrix,
        num_customers=args.num_customers,
        hidden_dim=args.hidden_dim,
        lr_policy=args.lr_policy,
        lr_value=args.lr_value,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        value_epochs=args.value_epochs,
        policy_epochs=args.policy_epochs,
        entropy_coef=args.entropy_coef,
        device=device,
    )

    # Decision lookup for accept/reject
    decisions_csv = Path(__file__).with_name("traveler_decisions_augmented.csv")
    decision_lookup = build_decision_lookup(str(decisions_csv))

    # Embedding model for personality prediction
    embedding_model = EmbeddingFFN(
        num_entities=args.num_customers,
        embed_dim=64, hidden_dim=128, output_dim=len(FLEXIBILITY_PERSONALITIES),
    ).to(device)
    embedding_model.eval()

    online_data: deque = deque(maxlen=512_000)

    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_improvement = -float("inf")

    log.info("Training: %d epochs x %d episodes x %d envs x %d steps",
             args.epochs, EPISODES_PER_EPOCH, args.num_envs, args.num_customers)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Accumulate trajectories across all episodes in this epoch
        all_states, all_actions, all_lp, all_val = [], [], [], []
        all_rewards, all_dones, all_masks = [], [], []
        all_ppo_costs, all_base_costs = [], []
        total_valid = 0
        total_nonzero = 0
        total_steps = 0
        all_penalties = []

        for ep in range(EPISODES_PER_EPOCH):
            obs = env.reset()
            trip_metadata = env.td["trip_metadata"]
            n_nodes = 2 * args.num_customers + 1
            no_pert_cost = env._eval_cost(env.original_tw, n_nodes)

            states_buf = []
            actions_buf, log_probs_buf, values_buf = [], [], []
            rewards_buf, dones_buf, masks_buf = [], [], []
            ep_penalty = torch.zeros(args.num_envs, device=device)

            for step in tqdm(range(args.num_customers),
                             desc=f"Epoch {epoch} ep {ep+1}/{EPISODES_PER_EPOCH}",
                             leave=False):

                # --- Agent action selection (feasibility mask only) --- #
                mask = obs["action_mask"]  # [B, 16]
                state = agent.encode_obs(obs)
                action, lp, val = agent.select_action(state, mask)

                # --- All actions accepted (acceptance disabled) --- #
                nonzero = (action != NO_PERTURBATION_ACTION)
                total_nonzero += nonzero.sum().item()
                total_steps += args.num_envs

                # --- Environment step --- #
                obs, reward, done, info = env.step(action)

                # Perturbation penalty: proportional to 1-norm of shifts
                p_shift = P_SHIFTS.to(device)[action].abs()
                d_shift = D_SHIFTS.to(device)[action].abs()
                step_penalty = args.pert_penalty * (p_shift + d_shift)
                ep_penalty += step_penalty
                reward = reward - step_penalty

                states_buf.append(state)
                actions_buf.append(action)
                log_probs_buf.append(lp)
                values_buf.append(val)
                rewards_buf.append(reward)
                dones_buf.append(done)
                masks_buf.append(mask)

            # Filter stuck envs for this episode
            valid = ~env.stuck_mask
            num_valid = valid.sum().item()
            if num_valid == 0:
                continue

            def interleave(lst, v=valid):
                stacked = torch.stack(lst, dim=0)
                return stacked[:, v].reshape(-1, *stacked.shape[2:]).squeeze(-1) \
                    if stacked.dim() > 2 else stacked[:, v].reshape(-1)

            C = states_buf[0].shape[1]
            all_states.append(torch.stack(states_buf, dim=0)[:, valid].reshape(-1, C, 6))
            all_actions.append(interleave(actions_buf))
            all_lp.append(interleave(log_probs_buf))
            all_val.append(interleave(values_buf))
            all_rewards.append(interleave(rewards_buf))
            all_dones.append(interleave(dones_buf))
            all_masks.append(torch.stack(masks_buf, dim=0)[:, valid].reshape(-1, NUM_ACTIONS))

            all_ppo_costs.append(info["ppo_cost"][valid])
            all_base_costs.append(no_pert_cost[valid])
            all_penalties.append(ep_penalty[valid])
            total_valid += num_valid

        if total_valid == 0:
            continue

        # --- PPO update (once per epoch over all buffered episodes) --- #
        buffer = {
            "states": torch.cat(all_states),
            "actions": torch.cat(all_actions),
            "old_log_probs": torch.cat(all_lp),
            "old_values": torch.cat(all_val),
            "rewards": torch.cat(all_rewards),
            "dones": torch.cat(all_dones),
            "masks": torch.cat(all_masks),
            "num_envs": total_valid,
        }

        update_info = agent.update(buffer)

        # --- Embedding update (disabled) --- #
        # if online_data and epoch % 5 == 0:
        #     log.info("[Embedding] Updating with %d samples...", len(online_data))
        #     update_embedding(embedding_model, online_data, device)

        # ------ Logging ----------------------------------------------------- #
        dt = time.time() - t0
        mean_ppo = torch.cat(all_ppo_costs).mean().item()
        mean_base = torch.cat(all_base_costs).mean().item()
        mean_penalty = torch.cat(all_penalties).mean().item()
        mean_reward = mean_base - mean_ppo - mean_penalty
        improvement = mean_reward / (abs(mean_base) + 1e-8) * 100
        perturb_rate = total_nonzero / max(total_steps, 1)

        writer.add_scalar("train/reward", mean_reward, epoch)
        writer.add_scalar("train/ppo_cost", mean_ppo, epoch)
        writer.add_scalar("train/baseline_cost", mean_base, epoch)
        writer.add_scalar("train/improvement_pct", improvement, epoch)
        writer.add_scalar("train/policy_loss", update_info["policy_loss"], epoch)
        writer.add_scalar("train/value_loss", update_info["value_loss"], epoch)
        writer.add_scalar("train/entropy", update_info["entropy"], epoch)
        writer.add_scalar("train/perturbation_rate", perturb_rate, epoch)
        writer.add_scalar("train/penalty", mean_penalty, epoch)

        last_actions = torch.stack(actions_buf)[:, -1]
        action_strs = [f"({int(P_SHIFTS[a])},{int(D_SHIFTS[a])})" for a in last_actions]
        log.info(
            "epoch %3d | reward %.1f | ppo %.1f | base %.1f | "
            "impr %.2f%% | pert %.1f%% | p_loss %.4f | v_loss %.4f | "
            "ent %.3f | valid %d | %.1fs",
            epoch, mean_reward, mean_ppo, mean_base, improvement,
            perturb_rate * 100,
            update_info["policy_loss"], update_info["value_loss"],
            update_info["entropy"], total_valid, dt,
        )
        log.info("  actions (last env): %s", " ".join(action_strs))

        # Log predicted flexibility types for all 30 travelers
        with torch.no_grad():
            all_ids = torch.arange(args.num_customers, device=device)
            proba = embedding_model(all_ids)  # [30, 4]
            pred_types = proba.argmax(dim=1)  # [30]
            short_names = ["late_do", "early_pu", "inflx", "both"]
            type_strs = [short_names[t] for t in pred_types.tolist()]
            log.info("  predicted flex: %s", " ".join(type_strs))

        # Checkpoint
        if improvement > best_improvement:
            best_improvement = improvement
            _save_checkpoint(agent, embedding_model, ckpt_dir / "best.pt",
                             epoch, improvement)

        if epoch % args.save_interval == 0:
            _save_checkpoint(agent, embedding_model, ckpt_dir / f"epoch_{epoch}.pt",
                             epoch, improvement)

    writer.close()
    log.info("Training complete. Best improvement: %.2f%%", best_improvement)


def _save_checkpoint(agent: PPOAgent, embedding_model: EmbeddingFFN,
                     path: Path, epoch: int, improvement: float):
    torch.save({
        "epoch": epoch,
        "improvement": improvement,
        "actor_critic_state_dict": agent.actor_critic.state_dict(),
        "policy_optim": agent.policy_optim.state_dict(),
        "value_optim": agent.value_optim.state_dict(),
        "embedding_state_dict": embedding_model.state_dict(),
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    # Environment
    p.add_argument("--num-customers", type=int, default=30)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--free-vehicles", type=int, default=5)
    p.add_argument("--vehicle-penalty", type=float, default=200.0)
    p.add_argument("--pert-penalty", type=float, default=0.001,
                   help="penalty coefficient on 1-norm of perturbation shifts")
    # PPO
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr-policy", type=float, default=3e-4)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-epochs", type=int, default=20)
    p.add_argument("--policy-epochs", type=int, default=6)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    # Training
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--solver-ckpt", type=str, default="checkpoints/refined/best.pt")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/ndvrp_ppo")
    p.add_argument("--log-dir", type=str, default="runs/ndvrp_ppo")
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
