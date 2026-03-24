"""PPO training loop for NDVRP perturbation optimisation.

Generates DARP instances via SFGenerator, runs the PPO agent through
sequential customer arrivals, and trains via clipped PPO with per-step
rewards  r_t = baseline_cost_t − ppo_cost_t − penalty.
"""

import argparse
import logging
import time
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from oracle_generator import SFGenerator
from oracle_policy import PDPTWAttentionPolicy
from ndvrp_env import NDVRPEnv
from ndvrp_ppo import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def load_solver_policy(ckpt_path: str, device: str) -> PDPTWAttentionPolicy:
    policy = PDPTWAttentionPolicy(embed_dim=128, num_encoder_layers=3, num_heads=8)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy.to(device)


def train(args):
    device = args.device
    log.info("Device: %s", device)

    # Generator
    generator = SFGenerator(
        num_customers=args.num_customers,
        perturbation=0,  # no generator-level perturbation; PPO handles it
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

    # PPO agent
    agent = PPOAgent(
        checkpoint_path=args.solver_ckpt,
        embed_dim=128,
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

    # Logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_improvement = -float("inf")

    for episode in range(args.episodes): #! Parallel env would load multiple episodes
        # Each episode runs num_envs (default 512) instances in parallel as a single batch.
        t0 = time.time()
        obs = env.reset() # This is a parallel env

        # Collection buffers
        states_buf = []
        actions_buf, log_probs_buf, values_buf = [], [], []
        rewards_buf, dones_buf, masks_buf = [], [], []
        final_ppo_cost = torch.zeros(args.num_envs, device=device)

        for step in tqdm(range(args.num_customers), desc=f"Ep {episode}", leave=False):
            state = agent.encode_obs(obs)
            mask = obs["action_mask"]
            action, lp, val = agent.select_action(state, mask)

            obs, reward, done, info = env.step(action)
            reward = reward - args.penalty  # per-step penalty

            states_buf.append(state)
            actions_buf.append(action)
            log_probs_buf.append(lp)
            values_buf.append(val)
            rewards_buf.append(reward)
            dones_buf.append(done)
            masks_buf.append(mask)

            final_ppo_cost = info["ppo_cost"]

        # Filter out stuck envs, stack buffer: [T*B'] where B' = valid envs
        valid = ~env.stuck_mask  # [B]
        num_valid = valid.sum().item()
        if num_valid == 0:
            continue  # entire batch stuck, skip update

        def interleave(lst):
            stacked = torch.stack(lst, dim=0)        # [T, B, ...]
            return stacked[:, valid].reshape(-1, *stacked.shape[2:]).squeeze(-1) \
                if stacked.dim() > 2 else stacked[:, valid].reshape(-1)

        buffer = {
            "states": torch.stack(states_buf, dim=0)[:, valid].reshape(-1, 256),
            "actions": interleave(actions_buf),
            "old_log_probs": interleave(log_probs_buf),
            "old_values": interleave(values_buf),
            "rewards": interleave(rewards_buf),
            "dones": interleave(dones_buf),
            "masks": torch.stack(masks_buf, dim=0)[:, valid].reshape(-1, 49),
            "num_envs": num_valid,
        }

        update_info = agent.update(buffer)

        # ------ Logging ----------------------------------------------------- #
        dt = time.time() - t0
        n_nodes = 2 * args.num_customers + 1
        no_pert_cost = env._eval_cost(env.original_tw, n_nodes)
        mean_ppo = final_ppo_cost[valid].mean().item()
        mean_base = no_pert_cost[valid].mean().item()
        mean_reward = (mean_base - mean_ppo)
        improvement = mean_reward / (abs(mean_base) + 1e-8) * 100

        writer.add_scalar("train/reward", mean_reward, episode)
        writer.add_scalar("train/ppo_cost", mean_ppo, episode)
        writer.add_scalar("train/baseline_cost", mean_base, episode)
        writer.add_scalar("train/improvement_pct", improvement, episode)
        writer.add_scalar("train/policy_loss", update_info["policy_loss"], episode)
        writer.add_scalar("train/value_loss", update_info["value_loss"], episode)
        writer.add_scalar("train/entropy", update_info["entropy"], episode)

        if episode % args.log_interval == 0 and episode > 0:
            log.info(
                "ep %4d | reward %.1f | ppo_cost %.1f | base_cost %.1f | "
                "impr %.2f%% | p_loss %.4f | v_loss %.4f | ent %.3f | %.1fs",
                episode, mean_reward, mean_ppo, mean_base, improvement,
                update_info["policy_loss"], update_info["value_loss"],
                update_info["entropy"], dt,
            )

        # Checkpoint
        if improvement > best_improvement:
            best_improvement = improvement
            _save_checkpoint(agent, ckpt_dir / "best.pt", episode, improvement)

        if (episode + 1) % args.save_interval == 0:
            _save_checkpoint(agent, ckpt_dir / f"ep_{episode+1}.pt", episode, improvement)

    writer.close()
    log.info("Training complete. Best improvement: %.2f%%", best_improvement)


def _save_checkpoint(agent: PPOAgent, path: Path, episode: int, improvement: float):
    torch.save({
        "episode": episode,
        "improvement": improvement,
        "policy_state_dict": agent.policy.state_dict(),
        "value_state_dict": agent.value.state_dict(),
        "policy_optim": agent.policy_optim.state_dict(),
        "value_optim": agent.value_optim.state_dict(),
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    # Environment
    p.add_argument("--num-customers", type=int, default=30)
    p.add_argument("--num-envs", type=int, default=512)
    p.add_argument("--free-vehicles", type=int, default=5)
    p.add_argument("--vehicle-penalty", type=float, default=200.0)
    p.add_argument("--penalty", type=float, default=0, help="per-step perturbation penalty")
    # PPO
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr-policy", type=float, default=3e-4)
    p.add_argument("--lr-value", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--value-epochs", type=int, default=10)
    p.add_argument("--policy-epochs", type=int, default=4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    # Training
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--solver-ckpt", type=str, default="checkpoints/refined/best.pt")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints/ndvrp_ppo")
    p.add_argument("--log-dir", type=str, default="runs/ndvrp_ppo")
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--save-interval", type=int, default=10)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
