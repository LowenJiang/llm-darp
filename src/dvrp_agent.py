"""
Proximal Policy Optimization (PPO) Agent

Implements PPO for the DVRP environment using PyTorch.

Changes from previous version:
  - Removed dead `mod` tensor addition in forward()
  - forward() now returns LOGITS (not probs) — masking and Categorical use logits directly
  - evaluate() accepts optional masks for consistent log-prob computation
  - select_action_batch() stores masks in buffer; removed dist_unmasked overwrite
  - Optimizers are created once in __init__, not recreated every update()
  - Value training calls value_network directly, not full forward()
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

# ==============================================================================
#   ENCODING BLOCK
# ==============================================================================

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, travel_time_matrix):
        super().__init__()
        self.register_buffer("travel_time_matrix", travel_time_matrix)
        self.register_buffer("travel_time_matrix_norm", self.normalize_adj(travel_time_matrix))
        self.gc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.gc2 = nn.Linear(hidden_dim, out_dim, bias=False)
    
    def normalize_adj(self, A):
        I = torch.eye(A.size(0), device=A.device)
        A_hat = A + I
        deg = A_hat.sum(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        return D_inv_sqrt @ A_hat @ D_inv_sqrt

    def forward(self, X):
        h = torch.relu(self.travel_time_matrix_norm @ self.gc1(X))
        h = self.travel_time_matrix_norm @ self.gc2(h)
        return h


class TripRequestEmbeddingModel(nn.Module):
    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=20,
        num_heads=4,
        num_layers=3,
        transformer_embed_dim=80
    ):
        super().__init__()

        num_nodes = travel_time_matrix.shape[0]
        self.time_vocab_size = time_vocab_size
        
        self.register_buffer("node_features", torch.eye(num_nodes))

        self.gcn = GCN(
            in_dim=num_nodes,
            hidden_dim=gcn_hidden,
            out_dim=gcn_out,
            travel_time_matrix=travel_time_matrix
        )

        self.pickup_embed  = nn.Embedding(time_vocab_size, time_embed_dim) 
        self.dropoff_embed = nn.Embedding(time_vocab_size, time_embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=num_heads,
            dim_feedforward=transformer_embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        total_dim = gcn_out * 2 + time_embed_dim * 4
        self.project = nn.Linear(total_dim, transformer_embed_dim)

    def forward(self, states):
        """
        states: (B, C, 6)
        """
        B, C, _ = states.shape

        node_emb = self.gcn(self.node_features)  # (N, gcn_out)

        origin_idx      = states[..., 0].long()
        pickup_tw_early = (states[..., 1].long() - 7*60) // 30
        pickup_tw_late  = (states[..., 2].long() - 7*60) // 30
        dest_idx        = states[..., 3].long()
        drop_tw_early   = (states[..., 4].long() - 7*60) // 30
        drop_tw_late    = (states[..., 5].long() - 7*60) // 30

        max_idx = self.time_vocab_size - 1
        pickup_tw_early = torch.clamp(pickup_tw_early, 0, max_idx) 
        pickup_tw_late  = torch.clamp(pickup_tw_late, 0, max_idx)
        drop_tw_early   = torch.clamp(drop_tw_early, 0, max_idx)
        drop_tw_late    = torch.clamp(drop_tw_late, 0, max_idx)

        origin_emb = node_emb[origin_idx]
        dest_emb   = node_emb[dest_idx]

        pickup_emb_early  = self.pickup_embed(pickup_tw_early)
        pickup_emb_late   = self.pickup_embed(pickup_tw_late)
        dropoff_emb_early = self.dropoff_embed(drop_tw_early)
        dropoff_emb_late  = self.dropoff_embed(drop_tw_late)

        trip_emb = torch.cat(
            [origin_emb, dest_emb, pickup_emb_early, pickup_emb_late,
             dropoff_emb_early, dropoff_emb_late], dim=-1
        )

        padding_mask = (states.abs().sum(dim=-1) == 0)
        trip_emb = trip_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        trip_emb = self.project(trip_emb)

        out = self.transformer(trip_emb, src_key_padding_mask=padding_mask)
        return out


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        time_embed_dim: int = 16,
        time_vocab_size=20,
        transformer_embed_dim: int = 64,
        action_dim: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super(PolicyNetwork, self).__init__()
        
        self.backbone = TripRequestEmbeddingModel(
            travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out,
            time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            num_heads=num_heads, num_layers=num_layers
        )

        self.network = nn.Sequential(
            nn.Linear(transformer_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, states):
        state_vector = self.backbone(states)       # (B, C, D) 
        state_vector = state_vector.mean(dim=1)    # (B, D)
        return self.network(state_vector)           # (B, action_dim) — RAW LOGITS


class ValueNetwork(nn.Module):
    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        time_embed_dim: int = 16,
        time_vocab_size=500,
        transformer_embed_dim: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super(ValueNetwork, self).__init__()
        
        self.backbone = TripRequestEmbeddingModel(
            travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out,
            time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            num_heads=num_heads, num_layers=num_layers
        )

        self.network = nn.Sequential(
            nn.Linear(transformer_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, states):
        state_vector = self.backbone(states)
        state_vector = state_vector.mean(dim=1)
        return self.network(state_vector)


class ActorCritic(nn.Module):
    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        time_embed_dim: int = 16,
        time_vocab_size=500,
        transformer_embed_dim: int = 64,
        action_dim: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super(ActorCritic, self).__init__()
        
        self.policy_network = PolicyNetwork(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden,
            gcn_out=gcn_out, time_embed_dim=time_embed_dim,
            time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            action_dim=action_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers
        )
        self.value_network = ValueNetwork(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden,
            gcn_out=gcn_out, time_embed_dim=time_embed_dim,
            time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            hidden_dim=hidden_dim, num_heads=num_heads,
            num_layers=num_layers
        )

    def get_logits_and_value(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns raw action logits and state value.
        No softmax — let Categorical(logits=...) handle numerics.
        """
        logits = self.policy_network(state)          # (B, action_dim)
        value  = self.value_network(state)            # (B, 1)
        return logits, value

    def get_distribution(
        self,
        states: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor]:
        """
        Build a (possibly masked) Categorical distribution and return value.

        Args:
            states: (B, C, 6)
            masks:  (B, action_dim) with 1=allowed, 0=forbidden.  None = no mask.

        Returns:
            dist:   Categorical distribution (masked if masks provided)
            values: (B, 1)
        """
        logits, values = self.get_logits_and_value(states)

        if masks is not None:
            # Set forbidden actions to -inf BEFORE softmax → zero probability
            logits = logits.masked_fill(masks == 0, float('-inf'))

        dist = Categorical(logits=logits)
        return dist, values

    def evaluate(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate states and actions for PPO update.
        Applies the SAME mask that was used during data collection.

        Returns:
            log_probs:    (B,)
            state_values: (B,)
            entropy:      (B,)
        """
        dist, values = self.get_distribution(states, masks)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent.
    """

    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        time_embed_dim: int = 16,
        time_vocab_size=500,
        transformer_embed_dim: int = 64,
        action_dim: int = 16,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        value_lr: Dict[str, float] | float | None = None,
        policy_lr: Dict[str, float] | float | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.1,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.policy = ActorCritic(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden,
            gcn_out=gcn_out, time_embed_dim=time_embed_dim,
            time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            action_dim=action_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers
        ).to(self.device)

        # --- Learning rate setup ---
        if value_lr is None:
            value_lr = {"backbone": 1e-4, "network": 1e-3}
        elif isinstance(value_lr, (int, float)):
            value_lr = {"backbone": float(value_lr), "network": float(value_lr)}
        if policy_lr is None:
            policy_lr = {"backbone": 1e-4, "network": 1e-3}
        elif isinstance(policy_lr, (int, float)):
            policy_lr = {"backbone": float(policy_lr), "network": float(policy_lr)}

        self.value_lr = value_lr
        self.policy_lr = policy_lr

        # --- FIX: Create optimizers ONCE so Adam momentum persists ---
        self.policy_optimizer = optim.Adam([
            {"params": self.policy.policy_network.backbone.parameters(),
             "lr": self.policy_lr["backbone"]},
            {"params": self.policy.policy_network.network.parameters(),
             "lr": self.policy_lr["network"]},
        ])
        self.value_optimizer = optim.Adam([
            {"params": self.policy.value_network.backbone.parameters(),
             "lr": self.value_lr["backbone"]},
            {"params": self.policy.value_network.network.parameters(),
             "lr": self.value_lr["network"]},
        ])

        # --- Rollout buffer ---
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []
        self.masks: List[Optional[torch.Tensor]] = []   # NEW: store masks

    def select_action_batch(
        self,
        states,
        masks: Optional[torch.Tensor] = None,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Select actions for a batch of states.

        Masking is done at the logit level (−inf before softmax).
        Epsilon mixes masked and unmasked distributions for exploration.
        Log-probs are ALWAYS computed from the distribution that was sampled.

        Args:
            states:  (B, C, 6) tensor or ndarray
            masks:   (B, action_dim) float tensor, 1=allowed 0=forbidden. None=no mask.
            epsilon: 0 → pure masked policy.  1 → ignore mask entirely.

        Returns:
            actions: (B,) numpy int array
        """
        batch_size = states.shape[0]

        # --- Tensor conversion ---
        if isinstance(states, torch.Tensor):
            state_tensor = states.to(self.device, dtype=torch.float32)
        else:
            state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)

        if masks is not None:
            if isinstance(masks, list):
                masks = torch.stack([
                    m if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.float32)
                    for m in masks
                ])
            elif not isinstance(masks, torch.Tensor):
                masks = torch.tensor(masks, dtype=torch.float32)
            masks = masks.to(self.device)

        with torch.no_grad():
            if torch.isnan(state_tensor).any():
                state_tensor = torch.nan_to_num(state_tensor, 0.0)

            # Get logits and values
            logits, state_values = self.policy.get_logits_and_value(state_tensor)

            # --- Masking at the logit level ---
            if masks is not None and epsilon < 1.0:
                masked_logits = logits.masked_fill(masks == 0, float('-inf'))

                if epsilon > 0.0:
                    # Mix: sample from masked with prob (1-eps), unmasked with prob eps
                    # Equivalent to mixing the two distributions' probs
                    masked_probs  = torch.softmax(masked_logits, dim=-1)
                    unmasked_probs = torch.softmax(logits, dim=-1)
                    mixed_probs = (1.0 - epsilon) * masked_probs + epsilon * unmasked_probs
                    mixed_probs = torch.clamp(mixed_probs, min=1e-8)
                    mixed_probs = mixed_probs / mixed_probs.sum(dim=-1, keepdim=True)
                    dist = Categorical(probs=mixed_probs)
                else:
                    dist = Categorical(logits=masked_logits)
            else:
                # No mask or epsilon=1 → use raw logits
                dist = Categorical(logits=logits)

            actions   = dist.sample()
            log_probs = dist.log_prob(actions)
            # ^^^ CRITICAL: log_probs come from the SAME dist we sampled from.
            #     No dist_unmasked overwrite.

        # --- Store rollout data ---
        state_cpu    = state_tensor.cpu()
        actions_cpu  = actions.cpu()
        log_probs_cpu = log_probs.cpu()
        values_cpu   = state_values.squeeze(-1).cpu()

        for i in range(batch_size):
            self.states.append(state_cpu[i])
            self.actions.append(actions_cpu[i].item())
            self.log_probs.append(log_probs_cpu[i])
            self.values.append(values_cpu[i])
            # Store per-sample mask for replay in evaluate()
            if masks is not None:
                self.masks.append(masks[i].cpu())
            else:
                self.masks.append(None)

        return actions.cpu().numpy()

    def store_reward(self, reward: float, done: bool) -> None:
        self.rewards.append(reward)
        self.dones.append(done)

    def store_rewards_batch(self, rewards, dones) -> None:
        for reward, done in zip(rewards, dones):
            self.rewards.append(float(reward))
            self.dones.append(bool(done))

    def update(
        self,
        num_value_epochs: int = 3,
        num_policy_epochs: int = 3,
        batch_size: int = 128,
        num_envs: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> dict:
        """
        PPO update with separate policy and value training.

        Order:
            1. Compute advantages with PRE-UPDATE value (frozen)
            2. Train policy with frozen advantages
            3. Train value with frozen returns

        Masks stored during rollout are replayed in evaluate() so that
        log-probs are computed under the same masked distribution.
        """
        if len(self.states) == 0:
            return {}

        # ========================================
        # Convert buffers
        # ========================================
        states       = torch.stack(self.states).to(self.device)
        actions      = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        rewards      = torch.FloatTensor(self.rewards).to(self.device)
        dones        = torch.FloatTensor(self.dones).to(self.device)
        num_samples  = len(states)

        # Reconstruct masks tensor: (num_samples, action_dim) or None
        has_masks = any(m is not None for m in self.masks)
        if has_masks:
            # For samples without masks, use all-ones (no masking)
            action_dim = self.policy.policy_network.network[-1].out_features
            masks_all = torch.stack([
                m if m is not None else torch.ones(action_dim)
                for m in self.masks
            ]).to(self.device)
        else:
            masks_all = None

        # ================================================
        # STEP 1: Compute advantages with PRE-UPDATE value
        # ================================================
        with torch.no_grad():
            current_values = self.policy.value_network(states).squeeze(-1)

        if num_envs is not None and num_steps is not None:
            advantages, returns = self._compute_gae_parallel(
                rewards, current_values, dones, num_envs, num_steps)
        else:
            advantages, returns = self._compute_gae(
                rewards, current_values, dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns    = returns.detach()
        advantages = advantages.detach()

        # ================================================
        # STEP 2: Train POLICY using frozen advantages
        # ================================================
        total_policy_loss = 0.0
        total_entropy     = 0.0
        num_policy_updates = 0

        for _ in range(num_policy_epochs):
            idx = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                bi  = idx[start:end]

                batch_masks = masks_all[bi] if masks_all is not None else None

                log_probs_new, _, entropy = self.policy.evaluate(
                    states[bi], actions[bi], masks=batch_masks)

                ratio = torch.exp(log_probs_new - old_log_probs[bi])
                surr1 = ratio * advantages[bi]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * advantages[bi]

                policy_loss  = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.entropy_coef * entropy_loss

                self.policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.policy_network.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_entropy     += entropy.mean().item()
                num_policy_updates += 1

        # ================================================
        # STEP 3: Train VALUE against frozen returns
        # ================================================
        total_value_loss  = 0.0
        num_value_updates = 0

        for _ in range(num_value_epochs):
            idx = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                bi  = idx[start:end]

                # FIX: call value_network directly, not full forward()
                predicted_values = self.policy.value_network(states[bi])
                value_loss = nn.MSELoss()(predicted_values.squeeze(-1), returns[bi])

                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.value_network.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                total_value_loss  += value_loss.item()
                num_value_updates += 1

        # ========================================
        # Clear buffer
        # ========================================
        self.clear_buffer()

        return {
            "policy_loss": total_policy_loss / max(num_policy_updates, 1),
            "value_loss":  total_value_loss / max(num_value_updates, 1),
            "entropy":     total_entropy / max(num_policy_updates, 1),
        }

    # ------------------------------------------------------------------
    # GAE
    # ------------------------------------------------------------------

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards.flatten()
        values  = values.flatten()
        dones   = dones.flatten()

        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )

        returns = advantages + values
        return advantages, returns

    def _compute_gae_parallel(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        num_envs: int,
        num_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards.flatten()
        values  = values.flatten()
        dones   = dones.flatten()

        total_samples = len(rewards)

        if total_samples != num_envs * num_steps:
            print(f"WARNING: Buffer size mismatch. Expected {num_envs * num_steps}, got {total_samples}")
            return self._compute_gae(rewards, values, dones)

        # Interleaved layout: [env0_t0, env1_t0, ..., env0_t1, env1_t1, ...]
        rewards_2d = rewards.reshape(num_steps, num_envs).T
        values_2d  = values.reshape(num_steps, num_envs).T
        dones_2d   = dones.reshape(num_steps, num_envs).T

        advantages_2d = torch.zeros_like(rewards_2d)

        for env_idx in range(num_envs):
            last_advantage = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1 or dones_2d[env_idx, t]:
                    next_value = 0.0
                else:
                    next_value = values_2d[env_idx, t + 1]

                delta = (
                    rewards_2d[env_idx, t]
                    + self.gamma * next_value * (1 - dones_2d[env_idx, t])
                    - values_2d[env_idx, t]
                )
                advantages_2d[env_idx, t] = last_advantage = (
                    delta
                    + self.gamma * self.gae_lambda * (1 - dones_2d[env_idx, t]) * last_advantage
                )

        advantages = advantages_2d.T.reshape(-1)
        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # Buffer / IO
    # ------------------------------------------------------------------

    def clear_buffer(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.masks.clear()

    def save(self, path: str) -> None:
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer":  self.policy_optimizer.state_dict(),
            "value_optimizer":   self.value_optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        if "policy_optimizer" in checkpoint:
            self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        if "value_optimizer" in checkpoint:
            self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------

def test_agent():
    print("Testing PPOAgent...")

    num_nodes = 30
    travel_time_matrix = torch.rand(num_nodes, num_nodes)

    agent = PPOAgent(
        travel_time_matrix=travel_time_matrix,
        gcn_hidden=16, gcn_out=16,
        time_embed_dim=8, time_vocab_size=50,
        transformer_embed_dim=32,
        action_dim=5, hidden_dim=64,
        num_heads=2, num_layers=2,
        device="cpu"
    )

    num_envs = 4
    num_steps = 6
    for step in range(num_steps):
        state = np.zeros((num_envs, 6, 6), dtype=np.float32)
        state[..., 0] = np.random.randint(0, num_nodes, size=(num_envs, 6))
        state[..., 1] = np.random.randint(420, 600, size=(num_envs, 6))
        state[..., 2] = np.random.randint(420, 600, size=(num_envs, 6))
        state[..., 3] = np.random.randint(0, num_nodes, size=(num_envs, 6))
        state[..., 4] = np.random.randint(420, 600, size=(num_envs, 6))
        state[..., 5] = np.random.randint(420, 600, size=(num_envs, 6))

        masks = torch.ones(num_envs, 5)
        masks[:, 3] = 0  # forbid action 3

        actions = agent.select_action_batch(state, masks=masks, epsilon=0.0)
        assert (actions != 3).all(), "Masked action was selected!"

        rewards = np.random.randn(num_envs)
        dones = np.full(num_envs, step == num_steps - 1)
        agent.store_rewards_batch(rewards, dones)

    stats = agent.update(
        num_value_epochs=2, num_policy_epochs=2, batch_size=4,
        num_envs=num_envs, num_steps=num_steps,
    )
    print(f"Stats: {stats}")
    print("✓ All checks passed.")


if __name__ == "__main__":
    test_agent()