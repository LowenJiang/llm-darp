"""PPO agent for NDVRP perturbation optimisation.

Uses a GCN-based encoder over the H3 travel-time graph, time-window
embeddings, and a Transformer to produce per-request representations.
Separate policy and value heads output over 16 discrete actions.

State format: (B, C, 6) where each request row is
    [origin_h3, pickup_tw_early, pickup_tw_late,
     dest_h3,   dropoff_tw_early, dropoff_tw_late]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Tuple

from ndvrp_env import NUM_ACTIONS


# =========================================================================== #
#   ENCODING BLOCK
# =========================================================================== #

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, travel_time_matrix):
        super().__init__()
        self.register_buffer("travel_time_matrix", travel_time_matrix)
        self.register_buffer(
            "travel_time_matrix_norm", self._normalize_adj(travel_time_matrix)
        )
        self.gc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.gc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    @staticmethod
    def _normalize_adj(A):
        A_hat = A + torch.eye(A.size(0), device=A.device)
        deg_inv_sqrt = A_hat.sum(1).pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
        D = torch.diag(deg_inv_sqrt)
        return D @ A_hat @ D

    def forward(self, X):
        h = torch.relu(self.travel_time_matrix_norm @ self.gc1(X))
        return self.travel_time_matrix_norm @ self.gc2(h)  # (N, out_dim)


class TripRequestEmbeddingModel(nn.Module):
    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=500,
        num_heads=4,
        num_layers=3,
        transformer_embed_dim=80,
    ):
        super().__init__()
        num_nodes = travel_time_matrix.shape[0]
        self.time_vocab_size = time_vocab_size

        self.register_buffer("node_features", torch.eye(num_nodes))

        self.gcn = GCN(num_nodes, gcn_hidden, gcn_out, travel_time_matrix)
        self.pickup_embed = nn.Embedding(time_vocab_size, time_embed_dim)
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
        Args:
            states: (B, C, 6) — columns are
                [origin, pickup_tw_early, pickup_tw_late,
                 dest,   dropoff_tw_early, dropoff_tw_late]
        Returns:
            (B, C, transformer_embed_dim)
        """
        B, C, _ = states.shape
        node_emb = self.gcn(self.node_features)  # (N, gcn_out)
        max_idx = self.time_vocab_size - 1

        origin_idx = states[..., 0].long()
        dest_idx = states[..., 3].long()

        pickup_tw_early = states[..., 1].long().clamp(0, max_idx)
        pickup_tw_late = states[..., 2].long().clamp(0, max_idx)
        drop_tw_early = states[..., 4].long().clamp(0, max_idx)
        drop_tw_late = states[..., 5].long().clamp(0, max_idx)

        trip_emb = torch.cat(
            [
                node_emb[origin_idx],
                node_emb[dest_idx],
                self.pickup_embed(pickup_tw_early),
                self.pickup_embed(pickup_tw_late),
                self.dropoff_embed(drop_tw_early),
                self.dropoff_embed(drop_tw_late),
            ],
            dim=-1,
        )  # (B, C, total_dim)

        # Zero-mask padded customers (all-zero rows)
        padding_mask = states.abs().sum(dim=-1) == 0  # (B, C)
        trip_emb = trip_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        trip_emb = self.project(trip_emb)

        return self.transformer(trip_emb, src_key_padding_mask=padding_mask)


class PolicyNetwork(nn.Module):
    def __init__(self, travel_time_matrix, gcn_hidden=32, gcn_out=32,
                 time_embed_dim=16, time_vocab_size=500,
                 transformer_embed_dim=64, action_dim=NUM_ACTIONS,
                 hidden_dim=512, num_heads=8, num_layers=3):
        super().__init__()
        self.backbone = TripRequestEmbeddingModel(
            travel_time_matrix, gcn_hidden, gcn_out, time_embed_dim,
            time_vocab_size, num_heads, num_layers, transformer_embed_dim,
        )
        self.network = nn.Sequential(
            nn.Linear(transformer_embed_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, states):
        return self.network(self.backbone(states).mean(dim=1))


class ValueNetwork(nn.Module):
    def __init__(self, travel_time_matrix, gcn_hidden=32, gcn_out=32,
                 time_embed_dim=16, time_vocab_size=500,
                 transformer_embed_dim=64, hidden_dim=512,
                 num_heads=8, num_layers=3):
        super().__init__()
        self.backbone = TripRequestEmbeddingModel(
            travel_time_matrix, gcn_hidden, gcn_out, time_embed_dim,
            time_vocab_size, num_heads, num_layers, transformer_embed_dim,
        )
        self.network = nn.Sequential(
            nn.Linear(transformer_embed_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, states):
        return self.network(self.backbone(states).mean(dim=1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, travel_time_matrix, gcn_hidden=32, gcn_out=32,
                 time_embed_dim=16, time_vocab_size=500,
                 transformer_embed_dim=64, action_dim=NUM_ACTIONS,
                 hidden_dim=512, num_heads=8, num_layers=3):
        super().__init__()
        common = dict(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden,
            gcn_out=gcn_out, time_embed_dim=time_embed_dim,
            time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
        )
        self.policy_network = PolicyNetwork(**common, action_dim=action_dim)
        self.value_network = ValueNetwork(**common)

    def forward(self, state):
        logits = self.policy_network(state)
        value = self.value_network(state)
        return logits, value


# =========================================================================== #
#   PPO Agent
# =========================================================================== #

class PPOAgent:
    def __init__(
        self,
        travel_time_matrix: torch.Tensor,
        num_customers: int = 30,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        time_embed_dim: int = 16,
        time_vocab_size: int = 500,
        transformer_embed_dim: int = 64,
        hidden_dim: int = 512,
        num_heads: int = 4,
        num_layers: int = 2,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_epochs: int = 10,
        policy_epochs: int = 4,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_epochs = value_epochs
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_customers = num_customers

        self.actor_critic = ActorCritic(
            travel_time_matrix=travel_time_matrix,
            gcn_hidden=gcn_hidden, gcn_out=gcn_out,
            time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size,
            transformer_embed_dim=transformer_embed_dim,
            action_dim=NUM_ACTIONS, hidden_dim=hidden_dim,
            num_heads=num_heads, num_layers=num_layers,
        ).to(self.device)

        # Separate optimisers with per-component learning rates
        self.policy_optim = torch.optim.Adam([
            {"params": self.actor_critic.policy_network.backbone.parameters(),
             "lr": lr_policy * 0.1},
            {"params": self.actor_critic.policy_network.network.parameters(),
             "lr": lr_policy},
        ])
        self.value_optim = torch.optim.Adam([
            {"params": self.actor_critic.value_network.backbone.parameters(),
             "lr": lr_value * 0.1},
            {"params": self.actor_critic.value_network.network.parameters(),
             "lr": lr_value},
        ])

    # ------------------------------------------------------------------ #
    # State encoding (from raw observation)
    # ------------------------------------------------------------------ #
    def encode_obs(self, obs: dict) -> torch.Tensor:
        """Convert env observation to state tensor [B, C, 6].

        Pads to num_customers with zeros for requests not yet revealed.
        """
        step = int(obs["step"][0].item())
        h3 = obs["h3_indices"]           # [B, 2*(step+1)+1]
        tw = obs["time_windows"]         # [B, 2*(step+1)+1, 2]
        B = h3.shape[0]
        C = self.num_customers

        state = torch.zeros(B, C, 6, device=self.device)
        num_visible = step + 1  # requests 1..step+1

        # request i (0-indexed): pickup_node = 2*i+1, dropoff_node = 2*i+2
        for i in range(num_visible):
            p_node = 2 * i + 1
            d_node = 2 * i + 2
            state[:, i, 0] = h3[:, p_node].float()       # origin h3
            state[:, i, 1] = tw[:, p_node, 0]             # pickup early
            state[:, i, 2] = tw[:, p_node, 1]             # pickup late
            state[:, i, 3] = h3[:, d_node].float()       # dest h3
            state[:, i, 4] = tw[:, d_node, 0]             # dropoff early
            state[:, i, 5] = tw[:, d_node, 1]             # dropoff late

        return state

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(
        self, state: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions; returns (action, log_prob, value) — all [B]."""
        with torch.no_grad():
            logits, value = self.actor_critic(state)
            logits[~mask] = -float("inf")
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action, dist.log_prob(action), value

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    def update(self, buffer: dict):
        """PPO update: compute advantages → train value → train policy.

        ``buffer`` keys (all tensors with leading dim = T*B):
            states, actions, old_log_probs,
            old_values, rewards, dones, masks
        """
        states = buffer["states"]        # [T*B, C, 6]
        actions = buffer["actions"]      # [T*B]
        old_lp = buffer["old_log_probs"] # [T*B]
        old_val = buffer["old_values"]   # [T*B]
        rewards = buffer["rewards"]      # [T*B]
        dones = buffer["dones"]          # [T*B]
        masks = buffer["masks"]          # [T*B, 16]
        num_envs = buffer["num_envs"]

        # ---- 1. GAE advantages (using old values, no grad) ------------ #
        advantages, returns = self._compute_gae(rewards, old_val, dones, num_envs)

        # ---- 2. Value network update --------------------------------- #
        value_params = list(self.actor_critic.value_network.parameters())
        for _ in range(self.value_epochs):
            v = self.actor_critic.value_network(states)
            value_loss = F.mse_loss(v, returns)
            self.value_optim.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(value_params, self.max_grad_norm)
            self.value_optim.step()

        # ---- 3. Policy update (clipped PPO) -------------------------- #
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_params = list(self.actor_critic.policy_network.parameters())
        for _ in range(self.policy_epochs):
            logits = self.actor_critic.policy_network(states)
            logits[~masks] = -float("inf")
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_lp - old_lp).exp()
            surr1 = ratio * adv_norm
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_norm
            policy_loss = -torch.min(surr1, surr2).mean()
            loss = policy_loss - self.entropy_coef * entropy

            self.policy_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_params, self.max_grad_norm)
            self.policy_optim.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "advantage_mean": advantages.mean().item(),
        }

    # ------------------------------------------------------------------ #
    # GAE
    # ------------------------------------------------------------------ #
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        num_envs: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard GAE over parallel-env buffer laid out as [env0_t0, env1_t0, …]."""
        T = rewards.shape[0] // num_envs
        r = rewards.view(T, num_envs)
        v = values.view(T, num_envs)
        d = dones.float().view(T, num_envs)

        advantages = torch.zeros_like(r)
        gae = torch.zeros(num_envs, device=rewards.device)

        for t in reversed(range(T)):
            next_val = v[t + 1] if t + 1 < T else torch.zeros(num_envs, device=rewards.device)
            delta = r[t] + self.gamma * next_val * (1 - d[t]) - v[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - d[t]) * gae
            advantages[t] = gae

        returns = (advantages + v).view(-1)
        advantages = advantages.view(-1)
        return advantages.detach(), returns.detach()
