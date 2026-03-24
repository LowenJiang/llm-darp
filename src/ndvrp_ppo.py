"""PPO agent for NDVRP perturbation optimisation.

Uses a frozen attention encoder (from oracle_policy checkpoint) to build a
fixed-dimensional state vector, then trains lightweight policy and value FFN
heads with PPO.

State construction:
    {h_1, …, h_t} = Enc_θ(ρ_1', …, ρ_t)
    h̄_{<t} = 𝟙{t>1} · mean(h_1, …, h_{t-1})
    s_t   = [ h̄_{<t} ‖ h_t ] ∈ ℝ^{2·d_h}

where each h_j is the mean of the pickup and dropoff node embeddings for
request j.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tensordict import TensorDict
from typing import Tuple

from ndvrp_env import NUM_ACTIONS


# --------------------------------------------------------------------------- #
# Frozen encoder wrapper
# --------------------------------------------------------------------------- #
class FrozenEncoder(nn.Module):
    """Loads init_embedding + encoder from an oracle_policy checkpoint and
    freezes them.  Exposes ``encode_nodes`` (no grad) and ``build_state``
    (differentiable through h0)."""

    def __init__(self, checkpoint_path: str, embed_dim: int = 128, device: str = "cpu"):
        super().__init__()
        from oracle_policy import PDPTWAttentionPolicy

        full = PDPTWAttentionPolicy(embed_dim=embed_dim)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        full.load_state_dict(ckpt["policy_state_dict"])

        self.init_embedding = full.init_embedding
        self.encoder = full.encoder
        for p in self.init_embedding.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.embed_dim = embed_dim

    @torch.no_grad()
    def encode_nodes(self, td: TensorDict) -> torch.Tensor:
        """Run frozen encoder → detached node embeddings [B, N, D]."""
        self.init_embedding.eval()
        self.encoder.eval()
        init_h = self.init_embedding(td)
        return self.encoder(init_h).detach()

    def extract_request_components(
        self, node_embs: torch.Tensor, step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """From node embeddings return (fixed_mean, current_emb).

        Args:
            node_embs: [B, 2*(step+1)+1, D] — depot + requests 1..step+1
            step: 0-indexed step (request index = step+1)

        Returns:
            fixed_mean:   [B, D]  mean embedding of requests 1..step (or zeros)
            current_emb:  [B, D]  embedding of request step+1
        """
        B, _, D = node_embs.shape
        num_requests = step + 1  # requests 1..step+1

        # Per-request embedding = mean(pickup, dropoff) — skip depot at idx 0
        pickup_embs = node_embs[:, 1::2]   # [B, num_requests, D]
        dropoff_embs = node_embs[:, 2::2]  # [B, num_requests, D]
        request_embs = (pickup_embs + dropoff_embs) / 2  # [B, num_requests, D]

        current_emb = request_embs[:, -1]  # [B, D]
        if num_requests > 1:
            fixed_mean = request_embs[:, :-1].mean(dim=1)
        else:
            fixed_mean = torch.zeros(B, D, device=node_embs.device)
        return fixed_mean, current_emb

    def build_state(
        self, fixed_mean: torch.Tensor, current_emb: torch.Tensor,
    ) -> torch.Tensor:
        """s_t = [ fixed_mean ‖ current_emb ] ∈ ℝ^{2D}."""
        return torch.cat([fixed_mean, current_emb], dim=-1)  # [B, 2D]


# --------------------------------------------------------------------------- #
# Policy / Value heads
# --------------------------------------------------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 512, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)  # raw logits [B, num_actions]


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)  # [B]


# --------------------------------------------------------------------------- #
# PPO Agent
# --------------------------------------------------------------------------- #
class PPOAgent:
    def __init__(
        self,
        checkpoint_path: str,
        embed_dim: int = 128,
        hidden_dim: int = 512,
        lr_policy: float = 3e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_epochs: int = 10,
        policy_epochs: int = 4,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_epochs = value_epochs
        self.policy_epochs = policy_epochs
        self.entropy_coef = entropy_coef

        state_dim = 2 * embed_dim

        self.encoder = FrozenEncoder(checkpoint_path, embed_dim, device).to(self.device)
        self.policy = PolicyNet(state_dim, hidden_dim).to(self.device)
        self.value = ValueNet(state_dim, hidden_dim).to(self.device)

        # Separate optimisers
        self.policy_optim = torch.optim.Adam(
            self.policy.parameters(), lr=lr_policy,
        )
        self.value_optim = torch.optim.Adam(
            self.value.parameters(), lr=lr_value,
        )

    # ------------------------------------------------------------------ #
    # State encoding (from raw observation)
    # ------------------------------------------------------------------ #
    def encode_obs(self, obs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an observation into a state vector [B, 2D]."""
        td = TensorDict({
            "h3_indices": obs["h3_indices"],
            "travel_time_matrix": obs["travel_time_matrix"],
            "locs": obs["locs"],
            "demand": obs["demand"],
            "time_windows": obs["time_windows"],
        }, batch_size=[obs["h3_indices"].shape[0]])

        node_embs = self.encoder.encode_nodes(td)
        step = int(obs["step"][0].item())
        fixed_mean, current_emb = self.encoder.extract_request_components(node_embs, step)
        return self.encoder.build_state(fixed_mean, current_emb)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def select_action(
        self, state: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions; returns (action, log_prob, value) — all [B]."""
        logits = self.policy(state)
        logits[~mask] = -float("inf")
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action).detach(), self.value(state).detach()

    # ------------------------------------------------------------------ #
    # PPO update
    # ------------------------------------------------------------------ #
    def update(self, buffer: dict):
        """PPO update: compute advantages → train value → train policy.

        ``buffer`` keys (all tensors with leading dim = T*B):
            states, actions, old_log_probs,
            old_values, rewards, dones, masks
        """
        states = buffer["states"]        # [T*B, 2D]
        actions = buffer["actions"]      # [T*B]
        old_lp = buffer["old_log_probs"] # [T*B]
        old_val = buffer["old_values"]   # [T*B]
        rewards = buffer["rewards"]      # [T*B]
        dones = buffer["dones"]          # [T*B]
        masks = buffer["masks"]          # [T*B, 49]
        num_envs = buffer["num_envs"]

        # ---- 1. GAE advantages (using old values, no grad) ------------ #
        advantages, returns = self._compute_gae(rewards, old_val, dones, num_envs)

        # ---- 2. Value network update --------------------------------- #
        for _ in range(self.value_epochs):
            v = self.value(states)
            value_loss = F.mse_loss(v, returns)
            self.value_optim.zero_grad()
            value_loss.backward()
            self.value_optim.step()

        # ---- 3. Policy update (clipped PPO) -------------------------- #
        adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.policy_epochs):
            logits = self.policy(states)
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
