"""
PPO Agent with Frozen Oracle Encoder and FFN Policy/Value Heads.

This module implements a PPO agent (TestPPOAgent) that re-uses the pre-trained
attention encoder from oracle_policy.PDPTWAttentionPolicy as a frozen feature
extractor. Only the lightweight FFN actor and critic heads are trained.

Key design decisions:
- The encoder (init_embedding + graph_encoder) is loaded from a checkpoint and
  its parameters are frozen — gradients are never computed through it.
- Rather than storing full TensorDicts in the rollout buffer (memory-expensive),
  we store the 128-dim mean-pooled node embedding produced by the frozen encoder.
  This means the encoder is called only once per timestep, at act-time.
- Separate Adam optimizers are created for the policy head and value head so
  their learning rates can be tuned independently.
- The PPO update follows the standard two-phase pattern from dvrp_ppo_agent_gcn.py:
  first train the value head to convergence, then update the policy head with the
  clipped surrogate objective.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torch.distributions import Categorical
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Local imports — oracle_policy lives alongside this file in src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from oracle_policy import PDPTWAttentionPolicy


# =============================================================================
# CONSTANTS
# =============================================================================

# Default embedding dimension produced by the oracle encoder.
DEFAULT_EMBED_DIM: int = 128

# PPO clipping range for the probability ratio.
DEFAULT_CLIP_EPSILON: float = 0.2

# Discount factor for future rewards.
DEFAULT_GAMMA: float = 0.995

# GAE smoothing parameter (λ).
DEFAULT_GAE_LAMBDA: float = 0.90

# Coefficient scaling the entropy bonus in the policy loss.
DEFAULT_ENTROPY_COEF: float = 0.01

# Maximum L2 norm for gradient clipping.
DEFAULT_MAX_GRAD_NORM: float = 0.5


# =============================================================================
# ACTOR-CRITIC MODULE
# =============================================================================

class TestActorCritic(nn.Module):
    """
    Actor-critic model built on top of a frozen oracle encoder.

    The encoder (PDPTWInitEmbedding + GraphAttentionNetwork) is loaded from a
    checkpoint and its weights are permanently frozen.  The trainable parts are
    two independent FFN heads — one for action logits (actor) and one for state
    value (critic) — that both operate on the mean-pooled node embedding.

    Args:
        encoder_checkpoint: Path to the oracle policy checkpoint (.pt file).
        action_dim: Number of discrete actions the actor can output.
        embed_dim: Dimensionality of the node embeddings produced by the encoder.
            Must match the checkpoint (default 128).
    """

    def __init__(
        self,
        encoder_checkpoint: str,
        action_dim: int = 49,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Load the pre-trained oracle encoder and freeze its parameters.
        # ------------------------------------------------------------------
        self.init_embedding, self.graph_encoder = self._load_frozen_encoder(
            encoder_checkpoint, embed_dim
        )

        # ------------------------------------------------------------------
        # Policy head (actor): embed_dim → 512 → 256 → action_dim
        # Uses Tanh activations to match the critic's dynamic range.
        # ------------------------------------------------------------------
        self.policy_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )

        # ------------------------------------------------------------------
        # Value head (critic): embed_dim → 512 → 256 → 1
        # Outputs a scalar state-value estimate.
        # ------------------------------------------------------------------
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frozen_encoder(
        checkpoint_path: str, embed_dim: int
    ) -> Tuple[nn.Module, nn.Module]:
        """
        Load and freeze the encoder components from an oracle policy checkpoint.

        The checkpoint may store the state dict under different keys depending
        on how it was saved.  We try both 'policy_state_dict' and 'state_dict'
        before falling back to treating the whole file as a state dict.

        Args:
            checkpoint_path: Filesystem path to the .pt checkpoint file.
            embed_dim: Expected embedding dimension; used to instantiate
                PDPTWAttentionPolicy with matching hyperparameters.

        Returns:
            Tuple of (init_embedding, graph_encoder) — both frozen nn.Modules.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Resolve the state dict regardless of how the checkpoint was written.
        state = ckpt.get("policy_state_dict", ckpt.get("state_dict", ckpt))

        # Instantiate a full oracle policy so we can borrow its submodules.
        full_model = PDPTWAttentionPolicy(embed_dim=embed_dim)
        full_model.load_state_dict(state, strict=False)

        # Extract the two encoder components.
        init_embedding = full_model.init_embedding
        graph_encoder = full_model.encoder  # GraphAttentionNetwork

        # Freeze all parameters — no gradients will flow through the encoder.
        for p in init_embedding.parameters():
            p.requires_grad = False
        for p in graph_encoder.parameters():
            p.requires_grad = False

        return init_embedding, graph_encoder

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode(self, encoder_input_td: TensorDict) -> torch.Tensor:
        """
        Run the frozen encoder and return mean-pooled node embeddings.

        This is separated from forward() so that the agent can call it once
        at act-time and store the pooled vector in the rollout buffer, avoiding
        a second forward pass through the (potentially large) encoder during the
        PPO update.

        Args:
            encoder_input_td: TensorDict with keys:
                - locs: [B, N, 2] node coordinates
                - time_windows: [B, N, 2] time windows in minutes
                - demand: [B, N] demand per node
                - h3_indices: [B, N] H3 cell indices
                - travel_time_matrix: [B, H, H] travel time matrix

        Returns:
            pooled: [B, embed_dim] mean-pooled node embeddings.
        """
        # init_embedding: TensorDict → [B, N, embed_dim]
        init_h = self.init_embedding(encoder_input_td)
        # graph_encoder: [B, N, embed_dim] → [B, N, embed_dim]
        node_embeddings = self.graph_encoder(init_h)
        # Mean-pool across the node dimension to get a single vector per instance.
        pooled = node_embeddings.mean(dim=1)  # [B, embed_dim]
        return pooled

    def forward(
        self,
        encoder_input_td: TensorDict,
        user_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass: encode then compute action logits and state value.

        Args:
            encoder_input_td: TensorDict of encoder inputs (see encode()).
            user_mask: Optional boolean tensor [B, action_dim].  Where False,
                the corresponding logit is set to -inf before softmax.  This
                allows an upstream user-embedding model to restrict the action
                space based on predicted user preferences.

        Returns:
            action_logits: [B, action_dim] raw (unmasked unless user_mask given).
            state_value:   [B, 1] scalar value estimates.
        """
        pooled = self.encode(encoder_input_td)  # [B, embed_dim]

        action_logits = self.policy_head(pooled)  # [B, action_dim]

        # Apply user-preference mask if provided (e.g. from an embedding model).
        if user_mask is not None:
            action_logits = action_logits.masked_fill(~user_mask.bool(), float("-inf"))

        state_value = self.value_head(pooled)  # [B, 1]

        return action_logits, state_value

    def forward_from_pooled(
        self, pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run only the FFN heads on a pre-computed pooled embedding.

        Used during the PPO update step, where we stored the pooled embedding in
        the rollout buffer and don't want to re-run the (frozen) encoder.

        Args:
            pooled: [B, embed_dim] mean-pooled node embeddings.

        Returns:
            action_logits: [B, action_dim]
            state_value:   [B, 1]
        """
        action_logits = self.policy_head(pooled)
        state_value = self.value_head(pooled)
        return action_logits, state_value


# =============================================================================
# PPO AGENT
# =============================================================================

class TestPPOAgent:
    """
    PPO agent that trains FFN policy/value heads on top of a frozen oracle encoder.

    The encoder runs at act-time to produce a compact 128-dim pooled embedding.
    This embedding is stored in the rollout buffer.  During the PPO update,
    only the two FFN heads are optimised — the encoder parameters never change.

    Args:
        encoder_checkpoint: Path to the oracle policy checkpoint (.pt file).
        action_dim: Size of the discrete action space.
        embed_dim: Encoder output dimensionality (must match checkpoint).
        policy_lr: Learning rate for the policy (actor) head.
        value_lr: Learning rate for the value (critic) head.
        gamma: Discount factor γ for future rewards.
        gae_lambda: GAE smoothing parameter λ.
        clip_epsilon: PPO clipping range ε.
        entropy_coef: Coefficient for the entropy bonus in the policy loss.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        device: PyTorch device string ('cpu', 'cuda', 'mps', …).
    """

    def __init__(
        self,
        encoder_checkpoint: str,
        action_dim: int = 49,
        embed_dim: int = DEFAULT_EMBED_DIM,
        policy_lr: float = 1e-3,
        value_lr: float = 1e-3,
        gamma: float = DEFAULT_GAMMA,
        gae_lambda: float = DEFAULT_GAE_LAMBDA,
        clip_epsilon: float = DEFAULT_CLIP_EPSILON,
        entropy_coef: float = DEFAULT_ENTROPY_COEF,
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Build the actor-critic and move it to the target device.
        self.policy = TestActorCritic(
            encoder_checkpoint=encoder_checkpoint,
            action_dim=action_dim,
            embed_dim=embed_dim,
        ).to(self.device)

        # Separate optimizers — only the FFN head parameters are registered.
        # The encoder parameters have requires_grad=False so they are already
        # excluded, but being explicit here makes the intent unmistakable.
        self.policy_optimizer = optim.Adam(
            self.policy.policy_head.parameters(), lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.policy.value_head.parameters(), lr=value_lr
        )

        # ------------------------------------------------------------------
        # Rollout buffer.  We store the mean-pooled embeddings (not the full
        # TensorDict) to avoid holding large tensors in memory across a rollout.
        # ------------------------------------------------------------------
        self.pooled_embeddings: List[torch.Tensor] = []  # [embed_dim] each
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []          # scalar log-prob each
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []             # scalar value each
        self.dones: List[bool] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action_batch(
        self,
        encoder_input_td: TensorDict,
        masks: Optional[torch.Tensor] = None,
        epsilon: float = 0.0,
    ) -> np.ndarray:
        """
        Select actions for a batch of environments and store rollout data.

        The encoder is called once under torch.no_grad() to produce the pooled
        embedding, which is immediately stored in the buffer.  The policy head
        then computes action logits.

        Masking logic (mirrors dvrp_ppo_agent_gcn.PPOAgent.select_action_batch):
        - Valid actions are those where masks == 1.
        - If a row has zero valid probability mass, fall back to uniform over
          valid actions.
        - If a row's mask is all zeros (invalid state), allow all actions.
        - With epsilon > 0, blend masked policy with unmasked policy.

        Args:
            encoder_input_td: TensorDict of encoder inputs [B, …].
            masks: Float tensor [B, action_dim] with 1 = allowed, 0 = blocked.
                If None, all actions are permitted.
            epsilon: Epsilon for exploration; blends (1-ε)*masked + ε*unmasked.

        Returns:
            actions: int64 numpy array of shape [B] with sampled action indices.
        """
        # Move encoder inputs to the agent's device.
        encoder_input_td = encoder_input_td.to(self.device)

        if masks is not None:
            masks = masks.to(self.device, dtype=torch.float32)

        with torch.no_grad():
            # Run the frozen encoder once.
            pooled = self.policy.encode(encoder_input_td)  # [B, embed_dim]

            # Run only the policy head (no encoder re-run needed).
            action_logits, state_values = self.policy.forward_from_pooled(pooled)
            # Convert logits → probabilities for masking arithmetic.
            action_probs = torch.softmax(action_logits, dim=-1)  # [B, action_dim]

            if masks is not None:
                masked_probs = action_probs * masks
                sum_probs = masked_probs.sum(dim=1, keepdim=True)  # [B, 1]

                # Identify rows where the policy assigns zero mass to valid actions.
                problematic_rows = sum_probs < 1e-8  # [B, 1]

                if problematic_rows.any():
                    valid_counts = masks.sum(dim=1, keepdim=True)  # [B, 1]

                    # If the mask itself is empty, open all actions as fallback.
                    mask_is_empty = valid_counts < 1e-8  # [B, 1]
                    if mask_is_empty.any():
                        masks = masks.clone()
                        masks[mask_is_empty.squeeze(1)] = 1.0
                        valid_counts = masks.sum(dim=1, keepdim=True)

                    # Uniform distribution over valid actions for troubled rows.
                    fallback_probs = masks / (valid_counts + 1e-8)
                    masked_probs = torch.where(
                        problematic_rows, fallback_probs, masked_probs
                    )
                    sum_probs = masked_probs.sum(dim=1, keepdim=True)

                # Normalise so probabilities sum to 1.
                masked_probs = masked_probs / (sum_probs + 1e-8)
            else:
                masked_probs = action_probs

            # Safety clamp: remove numerically negative/zero entries.
            masked_probs = torch.clamp(masked_probs, min=1e-8)
            masked_probs = masked_probs / masked_probs.sum(dim=1, keepdim=True)

            # Blend with unmasked probabilities for epsilon-greedy exploration.
            if masks is not None and epsilon > 0.0:
                mixed_probs = (
                    (1.0 - epsilon) * masked_probs + epsilon * action_probs
                )
                mixed_probs = torch.clamp(mixed_probs, min=1e-8)
                mixed_probs = mixed_probs / mixed_probs.sum(dim=1, keepdim=True)
                dist = Categorical(probs=mixed_probs)
            else:
                dist = Categorical(probs=masked_probs)

            actions = dist.sample()           # [B]
            log_probs = dist.log_prob(actions)  # [B]

        # ------------------------------------------------------------------
        # Store rollout data on CPU to keep GPU memory free between steps.
        # ------------------------------------------------------------------
        batch_size = pooled.shape[0]
        pooled_cpu = pooled.cpu()
        actions_cpu = actions.cpu()
        log_probs_cpu = log_probs.cpu()
        # squeeze(1) turns [B, 1] → [B]; if batch_size==1 we get a scalar,
        # so we use flatten() to always get a 1-D tensor.
        values_cpu = state_values.squeeze(-1).cpu().flatten()

        for i in range(batch_size):
            self.pooled_embeddings.append(pooled_cpu[i])
            self.actions.append(actions_cpu[i].item())
            self.log_probs.append(log_probs_cpu[i])
            self.values.append(values_cpu[i])

        return actions_cpu.numpy()

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def store_rewards_batch(
        self, rewards: np.ndarray, dones: np.ndarray
    ) -> None:
        """
        Append rewards and episode-termination flags for a batch of environments.

        This must be called once per timestep, after select_action_batch(), with
        arrays that match the batch size used at act-time.

        Args:
            rewards: Float array [B] of per-environment rewards.
            dones: Bool array [B]; True means the episode ended at this step.
        """
        for reward, done in zip(rewards, dones):
            self.rewards.append(float(reward))
            self.dones.append(bool(done))

    def clear_buffer(self) -> None:
        """Discard all rollout data accumulated since the last update."""
        self.pooled_embeddings.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalised Advantage Estimation (GAE) for a single sequential trajectory.

        δ_t = r_t + γ·V(s_{t+1})·(1 - d_t) - V(s_t)
        A_t = δ_t + γλ·(1 - d_t)·A_{t+1}

        Args:
            rewards: 1-D tensor of per-step rewards.
            values:  1-D tensor of value estimates V(s_t).
            dones:   1-D float tensor; 1.0 at terminal transitions.

        Returns:
            advantages: 1-D tensor of GAE advantages.
            returns:    1-D tensor of λ-return targets (advantages + values).
        """
        rewards = rewards.flatten()
        values = values.flatten()
        dones = dones.flatten()

        advantages = torch.zeros_like(rewards)
        last_adv = 0.0

        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1].item()
            delta = (
                rewards[t].item()
                + self.gamma * next_value * (1.0 - dones[t].item())
                - values[t].item()
            )
            last_adv = (
                delta
                + self.gamma * self.gae_lambda * (1.0 - dones[t].item()) * last_adv
            )
            advantages[t] = last_adv

        returns = advantages + values
        return advantages, returns

    def _compute_gae_parallel(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        num_envs: int,
        num_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE for data collected from multiple parallel environments.

        The buffer layout is interleaved:
            [env0_t0, env1_t0, …, env{E-1}_t0, env0_t1, env1_t1, …]

        We reshape to (num_envs, num_steps), compute GAE per trajectory, then
        flatten back to the original interleaved order so indexing is consistent
        with the stored pooled embeddings and actions.

        Args:
            rewards:   1-D tensor, length num_envs * num_steps.
            values:    1-D tensor, same length.
            dones:     1-D float tensor, same length.
            num_envs:  Number of parallel environments.
            num_steps: Number of timesteps collected per environment.

        Returns:
            advantages: 1-D tensor, length num_envs * num_steps.
            returns:    1-D tensor, same length.
        """
        total = len(rewards)
        if total != num_envs * num_steps:
            # Buffer mismatch — fall back gracefully with a warning.
            print(
                f"[WARNING] _compute_gae_parallel: expected {num_envs * num_steps} "
                f"samples, got {total}.  Falling back to sequential GAE."
            )
            return self._compute_gae(rewards, values, dones)

        # Reshape interleaved (num_steps * num_envs,) → (num_envs, num_steps).
        # Buffer order: step0[all envs], step1[all envs], …
        # reshape(num_steps, num_envs).T gives (num_envs, num_steps).
        rewards_2d = rewards.reshape(num_steps, num_envs).T   # (E, T)
        values_2d  = values.reshape(num_steps, num_envs).T    # (E, T)
        dones_2d   = dones.reshape(num_steps, num_envs).T     # (E, T)

        advantages_2d = torch.zeros_like(rewards_2d)

        for env_idx in range(num_envs):
            last_adv = 0.0
            for t in reversed(range(num_steps)):
                is_terminal = t == num_steps - 1 or bool(dones_2d[env_idx, t].item())
                next_value = 0.0 if is_terminal else values_2d[env_idx, t + 1].item()

                delta = (
                    rewards_2d[env_idx, t].item()
                    + self.gamma * next_value * (1.0 - dones_2d[env_idx, t].item())
                    - values_2d[env_idx, t].item()
                )
                last_adv = (
                    delta
                    + self.gamma
                    * self.gae_lambda
                    * (1.0 - dones_2d[env_idx, t].item())
                    * last_adv
                )
                advantages_2d[env_idx, t] = last_adv

        # Transpose back and flatten to restore interleaved order.
        advantages = advantages_2d.T.reshape(-1)
        returns    = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        num_value_epochs: int = 100,
        num_policy_epochs: int = 3,
        batch_size: int = 64,
        num_envs: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Run a full PPO update over the data in the rollout buffer.

        Phase 1 — Value head training:
            Train the value head for num_value_epochs using MSE against
            GAE-computed λ-returns.  The policy head is not touched.

        Phase 2 — Advantage computation:
            Re-compute values with the updated value head and derive advantages
            via GAE (parallel or sequential depending on arguments).

        Phase 3 — Policy head training:
            Update the policy head for num_policy_epochs using the PPO clipped
            surrogate objective with an entropy bonus.

        Args:
            num_value_epochs: Passes over the buffer for value regression.
            num_policy_epochs: Passes over the buffer for policy optimisation.
            batch_size: Mini-batch size for both phases.
            num_envs: If provided together with num_steps, enables parallel GAE.
            num_steps: Steps per environment; required with num_envs.

        Returns:
            Dictionary of scalar training statistics (losses, norms, KL, …).
            Returns an empty dict if the buffer is empty.
        """
        if not self.pooled_embeddings:
            return {}

        # ------------------------------------------------------------------
        # Materialise the rollout buffer into tensors on the target device.
        # ------------------------------------------------------------------
        # pooled_embeddings: list of [embed_dim] → stack to [N, embed_dim]
        pooled     = torch.stack(self.pooled_embeddings).to(self.device)
        actions    = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        rewards    = torch.FloatTensor(self.rewards).to(self.device)
        old_values = torch.stack(self.values).to(self.device)
        dones      = torch.FloatTensor([float(d) for d in self.dones]).to(self.device)

        num_samples = pooled.shape[0]

        # ------------------------------------------------------------------
        # Phase 1: Train value head to convergence.
        # ------------------------------------------------------------------
        total_value_loss = 0.0
        total_value_grad_norm = 0.0
        value_update_count = 0

        # Compute initial value targets once using the current (pre-update) values.
        _, initial_returns = self._compute_gae(rewards, old_values, dones)

        print("\tTraining value head...")
        for _ in tqdm(range(num_value_epochs)):
            idx = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                bi = idx[start:end]

                batch_pooled  = pooled[bi]
                batch_returns = initial_returns[bi]

                # Forward pass through value head only.
                _, predicted_values = self.policy.forward_from_pooled(batch_pooled)

                value_loss = nn.functional.mse_loss(
                    predicted_values.squeeze(-1), batch_returns
                )

                self.value_optimizer.zero_grad()
                value_loss.backward()
                vgn = nn.utils.clip_grad_norm_(
                    self.policy.value_head.parameters(), self.max_grad_norm
                )
                self.value_optimizer.step()

                total_value_loss += value_loss.item()
                total_value_grad_norm += vgn.item()
                value_update_count += 1

        # ------------------------------------------------------------------
        # Phase 2: Re-compute advantages with the now-updated value head.
        # ------------------------------------------------------------------
        with torch.no_grad():
            _, updated_values = self.policy.forward_from_pooled(pooled)
            updated_values = updated_values.squeeze(-1).flatten()  # [N]

        if num_envs is not None and num_steps is not None:
            advantages, returns = self._compute_gae_parallel(
                rewards, updated_values, dones, num_envs, num_steps
            )
        else:
            advantages, returns = self._compute_gae(rewards, updated_values, dones)

        # Normalise advantages to zero mean / unit variance for training stability.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ------------------------------------------------------------------
        # Phase 3: Update policy head using PPO clipped objective.
        # ------------------------------------------------------------------
        total_policy_loss = 0.0
        total_entropy = 0.0
        total_clip_fraction = 0.0
        total_approx_kl = 0.0
        total_ratio_mean = 0.0
        total_ratio_std = 0.0
        total_policy_grad_norm = 0.0
        num_policy_updates = 0

        print("\tTraining policy head...")
        for _ in tqdm(range(num_policy_epochs)):
            idx = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                bi = idx[start:end]

                batch_pooled      = pooled[bi]
                batch_actions     = actions[bi]
                batch_advantages  = advantages[bi]
                batch_old_log_probs = old_log_probs[bi]

                # Forward through policy head to get fresh log-probs and entropy.
                action_logits, _ = self.policy.forward_from_pooled(batch_pooled)
                dist = Categorical(logits=action_logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()  # [batch]

                # PPO importance-sampling ratio.
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective.
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus encourages exploration.
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.entropy_coef * entropy_loss

                self.policy_optimizer.zero_grad()
                loss.backward()
                pgn = nn.utils.clip_grad_norm_(
                    self.policy.policy_head.parameters(), self.max_grad_norm
                )
                self.policy_optimizer.step()

                # Diagnostics.
                total_policy_loss    += policy_loss.item()
                total_entropy        += entropy.mean().item()
                total_clip_fraction  += (
                    (ratio - 1.0).abs() > self.clip_epsilon
                ).float().mean().item()
                total_approx_kl      += (batch_old_log_probs - new_log_probs).mean().item()
                total_ratio_mean     += ratio.mean().item()
                total_ratio_std      += ratio.std().item()
                total_policy_grad_norm += pgn.item()
                num_policy_updates   += 1

        # ------------------------------------------------------------------
        # Clear buffer and return training statistics.
        # ------------------------------------------------------------------
        self.clear_buffer()

        safe_div = lambda total, n: total / max(n, 1)

        return {
            "policy_loss":       safe_div(total_policy_loss, num_policy_updates),
            "value_loss":        safe_div(total_value_loss, value_update_count),
            "entropy":           safe_div(total_entropy, num_policy_updates),
            "clip_fraction":     safe_div(total_clip_fraction, num_policy_updates),
            "approx_kl":         safe_div(total_approx_kl, num_policy_updates),
            "ratio_mean":        safe_div(total_ratio_mean, num_policy_updates),
            "ratio_std":         safe_div(total_ratio_std, num_policy_updates),
            "advantage_mean":    advantages.mean().item(),
            "advantage_std":     advantages.std().item(),
            "value_pred_mean":   updated_values.mean().item(),
            "returns_mean":      returns.mean().item(),
            "returns_std":       returns.std().item(),
            "policy_grad_norm":  safe_div(total_policy_grad_norm, num_policy_updates),
            "value_grad_norm":   safe_div(total_value_grad_norm, value_update_count),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the actor-critic model state to disk.

        Only the full model state dict is saved; the encoder weights within it
        are frozen but still included so the checkpoint is self-contained.

        Args:
            path: Destination file path (e.g. 'checkpoints/ppo_agent.pt').
        """
        torch.save(
            {"policy_state_dict": self.policy.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        """
        Restore the actor-critic model state from a checkpoint created by save().

        Args:
            path: Path to the checkpoint file written by save().
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])


# =============================================================================
# SMOKE TEST
# =============================================================================

if __name__ == "__main__":
    """
    Minimal end-to-end smoke test.

    Creates a TestPPOAgent, generates random dummy encoder inputs, runs one
    step of action selection and reward storage, then verifies the shapes.
    A full PPO update is intentionally not run here because the frozen
    encoder checkpoint is required for a real forward pass.
    """
    CHECKPOINT = (
        "/Users/jiangwolin/Desktop/Research/DARPSolver/checkpoints/refined/best.pt"
    )

    print("Initialising TestPPOAgent...")
    agent = TestPPOAgent(
        encoder_checkpoint=CHECKPOINT,
        action_dim=49,
        embed_dim=128,
        device="cpu",
    )
    print(f"  policy_head params: {sum(p.numel() for p in agent.policy.policy_head.parameters()):,}")
    print(f"  value_head params:  {sum(p.numel() for p in agent.policy.value_head.parameters()):,}")
    encoder_params = sum(
        p.numel()
        for p in list(agent.policy.init_embedding.parameters())
        + list(agent.policy.graph_encoder.parameters())
    )
    print(f"  encoder params (frozen): {encoder_params:,}")

    # ------------------------------------------------------------------
    # Build a dummy TensorDict mimicking the oracle environment output.
    # B=4 batch instances, N=11 nodes (depot + 5 pickup/delivery pairs).
    # ------------------------------------------------------------------
    B, N = 4, 11

    dummy_td = TensorDict(
        {
            "locs":                torch.randn(B, N, 2),
            "time_windows":        torch.rand(B, N, 2) * 1440.0,
            "demand":              torch.ones(B, N),
            "h3_indices":          torch.randint(0, 50, (B, N)),
            "travel_time_matrix":  torch.rand(B, 50, 50) * 60.0,
        },
        batch_size=[B],
    )

    masks = torch.ones(B, 49)  # All actions permitted.

    print("\nRunning select_action_batch...")
    actions = agent.select_action_batch(dummy_td, masks=masks)
    print(f"  actions shape: {actions.shape}  values: {actions}")
    assert actions.shape == (B,), f"Expected ({B},), got {actions.shape}"

    print("\nStoring rewards...")
    agent.store_rewards_batch(np.random.randn(B), np.zeros(B, dtype=bool))
    assert len(agent.rewards) == B
    assert len(agent.pooled_embeddings) == B

    print("\nTest passed!")
