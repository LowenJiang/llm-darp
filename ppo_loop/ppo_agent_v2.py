"""
Proximal Policy Optimization (PPO) Agent

Implements PPO for the DVRP environment using PyTorch.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

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
        return h  # shape (N, out_dim)

class TripRequestEmbeddingModel(nn.Module):
    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden=32,
        gcn_out=32,
        time_embed_dim=16,
        time_vocab_size=500,  # you must set this appropriately
        num_heads=4,
        num_layers=3,
        transformer_embed_dim=80
    ):
        super().__init__()

        num_nodes = travel_time_matrix.shape[0]
        self.time_vocab_size = time_vocab_size  # Store for clamping in forward

        # Node features (identity)
        self.register_buffer("node_features", torch.eye(num_nodes))

        # GCN
        self.gcn = GCN(
            in_dim=num_nodes,
            hidden_dim=gcn_hidden,
            out_dim=gcn_out,
            travel_time_matrix=travel_time_matrix
        )

        # Time embeddings (two fields per TW)
        self.pickup_embed  = nn.Embedding(time_vocab_size, time_embed_dim)
        self.dropoff_embed = nn.Embedding(time_vocab_size, time_embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=num_heads,
            dim_feedforward=transformer_embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Projection into transformer dimension
        total_dim = gcn_out * 2 + time_embed_dim * 4
        self.project = nn.Linear(total_dim, transformer_embed_dim)

    def forward(self, states):
        """
        states: (B, C, 6)
        """
        B, C, _ = states.shape

        # (1) Precompute node embeddings
        node_emb = self.gcn(self.node_features)  # (N, gcn_out)

        # Correct column mapping:
        # Col 0: h3_pickup, Col 1: pickup_tw_early, Col 2: pickup_tw_late
        # Col 3: h3_dropoff, Col 4: dropoff_tw_early, Col 5: dropoff_tw_late
        origin_idx = states[..., 0].long()      # h3_pickup
        pickup_tw_early = states[..., 1].long()  # pickup_tw_early
        pickup_tw_late  = states[..., 2].long()  # pickup_tw_late
        dest_idx   = states[..., 3].long()      # h3_dropoff
        drop_tw_early   = states[..., 4].long()  # dropoff_tw_early
        drop_tw_late    = states[..., 5].long()  # dropoff_tw_late

        # Clamp time window indices to valid range [0, time_vocab_size-1]
        max_idx = self.time_vocab_size - 1

        # Debug: Check for out-of-range values before clamping
        if pickup_tw_early.max() > max_idx or pickup_tw_late.max() > max_idx or \
           drop_tw_early.max() > max_idx or drop_tw_late.max() > max_idx:
            print(f"[WARNING] Time window indices exceed vocab size {self.time_vocab_size}:")
            print(f"  pickup_tw range: [{pickup_tw_early.min()}, {pickup_tw_early.max()}], [{pickup_tw_late.min()}, {pickup_tw_late.max()}]")
            print(f"  dropoff_tw range: [{drop_tw_early.min()}, {drop_tw_early.max()}], [{drop_tw_late.min()}, {drop_tw_late.max()}]")

        pickup_tw_early = torch.clamp(pickup_tw_early, 0, max_idx)
        pickup_tw_late = torch.clamp(pickup_tw_late, 0, max_idx)
        drop_tw_early = torch.clamp(drop_tw_early, 0, max_idx)
        drop_tw_late = torch.clamp(drop_tw_late, 0, max_idx)

        # (1) ORIGIN/DEST node embeddings
        origin_emb = node_emb[origin_idx]   # (B,C,gcn_out)
        dest_emb   = node_emb[dest_idx]     # (B,C,gcn_out)

        # (2) pickup time window embedding
        pickup_emb_early = self.pickup_embed(pickup_tw_early) # (B,C,time_emb)

        pickup_emb_late = self.pickup_embed(pickup_tw_late)  # (B,C,time_emb)

        # (3) dropoff TW embedding
        dropoff_emb_early = self.dropoff_embed(drop_tw_early) # (B,C,time_emb)
        dropoff_emb_late = self.dropoff_embed(drop_tw_late)  # (B,C,time_emb)

        # (4) concat
        trip_emb = torch.cat(
            [origin_emb, dest_emb, pickup_emb_early, pickup_emb_late, dropoff_emb_early, dropoff_emb_late], dim=-1
        )  # (B,C,total_dim)

        # (4.5) project into transformer dimension
        trip_emb = self.project(trip_emb)

        # (5) Transformer
        out = self.transformer(trip_emb)  # (B,C,transformer_dim)

        return out

class PolicyNetwork(nn.Module):
    """
    Policy network (Actor) for PPO.

    Input: State (state_rows, 2) flattened to state_dim-dimensional vector
           For SF dataset with 207 locations: (606, 2) -> 1212-dimensional
    Output: Action probabilities over discrete actions
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
        hidden_dim: int = 512,   # MLP Hidden Layer Size
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super(PolicyNetwork, self).__init__()
        
        # 1. The Encoder (Handles Graph)
        self.backbone = TripRequestEmbeddingModel(travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out, time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size, transformer_embed_dim=transformer_embed_dim, num_heads=num_heads, num_layers=num_layers)

        # 2. The MLP Head
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
        # Step 1: Encode Graph to vector
        state_vector = self.backbone(states)  # (B, C, D)
        state_vector = state_vector.mean(dim=1)  # (B, D)
        # Step 2: MLP
        return self.network(state_vector)


class ValueNetwork(nn.Module):
    """
    Value network (Critic) for PPO.

    Input: State (state_rows, 2) flattened to state_dim-dimensional vector
           For SF dataset with 207 locations: (606, 2) -> 1212-dimensional
    Output: State value estimation
    """

    def __init__(
        self,
        travel_time_matrix,
        gcn_hidden: int = 32,
        gcn_out: int = 32,
        time_embed_dim: int = 16,
        time_vocab_size=500,
        transformer_embed_dim: int = 64,
        hidden_dim: int = 512,   # MLP Hidden Layer Size
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super(ValueNetwork, self).__init__()
        
        # 1. The Encoder (Independent instance)
        self.backbone = TripRequestEmbeddingModel(travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out, time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size, transformer_embed_dim=transformer_embed_dim, num_heads=num_heads, num_layers=num_layers)

        # 2. The MLP Head
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
        state_vector = self.backbone(states)  # (B, C, D)
        state_vector = state_vector.mean(dim=1)  # (B, D)
        return self.network(state_vector)


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO with separate policy and value networks.

    Input: State (state_rows, 2) flattened to state_dim-dimensional vector
           For SF dataset with 207 locations: (606, 2) -> 1212-dimensional
    Actor output: Action probabilities over 16 discrete actions
    Critic output: State value estimation
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
        hidden_dim: int = 512,   # MLP Hidden Layer Size
        num_heads: int = 8,
        num_layers: int = 3
    ):
        super(ActorCritic, self).__init__()
        
        self.policy_network = PolicyNetwork(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out, time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size, transformer_embed_dim=transformer_embed_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers
        )
        self.value_network = ValueNetwork(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out, time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size, transformer_embed_dim=transformer_embed_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers
        )

        self.mask = torch.ones(action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch_size, state_rows, 2)
                   For SF dataset: (batch_size, 606, 2)

        Returns:
            action_probs: Action probabilities (batch_size, 16)
            state_value: State value (batch_size, 1)
        """
        # Policy network: action logits
        action_logits = self.policy_network(state)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Value network: state value
        state_value = self.value_network(state)

        return action_probs, state_value
    
    def mask_action(self, action_probs, mask, eps=0):
        """
        Apply action mask with epsilon-greedy forced exploration.

        Args:
            action_probs: Original action probabilities
            mask: Action mask (1 = allowed/unmasked, 0 = masked/forbidden)
            eps: Epsilon for forced exploration into masked actions
                 - With prob eps: uniform random over masked actions (mask=0)
                 - With prob 1-eps: sample from unmasked actions (mask=1) according to logits

        Returns:
            Combined action probabilities
        """
        unmasked_count = mask.sum()
        masked_count = (1 - mask).sum()

        # Compute probabilities for unmasked actions (according to policy)
        if unmasked_count > 0:
            unmasked_probs = mask * action_probs
            unmasked_sum = unmasked_probs.sum()
            if unmasked_sum > 0:
                unmasked_probs = unmasked_probs / unmasked_sum
            else:
                # Fallback to uniform over unmasked actions
                unmasked_probs = mask / unmasked_count
        else:
            # No unmasked actions available, use original probs
            unmasked_probs = action_probs

        # Compute uniform probabilities for masked actions (forced exploration)
        if masked_count > 0:
            masked_probs = (1 - mask) / masked_count
        else:
            # No masked actions, just use unmasked probs
            masked_probs = torch.zeros_like(action_probs)

        # Combine: eps * uniform_over_masked + (1-eps) * policy_over_unmasked
        return eps * masked_probs + (1 - eps) * unmasked_probs

    def act(
        self,
        state: torch.Tensor,
        mask: torch.Tensor = None,
        epsilon: float = 0.0
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor of shape (state_rows, 2)
                   For SF dataset: (606, 2)
            mask: Action mask tensor of shape (action_dim,) where 1 = allowed, 0 = masked
            epsilon: Epsilon for epsilon-greedy masking (forced exploration into masked actions)

        Returns:
            action: Sampled action index
            log_prob: Log probability of the action
            value: State value estimation
        """
        # Add batch dimension
        state = state.unsqueeze(0)

        action_probs, state_value = self.forward(state)

        # Apply mask with epsilon-greedy exploration
        if mask is not None:
            # Squeeze batch dimension for masking
            action_probs_squeezed = action_probs.squeeze(0)
            masked_action_probs = self.mask_action(action_probs_squeezed, mask, epsilon)
            action_probs = masked_action_probs.unsqueeze(0)

        # Sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob, state_value.squeeze()

    def evaluate(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate states and actions (for PPO update).

        Args:
            states: Batch of states (batch_size, state_rows, 2)
                    For SF dataset: (batch_size, 606, 2)
            actions: Batch of actions (batch_size,)

        Returns:
            log_probs: Log probabilities of actions
            state_values: State values
            entropy: Entropy of action distribution
        """
        action_probs, state_values = self.forward(states)

        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, state_values.squeeze(), entropy


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
        hidden_dim: int = 512,   # MLP Hidden Layer Size
        num_heads: int = 8,
        num_layers: int = 3,
        lr: float = 3e-4,
        gamma: float = 1.00,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create actor-critic network
        self.policy = ActorCritic(
            travel_time_matrix=travel_time_matrix, gcn_hidden=gcn_hidden, gcn_out=gcn_out, time_embed_dim=time_embed_dim, time_vocab_size=time_vocab_size, transformer_embed_dim=transformer_embed_dim, action_dim=action_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Rollout buffer
        self.states: List[torch.Tensor] = []
        self.actions: List[int] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.values: List[torch.Tensor] = []
        self.dones: List[bool] = []

    def select_action(
        self,
        state: np.ndarray,
        mask: torch.Tensor = None,
        epsilon: float = 0.0
    ) -> int:
        """
        Select action using current policy.

        Args:
            state: State array of shape (state_rows, customers, 6)
            mask: Action mask tensor where 1 = allowed, 0 = masked
            epsilon: Epsilon for epsilon-greedy masking

        Returns:
            action: Selected action index
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        # Move mask to device if provided
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_tensor, mask=mask, epsilon=epsilon)

        # Store for later update
        self.states.append(state_tensor)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)

        return action
    
    def select_action_batch(
            self,
            states: np.ndarray,
            masks: torch.Tensor = None
        ) -> np.ndarray:
            """
            Select actions for a batch of states (Vectorized).
            Robust against policy-mask disagreement and numerical instability.

            NOTE: Epsilon-greedy masking is handled BEFORE calling this function.
            This function always selects actions according to the policy distribution.
            """
            batch_size = states.shape[0]

            # 1. Convert State to Tensor
            state_tensor = torch.FloatTensor(states).to(self.device)

            # 2. Handle Masks (Ensure Tensor on Device)
            if masks is not None:
                if not isinstance(masks, torch.Tensor):
                    # If masks is a list, convert each element to tensor then stack
                    if isinstance(masks, list):
                        # Convert each element to tensor if it isn't already
                        tensor_list = []
                        for mask in masks:
                            if isinstance(mask, torch.Tensor):
                                tensor_list.append(mask)
                            else:
                                tensor_list.append(torch.tensor(mask, dtype=torch.float32))
                        masks = torch.stack(tensor_list)
                    else:
                        # Convert array to tensor
                        masks = torch.tensor(masks, dtype=torch.float32)
                masks = masks.to(self.device)

            with torch.no_grad():
                # Get raw probabilities from policy
                # Check for NaNs in input state to prevent immediate crash
                if torch.isnan(state_tensor).any():
                    state_tensor = torch.nan_to_num(state_tensor, 0.0)

                action_probs, state_values = self.policy.forward(state_tensor)

                # Safety: Handle NaNs from network output (exploding gradients)
                if torch.isnan(action_probs).any():
                    # Fallback to uniform distribution if network breaks
                    action_probs = torch.ones_like(action_probs) / action_probs.shape[1]

                # 3. VECTORIZED MASKING
                if masks is not None:
                    # A. Apply Mask: Multiply probabilities by 0 or 1
                    masked_probs = action_probs * masks

                    # B. Check for degenerate cases (Sum approx 0)
                    sum_probs = masked_probs.sum(dim=1, keepdim=True)

                    # Identify rows where policy assigns 0 prob to all valid actions
                    # OR where mask is all zeros
                    problematic_rows = (sum_probs < 1e-8)

                    if problematic_rows.any():
                        # Calculate count of valid actions per row
                        valid_counts = masks.sum(dim=1, keepdim=True)

                        # Case 1: Mask is all zeros (Invalid State). Force allow all.
                        mask_is_empty = (valid_counts < 1e-8)
                        if mask_is_empty.any():
                            # Update mask to all 1s for these specific rows
                            masks = masks.clone() # Clone to avoid in-place error if leaf
                            masks[mask_is_empty.squeeze()] = 1.0
                            valid_counts[mask_is_empty.squeeze()] = masks.shape[1]

                        # Case 2: Policy Disagreement.
                        # Create a uniform distribution over VALID actions for problematic rows
                        fallback_probs = masks / (valid_counts + 1e-8)

                        # Replace masked_probs with fallback_probs where sum was ~0
                        masked_probs = torch.where(problematic_rows, fallback_probs, masked_probs)

                        # Recompute sum for normalization
                        sum_probs = masked_probs.sum(dim=1, keepdim=True)

                    # Normalize to sum to 1
                    action_probs = masked_probs / (sum_probs + 1e-8)

                # Final Safety: Clamp to remove 0s or negatives (numerical errors)
                # Categorical requires > 0 and sum=1 (implied, but good to be safe)
                action_probs = torch.clamp(action_probs, min=1e-8)
                action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)

                # 4. Sample Actions
                try:
                    dist = torch.distributions.Categorical(probs=action_probs)
                    actions = dist.sample() # Returns shape (batch_size,)
                    log_probs = dist.log_prob(actions)
                except RuntimeError as e:
                    print(f"[ERROR] Categorical sampling failed: {e}")
                    print(f"  Max Prob: {action_probs.max()}, Min Prob: {action_probs.min()}")
                    print(f"  NaNs: {torch.isnan(action_probs).any()}")
                    # Emergency fallback: always pick action 0
                    actions = torch.zeros(batch_size, device=self.device, dtype=torch.long)
                    log_probs = torch.zeros(batch_size, device=self.device)

            # 5. Store data
            state_cpu = state_tensor.cpu()
            actions_cpu = actions.cpu()
            log_probs_cpu = log_probs.cpu()
            values_cpu = state_values.squeeze().cpu()
            
            for i in range(batch_size):
                self.states.append(state_cpu[i])
                self.actions.append(actions_cpu[i].item())
                self.log_probs.append(log_probs_cpu[i])
                self.values.append(values_cpu[i])

            return actions.cpu().numpy()

    def store_reward(self, reward: float, done: bool) -> None:
        """
        Store reward and done flag.

        Args:
            reward: Reward received
            done: Whether episode terminated
        """
        self.rewards.append(reward)
        self.dones.append(done)

    def store_rewards_batch(self, rewards: np.ndarray, dones: np.ndarray) -> None:
        """
        Store rewards and done flags for a batch of environments.

        Args:
            rewards: (batch_size,) array of rewards
            dones: (batch_size,) array of done flags
        """
        for reward, done in zip(rewards, dones):
            self.rewards.append(float(reward))
            self.dones.append(bool(done))

    def update(self, num_value_epochs=40, num_policy_epochs=10, batch_size=64,
               num_envs=None, num_steps=None) -> dict:
        """
        Update PPO agent:
            1. Train value function to convergence
            2. Compute advantages using updated value function
            3. Train policy using PPO objective

        Args:
            num_value_epochs: Number of epochs to train value network
            num_policy_epochs: Number of epochs to train policy network
            batch_size: Mini-batch size for updates
            num_envs: Number of parallel environments (for parallel GAE)
            num_steps: Number of steps per episode (for parallel GAE)

        Returns:
            Dictionary with training statistics
        """

        if len(self.states) == 0:
            return {}

        # ========================================
        # Convert buffers to tensors
        # ========================================
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        old_values = torch.stack(self.values).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        num_samples = len(states)

        # ========================================
        # 1. TRAIN VALUE FUNCTION ONLY
        # ========================================
        # Create dedicated value optimizer
        value_params = list(self.policy.value_network.parameters())
        value_optimizer = optim.Adam(value_params, lr=self.optimizer.param_groups[0]['lr'])
        total_value_loss = 0.0

        for _ in range(num_value_epochs):
            idx = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_rewards = rewards[batch_idx]
                batch_dones = dones[batch_idx]
                batch_old_values = old_values[batch_idx]

                # Compute updated values
                with torch.no_grad():
                    _, new_values = self.policy.forward(batch_states)

                # Compute new returns using updated values (bootstrapped)
                _, batch_returns = self._compute_gae(batch_rewards, new_values.squeeze(), batch_dones)

                # Forward again for gradient
                _, predicted_values = self.policy.forward(batch_states)

                # Value loss
                value_loss = nn.MSELoss()(predicted_values.squeeze(), batch_returns)
                total_value_loss += value_loss.item()

                value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(value_params, self.max_grad_norm)
                value_optimizer.step()

        # ========================================
        # 2. COMPUTE ADVANTAGES USING NEW VALUE FUNCTION
        # ========================================
        with torch.no_grad():
            _, updated_values = self.policy.forward(states)
            updated_values = updated_values.squeeze()

        # Use parallel GAE if num_envs and num_steps are provided
        if num_envs is not None and num_steps is not None:
            advantages, returns = self._compute_gae_parallel(
                rewards, updated_values, dones, num_envs, num_steps
            )
        else:
            advantages, returns = self._compute_gae(rewards, updated_values, dones)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ========================================
        # 3. UPDATE POLICY VIA PPO CLIPPED OBJECTIVE
        # ========================================
        policy_params = list(self.policy.policy_network.parameters())
        policy_optimizer = optim.Adam(policy_params, lr=self.optimizer.param_groups[0]['lr'])

        total_policy_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(num_policy_epochs):
            idx = np.random.permutation(num_samples)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_idx = idx[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Evaluate with current policy
                log_probs, _, entropy = self.policy.evaluate(batch_states, batch_actions)

                # PPO ratio
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # PPO clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.entropy_coef * entropy_loss

                policy_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_params, self.max_grad_norm)
                policy_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # ========================================
        # Clear buffer and return logs
        # ========================================
        self.clear_buffer()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def _compute_gae(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: Rewards tensor (1D)
            values: State values tensor (1D)
            dones: Done flags tensor (1D)

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        # Ensure all tensors are 1D (handle edge case of 0-dim tensors from squeeze())
        rewards = rewards.flatten()
        values = values.flatten()
        dones = dones.flatten()

        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1].item() if values[t + 1].dim() == 0 else values[t + 1]

            # TD error
            current_value = values[t].item() if values[t].dim() == 0 else values[t]
            current_done = dones[t].item() if dones[t].dim() == 0 else dones[t]
            current_reward = rewards[t].item() if rewards[t].dim() == 0 else rewards[t]

            delta = current_reward + self.gamma * next_value * (1 - current_done) - current_value

            # GAE
            advantages[t] = last_advantage = (
                delta + self.gamma * self.gae_lambda * (1 - current_done) * last_advantage
            )

        # Returns = advantages + values
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
        """
        Compute Generalized Advantage Estimation (GAE) for parallel environments.

        This method correctly handles the interleaved buffer structure from parallel
        environments by reshaping the data so each environment's trajectory is contiguous,
        computing GAE separately for each environment, then flattening back.

        Args:
            rewards: Rewards tensor, shape (num_envs * num_steps,) - interleaved
            values: State values tensor, shape (num_envs * num_steps,) - interleaved
            dones: Done flags tensor, shape (num_envs * num_steps,) - interleaved
            num_envs: Number of parallel environments
            num_steps: Number of steps per episode

        Returns:
            advantages: GAE advantages, shape (num_envs * num_steps,)
            returns: Discounted returns, shape (num_envs * num_steps,)
        """
        total_samples = len(rewards)

        # Verify buffer size matches expected
        if total_samples != num_envs * num_steps:
            print(f"WARNING: Buffer size mismatch. Expected {num_envs * num_steps}, got {total_samples}")
            print(f"  Falling back to sequential GAE (may be incorrect!)")
            return self._compute_gae(rewards, values, dones)

        # Reshape from interleaved to (num_envs, num_steps)
        # Current layout: [env0_t0, env1_t0, env2_t0, ..., env0_t1, env1_t1, ...]
        # After reshape: [[env0_t0, env0_t1, env0_t2, ...],
        #                 [env1_t0, env1_t1, env1_t2, ...],
        #                 ...]

        # First reshape to (num_steps, num_envs), then transpose to (num_envs, num_steps)
        rewards_2d = rewards.reshape(num_steps, num_envs).T  # (num_envs, num_steps)
        values_2d = values.reshape(num_steps, num_envs).T
        dones_2d = dones.reshape(num_steps, num_envs).T

        # Compute GAE for each environment separately
        advantages_2d = torch.zeros_like(rewards_2d)

        for env_idx in range(num_envs):
            last_advantage = 0.0

            for t in reversed(range(num_steps)):
                # Check if this is a terminal state
                if t == num_steps - 1 or dones_2d[env_idx, t]:
                    next_value = 0.0  # Terminal state
                else:
                    next_value = values_2d[env_idx, t + 1]  # Same env, next step

                # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
                delta = (
                    rewards_2d[env_idx, t]
                    + self.gamma * next_value * (1 - dones_2d[env_idx, t])
                    - values_2d[env_idx, t]
                )

                # GAE: A_t = δ_t + γλ*A_{t+1}
                advantages_2d[env_idx, t] = last_advantage = (
                    delta
                    + self.gamma * self.gae_lambda * (1 - dones_2d[env_idx, t]) * last_advantage
                )

        # Flatten back to original interleaved format
        # (num_envs, num_steps) → transpose → (num_steps, num_envs) → reshape → (num_envs * num_steps,)
        advantages = advantages_2d.T.reshape(-1)
        returns = advantages + values

        return advantages, returns

    def clear_buffer(self) -> None:
        """Clear rollout buffer."""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def test_agent():
    """Test the PPOAgent with a dummy travel time matrix and state space."""
    print("Testing PPOAgent with dummy state space...")

    # Create dummy travel time matrix for 10 nodes
    num_nodes = 10
    travel_time_matrix = torch.rand(num_nodes, num_nodes)

    # Initialize agent
    agent = PPOAgent(
        travel_time_matrix=travel_time_matrix,
        gcn_hidden=16,
        gcn_out=16,
        time_embed_dim=8,
        time_vocab_size=50,
        transformer_embed_dim=32,
        action_dim=5,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        device="cpu"
    )

    # Create dummy batch of trips: (num_trips, 6)
    # Columns: [origin_idx, dest_idx, pickup_tw_early, pickup_tw_late, drop_tw_early, drop_tw_late]
    num_trips = 6
    state = np.zeros((num_trips, 6), dtype=np.float32)
    state[:, 0] = np.random.randint(0, num_nodes, size=num_trips)  # origin
    state[:, 1] = np.random.randint(0, num_nodes, size=num_trips)  # dest
    state[:, 2] = np.random.randint(0, 50, size=num_trips)  # pickup early
    state[:, 3] = np.random.randint(0, 50, size=num_trips)  # pickup late
    state[:, 4] = np.random.randint(0, 50, size=num_trips)  # dropoff early
    state[:, 5] = np.random.randint(0, 50, size=num_trips)  # dropoff late

    print(f"Dummy state:\n{state}")

    # Select action
    action = agent.select_action(state)
    print(f"Selected action: {action}")

    # Store reward
    agent.store_reward(reward=10.0, done=False)

    # Collect more samples
    for _ in range(5):
        state = np.zeros((num_trips, 6), dtype=np.float32)
        state[:, 0] = np.random.randint(0, num_nodes, size=num_trips)
        state[:, 1] = np.random.randint(0, num_nodes, size=num_trips)
        state[:, 2] = np.random.randint(0, 50, size=num_trips)
        state[:, 3] = np.random.randint(0, 50, size=num_trips)
        state[:, 4] = np.random.randint(0, 50, size=num_trips)
        state[:, 5] = np.random.randint(0, 50, size=num_trips)

        action = agent.select_action(state)
        agent.store_reward(reward=np.random.randn(), done=False)

    # Final state
    state = np.zeros((num_trips, 6), dtype=np.float32)
    state[:, 0] = np.random.randint(0, num_nodes, size=num_trips)
    state[:, 1] = np.random.randint(0, num_nodes, size=num_trips)
    state[:, 2] = np.random.randint(0, 50, size=num_trips)
    state[:, 3] = np.random.randint(0, 50, size=num_trips)
    state[:, 4] = np.random.randint(0, 50, size=num_trips)
    state[:, 5] = np.random.randint(0, 50, size=num_trips)

    action = agent.select_action(state)
    agent.store_reward(reward=5.0, done=True)

    # Update agent
    print("\nPerforming PPO update...")
    stats = agent.update(num_value_epochs=5, num_policy_epochs=3, batch_size=2)
    print(f"Training stats: {stats}")
    print("\n✓ PPO agent test completed successfully!")

if __name__ == "__main__":
    test_agent()