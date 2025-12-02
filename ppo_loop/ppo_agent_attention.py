"""
Proximal Policy Optimization (PPO) Agent - Vectorized & Decoupled
Context: Dynamic VRP (New Node Querying Fixed History)
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tensordict.tensordict import TensorDict
from dvrp_env import DVRPEnv

# ==============================================================================
# 1. NETWORK ARCHITECTURE (Cross Attention)
# ==============================================================================

class CrossAttentionDVRPNetwork(nn.Module):
    """
    Standalone network for Actor or Critic.
    Vectorized to handle [Batch, N, Dim] inputs naturally.
    """
    def __init__(
        self,
        input_dim: int = 6,
        embed_dim: int = 128,
        output_dim: int = 1,  # 16 for Actor, 1 for Critic
        num_heads: int = 4,
        ffn_hidden: int = 256
    ):
        super().__init__()
        
        # 1. Projection Layers
        self.encoder = nn.Linear(input_dim, embed_dim)
        
        # 2. Multi-Head Cross Attention
        # batch_first=True -> [Batch, Seq_Len, Dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, ffn_hidden),
            nn.Tanh(),
            nn.Linear(ffn_hidden, ffn_hidden),
            nn.Tanh(),
            nn.Linear(ffn_hidden, ffn_hidden),
            nn.Tanh(),
            nn.Linear(ffn_hidden, output_dim)
        )

    def forward(self, tensordict: TensorDict) -> torch.Tensor:
        """
        Inputs:
            tensordict with keys:
            - 'fixed': [Batch, N, 6] (Key/Value) - N can vary between batches if padded
            - 'new':   [Batch, 1, 6] (Query)
        """
        fixed_nodes = tensordict['fixed'] # [Batch, N, 6]
        new_node = tensordict['new']      # [Batch, 1, 6]

        # --- A. Projection ---
        k_v_embed = self.encoder(fixed_nodes) # [Batch, N, 128]
        q_embed = self.encoder(new_node)      # [Batch, 1, 128]

        # --- B. Mask Generation ---
        # Identify empty slots (padding) in 'fixed'.
        # Assumes zero-padding implies invalid node.
        # Shape: [Batch, N]. True = Ignore.
        padding_mask = (fixed_nodes.abs().sum(dim=-1) < 1e-6)

        # Handle case where all nodes are padding (shouldn't happen but safeguard)
        all_padding = padding_mask.all(dim=1, keepdim=True)
        if all_padding.any():
            # For batches with all padding, mark first position as valid
            padding_mask[all_padding.squeeze(1), 0] = False

        # --- C. Cross Attention ---
        # Query: New Node | Key/Value: Fixed Nodes
        attn_output, _ = self.cross_attn(
            query=q_embed,
            key=k_v_embed,
            value=k_v_embed,
            key_padding_mask=padding_mask
        )
        # attn_output: [Batch, 1, 128]

        # --- D. Fusion & Output ---
        fusion = torch.cat([q_embed, attn_output], dim=-1) # [Batch, 1, 256]
        latent = fusion.squeeze(1) # [Batch, 256]

        output = self.ffn(latent)

        return output


class DecoupledActorCritic(nn.Module):
    """
    Container holding two separate networks.
    """
    def __init__(self, action_dim=16, embed_dim=128):
        super().__init__()
        self.actor = CrossAttentionDVRPNetwork(output_dim=action_dim, embed_dim=embed_dim)
        self.critic = CrossAttentionDVRPNetwork(output_dim=1, embed_dim=embed_dim)

    def act(self, td: TensorDict, mask: Optional[torch.Tensor] = None):
        """
        Batched action selection.
        """
        # 1. Get Logits [Batch, Action_Dim]
        action_logits = self.actor(td)
        action_probs = torch.softmax(action_logits, dim=-1)

        # 2. Get Value [Batch, 1]
        state_value = self.critic(td)

        # 3. Apply Mask [Batch, Action_Dim]
        if mask is not None:
            action_probs = self._apply_mask(action_probs, mask)

        # 4. Handle numerical instability
        # If a row is all zeros (shouldn't happen with valid masks), reset to uniform
        invalid_rows = action_probs.sum(dim=-1) < 1e-9
        if invalid_rows.any():
            action_probs[invalid_rows] = 1.0 / action_probs.shape[-1]

        # 5. Sample
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, state_value.squeeze(-1)

    def _apply_mask(self, probs, mask):
        # Mask is [Batch, Action_Dim]
        masked_probs = probs * mask
        sum_probs = masked_probs.sum(dim=-1, keepdim=True)
        
        # Renormalize
        masked_probs = torch.where(
            sum_probs > 1e-9, 
            masked_probs / sum_probs, 
            mask / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        )
        return masked_probs

    def get_value(self, td):
        return self.critic(td)

    def get_action_probs_value(self, td):
        logits = self.actor(td)
        value = self.critic(td)
        return torch.softmax(logits, dim=-1), value


# ==============================================================================
# 2. VECTORIZED PPO AGENT
# ==============================================================================

class PPOAgent:
    def __init__(
        self,
        embed_dim: int = 128,
        action_dim: int = 16,
        # Optimization Hyperparams
        actor_lr: float = 1e-4,
        critic_lr: float = 3e-4,
        actor_epochs: int = 10,   # UPDATED: Policy updates 10 times
        critic_epochs: int = 40,  # UPDATED: Critic updates 40 epochs
        # PPO Hyperparams
        gamma: float = 1,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        value_clip: float = 10.0,  # Clip value predictions
        normalize_rewards: bool = True,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.value_clip = value_clip
        self.normalize_rewards = normalize_rewards

        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs

        # Initialize Networks
        self.policy = DecoupledActorCritic(
            action_dim=action_dim,
            embed_dim=embed_dim
        ).to(self.device)

        # Initialize weights properly
        self._init_weights()

        # Separate Optimizers
        self.actor_optimizer = optim.Adam(self.policy.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.policy.critic.parameters(), lr=critic_lr)

        # Running statistics for reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_count = 0

        self.clear_buffer()

    def _init_weights(self):
        """Initialize network weights for stability"""
        def init_fn(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        self.policy.apply(init_fn)

        # Initialize final layers with smaller weights
        if hasattr(self.policy.actor.ffn, '__iter__'):
            final_layer = list(self.policy.actor.ffn.children())[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.orthogonal_(final_layer.weight, gain=0.01)

        if hasattr(self.policy.critic.ffn, '__iter__'):
            final_layer = list(self.policy.critic.ffn.children())[-1]
            if isinstance(final_layer, nn.Linear):
                nn.init.orthogonal_(final_layer.weight, gain=1.0)

    def clear_buffer(self) -> None:
        # Buffers store lists of tensors, where each list item corresponds to a time step
        # Tensors have shape [Batch_Size, ...]
        self.buffer = {
            'states': [],      # List[TensorDict]
            'actions': [],     # List[Tensor]
            'log_probs': [],   # List[Tensor]
            'rewards': [],     # List[Tensor]
            'dones': [],       # List[Tensor]
            'masks': [],       # List[Tensor]
            'values': []       # List[Tensor] (for GAE)
        }

    def select_action(self, state_td: TensorDict, mask: torch.Tensor = None):
        """
        Vectorized Action Selection.
        state_td: [Batch, ...]
        mask: [Batch, Action_Dim]
        """
        state_td = state_td.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_td, mask=mask)

            # Check for NaN in value predictions
            if torch.isnan(value).any():
                print(f"ERROR: NaN in value predictions during rollout!")
                print(f"  NaN count: {torch.isnan(value).sum().item()}/{value.numel()}")
                print(f"  Sample state['new']: {state_td['new'][0]}")
                raise ValueError("NaN in value predictions - network is unstable")

        # Move to CPU to save GPU memory during rollout
        self.buffer['states'].append(state_td.cpu())
        self.buffer['actions'].append(action.cpu())
        self.buffer['log_probs'].append(log_prob.cpu())
        self.buffer['values'].append(value.cpu())

        if mask is not None:
            self.buffer['masks'].append(mask.cpu())
        else:
            self.buffer['masks'].append(None)

        return action # Return on device (likely CPU based on env)

    def store_reward(self, reward: torch.Tensor, done: torch.Tensor):
        """
        reward: [Batch]
        done: [Batch] (bool or float)
        """
        # Check for NaN in rewards
        if torch.isnan(reward).any():
            nan_mask = torch.isnan(reward)
            print(f"ERROR: NaN detected in {nan_mask.sum().item()} rewards!")
            raise ValueError("NaN rewards from environment - check reward calculation")

        # Update running reward statistics
        if self.normalize_rewards:
            batch_mean = reward.mean().item()
            batch_std = reward.std().item()

            # Incremental update
            self.reward_count += 1
            alpha = 1.0 / self.reward_count if self.reward_count < 100 else 0.01
            self.reward_mean = (1 - alpha) * self.reward_mean + alpha * batch_mean
            self.reward_std = (1 - alpha) * self.reward_std + alpha * batch_std
            self.reward_std = max(self.reward_std, 1e-6)  # Prevent division by zero

            # Normalize rewards
            reward = (reward - self.reward_mean) / self.reward_std

        self.buffer['rewards'].append(reward.cpu())
        self.buffer['dones'].append(done.float().cpu())

    def _collate_and_pad(self):
        """
        Collates lists of [Batch, ...] into [Total_Samples, ...].
        Handles 'fixed' node padding because N grows over time.
        """
        # 1. Determine Max N in the trajectory history
        max_n = 0
        for td in self.buffer['states']:
            # td['fixed'] is [Batch, N, 6]
            max_n = max(max_n, td['fixed'].shape[1])

        batch_size = self.buffer['states'][0].batch_size[0]
        time_steps = len(self.buffer['states'])
        total_samples = batch_size * time_steps

        # 2. Pre-allocate flattened storage
        flat_fixed = torch.zeros(total_samples, max_n, 6)
        flat_new = torch.zeros(total_samples, 1, 6)
        
        # 3. Flatten and Pad
        # We iterate through time, padding each batch to max_n, then placing in flat buffer
        for t, td in enumerate(self.buffer['states']):
            current_n = td['fixed'].shape[1]
            
            # Helper: Calculate slice indices for this timestep
            # Strategy: Interleave? Or concatenate? 
            # Concatenating time chunks is easier for GAE but PPO shuffles anyway.
            # Here we fill sequentially: [Batch_t0, Batch_t1, ...]
            start_idx = t * batch_size
            end_idx = start_idx + batch_size
            
            # Copy 'fixed' with padding
            flat_fixed[start_idx:end_idx, :current_n, :] = td['fixed']
            
            # Copy 'new'
            flat_new[start_idx:end_idx] = td['new']

        flat_td = TensorDict({
            'fixed': flat_fixed,
            'new': flat_new
        }, batch_size=[total_samples])

        # 4. Flatten simple tensors
        def flatten_list(k):
            # Stack [Time, Batch] -> Flatten to [Time*Batch]
            if all(x is None for x in self.buffer[k]): return None
            return torch.stack(self.buffer[k]).view(-1, *self.buffer[k][0].shape[1:])

        flat_actions = flatten_list('actions')
        flat_log_probs = flatten_list('log_probs')
        flat_rewards = flatten_list('rewards') # [Time*Batch]
        flat_dones = flatten_list('dones')
        flat_masks = flatten_list('masks')
        
        # For GAE, we keep Rewards/Values/Dones as [Time, Batch] before flattening
        t_rewards = torch.stack(self.buffer['rewards']) # [T, B]
        t_dones = torch.stack(self.buffer['dones'])     # [T, B]
        t_values = torch.stack(self.buffer['values'])   # [T, B]

        return flat_td, flat_actions, flat_log_probs, flat_masks, t_rewards, t_dones, t_values

    def update(self, batch_size=256):
        if len(self.buffer['states']) == 0: return {}

        # 1. Prepare Data
        (flat_td, flat_actions, flat_old_log_probs, flat_masks, 
         t_rewards, t_dones, t_values) = self._collate_and_pad()
        
        # Move relevant tensors to device for GAE calc
        t_rewards = t_rewards.to(self.device)
        t_dones = t_dones.to(self.device)
        t_values = t_values.to(self.device)

        # 2. Compute GAE (Vectorized over Batch dimension)
        # We need next_value for the last step.
        # Assumption: The buffer ends when the batch is Done or we bootstrap.
        # For simplicity here, we assume 0 value for terminal step if not provided.
        # Ideally, PPOAgent.store_reward should handle "next_obs" value for truncation.
        # Here we assume standard episodic rollout where last done=1.

        num_steps, num_envs = t_rewards.shape
        advantages = torch.zeros_like(t_rewards)
        last_gae = torch.zeros(num_envs, device=self.device)

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = torch.zeros(num_envs, device=self.device)
                next_value = torch.zeros(num_envs, device=self.device)
            else:
                next_non_terminal = 1.0 - t_dones[t + 1]
                next_value = t_values[t + 1]

            delta = t_rewards[t] + self.gamma * next_value * next_non_terminal - t_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + t_values

        # Check for NaN after GAE computation
        if torch.isnan(advantages).any() or torch.isnan(returns).any():
            print(f"ERROR: NaN detected after GAE computation!")
            print(f"  - NaN in advantages: {torch.isnan(advantages).sum().item()}")
            print(f"  - NaN in returns: {torch.isnan(returns).sum().item()}")
            print(f"  - NaN in t_rewards: {torch.isnan(t_rewards).sum().item()}")
            print(f"  - NaN in t_values: {torch.isnan(t_values).sum().item()}")
            print(f"  - t_rewards range: [{t_rewards.min().item():.2f}, {t_rewards.max().item():.2f}]")
            print(f"  - t_values range: [{t_values.min().item():.2f}, {t_values.max().item():.2f}]")
            raise ValueError("NaN in GAE computation")
        
        # Flatten advantages/returns for batch training
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)
        
        # Normalize Advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # Move flattened training data to device
        flat_td = flat_td.to(self.device)
        flat_actions = flat_actions.to(self.device)
        flat_old_log_probs = flat_old_log_probs.to(self.device)
        flat_masks = flat_masks.to(self.device) if flat_masks is not None else None
        
        num_samples = flat_actions.shape[0]
        indices = np.arange(num_samples)

        # Metrics
        loss_actor_accum = 0
        loss_critic_accum = 0

        # ======================================================================
        # 3. SEPARATE UPDATE LOOPS
        # ======================================================================

        # --- A. Critic Update Loop ---
        # Runs for self.critic_epochs (set to 40)
        for _ in range(self.critic_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                mb_idx = indices[start:end]

                mb_td = flat_td[mb_idx]
                mb_returns = flat_returns[mb_idx]
                mb_old_values = t_values.view(-1)[mb_idx]

                # Critic Forward
                current_values = self.policy.get_value(mb_td).squeeze(-1)

                # Check for NaN in critic outputs
                if torch.isnan(current_values).any():
                    print(f"ERROR: NaN in current_values during training!")
                    print(f"  Sample input: {mb_td['new'][0]}")
                    raise ValueError("NaN in critic output - network is unstable")

                # Clipped Value Loss (PPO-style)
                # Prevents value function from changing too much
                value_clipped = mb_old_values + torch.clamp(
                    current_values - mb_old_values,
                    -self.value_clip,
                    self.value_clip
                )
                loss_unclipped = (current_values - mb_returns).pow(2)
                loss_clipped = (value_clipped - mb_returns).pow(2)
                critic_loss = 0.5 * torch.max(loss_unclipped, loss_clipped).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()

                # Check for NaN in gradients (will raise error if found)
                for name, param in self.policy.critic.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"ERROR: NaN gradient in critic.{name}")
                        print(f"  Current values range: [{current_values.min().item():.2f}, {current_values.max().item():.2f}]")
                        print(f"  Returns range: [{mb_returns.min().item():.2f}, {mb_returns.max().item():.2f}]")
                        raise ValueError(f"NaN gradient in {name}")

                nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                loss_critic_accum += critic_loss.item()

        # --- B. Actor Update Loop ---
        # Runs for self.actor_epochs (set to 10)
        for _ in range(self.actor_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                mb_idx = indices[start:end]

                mb_td = flat_td[mb_idx]
                mb_actions = flat_actions[mb_idx]
                mb_old_lp = flat_old_log_probs[mb_idx]
                mb_adv = flat_advantages[mb_idx]
                mb_mask = flat_masks[mb_idx] if flat_masks is not None else None

                # Actor Forward
                logits = self.policy.actor(mb_td)
                probs = torch.softmax(logits, dim=-1)
                
                if mb_mask is not None:
                    probs = self.policy._apply_mask(probs, mb_mask)
                
                # Stability
                probs = probs.clamp(min=1e-9)
                dist = Categorical(probs)
                
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Ratio
                ratio = torch.exp(new_log_probs - mb_old_lp)
                
                # Surrogate Loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # Check for NaN in gradients (will raise error if found)
                for name, param in self.policy.actor.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"ERROR: NaN gradient in actor.{name}")
                        raise ValueError(f"NaN gradient in {name}")

                nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                loss_actor_accum += actor_loss.item()

        self.clear_buffer()

        # Calculate averages based on total batches processed
        critic_batches = self.critic_epochs * (num_samples // batch_size + (1 if num_samples % batch_size else 0))
        actor_batches = self.actor_epochs * (num_samples // batch_size + (1 if num_samples % batch_size else 0))

        return {
            "actor_loss": loss_actor_accum / max(actor_batches, 1),
            "critic_loss": loss_critic_accum / max(critic_batches, 1),
            "value_mean": t_values.mean().item(),
            "value_std": t_values.std().item(),
            "return_mean": flat_returns.mean().item(),
            "return_std": flat_returns.std().item(),
            "advantage_mean": flat_advantages.mean().item(),
            "advantage_std": flat_advantages.std().item(),
        }

if __name__ == "__main__":
    print("=== Starting Vectorized Integration Test ===")
    
    # 1. Configuration
    BATCH_SIZE = 400  # Small batch for local testing
    NUM_CUSTOMERS = 10
    DEVICE = "cpu"   # Use "cuda" if available

    # 2. Initialize Vectorized Env
    # Note: DVRPEnv must support receiving a tensor action [Batch] and returning tensor obs
    env = DVRPEnv(
        num_customers=NUM_CUSTOMERS, 
        device=DEVICE,
        batch_size=BATCH_SIZE
    )

    # 3. Initialize Agent
    agent = PPOAgent(
        embed_dim=128,
        action_dim=16,
        actor_lr=1e-4,
        critic_lr=5e-4,
        actor_epochs=10, # Explicit test value
        critic_epochs=40, # Explicit test value
        device=DEVICE
    )

    print(f"Initialized Agent with Batch Size {BATCH_SIZE}")
    print(f"Updates per step: Actor={agent.actor_epochs}, Critic={agent.critic_epochs}")

    # 4. Training Loop (Episodes)
    for episode in range(2):
        obs, _ = env.reset()
        
        done_batch = torch.zeros(BATCH_SIZE, dtype=torch.bool, device=DEVICE)
        total_rewards = torch.zeros(BATCH_SIZE, device=DEVICE)
        steps = 0
        
        while not done_batch.all():
            mask = torch.ones((BATCH_SIZE, 16), device=DEVICE)

            # Select Action
            action = agent.select_action(obs, mask=mask)
            
            # Step Env
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            dones = terminated | truncated
            
            agent.store_reward(reward, dones)
            
            obs = next_obs
            total_rewards += reward * (~done_batch).float()
            done_batch |= dones
            steps += 1
            
        print(f"Ep {episode+1} Complete. Avg Reward: {total_rewards.mean().item():.2f}, Steps: {steps}")

        # Update Agent
        logs = agent.update(batch_size=64)
        if logs:
            print(f"  > Actor Loss: {logs['actor_loss']:.4f} | Critic Loss: {logs['critic_loss']:.4f}")
            print(f"  > Value: {logs['value_mean']:.2f}±{logs['value_std']:.2f} | Return: {logs['return_mean']:.2f}±{logs['return_std']:.2f}")