"""
Proximal Policy Optimization (PPO) Agent

Implements PPO for the DVRP environment using PyTorch.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """
    Policy network (Actor) for PPO.

    Input: State (30, 6) flattened to 180-dimensional vector
    Output: Action probabilities over discrete actions
    """

    def __init__(
        self,
        state_dim: int = 180,
        action_dim: int = 16,
        hidden_dim: int = 256,
    ):
        """
        Args:
            state_dim: Flattened state dimension (default: 180)
            action_dim: Number of discrete actions (default: 16)
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super(PolicyNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, state_flat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state_flat: Flattened state tensor (batch_size, state_dim)

        Returns:
            action_logits: Action logits (batch_size, action_dim)
        """
        return self.network(state_flat)


class ValueNetwork(nn.Module):
    """
    Value network (Critic) for PPO.

    Input: State (30, 6) flattened to 180-dimensional vector
    Output: State value estimation
    """

    def __init__(
        self,
        state_dim: int = 180,
        hidden_dim: int = 256,
    ):
        """
        Args:
            state_dim: Flattened state dimension (default: 180)
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super(ValueNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_flat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state_flat: Flattened state tensor (batch_size, state_dim)

        Returns:
            state_value: State value (batch_size, 1)
        """
        return self.network(state_flat)


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO with separate policy and value networks.

    Input: State (30, 6) flattened to 180-dimensional vector
    Actor output: Action probabilities over 16 discrete actions
    Critic output: State value estimation
    """

    def __init__(
        self,
        state_dim: int = 180,  # 30 requests * 6 features
        action_dim: int = 16,
        hidden_dim: int = 256,
    ):
        """
        Args:
            state_dim: Flattened state dimension (default: 180)
            action_dim: Number of discrete actions (default: 16)
            hidden_dim: Hidden layer dimension (default: 256)
        """
        super(ActorCritic, self).__init__()

        # Separate policy network (actor)
        self.policy_network = PolicyNetwork(state_dim, action_dim, hidden_dim)

        # Separate value network (critic)
        self.value_network = ValueNetwork(state_dim, hidden_dim)

        self.mask = torch.ones(action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch_size, 30, 6)

        Returns:
            action_probs: Action probabilities (batch_size, 16)
            state_value: State value (batch_size, 1)
        """
        # Flatten state: (batch_size, 30, 6) -> (batch_size, 180)
        state_flat = state.reshape(state.shape[0], -1)

        # Policy network: action logits
        action_logits = self.policy_network(state_flat)
        action_probs = torch.softmax(action_logits, dim=-1)

        # Value network: state value
        state_value = self.value_network(state_flat)

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
            state: State tensor of shape (30, 6)
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
            states: Batch of states (batch_size, 30, 6)
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
        state_dim: int = 240,
        action_dim: int = 16,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        """
        Args:
            state_dim: Flattened state dimension
            action_dim: Number of discrete actions
            hidden_dim: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            device: Device to run on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
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
            state: State array of shape (30, 6)
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
        masks: torch.Tensor = None,
        epsilon: float = 0.0
    ) -> np.ndarray:
        """
        Select actions for a batch of states (for parallel environments).

        Args:
            states: State array of shape (batch_size, 30, 6)
            masks: Action mask tensor of shape (batch_size, action_dim) where 1 = allowed, 0 = masked
            epsilon: Epsilon for epsilon-greedy masking

        Returns:
            actions: (batch_size,) array of action indices
        """
        batch_size = states.shape[0]
        state_tensor = torch.FloatTensor(states).to(self.device)  # (batch_size, 30, 6)

        # Move masks to device if provided
        if masks is not None:
            masks = masks.to(self.device)

        with torch.no_grad():
            # Get action probabilities and values for all states
            action_probs, state_values = self.policy.forward(state_tensor)  # (batch_size, action_dim), (batch_size, 1)

            # Apply masks if provided
            if masks is not None:
                # Apply masking for each state in the batch
                for i in range(batch_size):
                    action_probs[i] = self.policy.mask_action(
                        action_probs[i], masks[i], epsilon
                    )

            # Sample actions
            dist = Categorical(action_probs)
            actions = dist.sample()  # (batch_size,)
            log_probs = dist.log_prob(actions)  # (batch_size,)

        # Store for later update
        for i in range(batch_size):
            self.states.append(state_tensor[i])
            self.actions.append(actions[i].item())
            self.log_probs.append(log_probs[i])
            self.values.append(state_values[i].squeeze())

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

    def update(self, num_epochs: int = 10, batch_size: int = 64) -> dict:
        """
        Update policy using PPO algorithm.

        Args:
            num_epochs: Number of PPO epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary with training statistics
        """
        if len(self.states) == 0:
            return {}

        # Ensure buffers are aligned
        assert len(self.states) == len(self.rewards), \
            f"Mismatch: {len(self.states)} states vs {len(self.rewards)} rewards"
        assert len(self.actions) == len(self.rewards), \
            f"Mismatch: {len(self.actions)} actions vs {len(self.rewards)} rewards"
        assert len(self.values) == len(self.rewards), \
            f"Mismatch: {len(self.values)} values vs {len(self.rewards)} rewards"

        # Convert lists to tensors
        states = torch.stack(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.stack(self.values).to(self.device)
        dones = torch.FloatTensor(self.dones).to(self.device)

        # Compute advantages using GAE
        advantages, returns = self._compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update for multiple epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(num_epochs):
            # Generate random mini-batches
            num_samples = len(states)
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                # Compute ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(state_values, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Gradient descent
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Clear buffer
        self.clear_buffer()

        # Return statistics
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
            rewards: Rewards tensor
            values: State values tensor
            dones: Done flags tensor

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            advantages[t] = last_advantage = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            )

        # Returns = advantages + values
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
    """Test the PPO agent."""
    print("Testing PPOAgent...")

    agent = PPOAgent(state_dim=180, action_dim=16, hidden_dim=128)

    # Create dummy state
    state = np.random.randn(30, 6).astype(np.float32)

    # Select action
    action = agent.select_action(state)
    print(f"Selected action: {action}")

    # Store reward
    agent.store_reward(reward=10.0, done=False)

    # Collect more samples
    for _ in range(10):
        state = np.random.randn(30, 6).astype(np.float32)
        action = agent.select_action(state)
        agent.store_reward(reward=np.random.randn(), done=False)

    # Final state
    state = np.random.randn(30, 6).astype(np.float32)
    action = agent.select_action(state)
    agent.store_reward(reward=5.0, done=True)

    # Update
    print("\nPerforming PPO update...")
    stats = agent.update(num_epochs=5, batch_size=4)
    print(f"Training stats: {stats}")


if __name__ == "__main__":
    test_agent()
