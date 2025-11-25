"""
Vectorized DVRP Environment for Parallel Rollouts

Enables running multiple DVRP environments in parallel with batched neural oracle calls.
This provides significant speedup compared to sequential execution.
"""

from __future__ import annotations

from typing import List, Tuple, Dict
import numpy as np
import torch
from .dvrp_env import DVRPEnv
from tensordict.tensordict import TensorDict


class VectorizedDVRPEnv:
    """
    Wrapper for running multiple DVRP environments in parallel.

    Key optimization: Batches all neural oracle (routing solver) calls into
    a single forward pass instead of 64 sequential calls.

    Args:
        num_envs: Number of parallel environments (default: 64)
        **env_kwargs: Arguments passed to each DVRPEnv instance
    """

    def __init__(self, num_envs: int = 64, **env_kwargs):
        """Initialize parallel environments with shared neural oracle."""
        self.num_envs = num_envs

        # Create parallel environments
        # Each gets a unique seed for diversity
        base_seed = env_kwargs.get('seed', 42)
        self.envs = []
        for i in range(num_envs):
            env_kwargs_copy = env_kwargs.copy()
            env_kwargs_copy['seed'] = base_seed + i if base_seed is not None else None
            self.envs.append(DVRPEnv(**env_kwargs_copy))

        # Extract shared policy (all envs use the same neural oracle)
        self.policy = self.envs[0].policy

        # Store environment parameters
        self.num_customers = self.envs[0].num_customers
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self) -> Tuple[np.ndarray, List[Dict]]:
        """
        Reset all environments.

        Returns:
            states: (num_envs, num_customers, 6) array of initial states
            infos: List of info dicts from each environment
        """
        states = []
        infos = []

        for env in self.envs:
            state, info = env.reset()
            states.append(state)
            infos.append(info)

        return np.stack(states, axis=0), infos

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """
        Step all environments in parallel with batched neural oracle.

        Args:
            actions: (num_envs,) array of action indices

        Returns:
            states: (num_envs, num_customers, 6) next states
            rewards: (num_envs,) rewards
            dones: (num_envs,) termination flags
            truncated: (num_envs,) truncation flags
            infos: List of info dicts
        """
        # Phase 1: Apply perturbations and build routing problems
        new_requests_batch = []
        accepted_batch = []
        patience_penalties = []
        previous_costs = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            # Get current incoming request
            new_request = env._slice(env.pending_requests, env.current_step)

            # Decode action to perturbation
            pickup_shift, dropoff_shift = env.ACTION_SPACE_MAP[action]

            # Check acceptance based on flexibility
            flexibility = new_request["flexibility"][0].cpu().numpy()  # [2]
            flexibility_pickup_earlier = flexibility[0]
            flexibility_dropoff_later = flexibility[1]

            accepted = (abs(pickup_shift) <= flexibility_pickup_earlier) and \
                      (dropoff_shift <= flexibility_dropoff_later)

            if accepted:
                perturbed_request = env._apply_perturbation(
                    new_request, pickup_shift, dropoff_shift
                )
                patience_penalty = (abs(pickup_shift) + abs(dropoff_shift)) * env.patience_factor
            else:
                perturbed_request = new_request.clone()
                patience_penalty = 0.0

            # Add to current requests
            env.current_requests = env._append_to_current(
                env.current_requests, perturbed_request
            )

            new_requests_batch.append(env.current_requests)
            accepted_batch.append(accepted)
            patience_penalties.append(patience_penalty)
            previous_costs.append(env.previous_cost)

        # Phase 2: Batched neural oracle call (KEY OPTIMIZATION!)
        new_costs = self._batch_solve_routing(new_requests_batch)

        # Phase 3: Compute rewards and collect results
        states, rewards, dones, truncs, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            new_cost = new_costs[i]

            # Check for solver failure
            if new_cost == float('inf'):
                reward = -5000.0
                terminated = True
                truncated = False
            else:
                # Calculate reward
                reward = previous_costs[i] - new_cost - patience_penalties[i]
                env.previous_cost = new_cost
                env.current_step += 1
                terminated = env.current_step >= env.num_customers
                truncated = False

            # Get next observation
            observation = env._get_observation()

            # Get user_id for current request
            user_id = None
            if "user_id" in env.pending_requests.keys():
                pickup_idx = 2 * (env.current_step - 1) + 1
                if pickup_idx > 0:  # Ensure valid index
                    user_id = env.pending_requests["user_id"][0, pickup_idx].item()

            # Build info dict
            info = {
                "step": env.current_step,
                "current_cost": new_cost,
                "accepted": accepted_batch[i],
                "patience_penalty": patience_penalties[i],
                "num_requests": (env.current_requests["h3_indices"].shape[1] - 1) // 2,
                "user_id": user_id,
                "solver_failed": new_cost == float('inf')
            }

            states.append(observation)
            rewards.append(reward)
            dones.append(terminated)
            truncs.append(truncated)
            infos.append(info)

        return (
            np.stack(states, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(truncs, dtype=bool),
            infos
        )

    def _batch_solve_routing(self, requests_batch: List[TensorDict]) -> np.ndarray:
        """
        Solve all routing problems in a single batched neural oracle call.

        This is the key optimization: instead of 64 sequential forward passes,
        we do 1 batched forward pass.

        Args:
            requests_batch: List of num_envs TensorDicts, each with batch_size=[1]

        Returns:
            costs: (num_envs,) array of routing costs
        """
        if self.policy is None:
            # Fallback: if no policy, return inf costs
            # (This shouldn't happen with neural oracle)
            return np.full(self.num_envs, float('inf'), dtype=np.float32)

        # Stack all TensorDicts into a single batch
        # Each TensorDict has batch_size=[1], concatenate to batch_size=[num_envs]
        batched_td = torch.cat(requests_batch, dim=0)  # batch_size=[num_envs]

        # Single batched forward pass through neural oracle
        with torch.no_grad():
            out = self.policy(
                batched_td,
                phase='test',
                decode_type="greedy",
                return_actions=True
            )
            # Extract costs (negative rewards)
            costs = -out["reward"].cpu().numpy()  # (num_envs,)

        return costs

    def get_current_user_ids(self) -> np.ndarray:
        """
        Get user_id of current incoming request for all environments.

        Returns:
            user_ids: (num_envs,) array of user IDs
        """
        user_ids = []
        for env in self.envs:
            user_ids.append(env.get_current_user_id())
        return np.array(user_ids, dtype=np.int64)

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def test_vectorized_env():
    """Test the vectorized environment."""
    print("Testing VectorizedDVRPEnv...")
    print("=" * 80)

    # Create vectorized environment with 4 envs for testing
    num_envs = 4
    num_customers = 5

    vec_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        seed=42
    )

    print(f"Created {num_envs} parallel environments")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")

    # Reset
    print("\n--- Resetting Environments ---")
    states, infos = vec_env.reset()
    print(f"States shape: {states.shape}")
    print(f"Expected: ({num_envs}, {num_customers}, 6)")
    assert states.shape == (num_envs, num_customers, 6)

    # Run a few steps
    print("\n--- Running Steps ---")
    total_rewards = np.zeros(num_envs)

    for step in range(3):
        # Random actions for all envs
        actions = np.random.randint(0, 16, size=num_envs)

        states, rewards, dones, truncs, infos = vec_env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"  States shape: {states.shape}")
        print(f"  Rewards: {rewards}")
        print(f"  Dones: {dones}")
        print(f"  Accepted: {[info['accepted'] for info in infos]}")

        total_rewards += rewards

        if dones.all():
            print("All environments terminated")
            break

    print(f"\nTotal rewards per env: {total_rewards}")
    print("\nTest completed successfully!")

    vec_env.close()


if __name__ == "__main__":
    test_vectorized_env()
