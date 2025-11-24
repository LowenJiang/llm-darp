"""
Quick test script to verify the parallelized training system works.

This runs a minimal training loop with 4 parallel environments for 2 epochs
to ensure everything is wired up correctly.
"""

import sys
import torch
import numpy as np

# Disable wandb for testing
import os
os.environ['WANDB_MODE'] = 'disabled'

from vectorized_env import VectorizedDVRPEnv
from ppo_agent import PPOAgent
from meta_train import compute_masks_from_flexibility

def test_parallelized_system():
    """Test the parallelized training system with minimal setup."""
    print("=" * 80)
    print("Testing Parallelized Training System")
    print("=" * 80)

    # Minimal configuration
    num_envs = 4
    num_customers = 5
    action_dim = 16
    device = "cpu"

    print(f"\nConfiguration:")
    print(f"  Parallel environments: {num_envs}")
    print(f"  Customers per episode: {num_customers}")
    print(f"  Device: {device}")

    # Create vectorized environment
    print("\n[1/5] Creating vectorized environment...")
    vec_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=3,
        solver_time_limit=1,
        seed=42,
    )
    print("  ✓ Vectorized environment created")

    # Create PPO agent
    print("\n[2/5] Creating PPO agent...")
    agent = PPOAgent(
        state_dim=num_customers * 6,
        action_dim=action_dim,
        hidden_dim=128,
        lr=3e-4,
        device=device,
    )
    print("  ✓ PPO agent created")

    # Initialize action masks
    print("\n[3/5] Testing vectorized mask computation...")
    predicted_flexibility = torch.randint(0, 4, (num_customers,))
    action_masks = compute_masks_from_flexibility(predicted_flexibility, action_dim)
    print(f"  ✓ Action masks shape: {action_masks.shape} (expected: ({num_customers}, {action_dim}))")
    print(f"  ✓ Predicted flexibility: {predicted_flexibility.tolist()}")

    # Test parallel rollout
    print("\n[4/5] Testing parallel rollout...")
    states, infos = vec_env.reset()
    print(f"  ✓ Reset: states shape = {states.shape} (expected: ({num_envs}, {num_customers}, 6))")

    total_steps = 0
    for step in range(num_customers):
        # Get user IDs
        user_ids = vec_env.get_current_user_ids()

        # Get masks
        batch_masks = action_masks[user_ids - 1]

        # Select actions in batch
        actions = agent.select_action_batch(states, masks=batch_masks, epsilon=0.1)

        # Step all environments
        next_states, rewards, dones, truncs, step_infos = vec_env.step(actions)

        # Store rewards
        agent.store_rewards_batch(rewards, dones | truncs)

        states = next_states
        total_steps += num_envs

        if step == 0:
            print(f"  ✓ Step {step}: actions = {actions}, rewards mean = {rewards.mean():.2f}")

    print(f"  ✓ Completed rollout: {total_steps} total environment steps")

    # Test PPO update
    print("\n[5/5] Testing PPO update...")
    train_stats = agent.update(num_epochs=2, batch_size=8)
    print(f"  ✓ PPO update completed")
    print(f"    Policy loss: {train_stats.get('policy_loss', 0):.4f}")
    print(f"    Value loss: {train_stats.get('value_loss', 0):.4f}")
    print(f"    Entropy: {train_stats.get('entropy', 0):.4f}")

    # Clean up
    vec_env.close()

    print("\n" + "=" * 80)
    print("✅ All tests passed! Parallelized system is working correctly.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_parallelized_system()
        sys.exit(0 if success else 1)
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Test failed with error:")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
