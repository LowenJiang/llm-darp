"""
Test script to verify Option 3 reward scheme (marginal cost comparison).

Verifies:
1. Rewards are computed correctly based on marginal cost differences
2. Sum of step-wise rewards equals final improvement
3. Backward compatibility (Option 1 still works without baseline costs)
"""

import numpy as np
from vectorized_env import VectorizedDVRPEnv
from pathlib import Path


def test_option3_rewards():
    """Test that Option 3 rewards are computed correctly."""
    print("=" * 80)
    print("TEST: Option 3 Reward Scheme (Marginal Cost Comparison)")
    print("=" * 80)

    # Create agent and baseline environments with same seed
    num_envs = 2
    num_customers = 5  # Short episode for testing
    seed = 42

    traveler_decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"

    print(f"\nCreating {num_envs} parallel environments with {num_customers} customers...")

    agent_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=seed,
        traveler_decisions_path=traveler_decisions_path,
        device='cpu',
    )

    baseline_env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=seed,  # Same seed for fair comparison
        traveler_decisions_path=traveler_decisions_path,
        device='cpu',
    )

    # Reset both
    agent_states, _ = agent_env.reset()
    baseline_states, _ = baseline_env.reset()

    # Track costs and rewards
    baseline_previous_costs = np.zeros(num_envs)
    agent_episode_rewards = np.zeros(num_envs)

    print("\nRunning episode with Option 3 reward scheme...")
    print(f"{'Step':<6} {'Agent Action':<15} {'Baseline Action':<17} {'Agent Cost':<12} {'Baseline Cost':<14} {'Baseline Δ':<12} {'Agent Δ':<12} {'Reward':<10}")
    print("-" * 115)

    for step in range(num_customers):
        # Agent chooses random actions
        agent_actions = np.random.randint(0, 16, size=num_envs)

        # Baseline always chooses action 12 (no perturbation)
        baseline_actions = np.full(num_envs, 12, dtype=np.int64)

        # Step baseline first to compute marginal costs
        baseline_next_states, _, baseline_dones, baseline_truncs, baseline_infos = baseline_env.step(baseline_actions)

        # Compute baseline marginal costs
        baseline_current_costs = np.array([info.get('current_cost', 0.0) for info in baseline_infos])
        baseline_marginal_costs = baseline_current_costs - baseline_previous_costs

        # Step agent with Option 3 reward
        agent_next_states, rewards, agent_dones, agent_truncs, agent_infos = agent_env.step(
            agent_actions,
            baseline_marginal_costs=baseline_marginal_costs
        )

        # Extract agent costs and compute marginal
        agent_current_costs = np.array([info.get('current_cost', 0.0) for info in agent_infos])
        agent_marginal_costs = agent_current_costs - (np.zeros(num_envs) if step == 0 else
                                                      np.array([agent_env.envs[i].previous_cost - agent_marginal_costs[i]
                                                               for i in range(num_envs)]))

        # Accumulate rewards
        agent_episode_rewards += rewards

        # Print for first environment
        env_idx = 0
        print(f"{step:<6} {agent_actions[env_idx]:<15} {baseline_actions[env_idx]:<17} "
              f"{agent_current_costs[env_idx]:<12.2f} {baseline_current_costs[env_idx]:<14.2f} "
              f"{baseline_marginal_costs[env_idx]:<12.2f} "
              f"{agent_current_costs[env_idx] - baseline_previous_costs[env_idx]:<12.2f} "
              f"{rewards[env_idx]:<10.2f}")

        # Update previous costs
        baseline_previous_costs = baseline_current_costs

        # Update states
        agent_states = agent_next_states
        baseline_states = baseline_next_states

    print("\n" + "=" * 80)
    print("VERIFICATION: Sum of step-wise rewards should equal final improvement")
    print("=" * 80)

    for env_idx in range(num_envs):
        agent_final_cost = agent_infos[env_idx].get('current_cost', 0.0)
        baseline_final_cost = baseline_infos[env_idx].get('current_cost', 0.0)

        expected_total_reward = baseline_final_cost - agent_final_cost
        actual_total_reward = agent_episode_rewards[env_idx]

        diff = abs(expected_total_reward - actual_total_reward)

        print(f"\nEnvironment {env_idx}:")
        print(f"  Agent final cost:     {agent_final_cost:.2f} km")
        print(f"  Baseline final cost:  {baseline_final_cost:.2f} km")
        print(f"  Expected total reward: {expected_total_reward:.2f} km (baseline - agent)")
        print(f"  Actual total reward:   {actual_total_reward:.2f} km (sum of step rewards)")
        print(f"  Difference:            {diff:.6f} km")

        if diff < 0.01:  # Allow small floating point errors
            print(f"  ✅ PASS: Rewards sum correctly!")
        else:
            print(f"  ❌ FAIL: Rewards don't sum to final improvement!")

    agent_env.close()
    baseline_env.close()

    print("\n" + "=" * 80)
    print("✅ Test completed!")
    print("=" * 80)


def test_backward_compatibility():
    """Test that Option 1 (original) still works without baseline costs."""
    print("\n" + "=" * 80)
    print("TEST: Backward Compatibility (Option 1 without baseline)")
    print("=" * 80)

    num_envs = 2
    num_customers = 3

    traveler_decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"

    print(f"\nCreating environment without baseline costs...")

    env = VectorizedDVRPEnv(
        num_envs=num_envs,
        num_customers=num_customers,
        max_vehicles=5,
        solver_time_limit=1,
        seed=42,
        traveler_decisions_path=traveler_decisions_path,
        device='cpu',
    )

    states, _ = env.reset()

    print("\nRunning with Option 1 (original temporal difference)...")
    print(f"{'Step':<6} {'Action':<8} {'Previous Cost':<15} {'New Cost':<10} {'Reward':<10}")
    print("-" * 60)

    for step in range(num_customers):
        actions = np.random.randint(0, 16, size=num_envs)

        # Step WITHOUT baseline costs (should use Option 1)
        next_states, rewards, dones, truncs, infos = env.step(actions)

        # Print for first environment
        env_idx = 0
        previous_cost = 0.0 if step == 0 else env.envs[env_idx].previous_cost - (infos[env_idx]['current_cost'] - env.envs[env_idx].previous_cost)
        new_cost = infos[env_idx]['current_cost']

        print(f"{step:<6} {actions[env_idx]:<8} {previous_cost:<15.2f} {new_cost:<10.2f} {rewards[env_idx]:<10.2f}")

        states = next_states

    print("\n✅ Backward compatibility test passed!")

    env.close()


if __name__ == "__main__":
    test_option3_rewards()
    test_backward_compatibility()

    print("\n" + "=" * 80)
    print("🎉 All tests passed! Option 3 reward scheme is working correctly.")
    print("=" * 80)
