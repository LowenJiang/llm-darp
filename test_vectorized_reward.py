"""Test script to validate vectorized reward computation."""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator

def test_reward_equivalence():
    """Test that vectorized and legacy reward computation produce identical results."""

    print("=" * 60)
    print("Testing Vectorized Reward Computation")
    print("=" * 60)

    # Setup
    device = torch.device("cpu")  # Use CPU for testing
    batch_size = 8

    csv_path = Path("src/traveler_trip_types_res_7.csv")
    ttm_path = Path("src/travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        device=device,
        num_customers=10  # Smaller for faster testing
    )

    env = PDPTWEnv(generator=generator)

    # Generate test episode
    print(f"\n1. Generating {batch_size} problem instances...")
    state = env.reset(batch_size=[batch_size])

    # Run random rollout
    print("2. Running random rollout...")
    actions_taken = []
    max_steps = 100

    for step in range(max_steps):
        if state["done"].all():
            break

        mask = state["action_mask"]
        # Handle case where some envs have no valid actions
        valid_counts = mask.sum(dim=-1)
        if (valid_counts == 0).any():
            # Force depot for envs with no valid actions
            action = torch.zeros(batch_size, dtype=torch.long, device=device)
            action[valid_counts > 0] = torch.multinomial(
                mask[valid_counts > 0].float(), 1
            ).squeeze(-1)
        else:
            probs = mask.float() / valid_counts.unsqueeze(-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        actions_taken.append(action)
        state["action"] = action
        state = env.step(state)["next"]

    actions_tensor = torch.stack(actions_taken, dim=1)
    print(f"   Rollout completed in {len(actions_taken)} steps")

    # Test both implementations
    print("\n3. Computing rewards...")
    print("   a) Vectorized implementation...")
    import time
    t0 = time.time()
    reward_vec = env.get_reward(state, actions_tensor, use_vectorized=True)
    time_vec = time.time() - t0

    print("   b) Legacy implementation...")
    t0 = time.time()
    reward_legacy = env.get_reward(state, actions_tensor, use_vectorized=False)
    time_legacy = time.time() - t0

    # Compare results
    print("\n4. Results:")
    print(f"   Vectorized time: {time_vec*1000:.2f} ms")
    print(f"   Legacy time:     {time_legacy*1000:.2f} ms")
    print(f"   Speedup:         {time_legacy/time_vec:.2f}x")
    print()
    print(f"   Vectorized rewards: {reward_vec.tolist()}")
    print(f"   Legacy rewards:     {reward_legacy.tolist()}")
    print()

    # Check if identical
    if torch.allclose(reward_vec, reward_legacy, rtol=1e-5):
        print("✅ PASS: Rewards match perfectly!")
        return True
    else:
        print("❌ FAIL: Rewards don't match!")
        diff = (reward_vec - reward_legacy).abs()
        print(f"   Max difference: {diff.max().item()}")
        print(f"   Mean difference: {diff.mean().item()}")
        return False

def benchmark_scaling():
    """Benchmark performance with increasing batch sizes."""

    print("\n" + "=" * 60)
    print("Benchmarking Reward Computation Scaling")
    print("=" * 60)

    device = torch.device("mps:0" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    csv_path = Path("src/traveler_trip_types_res_7.csv")
    ttm_path = Path("src/travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        device=device,
        num_customers=15
    )

    env = PDPTWEnv(generator=generator)

    batch_sizes = [16, 32, 64, 128]

    print("\nBatch Size | Vectorized | Legacy   | Speedup")
    print("-" * 50)

    for batch_size in batch_sizes:
        # Generate episode
        state = env.reset(batch_size=[batch_size])

        actions_taken = []
        for step in range(80):
            if state["done"].all():
                break

            mask = state["action_mask"]
            valid_counts = mask.sum(dim=-1)
            if (valid_counts == 0).any():
                action = torch.zeros(batch_size, dtype=torch.long, device=device)
                action[valid_counts > 0] = torch.multinomial(
                    mask[valid_counts > 0].float(), 1
                ).squeeze(-1)
            else:
                probs = mask.float() / valid_counts.unsqueeze(-1)
                action = torch.multinomial(probs, 1).squeeze(-1)

            actions_taken.append(action)
            state["action"] = action
            state = env.step(state)["next"]

        actions_tensor = torch.stack(actions_taken, dim=1)

        # Warmup
        _ = env.get_reward(state, actions_tensor, use_vectorized=True)

        # Benchmark vectorized
        import time
        n_runs = 10
        t0 = time.time()
        for _ in range(n_runs):
            _ = env.get_reward(state, actions_tensor, use_vectorized=True)
        time_vec = (time.time() - t0) / n_runs

        # Benchmark legacy
        t0 = time.time()
        for _ in range(n_runs):
            _ = env.get_reward(state, actions_tensor, use_vectorized=False)
        time_legacy = (time.time() - t0) / n_runs

        speedup = time_legacy / time_vec
        print(f"{batch_size:10d} | {time_vec*1000:9.2f}ms | {time_legacy*1000:8.2f}ms | {speedup:6.2f}x")

if __name__ == "__main__":
    # Run tests
    success = test_reward_equivalence()

    if success:
        benchmark_scaling()
    else:
        print("\n⚠️  Skipping benchmark due to test failure")
