"""Test vectorized lookahead validity implementation."""

import torch
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator

def create_legacy_lookahead():
    """Create a legacy version of lookahead for comparison."""

    def legacy_lookahead(env, td, time_at_candidate, node_indices, num_nodes):
        """Legacy implementation with capacity loop."""
        batch_size = node_indices.shape[0]
        device = node_indices.device
        capacity = td["pending_schedule"].shape[1]

        star_valid = torch.ones_like(time_at_candidate, dtype=torch.bool)

        # Original loop-based implementation
        for k in range(capacity):
            target = td["pending_schedule"][:, k].unsqueeze(-1)
            target_expanded = target.expand(batch_size, num_nodes)

            has_target = target_expanded != 0
            travel_to_target = env._get_travel_time(td, node_indices, target_expanded)
            arrival_at_target = time_at_candidate + travel_to_target

            from oracle_env import gather_by_index
            target_late = gather_by_index(td["time_windows"], target_expanded)[..., 1]

            is_late = arrival_at_target > target_late
            star_valid = star_valid & ~(has_target & is_late)

        # Check new dropoff
        candidate_dropoff = torch.where(
            (node_indices % 2 != 0) & (node_indices != 0),
            torch.clamp(node_indices + 1, max=num_nodes - 1),
            torch.zeros_like(node_indices)
        )
        has_new_drop = candidate_dropoff != 0

        if has_new_drop.any():
            travel_new = env._get_travel_time(td, node_indices, candidate_dropoff)
            arrival_new = time_at_candidate + travel_new
            from oracle_env import gather_by_index
            drop_late = gather_by_index(td["time_windows"], candidate_dropoff)[..., 1]
            star_valid = star_valid & ~(has_new_drop & (arrival_new > drop_late))

        return star_valid

    return legacy_lookahead


def test_lookahead_correctness():
    """Test that vectorized lookahead produces identical results."""

    print("=" * 70)
    print("Testing Vectorized Lookahead Validity")
    print("=" * 70)

    device = torch.device("cpu")
    batch_size = 16

    csv_path = Path("src/traveler_trip_types_res_7.csv")
    ttm_path = Path("src/travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        device=device,
        num_customers=15
    )

    env = PDPTWEnv(generator=generator)
    legacy_lookahead = create_legacy_lookahead()

    print(f"\n1. Setup: batch_size={batch_size}, num_customers=15")

    # Generate initial state
    state = env.reset(batch_size=[batch_size])

    # Run several environment steps to get diverse states
    print("\n2. Running rollout to generate diverse test states...")
    test_states = [state]

    for step in range(20):
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

        state["action"] = action
        state = env.step(state)["next"]

        if step % 5 == 0:
            test_states.append(state.clone())

    print(f"   Collected {len(test_states)} diverse states")

    # Test lookahead on all states
    print("\n3. Testing lookahead correctness...")

    all_match = True
    max_diff = 0

    for i, td in enumerate(test_states):
        num_nodes = td["h3_indices"].shape[1]
        current_time = td["current_time"]
        current_node = td["current_node"].squeeze(-1)

        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, -1)

        # Compute time at candidate (mimicking action mask logic)
        travel_time_to_candidate = env._get_travel_time(
            td,
            current_node.unsqueeze(-1),
            node_indices
        )
        arrival_at_candidate = current_time + travel_time_to_candidate
        candidate_early = td["time_windows"][..., 0]
        time_at_candidate = torch.max(arrival_at_candidate, candidate_early)

        # Compute with both methods
        vectorized = env._lookahead_validity(td, time_at_candidate, node_indices, num_nodes)
        legacy = legacy_lookahead(env, td, time_at_candidate, node_indices, num_nodes)

        # Compare
        matches = (vectorized == legacy).all().item()
        if not matches:
            all_match = False
            diff = (vectorized != legacy).sum().item()
            max_diff = max(max_diff, diff)
            print(f"   State {i}: ❌ MISMATCH - {diff} differences")
        else:
            print(f"   State {i}: ✓ Match")

    print("\n4. Results:")
    if all_match:
        print("   ✅ PASS: All states produce identical results!")
    else:
        print(f"   ❌ FAIL: Found mismatches (max {max_diff} differences)")

    return all_match


def benchmark_lookahead():
    """Benchmark vectorized vs legacy lookahead."""

    print("\n" + "=" * 70)
    print("Benchmarking Lookahead Performance")
    print("=" * 70)

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
    legacy_lookahead = create_legacy_lookahead()

    batch_sizes = [16, 32, 64, 128]

    print("\nBatch Size | Vectorized | Legacy   | Speedup")
    print("-" * 55)

    for batch_size in batch_sizes:
        # Generate test state
        state = env.reset(batch_size=[batch_size])

        # Run a few steps to get realistic state
        for _ in range(10):
            if state["done"].all():
                break
            mask = state["action_mask"]
            valid_counts = mask.sum(dim=-1)
            if (valid_counts == 0).any():
                action = torch.zeros(batch_size, dtype=torch.long, device=device)
            else:
                probs = mask.float() / valid_counts.unsqueeze(-1)
                action = torch.multinomial(probs, 1).squeeze(-1)
            state["action"] = action
            state = env.step(state)["next"]

        # Prepare inputs
        num_nodes = state["h3_indices"].shape[1]
        current_time = state["current_time"]
        current_node = state["current_node"].squeeze(-1)
        node_indices = torch.arange(num_nodes, device=device).expand(batch_size, -1)

        travel_time_to_candidate = env._get_travel_time(
            state, current_node.unsqueeze(-1), node_indices
        )
        arrival_at_candidate = current_time + travel_time_to_candidate
        candidate_early = state["time_windows"][..., 0]
        time_at_candidate = torch.max(arrival_at_candidate, candidate_early)

        # Warmup
        _ = env._lookahead_validity(state, time_at_candidate, node_indices, num_nodes)

        # Benchmark vectorized
        n_runs = 100
        t0 = time.time()
        for _ in range(n_runs):
            _ = env._lookahead_validity(state, time_at_candidate, node_indices, num_nodes)
        if device.type == "mps":
            torch.mps.synchronize()
        time_vec = (time.time() - t0) / n_runs

        # Benchmark legacy
        t0 = time.time()
        for _ in range(n_runs):
            _ = legacy_lookahead(env, state, time_at_candidate, node_indices, num_nodes)
        if device.type == "mps":
            torch.mps.synchronize()
        time_legacy = (time.time() - t0) / n_runs

        speedup = time_legacy / time_vec
        print(f"{batch_size:10d} | {time_vec*1000:9.2f}ms | {time_legacy*1000:8.2f}ms | {speedup:6.2f}x")


if __name__ == "__main__":
    # Test correctness
    success = test_lookahead_correctness()

    if success:
        # Benchmark performance
        benchmark_lookahead()
    else:
        print("\n⚠️  Skipping benchmark due to correctness test failure")
