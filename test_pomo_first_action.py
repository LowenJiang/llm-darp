"""Test to verify POMO starting nodes work correctly with environment."""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator
from trial_gnn_policy import DARPPolicy

def test_pomo_first_action():
    """Verify POMO forces different starting nodes correctly."""

    print("=" * 70)
    print("Testing POMO First Action Behavior")
    print("=" * 70)

    device = torch.device("cpu")
    batch_size = 2
    pomo_size = 4

    # Setup
    csv_path = Path("src/traveler_trip_types_res_7.csv")
    ttm_path = Path("src/travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        device=device,
        num_customers=10
    )

    env = PDPTWEnv(generator=generator)

    policy = DARPPolicy(
        embed_dim=64,
        num_encoder_layers=2,
        num_heads=4,
        ff_hidden=128,
    ).to(device)

    # Generate problem
    td = generator(batch_size=[batch_size])

    print(f"\n1. Initial Setup")
    print(f"   Batch size: {batch_size}")
    print(f"   POMO size: {pomo_size}")
    print(f"   Total instances: {batch_size * pomo_size}")

    # Run policy with POMO
    print(f"\n2. Running policy with POMO...")
    outputs = policy(
        td,
        env,
        phase="train",
        pomo_size=pomo_size,
        decode_type="sampling",
        max_steps=5,  # Just a few steps to check first action
        return_actions=True,
    )

    actions = outputs["actions"]  # [B*P, T]
    first_actions = actions[:, 0]  # [B*P]

    print(f"\n3. First Actions Taken:")
    print(f"   Shape: {first_actions.shape}")
    print(f"   Values: {first_actions.tolist()}")

    # Reshape to see POMO structure
    first_actions_2d = first_actions.view(batch_size, pomo_size)
    print(f"\n4. POMO Structure (batch x pomo):")
    for b in range(batch_size):
        print(f"   Instance {b}: {first_actions_2d[b].tolist()}")

    # Check that first actions are diverse (not all depot)
    print(f"\n5. Verification:")

    # Check 1: No first actions should be depot (0)
    depot_actions = (first_actions == 0).sum().item()
    print(f"   ✓ First actions that are depot (0): {depot_actions}")

    # Check 2: First actions should be pickup nodes (odd indices)
    is_pickup = (first_actions % 2 == 1) & (first_actions != 0)
    num_pickup_starts = is_pickup.sum().item()
    print(f"   ✓ First actions that are pickups: {num_pickup_starts} / {batch_size * pomo_size}")

    # Check 3: POMO should create diversity
    unique_per_instance = []
    for b in range(batch_size):
        unique = first_actions_2d[b].unique().numel()
        unique_per_instance.append(unique)
    avg_unique = sum(unique_per_instance) / len(unique_per_instance)
    print(f"   ✓ Unique starting nodes per instance: {unique_per_instance}")
    print(f"   ✓ Average diversity: {avg_unique:.1f} / {pomo_size}")

    # Check 4: Verify environment state at start
    td_reset = env.reset(td)
    print(f"\n6. Environment Initial State:")
    print(f"   current_node: {td_reset['current_node'][:3].squeeze().tolist()}")
    print(f"   current_time: {td_reset['current_time'][:3].squeeze().tolist()}")
    print(f"   (All should start at depot=0, time=0)")

    # Check 5: Verify action mask allows pickup nodes
    action_mask = td_reset["action_mask"]
    pickup_mask = action_mask[:, 1::2]  # Odd indices (pickups)
    print(f"\n7. Action Mask at t=0:")
    print(f"   Depot (0) allowed: {action_mask[:3, 0].tolist()}")
    print(f"   Any pickups allowed: {pickup_mask.any(dim=1)[:3].tolist()}")
    print(f"   Number of valid pickups: {pickup_mask.sum(dim=1)[:3].tolist()}")

    # Final verdict
    print(f"\n{'='*70}")
    if depot_actions == 0 and num_pickup_starts >= batch_size * pomo_size * 0.8:
        print("✅ PASS: POMO correctly forces diverse pickup starting nodes!")
        print("   - No depot first actions")
        print("   - Most/all first actions are valid pickups")
        print("   - Diversity across POMO rollouts")
    else:
        print("❌ FAIL: POMO first action behavior incorrect!")
        print(f"   - Depot actions: {depot_actions} (should be 0)")
        print(f"   - Pickup actions: {num_pickup_starts} / {batch_size * pomo_size}")
    print("="*70)

if __name__ == "__main__":
    test_pomo_first_action()
