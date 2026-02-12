"""Demonstrate POMO starting node selection strategy."""

import torch

def _get_pomo_starting_nodes(num_nodes: int, pomo_size: int, device: torch.device) -> torch.Tensor:
    """Select POMO starting nodes (pickup nodes with odd indices)."""
    pickup_nodes = torch.arange(1, num_nodes, 2, device=device)  # [1, 3, 5, ...]
    num_pickups = pickup_nodes.shape[0]

    if pomo_size <= num_pickups:
        # Select a spread of pickup nodes
        indices = torch.linspace(0, num_pickups - 1, pomo_size, device=device).long()
        return pickup_nodes[indices]
    else:
        # Cycle through pickup nodes
        repeats = (pomo_size + num_pickups - 1) // num_pickups
        repeated = pickup_nodes.repeat(repeats)
        return repeated[:pomo_size]

print("=" * 70)
print("POMO Starting Node Selection Strategy")
print("=" * 70)

# Example: 61 nodes total = 1 depot + 30 pickups + 30 dropoffs
num_nodes = 61  # 0 (depot), 1,3,5,...,59 (pickups), 2,4,6,...,60 (dropoffs)
num_customers = 30

print(f"\n📊 Problem Setup:")
print(f"   Total nodes: {num_nodes}")
print(f"   Customers: {num_customers}")
print(f"   Valid starting nodes (pickups): {num_customers}")
print(f"   Pickup node indices: 1, 3, 5, ..., {2*num_customers-1}")

# Test different POMO sizes
pomo_sizes = [5, 10, 20, 30, 40]

print(f"\n" + "=" * 70)
print("POMO Selection for Different Sizes")
print("=" * 70)

for pomo_size in pomo_sizes:
    starting_nodes = _get_pomo_starting_nodes(num_nodes, pomo_size, torch.device("cpu"))

    print(f"\n🎯 POMO Size = {pomo_size}")
    print(f"   Starting nodes: {starting_nodes.tolist()}")

    # Check uniqueness
    unique = starting_nodes.unique()
    print(f"   Unique nodes: {len(unique)} / {pomo_size}")

    # Check spacing
    if pomo_size <= num_customers:
        # Calculate ideal spacing
        spacing = (num_customers - 1) / (pomo_size - 1) if pomo_size > 1 else 0
        print(f"   Strategy: EVENLY SPACED (spacing ≈ {spacing:.1f} customers apart)")
    else:
        print(f"   Strategy: CYCLING (more POMO than pickups, some repeats)")

# Detailed example for pomo_size=20
print(f"\n" + "=" * 70)
print("Detailed Example: POMO Size = 20 with 30 Customers")
print("=" * 70)

pomo_size = 20
starting_nodes = _get_pomo_starting_nodes(num_nodes, pomo_size, torch.device("cpu"))

print(f"\n1️⃣  Available pickup nodes (30 total):")
pickup_nodes = torch.arange(1, num_nodes, 2)
print(f"   {pickup_nodes.tolist()}")

print(f"\n2️⃣  POMO selects using linspace(0, 29, 20):")
indices = torch.linspace(0, num_customers - 1, pomo_size).long()
print(f"   Indices into pickup array: {indices.tolist()}")
print(f"   (Evenly spaced from 0 to 29)")

print(f"\n3️⃣  Selected starting nodes:")
print(f"   {starting_nodes.tolist()}")

print(f"\n4️⃣  Which customers are selected:")
customer_ids = [(node - 1) // 2 + 1 for node in starting_nodes.tolist()]
print(f"   Customers: {customer_ids}")
print(f"   (Approximately every {30/19:.1f}th customer)")

# Compare with random selection
print(f"\n" + "=" * 70)
print("Comparison: POMO (Deterministic) vs Random Sampling")
print("=" * 70)

pomo_size = 20

# POMO selection (deterministic, evenly spaced)
pomo_nodes = _get_pomo_starting_nodes(num_nodes, pomo_size, torch.device("cpu"))

# Random selection (for comparison)
torch.manual_seed(42)
random_nodes = torch.arange(1, num_nodes, 2)[torch.randperm(num_customers)[:pomo_size]]

print(f"\n📐 POMO (evenly spaced):")
print(f"   {pomo_nodes.tolist()}")
print(f"   Coverage: {len(pomo_nodes.unique())} unique nodes")
print(f"   Min gap: {(pomo_nodes[1:] - pomo_nodes[:-1]).min().item()}")
print(f"   Max gap: {(pomo_nodes[1:] - pomo_nodes[:-1]).max().item()}")

print(f"\n🎲 Random sampling:")
random_sorted = random_nodes.sort().values
print(f"   {random_sorted.tolist()}")
print(f"   Coverage: {len(random_nodes.unique())} unique nodes")
gaps = random_sorted[1:] - random_sorted[:-1]
print(f"   Min gap: {gaps.min().item()}")
print(f"   Max gap: {gaps.max().item()}")

print(f"\n💡 Key Insight:")
print(f"   POMO uses DETERMINISTIC, EVENLY-SPACED selection")
print(f"   - Ensures good coverage of the solution space")
print(f"   - Same starting nodes every time (reproducible)")
print(f"   - No randomness or exclusion logic needed")

print("\n" + "=" * 70)
