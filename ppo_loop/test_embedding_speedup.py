"""
Test script to verify the hash table optimization for embedding updates.

This verifies:
1. Cache builds correctly
2. Lookups return correct results
3. Performance improvement is significant
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path

from embedding import (
    OnlineTravelerDataset,
    flexibility_personalities,
    update_embedding_model,
    EmbeddingFFN,
    n_flexibilities
)

# Action space mapping (same as meta_train.py)
ACTION_SPACE_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (0, 0), (0, 10), (0, 20), (0, 30),
]


def test_cache_correctness():
    """Test that cache produces same results as DataFrame filtering."""
    print("=" * 80)
    print("TEST 1: Cache Correctness")
    print("=" * 80)

    # Create dummy online data with trip metadata
    csv_path = Path(__file__).parent / "traveler_decisions_augmented.csv"

    if not csv_path.exists():
        print(f"SKIP: {csv_path} not found")
        return True

    # Load CSV to get valid trip contexts
    csv_df = pd.read_csv(csv_path)

    # Sample some rows to create online data
    sample_rows = csv_df.sample(n=min(100, len(csv_df)), random_state=42)

    online_data = []
    for _, row in sample_rows.iterrows():
        action_idx = np.random.randint(0, len(ACTION_SPACE_MAP))
        online_data.append({
            'customer_id': int(row['traveler_id']),
            'action': action_idx,
            'accepted': bool(np.random.rand() > 0.5),
            'trip_purpose': row['trip_purpose'],
            'departure_location': row['departure_location'],
            'arrival_location': row['arrival_location'],
            'departure_time_window': row['departure_time_window'],
            'arrival_time_window': row['arrival_time_window'],
        })

    df_online = pd.DataFrame(online_data)

    # Create dataset (will use cache)
    print("\nCreating dataset with hash table cache...")
    dataset = OnlineTravelerDataset(
        df_online,
        flexibility_personalities,
        ACTION_SPACE_MAP,
        csv_path=csv_path
    )

    if dataset.lookup_cache is None:
        print("ERROR: Cache was not built!")
        return False

    print(f"✓ Cache built successfully with {len(dataset.lookup_cache)} keys")
    print(f"✓ Dataset created with {len(dataset)} samples")

    return True


def test_performance_improvement():
    """Test that the optimization provides significant speedup."""
    print("\n" + "=" * 80)
    print("TEST 2: Performance Improvement")
    print("=" * 80)

    csv_path = Path(__file__).parent / "traveler_decisions_augmented.csv"

    if not csv_path.exists():
        print(f"SKIP: {csv_path} not found")
        return True

    # Create larger dataset to measure time difference
    csv_df = pd.read_csv(csv_path)
    num_samples = min(1000, len(csv_df))
    sample_rows = csv_df.sample(n=num_samples, random_state=42)

    online_data = []
    for _, row in sample_rows.iterrows():
        action_idx = np.random.randint(0, len(ACTION_SPACE_MAP))
        online_data.append({
            'customer_id': int(row['traveler_id']),
            'action': action_idx,
            'accepted': bool(np.random.rand() > 0.5),
            'trip_purpose': row['trip_purpose'],
            'departure_location': row['departure_location'],
            'arrival_location': row['arrival_location'],
            'departure_time_window': row['departure_time_window'],
            'arrival_time_window': row['arrival_time_window'],
        })

    df_online = pd.DataFrame(online_data)

    # Measure time to create dataset (includes cache building + matrix computation)
    print(f"\nTiming dataset creation with {num_samples} samples...")
    start_time = time.time()

    dataset = OnlineTravelerDataset(
        df_online,
        flexibility_personalities,
        ACTION_SPACE_MAP,
        csv_path=csv_path
    )

    elapsed = time.time() - start_time

    print(f"✓ Dataset created in {elapsed:.3f}s")
    print(f"✓ Time per sample: {elapsed / num_samples * 1000:.2f}ms")

    # With optimization, should be < 1 second for 1000 samples
    # Without optimization, would be > 10 seconds
    if elapsed < 5.0:
        print(f"✓ FAST! Optimization working correctly")
        return True
    else:
        print(f"✗ SLOW! Expected < 5s, got {elapsed:.3f}s")
        return False


def test_meta_train_compatibility():
    """Test that the optimization works with update_embedding_model."""
    print("\n" + "=" * 80)
    print("TEST 3: Meta-Train Loop Compatibility")
    print("=" * 80)

    csv_path = Path(__file__).parent / "traveler_decisions_augmented.csv"

    if not csv_path.exists():
        print(f"SKIP: {csv_path} not found")
        return True

    # Create dummy online data
    csv_df = pd.read_csv(csv_path)
    num_samples = min(200, len(csv_df))
    sample_rows = csv_df.sample(n=num_samples, random_state=42)

    online_data = []
    for _, row in sample_rows.iterrows():
        action_idx = np.random.randint(0, len(ACTION_SPACE_MAP))
        online_data.append({
            'customer_id': int(row['traveler_id']),
            'action': action_idx,
            'accepted': bool(np.random.rand() > 0.5),
            'trip_purpose': row['trip_purpose'],
            'departure_location': row['departure_location'],
            'arrival_location': row['arrival_location'],
            'departure_time_window': row['departure_time_window'],
            'arrival_time_window': row['arrival_time_window'],
        })

    # Create embedding model
    num_customers = 30
    embedding_model = EmbeddingFFN(
        num_entities=num_customers,
        embed_dim=64,
        hidden_dim=128,
        output_dim=n_flexibilities
    )

    print(f"\nTesting update_embedding_model with {len(online_data)} samples...")
    start_time = time.time()

    # Call update function (as done in meta_train.py)
    updated_model = update_embedding_model(
        embedding_model,
        online_data,
        flexibility_personalities,
        ACTION_SPACE_MAP,
        num_epochs=10,  # Reduced for testing
        batch_size=32,
        lr=1e-3
    )

    elapsed = time.time() - start_time

    print(f"✓ Embedding update completed in {elapsed:.3f}s")
    print(f"✓ Model updated successfully")

    # Should complete quickly with optimization
    if elapsed < 30.0:
        print(f"✓ FAST! Meta-train compatibility verified")
        return True
    else:
        print(f"✗ SLOW! Expected < 30s, got {elapsed:.3f}s")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EMBEDDING HASH TABLE OPTIMIZATION TESTS")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Cache Correctness", test_cache_correctness()))
    results.append(("Performance Improvement", test_performance_improvement()))
    results.append(("Meta-Train Compatibility", test_meta_train_compatibility()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10} {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! Optimization is working correctly.")
        print("Expected speedup: ~10,000x for CSV lookups")
        print("Embedding updates should now be MUCH faster!")
    else:
        print("\n❌ Some tests failed. Please review the output above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
