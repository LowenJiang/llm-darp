#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
Run this before running meta_train.py to catch import issues early.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("Testing RL4CO Import Setup")
print("=" * 80)

# Test 1: Basic imports
print("\n[Test 1] Testing basic Python imports...")
try:
    import numpy as np
    import pandas as pd
    import torch
    print("  ✓ numpy, pandas, torch imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import basic packages: {e}")
    print("  Please install required packages first")
    sys.exit(1)

# Test 2: RL4CO imports
print("\n[Test 2] Testing rl4co imports...")
try:
    from rl4co.envs.routing import SFGenerator, PDPTWEnv
    from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
    print("  ✓ rl4co.envs.routing imported successfully")
    print("  ✓ rl4co.models.zoo imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import rl4co: {e}")
    print("  Make sure PYTHONPATH includes the parent directory")
    sys.exit(1)

# Test 3: Local module imports
print("\n[Test 3] Testing local module imports...")
try:
    from dvrp_env import DVRPEnv
    from vectorized_env import VectorizedDVRPEnv
    from ppo_agent import PPOAgent
    from embedding import EmbeddingFFN
    print("  ✓ dvrp_env imported successfully")
    print("  ✓ vectorized_env imported successfully")
    print("  ✓ ppo_agent imported successfully")
    print("  ✓ embedding imported successfully")
except ImportError as e:
    print(f"  ✗ Failed to import local modules: {e}")
    sys.exit(1)

# Test 4: CUDA availability
print("\n[Test 4] Checking CUDA availability...")
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"  ✓ CUDA is available")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Current device: {torch.cuda.current_device()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
else:
    print("  ⚠ CUDA is NOT available (CPU only)")

# Test 5: Device parameter propagation
print("\n[Test 5] Testing device parameter in DVRPEnv...")
try:
    device = 'cuda' if cuda_available else 'cpu'
    print(f"  Creating DVRPEnv with device='{device}'...")

    # Don't actually create the env to avoid loading the large model
    # Just check that the parameter is accepted
    from inspect import signature
    sig = signature(DVRPEnv.__init__)
    params = list(sig.parameters.keys())

    if 'device' in params:
        print("  ✓ DVRPEnv accepts 'device' parameter")
    else:
        print("  ✗ DVRPEnv does NOT accept 'device' parameter")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ Error testing DVRPEnv: {e}")
    sys.exit(1)

# Test 6: Check required data files
print("\n[Test 6] Checking required data files...")
data_file = Path(__file__).parent / "traveler_decisions_augmented.csv"
if data_file.exists():
    print(f"  ✓ Found {data_file.name}")
else:
    print(f"  ✗ Missing {data_file.name}")
    print(f"  Expected location: {data_file}")

model_path = Path("/home/jiangwolin/rl4co git/examples/checkpoints/sf_newenv_2/epoch_epoch=067.ckpt")
if model_path.exists():
    print(f"  ✓ Found oracle model checkpoint")
else:
    print(f"  ⚠ Oracle model checkpoint not found at {model_path}")
    print(f"  The environment will still work but without the neural oracle")

# Summary
print("\n" + "=" * 80)
print("All import tests passed! ✓")
print("=" * 80)
print("\nYou can now run:")
print("  python3 meta_train.py --episodes 1000 --device cuda")
print("\nOr use the run script:")
print("  ./run_meta_train.sh --episodes 1000 --device cuda")
print("=" * 80)
