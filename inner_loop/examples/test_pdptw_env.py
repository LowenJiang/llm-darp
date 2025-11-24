"""Test script for PDPTWEnv initialization"""

import traceback
from rl4co.envs.routing.pdptw.env import PDPTWEnv

try:
    print("Creating PDPTWEnv...")
    env = PDPTWEnv()
    print("✓ Environment created successfully")

    print("\nResetting environment with batch_size=[1]...")
    td = env.reset(batch_size=[1])
    print("✓ Reset successful!")

    print("\n=== TensorDict Contents ===")
    print(f"Batch size: {td.batch_size}")
    print(f"\nKeys: {list(td.keys())}")

    print("\n=== Shapes ===")
    for key in sorted(td.keys()):
        print(f"{key:20s}: {td[key].shape}")

    print("\n=== Action Mask ===")
    action_mask = td["action_mask"]
    print(f"Shape: {action_mask.shape}")
    print(f"Valid actions: {action_mask.sum().item()} out of {action_mask.numel()}")
    print(f"First few mask values: {action_mask[0, :10].tolist()}")

    print("\n" + "="*50)
    print("SUCCESS! Environment is working correctly.")
    print("="*50)

except Exception as e:
    print("\n" + "="*50)
    print("ERROR OCCURRED:")
    print("="*50)
    print(f"\nError type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nFull traceback:")
    traceback.print_exc()
