"""Test PDPTWEnv with a full step"""

import torch
from rl4co.envs.routing.pdptw.env import PDPTWEnv

try:
    print("Creating PDPTWEnv...")
    env = PDPTWEnv()

    print("Resetting environment...")
    td = env.reset(batch_size=[2])  # Test with batch_size=2
    print(f"✓ Reset successful! Batch size: {td.batch_size}")

    print("\nAction mask shape:", td["action_mask"].shape)
    print("Action mask for first batch:", td["action_mask"][0].shape)

    # Get valid actions
    action_mask = td["action_mask"]
    print(f"\nAction mask dimensions: {action_mask.dim()}")

    # Take a greedy action (first valid action)
    if action_mask.dim() == 3:
        # If mask is [B, N, N], we might need to handle it differently
        print("Warning: Action mask has 3 dimensions, using first row")
        valid_actions_batch0 = action_mask[0, 0]  # First batch, first row
    else:
        # If mask is [B, N], use it directly
        valid_actions_batch0 = action_mask[0]

    valid_indices = torch.where(valid_actions_batch0)[0]
    print(f"Number of valid actions for batch 0: {len(valid_indices)}")
    print(f"First few valid actions: {valid_indices[:5].tolist()}")

    # Select first valid action for each batch
    if action_mask.dim() == 3:
        actions = torch.tensor([torch.where(action_mask[i, 0])[0][0].item() for i in range(td.batch_size[0])])
    else:
        actions = torch.tensor([torch.where(action_mask[i])[0][0].item() for i in range(td.batch_size[0])])

    print(f"\nSelected actions: {actions}")

    # Take a step
    print("\nTaking a step...")
    td["action"] = actions
    td_next = env.step(td)["next"]

    print("✓ Step successful!")
    print(f"Current node after step: {td_next['current_node']}")
    print(f"Done: {td_next['done']}")

    print("\n" + "="*50)
    print("SUCCESS! Environment works end-to-end!")
    print("="*50)

except Exception as e:
    print("\n" + "="*50)
    print("ERROR:")
    print("="*50)
    import traceback
    traceback.print_exc()
