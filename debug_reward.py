"""Debug script to compare routes between implementations."""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator

def debug_single_instance():
    """Debug a single instance to see the difference."""

    device = torch.device("cpu")
    batch_size = 1  # Just one instance

    csv_path = Path("src/traveler_trip_types_res_7.csv")
    ttm_path = Path("src/travel_time_matrix_res_7.csv")

    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path,
        device=device,
        num_customers=5  # Small for easy debugging
    )

    env = PDPTWEnv(generator=generator)

    # Generate and run
    state = env.reset(batch_size=[batch_size])

    actions_taken = []
    for step in range(50):
        if state["done"].all():
            break

        mask = state["action_mask"]
        valid_counts = mask.sum(dim=-1)
        if (valid_counts == 0).any():
            action = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            probs = mask.float() / valid_counts.unsqueeze(-1)
            action = torch.multinomial(probs, 1).squeeze(-1)

        actions_taken.append(action)
        state["action"] = action
        state = env.step(state)["next"]

    actions_tensor = torch.stack(actions_taken, dim=1)

    print("=" * 60)
    print("Action sequence:", actions_tensor[0].tolist())
    print("=" * 60)

    # Get effective routes from legacy
    effective_routes = env._build_effective_routes(actions_tensor)
    print("\nLegacy effective route:", effective_routes[0])

    # Get valid mask from vectorized
    valid_mask = env._build_valid_action_mask(actions_tensor)
    print("\nVectorized valid mask:", valid_mask[0].tolist())
    print("Vectorized valid actions:", actions_tensor[0][valid_mask[0]].tolist())

    # Compute rewards
    reward_vec = env.get_reward(state, actions_tensor, use_vectorized=True)
    reward_legacy = env.get_reward(state, actions_tensor, use_vectorized=False)

    print("\nVectorized reward:", reward_vec[0].item())
    print("Legacy reward:", reward_legacy[0].item())
    print("Difference:", (reward_vec[0] - reward_legacy[0]).abs().item())

if __name__ == "__main__":
    debug_single_instance()
