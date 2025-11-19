"""Utility functions for DARP environment."""

import torch


def calculate_successful_requests(actions: torch.Tensor) -> torch.Tensor:
    """Calculate the number of successfully completed pickup-dropoff pairs.

    A request is considered successful if both its pickup (odd index) and
    corresponding dropoff (even index) are visited by the same vehicle in
    the correct order (pickup before dropoff).

    Args:
        actions: Tensor of shape [batch_size, max_steps] containing the action sequences.
                Actions are node indices where 0 is depot, odd indices are pickups,
                and even indices are dropoffs.

    Returns:
        Tensor of shape [batch_size] containing the count of successful requests per instance.

    Example:
        >>> actions = torch.tensor([[11, 25, 5, 12, 21, 22, 0, 3, 4, 0]])
        >>> calculate_successful_requests(actions)
        tensor([2])  # Pairs (5,6) and (21,22) are both visited in same tour
    """
    batch_size = actions.shape[0]
    successful_counts = torch.zeros(batch_size, dtype=torch.long, device=actions.device)

    for b in range(batch_size):
        action_seq = actions[b]

        # Split into vehicle tours (separated by depot returns, index 0)
        tours = []
        current_tour = []

        for action in action_seq:
            if action == 0:
                if current_tour:  # Don't add empty tours
                    tours.append(set(current_tour))
                    current_tour = []
            else:
                current_tour.append(action.item())

        # Add final tour if exists
        if current_tour:
            tours.append(set(current_tour))

        # Count valid pickup-dropoff pairs
        valid_pairs = set()

        for tour in tours:
            # Check each potential pickup node (odd indices)
            for node in tour:
                if node % 2 == 1:  # This is a pickup node
                    pickup_idx = node
                    dropoff_idx = node + 1

                    # Check if corresponding dropoff is also in this tour
                    if dropoff_idx in tour:
                        # The pair ID is node // 2 (since pickup=2i-1, dropoff=2i)
                        pair_id = node // 2
                        valid_pairs.add(pair_id)

        successful_counts[b] = len(valid_pairs)

    return successful_counts
