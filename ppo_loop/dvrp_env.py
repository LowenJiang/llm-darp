"""
Dynamic Vehicle Routing Problem with Time Windows (DVRP-TW) Environment

MDP formulation:
- State: Current accepted requests + new incoming request (padded to max 30)
- Action: 16 discrete actions for time window perturbation
- Reward: old_routing_cost - new_routing_cost - patience_penalty
- Episode: 30 sequential request arrivals
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path

# Add parent directory to sys.path to import rl4co
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from rl4co.envs.routing import SFGenerator, PDPTWEnv
from ortools_solver import darp_solver
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from tensordict.tensordict import TensorDict


class DVRPEnv(gym.Env):
    """
    Gym environment for Dynamic Vehicle Routing Problem with Time Windows.

    State Space:
        - Shape: (30, 6) - 30 requests max, 6 features per request
        - Features: [pickup_h3, pickup_tw_early, pickup_tw_late,
                    dropoff_h3, dropoff_tw_early, dropoff_tw_late]
        - Padding: zero vectors for empty slots

    Action Space:
        - Discrete(16): 4 pickup shifts × 4 dropoff shifts
        - Pickup shifts: {-30, -20, -10, 0} minutes (earlier)
        - Dropoff shifts: {0, +10, +20, +30} minutes (later)

    Reward:
        - reward = old_cost - new_cost - patience_penalty
        - patience_penalty = |pickup_shift| + |dropoff_shift| (if accepted)
        - If OR-Tools fails: reward = -5000, episode terminates
    """

    # Action space mapping: (pickup_shift, dropoff_shift)
    ACTION_SPACE_MAP = [
        (-30, 0),
        (-30, 10),
        (-30, 20),
        (-30, 30),
        (-20, 0),
        (-20, 10),
        (-20, 20),
        (-20, 30),
        (-10, 0),
        (-10, 10),
        (-10, 20),
        (-10, 30),
        (0, 0),
        (0, 10),
        (0, 20),
        (0, 30),
    ]

    # Fields that need to be sliced (node-indexed fields)
    SLICE_FIELDS = {"h3_indices", "time_windows", "demand", "locs", "action_mask", "visited"}

    def __init__(
        self,
        num_customers: int = 30,
        max_vehicles: int = 5,
        solver_time_limit: int = 1,
        acceptance_rate: float = 0.5,
        depot: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        patience_factor : int=0.2,
        model_path : Optional[str] = '/home/jiangwolin/rl4co git/examples/checkpoints/sf_newenv_2/epoch_epoch=067.ckpt',
        traveler_decisions_path: Optional[str] = None,
        device: str = 'cpu'
    ):  
        """
        Args:
            num_customers: Number of requests per episode (default: 30)
            vehicle_capacity: Vehicle capacity (default: 4)
            max_vehicles: Maximum vehicles for OR-Tools (default: 5)
            solver_time_limit: OR-Tools time limit in seconds (default: 1)
            acceptance_rate: Probability of customer accepting perturbation (default: 0.5)
            depot: Depot GPS coordinates (default: SF downtown)
            seed: Random seed
        """
        super().__init__()
        self.patience_factor = patience_factor
        self.num_customers = num_customers
        self.max_vehicles = max_vehicles
        self.solver_time_limit = solver_time_limit
        self.acceptance_rate = acceptance_rate
        self.seed_val = seed
        self.device = device

        # Initialize data generator
        self.data_generator = PDPTWEnv(generator_params={
            "num_customers": num_customers
        })

        # Define observation and action spaces
        # State: (30 requests, 6 features per request)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_customers, 6),
            dtype=np.float32,
        )

        # Action: 16 discrete actions
        self.action_space = spaces.Discrete(16)

        # Episode state
        self.current_requests: Optional[TensorDict] = None
        self.pending_requests: Optional[TensorDict] = None
        self.current_step = 0
        self.previous_cost = 0.0

        # Only load policy if model_path is provided
        self.model_path = model_path
        if model_path is not None:
            env = PDPTWEnv()
            torch.serialization.add_safe_globals([PDPTWEnv])
            # Always load checkpoint to CPU first to avoid RNG state issues
            ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
            state = ckpt["state_dict"]

            model = AttentionModel(env=PDPTWEnv)
            model.load_state_dict(state, strict=False)
            # Then move the model to the desired device
            self.policy = model.policy.to(self.device)
            self.policy.eval()  # Set to evaluation mode
        else:
            self.policy = None

        # Load traveler decisions CSV for acceptance lookup
        self.traveler_decisions_path = traveler_decisions_path
        if traveler_decisions_path is not None:
            self.traveler_decisions_df = pd.read_csv(traveler_decisions_path)
        else:
            self.traveler_decisions_df = None
        self.df_mask = self._compute_mask_all_flexs()
        self.current_request_masks = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment and start a new episode.

        Returns:
            observation: Initial state (first request + padding)
            info: Additional information
        """
        super().reset(seed=seed)


        # Generate new episode of requests using correct syntax
        # pending_requests is a tensordict with batch_size=[1]
        self.pending_requests = self.data_generator.reset(batch_size=[1])

        # Initialize current_requests with depot only (index 0)
        self.current_requests = self._init_depot_only(self.pending_requests)

        self.current_step = 0
        self.previous_cost = 0.0

        # Initial state: only first request (no previous requests)
        initial_state = self._get_observation()

        info = {
            "episode": 0,
            "total_requests": self.num_customers,
            "current_cost": 0.0,
        }

        return initial_state, info

    def _init_depot_only(self, td: TensorDict) -> TensorDict:
        """
        Initialize current_requests with only the depot (index 0).

        This creates a TensorDict containing only the depot information
        as the starting point for building up the route.
        All non-sliced fields are preserved for compatibility.
        """
        batch_size = td.batch_size

        # Start with a clone to preserve all fields
        depot_td = td.clone()

        # Slice node-indexed fields to only include depot (index 0)
        for key in td.keys():
            if key in self.SLICE_FIELDS and key in td.keys():
                tensor = td[key]
                if key == "time_windows" or key == "locs":
                    # Shape: [batch, num_nodes, 2]
                    depot_td[key] = tensor[:, 0:1, :]
                else:
                    # Shape: [batch, num_nodes]
                    depot_td[key] = tensor[:, 0:1]

        return depot_td

    def _slice(self, td: TensorDict, current_step: int) -> TensorDict:
        """
        Extract the pickup and dropoff nodes for a given step from the TensorDict.

        For current_step = i, this function extracts indices 2i+1 (pickup) and 2i+2 (dropoff).

        Input TensorDict has nodes ordered as: [Depot, P1, D1, P2, D2, ...]
        So for step i (0-indexed), pickup is at 2*i+1 and dropoff is at 2*i+2.

        All non-sliced fields are preserved for compatibility.

        Returns:
            TensorDict containing only the pickup and dropoff for this step,
            plus all other non-node-indexed fields preserved.
        """
        # Calculate indices for pickup and dropoff
        pickup_idx = 2 * current_step + 1
        dropoff_idx = 2 * current_step + 2
        indices = [pickup_idx, dropoff_idx]

        # Start with a clone to preserve all fields
        sliced_td = td.clone()

        # Slice node-indexed fields
        for key in td.keys():
            if key in self.SLICE_FIELDS and key in td.keys():
                tensor = td[key]
                if key == "time_windows" or key == "locs":
                    # Shape: [batch, num_nodes, 2]
                    sliced_td[key] = tensor[:, indices, :]
                else:
                    # Shape: [batch, num_nodes]
                    sliced_td[key] = tensor[:, indices]

        return sliced_td
    
    def _get_meta_data_single_traveler(self, traveler_id):
        # Use direct attribute access to avoid TensorDict indexing issues
        trip_metadata_list = self.pending_requests["trip_metadata"]
        # trip_metadata_list is a list of dicts, get the first one (batch index 0)
        if isinstance(trip_metadata_list, list) and len(trip_metadata_list) > 0:
            trip_metadata = trip_metadata_list[0]
        else:
            trip_metadata = trip_metadata_list
        metadata = trip_metadata[traveler_id]
        return metadata
    
    def get_mask(self, traveler_id, predicted_flex_index):
        metadata = self._get_meta_data_single_traveler(traveler_id)
        trip_purpose = metadata["trip_purpose"]
        departure_location = metadata["departure_location"]
        arrival_location = metadata["arrival_location"]
        departure_time_window = metadata["departure_time_window"]
        arrival_time_window = metadata["arrival_time_window"]
        mask = self._get_mask_from_flex(traveler_id, trip_purpose, departure_location, arrival_location, departure_time_window, arrival_time_window, predicted_flex_index)


        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Args:
            action: Integer in [0, 15] representing perturbation choice

        Returns:
            observation: Next state
            reward: Reward for this step
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Get current incoming request
        new_request = self._slice(self.pending_requests, self.current_step)

        # Decode action to perturbation
        pickup_shift, dropoff_shift = self.ACTION_SPACE_MAP[action]

        # Get traveler_id for the current request
        pickup_idx = 2 * self.current_step + 1
        traveler_id = self.pending_requests["user_id"][0, pickup_idx].item()

        # Get trip metadata from pending_requests
        # trip_metadata is stored as a list (not a tensor), so access it differently
        if "trip_metadata" in self.pending_requests.keys():
            # Use direct attribute access to avoid TensorDict indexing issues
            trip_metadata_list = self.pending_requests["trip_metadata"]
            # trip_metadata_list is a list of dicts, get the first one (batch index 0)
            if isinstance(trip_metadata_list, list) and len(trip_metadata_list) > 0:
                trip_metadata = trip_metadata_list[0]
            else:
                trip_metadata = trip_metadata_list

            if traveler_id in trip_metadata:
                metadata = trip_metadata[traveler_id]
                flexibility = metadata["flexibility"]
                trip_purpose = metadata["trip_purpose"]
                departure_location = metadata["departure_location"]
                arrival_location = metadata["arrival_location"]

                # Look up acceptance decision
                accepted = self._get_acceptance_decision(
                    traveler_id=traveler_id,
                    flexibility=flexibility,
                    trip_purpose=trip_purpose,
                    departure_location=departure_location,
                    arrival_location=arrival_location,
                    pickup_shift=pickup_shift,
                    dropoff_shift=dropoff_shift
                )
            else:
                # Fallback to random acceptance if traveler_id not in metadata
                accepted = np.random.random() < self.acceptance_rate
        else:
            # Fallback to random acceptance if no trip metadata available
            accepted = np.random.random() < self.acceptance_rate

        if accepted:
            # Apply perturbation to time windows
            perturbed_request = self._apply_perturbation(
                new_request, pickup_shift, dropoff_shift
            )
            # Calculate patience penalty
            patience_penalty = (abs(pickup_shift) + abs(dropoff_shift)) * self.patience_factor
        else:
            # Use original time windows (no perturbation)
            perturbed_request = new_request.clone()
            patience_penalty = 0.0

        # Add perturbed request to current requests
        self.current_requests = self._append_to_current(
            self.current_requests, perturbed_request
        )

        # Solve routing problem
        if self.policy is None:
            # Use OR-Tools darp_solver
            result = darp_solver(
                self.current_requests,
                max_vehicles=self.max_vehicles,
                time_limit_seconds=self.solver_time_limit
            )
            new_cost = result["total_time"]
        else:
            # Use trained policy model
            with torch.no_grad():
                # Move current_requests to the same device as the policy
                current_requests_on_device = self.current_requests.to(self.device)
                out = self.policy(current_requests_on_device, phase='test', decode_type="greedy", return_actions=True)
                # Extract cost from policy output (reward is negative cost)
                new_cost = -out["reward"].item()

        

        # DEBUG: Track node counts
        num_nodes = self.current_requests["h3_indices"].shape[1]
        expected_nodes = 1 + 2 * (self.current_step + 1)  # depot + 2*(step+1)
        if num_nodes != expected_nodes:
            print(f"WARNING: num_nodes={num_nodes}, expected={expected_nodes} at step {self.current_step}")

        # Check for solver failure
        if new_cost == float('inf'):
            # Solver failed, terminate episode with penalty
            reward = -5000.0
            terminated = True
            truncated = False
            observation = self._get_observation()
            info = {
                "step": self.current_step,
                "current_cost": new_cost,
                "accepted": accepted,
                "patience_penalty": patience_penalty,
                "num_requests": (self.current_requests["h3_indices"].shape[1] - 1) // 2,
                "solver_failed": True,
            }
            return observation, reward, terminated, truncated, info

        # Calculate reward: old_cost - new_cost - patience_penalty
        reward = self.previous_cost - new_cost - patience_penalty

        # Update previous cost
        self.previous_cost = new_cost

        # Move to next step
        self.current_step += 1

        # Check if episode is done (all requests processed)
        terminated = self.current_step >= self.num_customers
        truncated = False

        # Get next observation
        observation = self._get_observation()

        # Get user_id for current request if available
        user_id = None
        if "user_id" in self.pending_requests.keys():
            # user_id is indexed by node, get pickup node's user_id
            pickup_idx = 2 * (self.current_step - 1) + 1
            user_id = self.pending_requests["user_id"][0, pickup_idx].item()

        # Info to register to PPO agent
        info = {
            "step": self.current_step,
            "current_cost": new_cost,
            "accepted": accepted,
            "patience_penalty": patience_penalty,
            "num_requests": (self.current_requests["h3_indices"].shape[1] - 1) // 2,
            "user_id": user_id,
        }

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Build observation state.

        State format: (num_customers, 6) where each row contains:
        [pickup_h3, pickup_tw_early, pickup_tw_late, dropoff_h3, dropoff_tw_early, dropoff_tw_late]

        Fixed requests fill from top (row 0) downward, new request is always at bottom row,
        and zeros for padding in between.
        """
        # Initialize observation with zeros
        obs = np.zeros((self.num_customers, 6), dtype=np.float32)

        # Get current number of accepted requests (excluding depot)
        num_nodes = self.current_requests["h3_indices"].shape[1]
        num_fixed_requests = (num_nodes - 1) // 2

        # Fill in fixed requests from top to bottom (rows 0, 1, 2, ...)
        h3_indices = self.current_requests["h3_indices"][0].cpu().numpy()  # [num_nodes]
        time_windows = self.current_requests["time_windows"][0].cpu().numpy()  # [num_nodes, 2]

        for i in range(num_fixed_requests):
            pickup_idx = 2 * i + 1
            dropoff_idx = 2 * i + 2

            if pickup_idx < num_nodes and dropoff_idx < num_nodes:
                obs[i] = [
                    h3_indices[pickup_idx],
                    time_windows[pickup_idx, 0],
                    time_windows[pickup_idx, 1],
                    h3_indices[dropoff_idx],
                    time_windows[dropoff_idx, 0],
                    time_windows[dropoff_idx, 1],
                ]

        # Add new request at bottom row (if not done)
        if self.current_step < self.num_customers:
            new_request = self._slice(self.pending_requests, self.current_step)
            new_h3 = new_request["h3_indices"][0].cpu().numpy()  # [2]
            new_tw = new_request["time_windows"][0].cpu().numpy()  # [2, 2]

            # Place new request at the bottom row (last index)
            obs[self.num_customers - 1] = [
                new_h3[0], new_tw[0, 0], new_tw[0, 1],  # Pickup
                new_h3[1], new_tw[1, 0], new_tw[1, 1],  # Dropoff
            ]

        return obs

    def _append_to_current(self, current_requests: TensorDict, perturbed_request: TensorDict) -> TensorDict:
        """
        Combine current_requests with a new perturbed_request.

        Concatenates node-indexed fields along the node dimension.
        All other fields are preserved from current_requests.

        Returns:
            TensorDict with appended request.
        """
        batch_size = current_requests.batch_size
        device = current_requests.device

        # Start with a clone of current_requests to preserve all non-sliced fields
        appended_td = current_requests.clone()

        # Concatenate node-indexed fields
        for key in self.SLICE_FIELDS:
            if key in current_requests.keys() and key in perturbed_request.keys():
                if key == "time_windows" or key == "locs":
                    # Shape: [batch, num_nodes, 2]
                    appended_td[key] = torch.cat([
                        current_requests[key],
                        perturbed_request[key]
                    ], dim=1)
                else:
                    # Shape: [batch, num_nodes]
                    appended_td[key] = torch.cat([
                        current_requests[key],
                        perturbed_request[key]
                    ], dim=1)

        return appended_td

    def _apply_perturbation(
        self, new_request: TensorDict, pickup_shift: int, dropoff_shift: int
    ) -> TensorDict:
        """
        Apply time window perturbation to a request.

        Args:
            new_request: TensorDict containing the request
            pickup_shift: Minutes to shift pickup (negative = earlier)
            dropoff_shift: Minutes to shift dropoff (positive = later)

        Returns:
            Perturbed TensorDict

        Perturbs time_windows:
        - Pickup (index 0): shifts both early and late by pickup_shift
        - Dropoff (index 1): shifts both early and late by dropoff_shift
        """
        perturbed = new_request.clone()

        # Get time windows [batch, 2, 2]
        tw = perturbed["time_windows"].clone()

        # Apply pickup shift (index 0)
        tw[:, 0, 0] = tw[:, 0, 0] + pickup_shift  # Early time
        tw[:, 0, 1] = tw[:, 0, 1] + pickup_shift  # Late time

        # Apply dropoff shift (index 1)
        tw[:, 1, 0] = tw[:, 1, 0] + dropoff_shift  # Early time
        tw[:, 1, 1] = tw[:, 1, 1] + dropoff_shift  # Late time

        # Ensure non-negative time windows
        tw = torch.clamp(tw, min=0)

        perturbed["time_windows"] = tw

        return perturbed
    
    def _compute_mask_single_flex(self, index_cols, action_cols, flexibility):
            # 1. Prepare data
            df = self.traveler_decisions_df[index_cols + action_cols + [flexibility]].copy()
            df["indicator"] = (df[flexibility] == "accept").astype(int)

            # 2. Pivot to wide form (MultiIndex columns: pickup_shift_min, dropoff_shift_min)
            # fill_value=0 means if a specific shift combo is missing from CSV, we assume REJECT
            wide = df.pivot_table(
                index=index_cols,
                columns=action_cols,
                values="indicator",
                fill_value=0
            )

            # 3. Create the strict ordering required by ACTION_SPACE_MAP
            # The CSV uses absolute values, so we convert action map to match CSV headers
            strict_ordering = []
            for pickup_shift, dropoff_shift in self.ACTION_SPACE_MAP:
                # Matches CSV columns: pickup_shift_min, dropoff_shift_min
                strict_ordering.append((abs(pickup_shift), abs(dropoff_shift)))
            
            # 4. Reindex the columns to enforce the specific Action Space order
            # This fixes the sorting bug. 
            # Note: We must ensure the pivot table columns are integers to match the tuple list
            wide = wide.reindex(columns=strict_ordering, fill_value=0)

            # 5. Collapse into list
            # Now wide.iloc[row, 0] corresponds exactly to ACTION_SPACE_MAP[0]
            wide[flexibility] = wide.apply(lambda row: row.values.tolist(), axis=1)

            # 6. Clean up
            result = wide[[flexibility]].reset_index()
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = result.columns.get_level_values(0)
            result.columns.name = None
            
            return result
    
    def _compute_mask_all_flexs(self):
        index_cols = ["traveler_id", "trip_purpose", "departure_location", "arrival_location", "departure_time_window", "arrival_time_window"]
        action_cols = ["pickup_shift_min", "dropoff_shift_min"]
        df_mask = None
        # IMPORTANT: This order MUST match embedding.py:21-26 for correct mask lookup!
        for i, flexibility in enumerate(["flexible for late dropoff, but inflexible for early pickup", "flexible for early pickup, but inflexible for late dropoff", "inflexible for any schedule changes", "flexible for both early pickup and late dropoff"]):
            result = self._compute_mask_single_flex(index_cols, action_cols, flexibility)
            result = result.rename(columns = {flexibility: i})
            if df_mask is None:
                df_mask = result
            else:
                df_mask = df_mask.merge(result, on = index_cols)
        return df_mask
    
    def _get_mask_from_flex(self, traveler_id, trip_purpose, departure_location, arrival_location, departure_time_window, arrival_time_window, predicted_flex_index):
        # Convert tensor inputs to Python scalars if needed
        if isinstance(predicted_flex_index, torch.Tensor):
            predicted_flex_index = int(predicted_flex_index.item())

        selection = (
            (self.df_mask["traveler_id"] == traveler_id) &
            (self.df_mask["trip_purpose"] == trip_purpose) &
            (self.df_mask["departure_location"] == departure_location) &
            (self.df_mask["arrival_location"] == arrival_location) & 
            (self.df_mask["departure_time_window"] == departure_time_window) & 
            (self.df_mask["arrival_time_window"] == arrival_time_window)
        )
        mask_series = self.df_mask[selection][predicted_flex_index]
        # Extract the actual value from the Series (should be a list)
        mask = mask_series.iloc[0] if len(mask_series) > 0 else [1]*16

        return mask

    def _get_acceptance_decision(
        self,
        traveler_id: int,
        flexibility: str,
        trip_purpose: str,
        departure_location: str,
        arrival_location: str,
        pickup_shift: int,
        dropoff_shift: int
    ) -> bool:


        """
        Look up the acceptance decision from traveler_decisions_augmented.csv.

        Args:
            traveler_id: ID of the traveler
            flexibility: Flexibility type string (e.g., "flexible for both early pickup and late dropoff")
            trip_purpose: Purpose of the trip
            departure_location: Departure location name
            arrival_location: Arrival location name
            pickup_shift: Pickup shift in minutes (negative = earlier)
            dropoff_shift: Dropoff shift in minutes (positive = later)

        Returns:
            True if accepted, False if rejected
        """
        if self.traveler_decisions_df is None:
            # Fallback to random acceptance if CSV not provided
            return np.random.random() < self.acceptance_rate

        # Convert shifts to absolute values for matching
        pickup_shift_abs = abs(pickup_shift)
        dropoff_shift_abs = abs(dropoff_shift)

        # Find matching row
        mask = (
            (self.traveler_decisions_df["traveler_id"] == traveler_id) &
            (self.traveler_decisions_df["trip_purpose"] == trip_purpose) &
            (self.traveler_decisions_df["departure_location"] == departure_location) &
            (self.traveler_decisions_df["arrival_location"] == arrival_location) &
            (self.traveler_decisions_df["pickup_shift_min"] == pickup_shift_abs) &
            (self.traveler_decisions_df["dropoff_shift_min"] == dropoff_shift_abs)           
        )

        matching_rows = self.traveler_decisions_df[mask]


        if len(matching_rows) == 0:
            print(f"WARNING: No matching decision found for traveler {traveler_id}, "
                    f"flexibility='{flexibility}', trip_purpose='{trip_purpose}', "
                  f"departure='{departure_location}', arrival='{arrival_location}', "
                  f"pickup_shift={pickup_shift}, dropoff_shift={dropoff_shift}")

            return np.random.random() < self.acceptance_rate

        # Get the first matching row
        row = matching_rows.iloc[0]

        ## TODO: Note that this should never happen
        # The acceptance decision is stored in a column named after the flexibility type
        # Column name should match the flexibility string
#        if flexibility not in row.index:
#            print(f"WARNING: Flexibility column '{flexibility}' not found in CSV")
#            return np.random.random() < self.acceptance_rate

        decision = row[flexibility]
        return decision == "accept"

    def get_current_user_id(self) -> int:
        """Get the user_id of the current incoming request."""
        if "user_id" in self.pending_requests.keys():
            pickup_idx = 2 * self.current_step + 1
            return self.pending_requests["user_id"][0, pickup_idx].item()
        return self.current_step  # Fallback to step index

    def render(self, mode: str = "human") -> None:
        """Render environment (not implemented)."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass


def test_env():
    """Test the DVRP environment."""
    print("Testing DVRPEnv...")
    print("=" * 50)

    # Create environment with small number of customers for testing
    decisions_path = Path(__file__).parent / "traveler_decisions_augmented.csv"
    env = DVRPEnv(num_customers=5, seed=42, traveler_decisions_path=decisions_path)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    print("\n--- Resetting Environment ---")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    print(f"First few observation rows:\n{obs[:]}")

    # Run a few steps
    print("\n=== Running 5 steps ===")
    total_reward = 0.0

    for step in range(5):
        # Random action
        action = env.action_space.sample()
        pickup_shift, dropoff_shift = env.ACTION_SPACE_MAP[action]

        obs, reward, terminated, truncated, info = env.step(action)
        print(f"First few observation rows:\n{obs[:]}")
        print(f"\nStep {step + 1}:")
        print(f"  Action: ({pickup_shift:3d}, {dropoff_shift:3d})")
        print(f"  Accepted: {info['accepted']}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Cost: {info.get('current_cost', 'N/A')}")
        print(f"  Num requests: {info.get('num_requests', 'N/A')}")
        print(f"  Terminated: {terminated}")
        print("- -" * 25)
        if 'solver_failed' in info:
            print(f"  Solver failed!")

        total_reward += reward

        if terminated or truncated:
            break

    print(f"\n{'=' * 50}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final observation shape: {obs.shape}")

    # Test multiple resets
    print("\n--- Testing Multiple Resets ---")
    for i in range(3):
        obs, info = env.reset(seed=i)
        print(f"Reset {i+1}: obs shape = {obs.shape}, info = {info}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_env()
