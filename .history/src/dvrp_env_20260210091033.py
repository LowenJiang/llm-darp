"""
Dynamic Vehicle Routing Problem with Time Windows (DVRP-TW) Environment
(Batched Version)

MDP formulation:
- State: Batched TensorDict with keys:
  - fixed: [batch_size, num_customers, 6]
  - new: [batch_size, 1, 6]
  - distance_matrix: [batch_size, num_locations, num_locations]
  - step: [batch_size, 1]
- Action: 16 discrete actions for time window perturbation
- Reward: old_routing_cost - new_routing_cost - patience_penalty
- Episode: 30 sequential request arrivals
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import sys
from pathlib import Path

# Add parent directory to sys.path to import rl4co
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces
from oracle_generator import SFGenerator
from oracle_env import PDPTWEnv
from or_tools import darp_solver
from oracle_policy import PDPTWAttentionPolicy
from tensordict.tensordict import TensorDict


class DVRPEnv(gym.Env):
    """
    Gym environment for Dynamic Vehicle Routing Problem with Time Windows.
    Supports batched execution.

    State Space (Batched TensorDict):
        - fixed: [batch_size, num_customers, 6]
        - new: [batch_size, 1, 6]
        - distance_matrix: [batch_size, num_locations, num_locations]
        - step: [batch_size, 1]

    Action Space:
        - Discrete(16): 4 pickup shifts × 4 dropoff shifts

    Reward:
        - reward = old_cost - new_cost - patience_penalty
        - patience_penalty = |pickup_shift| + |dropoff_shift| (if accepted)
        - If OR-Tools fails: reward = -5000
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

    FLEXIBILITY_COLUMNS = [
        "flexible for both early pickup and late dropoff",
        "flexible for early pickup, but inflexible for late dropoff",
        "flexible for late dropoff, but inflexible for early pickup",
        "inflexible for any schedule changes",
    ]

    # Fields that need to be sliced (node-indexed fields)
    SLICE_FIELDS = {
        "h3_indices",
        "time_windows",
        "demand",
        "locs",
        "action_mask",
        "visited",
        "completed",
        "user_id",
    }

    def __init__(
        self,
        num_customers: int = 30,
        max_vehicles: int = 5,
        solver_time_limit: int = 1,
        acceptance_rate: float = 0.5,
        depot: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        patience_factor : float=0.002,
        batch_size: int = 1,
        model_path : Optional[str] = '/Users/jiangwolin/Downloads/llm-darp-main/checkpoints/sf_newenv_3/epoch_epoch=062.ckpt',
        traveler_decisions_path: Optional[str] = None,
        device: str = 'cpu',
        force_accept: bool = False,
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
        self.force_accept = force_accept
        self.patience_factor = patience_factor
        self.num_customers = num_customers
        self.max_vehicles = max_vehicles
        self.solver_time_limit = solver_time_limit
        self.acceptance_rate = acceptance_rate
        self.seed_val = seed
        self.device = device
        self.batch_size = int(batch_size)

        # Precompute action tensor for vectorization [16, 2]
        self.action_tensor = torch.tensor(self.ACTION_SPACE_MAP, device=self.device)

        # Initialize data generator
        csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
        ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
        generator = SFGenerator(
            csv_path=csv_path,
            travel_time_matrix_path=ttm_path,
            num_customers=num_customers,
            seed=seed,
        )
        self.data_generator = PDPTWEnv(generator=generator)

        # Get number of distinct locations from the travel time matrix
        self.num_distinct_locations = generator.travel_time_matrix.shape[0]

        # Define observation and action spaces
        # New state space: (2*num_distinct_locations + 4*48, 2)
        # Rows: pickup_loc_onehot + dropoff_loc_onehot + 4 time_window sections (each 48 rows)
        # Columns: [aggregated_previous_requests, current_new_request]
        self.num_time_intervals = 48  # Day divided into 48 intervals of 30 minutes each
        self.state_rows = 2 * self.num_distinct_locations + 4 * self.num_time_intervals

        self.observation_space = spaces.Dict({
            "fixed": spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.batch_size, self.num_customers, 6),
                dtype=np.float32,
            ),
            "new": spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.batch_size, 1, 6),
                dtype=np.float32,
            ),
            "distance_matrix": spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.batch_size, self.num_distinct_locations, self.num_distinct_locations),
                dtype=np.float32,
            ),
            "step": spaces.Box(
                low=0,
                high=self.num_customers,
                shape=(self.batch_size, 1),
                dtype=np.int64,
            ),
        })

        # Action: 16 discrete actions
        self.action_space = spaces.Discrete(16)

        # Episode state
        self.current_requests: Optional[TensorDict] = None  # Perturbed TWs for routing
        self.real_requests: Optional[TensorDict] = None  # Original TWs for oracle evaluation
        self.pending_requests: Optional[TensorDict] = None
        self.current_step = 0
        self.previous_cost = torch.zeros(self.batch_size, device=self.device)

        # Only load policy if model_path is provided
        self.model_path = model_path
        self.policy = None
        if model_path is not None:
            model_path_obj = Path(model_path)
            if model_path_obj.exists():
                torch.serialization.add_safe_globals([PDPTWEnv])
                # Always load checkpoint to CPU first to avoid RNG state issues
                ckpt = torch.load(model_path_obj, map_location="cpu", weights_only=False)
                state = ckpt.get("policy_state_dict", ckpt.get("state_dict", ckpt))

                model = PDPTWAttentionPolicy(num_h3=self.num_distinct_locations)
                model.load_state_dict(state, strict=False)
                # Then move the model to the desired device
                self.policy = model.to(self.device)
                self.policy.eval()  # Set to evaluation mode
            else:
                print(f"Warning: model_path not found ({model_path_obj}); running without policy.")

        # Load traveler decisions CSV for acceptance lookup
        if traveler_decisions_path is None:
            traveler_decisions_path = str(Path(__file__).with_name("traveler_decisions_augmented.csv"))
        self.traveler_decisions_path = traveler_decisions_path
        if traveler_decisions_path is not None:
            self.traveler_decisions_df = pd.read_csv(traveler_decisions_path)
        else:
            self.traveler_decisions_df = None
        self.decision_lookup = self._build_decision_lookup()
        self.df_mask = self._compute_mask_all_flexs() if self.traveler_decisions_df is not None else None
        self.current_request_masks = None

    def _build_decision_lookup(self) -> Optional[Dict[Tuple[int, str, str, str, int, int], Dict[str, bool]]]:
        if self.traveler_decisions_df is None:
            return None
        df = self.traveler_decisions_df
        missing_cols = [c for c in self.FLEXIBILITY_COLUMNS if c not in df.columns]
        if missing_cols:
            print(f"WARNING: Missing flexibility columns in decisions CSV: {missing_cols}")
        lookup: Dict[Tuple[int, str, str, str, int, int], Dict[str, bool]] = {}
        for _, row in df.iterrows():
            key = (
                int(row["traveler_id"]),
                str(row["trip_purpose"]).strip(),
                str(row["departure_location"]).strip(),
                str(row["arrival_location"]).strip(),
                int(row["pickup_shift_min"]),
                int(row["dropoff_shift_min"]),
            )
            decisions: Dict[str, bool] = {}
            for col in self.FLEXIBILITY_COLUMNS:
                if col in df.columns:
                    decisions[col] = str(row[col]).strip().lower() == "accept"
            lookup[key] = decisions
        return lookup

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[TensorDict, dict]:
        """
        Reset the environment and start a new batched episode.

        Returns:
            observation: Batched TensorDict
            info: Additional information
        """
        super().reset(seed=seed)

        # Generate new batch of episodes
        self.pending_requests = self.data_generator.reset(batch_size=[self.batch_size]).to(self.device)

        # Initialize current_requests with depot only (index 0)
        self.current_requests = self._init_depot_only(self.pending_requests)

        # Initialize real_requests (original unperturbed TWs for oracle)
        self.real_requests = self._init_depot_only(self.pending_requests)

        self.current_step = 0
        self.previous_cost = torch.zeros(self.batch_size, device=self.device)

        # Initial state: only first request (no previous requests)
        initial_state = self._get_observation()

        info = {
            "episode": 0,
            "total_requests": self.num_customers,
            "current_cost": self.previous_cost,
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
                if tensor.dim() == 3:
                    # Shape: [batch, num_nodes, 2]
                    depot_td[key] = tensor[:, 0:1, :]
                elif tensor.dim() == 2:
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
                if tensor.dim() == 3:
                    # Shape: [batch, num_nodes, 2]
                    sliced_td[key] = tensor[:, indices, :]
                elif tensor.dim() == 2:
                    # Shape: [batch, num_nodes]
                    sliced_td[key] = tensor[:, indices]

        return sliced_td
    
    def _parse_time_window_string(self, tw_string: str) -> Tuple[int, int]:
        """
        Parse time window string (e.g., "08:00-08:30") into minutes.

        Args:
            tw_string: Time window string in format "HH:MM-HH:MM"

        Returns:
            Tuple of (early_minutes, late_minutes)
        """
        try:
            early_str, late_str = tw_string.split('-')
            early_h, early_m = early_str.strip().split(':')
            late_h, late_m = late_str.strip().split(':')
            early_minutes = int(early_h) * 60 + int(early_m)
            late_minutes = int(late_h) * 60 + int(late_m)
            return early_minutes, late_minutes
        except (ValueError, AttributeError):
            # Fallback to 0-1440 if parsing fails
            return 0, 1440

    def _get_meta_data_single_traveler(self, traveler_id):
        # Use direct attribute access to avoid TensorDict indexing issues
        trip_metadata_list = self.pending_requests["trip_metadata"]
        # trip_metadata_list may be a list or NonTensorStack; unwrap to a dict
        trip_metadata = trip_metadata_list
        if not isinstance(trip_metadata, dict) and hasattr(trip_metadata, "__len__"):
            trip_metadata = trip_metadata[0] if len(trip_metadata) > 0 else {}
        if hasattr(trip_metadata, "data"):
            trip_metadata = trip_metadata.data
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

    def step(
        self, action: Union[int, torch.Tensor, np.ndarray]
    ) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one environment step (vectorized).

        Args:
            action: Tensor/Array of shape [batch_size] containing int actions [0-15].
        """
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0).expand(self.batch_size)

        # Get current incoming request for the whole batch
        new_request = self._slice(self.pending_requests, self.current_step)

        # Vectorized action shifts
        shifts = self.action_tensor[action]
        pickup_shifts = shifts[:, 0]
        dropoff_shifts = shifts[:, 1]

        accepted_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        if self.force_accept:
            accepted_mask[:] = True
        else:
            trip_metadata_list = self.pending_requests.get("trip_metadata", None)
            pickup_idx = 2 * self.current_step + 1
            if "user_id" in self.pending_requests.keys():
                current_user_ids = self.pending_requests["user_id"][:, pickup_idx]
            else:
                current_user_ids = torch.arange(1, self.batch_size + 1, device=self.device)

            for b in range(self.batch_size):
                u_id = int(current_user_ids[b].item())
                p_shift = int(pickup_shifts[b].item())
                d_shift = int(dropoff_shifts[b].item())
                accepted = np.random.random() < self.acceptance_rate

                if trip_metadata_list is not None:
                    md = trip_metadata_list
                    if not isinstance(md, dict) and hasattr(md, "__len__"):
                        md = md[b] if b < len(md) else {}
                    if hasattr(md, "data"):
                        md = md.data
                    if isinstance(md, dict):
                        user_meta = md.get(u_id)
                        if user_meta is not None:
                            flexibility = user_meta.get("flexibility")
                            trip_purpose = user_meta.get("trip_purpose")
                            departure_location = user_meta.get("departure_location")
                            arrival_location = user_meta.get("arrival_location")
                            if flexibility is not None:
                                accepted = self._get_acceptance_decision(
                                    traveler_id=u_id,
                                    flexibility=flexibility,
                                    trip_purpose=trip_purpose,
                                    departure_location=departure_location,
                                    arrival_location=arrival_location,
                                    pickup_shift=p_shift,
                                    dropoff_shift=d_shift,
                                )

                accepted_mask[b] = accepted

        # Apply perturbations batch-wise (zero shift if rejected)
        final_p_shifts = pickup_shifts * accepted_mask.float()
        final_d_shifts = dropoff_shifts * accepted_mask.float()

        if self.force_accept:
            patience_penalties = torch.zeros(self.batch_size, device=self.device)
        else:
            patience_penalties = (final_p_shifts.abs() + final_d_shifts.abs()) * self.patience_factor

        perturbed_request = self._apply_perturbation_batch(
            new_request, final_p_shifts, final_d_shifts
        )

        # Append to current/real requests
        self.current_requests = self._append_to_current(self.current_requests, perturbed_request)
        self.real_requests = self._append_to_current(self.real_requests, new_request)

        # Evaluate costs
        new_costs = torch.zeros(self.batch_size, device=self.device)
        solver_failures = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        solver_input = self._get_solver_input(self.current_requests)

        if self.policy is not None:
            with torch.no_grad():
                out = self.policy(
                    solver_input.to(self.device),
                    env=self.data_generator,
                    phase="test",
                    decode_type="greedy",
                    return_actions=True,
                )
                new_costs = -out["reward"]
        else:
            for b in range(self.batch_size):
                single_td = solver_input[b:b+1]
                result = darp_solver(
                    single_td,
                    max_vehicles=self.max_vehicles,
                    time_limit_seconds=self.solver_time_limit,
                )
                cost = result["total_time"]
                if cost == float("inf"):
                    solver_failures[b] = True
                    new_costs[b] = cost
                else:
                    new_costs[b] = cost

        rewards = self.previous_cost - new_costs - patience_penalties
        rewards = torch.where(
            solver_failures,
            torch.tensor(-5000.0, device=self.device),
            rewards,
        )

        self.previous_cost = new_costs
        self.current_step += 1

        terminated = torch.full(
            (self.batch_size,),
            self.current_step >= self.num_customers,
            dtype=torch.bool,
            device=self.device,
        )
        truncated = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        observation = self._get_observation()

        info = {
            "step": self.current_step,
            "current_cost": new_costs,
            "accepted": accepted_mask,
            "patience_penalty": patience_penalties,
            "solver_failed": solver_failures,
        }

        return observation, rewards, terminated, truncated, info

    def _encode_location_onehot(self, location_idx: int) -> np.ndarray:
        """
        Encode a location index as one-hot vector.

        Args:
            location_idx: Integer index of location in [0, num_distinct_locations)

        Returns:
            One-hot numpy array of shape (num_distinct_locations,)
        """
        onehot = np.zeros(self.num_distinct_locations, dtype=np.float32)
        if 0 <= location_idx < self.num_distinct_locations:
            onehot[location_idx] = 1.0
        return onehot

    def _encode_time_onehot(self, time_value: float) -> np.ndarray:
        """
        Encode a time value (in minutes) as one-hot vector for 48 intervals.

        A day (1440 minutes) is divided into 48 intervals of 30 minutes each.
        For a time value, we find which interval(s) it belongs to and set those to 1.

        Example: time_value = 320 minutes -> interval 10 (320 // 30 = 10.67 -> floor = 10)
                 time_value = 350 minutes -> interval 11 (350 // 30 = 11.67 -> floor = 11)

        Args:
            time_value: Time in minutes [0, 1440]

        Returns:
            One-hot numpy array of shape (48,)
        """
        onehot = np.zeros(self.num_time_intervals, dtype=np.float32)
        interval_idx = int(time_value // 30)  # 30 minutes per interval
        # Clamp to valid range [0, 47]
        interval_idx = max(0, min(interval_idx, self.num_time_intervals - 1))
        onehot[interval_idx] = 1.0
        return onehot

    def _encode_request_onehot(
        self,
        pickup_loc_idx: int,
        dropoff_loc_idx: int,
        pickup_tw_early: float,
        pickup_tw_late: float,
        dropoff_tw_early: float,
        dropoff_tw_late: float
    ) -> np.ndarray:
        """
        Encode a single request as a one-hot vector.

        Returns a vector of shape (state_rows,) with the following structure:
        - [0:num_distinct_locations]: pickup location one-hot
        - [num_distinct_locations:2*num_distinct_locations]: dropoff location one-hot
        - [2*num_distinct_locations:2*num_distinct_locations+48]: pickup TW early one-hot
        - [2*num_distinct_locations+48:2*num_distinct_locations+96]: pickup TW late one-hot
        - [2*num_distinct_locations+96:2*num_distinct_locations+144]: dropoff TW early one-hot
        - [2*num_distinct_locations+144:2*num_distinct_locations+192]: dropoff TW late one-hot

        Args:
            pickup_loc_idx: Pickup location index
            dropoff_loc_idx: Dropoff location index
            pickup_tw_early: Pickup time window early bound (minutes)
            pickup_tw_late: Pickup time window late bound (minutes)
            dropoff_tw_early: Dropoff time window early bound (minutes)
            dropoff_tw_late: Dropoff time window late bound (minutes)

        Returns:
            One-hot encoded vector of shape (state_rows,)
        """
        encoding = np.zeros(self.state_rows, dtype=np.float32)

        # Encode pickup location
        encoding[0:self.num_distinct_locations] = self._encode_location_onehot(pickup_loc_idx)

        # Encode dropoff location
        encoding[self.num_distinct_locations:2*self.num_distinct_locations] = self._encode_location_onehot(dropoff_loc_idx)

        # Encode time windows
        offset = 2 * self.num_distinct_locations
        encoding[offset:offset+48] = self._encode_time_onehot(pickup_tw_early)
        encoding[offset+48:offset+96] = self._encode_time_onehot(pickup_tw_late)
        encoding[offset+96:offset+144] = self._encode_time_onehot(dropoff_tw_early)
        encoding[offset+144:offset+192] = self._encode_time_onehot(dropoff_tw_late)

        return encoding

    def _get_observation(self) -> TensorDict:
        """Build batched observation for PPO."""
        fixed = torch.zeros((self.batch_size, self.num_customers, 6), device=self.device)

        if self.current_step > 0:
            p_indices = torch.arange(1, 2 * self.current_step + 1, 2, device=self.device)
            d_indices = torch.arange(2, 2 * self.current_step + 2, 2, device=self.device)

            p_h3 = self.current_requests["h3_indices"][:, p_indices]
            p_tw = self.current_requests["time_windows"][:, p_indices]
            d_h3 = self.current_requests["h3_indices"][:, d_indices]
            d_tw = self.current_requests["time_windows"][:, d_indices]

            fixed[:, :self.current_step, 0] = p_h3
            fixed[:, :self.current_step, 1] = p_tw[:, :, 0]
            fixed[:, :self.current_step, 2] = p_tw[:, :, 1]
            fixed[:, :self.current_step, 3] = d_h3
            fixed[:, :self.current_step, 4] = d_tw[:, :, 0]
            fixed[:, :self.current_step, 5] = d_tw[:, :, 1]

        if self.current_step < self.num_customers:
            req = self._slice(self.pending_requests, self.current_step)
            new = torch.zeros((self.batch_size, 1, 6), device=self.device)
            new[:, 0, 0] = req["h3_indices"][:, 0]
            new[:, 0, 1] = req["time_windows"][:, 0, 0]
            new[:, 0, 2] = req["time_windows"][:, 0, 1]
            new[:, 0, 3] = req["h3_indices"][:, 1]
            new[:, 0, 4] = req["time_windows"][:, 1, 0]
            new[:, 0, 5] = req["time_windows"][:, 1, 1]
        else:
            new = torch.zeros((self.batch_size, 1, 6), device=self.device)

        d_mat = self.pending_requests["travel_time_matrix"]

        return TensorDict(
            {
                "fixed": fixed,
                "new": new,
                "distance_matrix": d_mat,
                "step": torch.full(
                    (self.batch_size, 1),
                    self.current_step,
                    dtype=torch.int64,
                    device=self.device,
                ),
            },
            batch_size=[self.batch_size],
        )

    def _append_to_current(self, current_requests: TensorDict, perturbed_request: TensorDict) -> TensorDict:
        """
        Combine current_requests with a new perturbed_request.

        Concatenates node-indexed fields along the node dimension.
        All other fields are preserved from current_requests.

        Returns:
            TensorDict with appended request.
        """
        # Start with a clone of current_requests to preserve all non-sliced fields
        appended_td = current_requests.clone()

        # Concatenate node-indexed fields
        for key in self.SLICE_FIELDS:
            if key in current_requests.keys() and key in perturbed_request.keys():
                tensor = current_requests[key]
                if tensor.dim() == 3:
                    appended_td[key] = torch.cat([current_requests[key], perturbed_request[key]], dim=1)
                elif tensor.dim() == 2:
                    appended_td[key] = torch.cat([current_requests[key], perturbed_request[key]], dim=1)

        return appended_td

    def _get_solver_input(self, td: TensorDict) -> TensorDict:
        """
        Prepare a TensorDict for the solver by resetting dynamic state.
        """
        solver_td = td.clone()

        if "visited" in solver_td.keys():
            solver_td["visited"] = torch.zeros_like(solver_td["visited"])
        if "completed" in solver_td.keys():
            solver_td["completed"] = torch.zeros_like(solver_td["completed"])
        if "used_capacity" in solver_td.keys():
            solver_td["used_capacity"] = torch.zeros_like(solver_td["used_capacity"])
        if "current_node" in solver_td.keys():
            solver_td["current_node"] = torch.zeros_like(solver_td["current_node"])
        if "i" in solver_td.keys():
            solver_td["i"] = torch.zeros_like(solver_td["i"])
        if "pending_schedule" in solver_td.keys():
            solver_td["pending_schedule"] = torch.zeros_like(solver_td["pending_schedule"])
        if "pending_count" in solver_td.keys():
            solver_td["pending_count"] = torch.zeros_like(solver_td["pending_count"])

        return solver_td

    def _apply_perturbation_batch(
        self, new_request: TensorDict, pickup_shifts: torch.Tensor, dropoff_shifts: torch.Tensor
    ) -> TensorDict:
        perturbed = new_request.clone()

        tw = perturbed["time_windows"].clone()
        ps = pickup_shifts.view(-1, 1)
        ds = dropoff_shifts.view(-1, 1)

        tw[:, 0, 0] += ps[:, 0]
        tw[:, 0, 1] += ps[:, 0]
        tw[:, 1, 0] += ds[:, 0]
        tw[:, 1, 1] += ds[:, 0]

        tw = torch.clamp(tw, min=0, max=1410)
        perturbed["time_windows"] = tw
        return perturbed

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

        if self.decision_lookup is not None:
            key = (
                int(traveler_id),
                str(trip_purpose).strip(),
                str(departure_location).strip(),
                str(arrival_location).strip(),
                int(pickup_shift_abs),
                int(dropoff_shift_abs),
            )
            decisions = self.decision_lookup.get(key)
            if decisions is None:
                print(f"WARNING: No matching decision found for traveler {traveler_id}, "
                        f"flexibility='{flexibility}', trip_purpose='{trip_purpose}', "
                      f"departure='{departure_location}', arrival='{arrival_location}', "
                      f"pickup_shift={pickup_shift}, dropoff_shift={dropoff_shift}")
                return np.random.random() < self.acceptance_rate
            decision = decisions.get(flexibility)
            if decision is None:
                print(f"WARNING: Flexibility column '{flexibility}' not found in CSV decisions")
                return np.random.random() < self.acceptance_rate
            return bool(decision)

        # Find matching row (fallback path)
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

    def get_real_requests(self) -> TensorDict:
        """
        Get requests with ORIGINAL (unperturbed) time windows for oracle evaluation.

        Returns:
            TensorDict containing requests with original time windows from trip_metadata.
            This is used by oracle evaluators to assess performance against ground truth.
        """
        return self.real_requests

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
    decisions_path = Path(__file__).with_name("traveler_decisions_augmented.csv")
    print(decisions_path)
    solver_ckpt_path = "/Users/jiangwolin/Desktop/Research/DARPSolver/checkpoints/Jan1501/best.pt"
    env = DVRPEnv(
        num_customers=5,
        batch_size=2,
        seed=42,
        traveler_decisions_path=decisions_path,
        model_path=solver_ckpt_path,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    print("\n--- Resetting Environment ---")
    obs, info = env.reset()
    print(f"Initial observation keys: {list(obs.keys())}")
    print(f"fixed shape: {obs['fixed'].shape}, new shape: {obs['new'].shape}")
    print(f"Initial info: {info}")
    print(f"First few observation rows:\n{obs[:]}")

    # Run a few steps
    print("\n=== Running 5 steps ===")
    total_reward = 0.0

    for step in range(5):
        # Random actions for batch
        action = torch.randint(0, 16, (env.batch_size,))
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}:")
        print(f"  Actions: {action.tolist()}")
        print(f"  Accepted: {info['accepted'].tolist()}")
        print(f"  Reward: {reward.tolist()}")
        print(f"  Cost: {info.get('current_cost', 'N/A')}")
        print(f"  Terminated: {terminated.tolist()}")
        print("- -" * 25)
        total_reward += float(reward.mean().item())

        if terminated.any() or truncated.any():
            break

    print(f"\n{'=' * 50}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final fixed shape: {obs['fixed'].shape}, new shape: {obs['new'].shape}")

    # Test multiple resets
    print("\n--- Testing Multiple Resets ---")
    for i in range(3):
        obs, info = env.reset(seed=i)
        print(f"Reset {i+1}: fixed shape = {obs['fixed'].shape}, info = {info}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_env()
