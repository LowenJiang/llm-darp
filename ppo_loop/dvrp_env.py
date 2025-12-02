"""
Dynamic Vehicle Routing Problem with Time Windows (DVRP-TW) Environment
(Batched Version)

MDP formulation:
- State: One-hot encoded representation (2*num_locations + 192, 2)
  - Column 0: Aggregated sum of all previously accepted requests
  - Column 1: Current new incoming request
  - Rows encode: pickup_loc, dropoff_loc, 4 time windows (each 48 intervals)
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
from rl4co.envs.routing import SFGenerator, PDPTWEnv
from ortools_solver import darp_solver
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy
from tensordict.tensordict import TensorDict, TensorDictBase


class DVRPEnv(gym.Env):
    """
    Gym environment for Dynamic Vehicle Routing Problem with Time Windows.
    Supports Batch Execution.

    State Space (Batched):
        - TensorDict with keys:
          * 'fixed': Tensor [batch_size, num_customers, 6]
          * 'new': Tensor [batch_size, 1, 6]
          * 'distance_matrix': Tensor [batch_size, num_locations, num_locations]
          * 'step': Tensor [batch_size, 1]
    
    Action Space:
        - Discrete(16) -> Expects input [batch_size] of integers.
    """

    # Action space mapping: (pickup_shift, dropoff_shift)
    ACTION_SPACE_MAP = [
        (-30, 0), (-30, 10), (-30, 20), (-30, 30),
        (-20, 0), (-20, 10), (-20, 20), (-20, 30),
        (-10, 0), (-10, 10), (-10, 20), (-10, 30),
        (0, 0),   (0, 10),   (0, 20),   (0, 30),
    ]

    # Fields that grow with the number of requests (must be sliced/concatenated)
    # Note: Global fields like 'travel_time_matrix', 'capacity', 'vehicle_capacity', 'trip_metadata'
    # are NOT in here, so they are preserved automatically by clone().
    SLICE_FIELDS = {
        "h3_indices", 
        "time_windows", 
        "demand", 
        "locs", 
        "action_mask", 
        "visited", 
        "user_id",      # Added to track IDs in history
        "service_time"  # Added if present in data
    }

    def __init__(
        self,
        num_customers: int = 30,
        max_vehicles: int = 5,
        solver_time_limit: int = 1,
        acceptance_rate: float = 0.9,
        depot: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        patience_factor: float = 0.002,
        batch_size: int = 1,
        model_path: Optional[str] = "/Users/jiangwolin/Downloads/llm-darp-main/checkpoints/sf_newenv_3/epoch_epoch=062.ckpt",
        traveler_decisions_path: Optional[str] = "/Users/jiangwolin/Downloads/llm-darp-main/ppo_loop/traveler_decisions_augmented.csv",
        device: str = 'cpu'
    ):
        super().__init__()
        self.patience_factor = patience_factor
        self.num_customers = num_customers
        self.max_vehicles = max_vehicles
        self.solver_time_limit = solver_time_limit
        self.acceptance_rate = acceptance_rate
        self.seed_val = seed
        self.device = device
        self.batch_size = batch_size

        # Precompute action tensor for vectorization [16, 2]
        self.action_tensor = torch.tensor(self.ACTION_SPACE_MAP, device=self.device)

        # Initialize data generator
        self.data_generator = PDPTWEnv(generator_params={
            "num_customers": num_customers
        })

        # Get number of distinct locations
        self.num_distinct_locations = self.data_generator.generator.travel_time_matrix.shape[0]

        # Define observation space (Batched)
        self.observation_space = spaces.Dict({
            'fixed': spaces.Box(low=0, high=np.inf, shape=(batch_size, self.num_customers, 6), dtype=np.float32),
            'new': spaces.Box(low=0, high=np.inf, shape=(batch_size, 1, 6), dtype=np.float32),
            'distance_matrix': spaces.Box(low=0, high=np.inf,
                                         shape=(batch_size, self.num_distinct_locations, self.num_distinct_locations),
                                         dtype=np.float32),
            'step': spaces.Box(low=0, high=self.num_customers, shape=(batch_size, 1), dtype=np.int64)
        })

        self.action_space = spaces.Discrete(16)

        # Episode state
        self.current_requests: Optional[TensorDict] = None
        self.real_requests: Optional[TensorDict] = None
        self.pending_requests: Optional[TensorDict] = None
        self.current_step = 0
        self.previous_cost = torch.zeros(self.batch_size, device=self.device)

        # Load Policy
        self.model_path = model_path
        if model_path is not None:
            env = PDPTWEnv()
            torch.serialization.add_safe_globals([PDPTWEnv])
            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                state = ckpt["state_dict"]
                model = AttentionModel(env=PDPTWEnv)
                model.load_state_dict(state, strict=False)
                self.policy = model.policy.to(self.device)
                self.policy.eval()
            except Exception as e:
                print(f"Warning: Could not load model from {model_path}. Error: {e}")
                self.policy = None
        else:
            self.policy = None

        # Load traveler decisions
        self.traveler_decisions_path = traveler_decisions_path
        if traveler_decisions_path is not None:
            self.traveler_decisions_df = pd.read_csv(traveler_decisions_path)
            self.df_mask = self._compute_mask_all_flexs()
        else:
            self.traveler_decisions_df = None
            self.df_mask = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[TensorDict, dict]:
        """Reset the environment (batched)."""
        super().reset(seed=seed)

        # Generate new batch of episodes
        # pending_requests will contain full batch: [batch_size, 61, ...]
        self.pending_requests = self.data_generator.reset(batch_size=[self.batch_size]).to(self.device)

        # Initialize current/real requests with depot only [batch_size, 1, ...]
        # NOTE: This preserves global fields like travel_time_matrix automatically
        self.current_requests = self._init_depot_only(self.pending_requests)
        self.real_requests = self._init_depot_only(self.pending_requests)

        self.current_step = 0
        self.previous_cost = torch.zeros(self.batch_size, device=self.device)

        initial_state = self._get_observation()
        
        info = {
            "episode": 0,
            "total_requests": self.num_customers,
            "current_cost": self.previous_cost,
        }

        return initial_state, info

    def step(self, action: Union[int, torch.Tensor, np.ndarray]) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Execute one environment step (Vectorized).
        Args:
            action: Tensor/Array of shape [batch_size] containing int actions [0-15].
        """
        # Ensure action is a tensor on device
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0).expand(self.batch_size)
        
        # Get current incoming request for the whole batch
        # Slice returns [Batch, 2 (P+D), Attributes]
        new_request = self._slice(self.pending_requests, self.current_step)

        # 1. Vectorized Acceptance Logic
        shifts = self.action_tensor[action] 
        pickup_shifts = shifts[:, 0]
        dropoff_shifts = shifts[:, 1]

        accepted_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        
        # Metadata extraction
        trip_metadata_col = self.pending_requests.get("trip_metadata")
        
        # Identify User IDs
        pickup_idx = 2 * self.current_step + 1
        current_user_ids = self.pending_requests["user_id"][:, pickup_idx]

        for b in range(self.batch_size):
            u_id = current_user_ids[b].item()
            p_shift = pickup_shifts[b].item()
            d_shift = dropoff_shifts[b].item()
            
            decision = False
            if trip_metadata_col is not None:
                try:
                    md = trip_metadata_col[b]
                    if md is not None and not isinstance(md, (TensorDict, TensorDictBase)) and u_id in md:
                        user_meta = md[u_id]
                        decision = self._get_acceptance_decision(
                            traveler_id=u_id,
                            flexibility=user_meta["flexibility"],
                            trip_purpose=user_meta["trip_purpose"],
                            departure_location=user_meta["departure_location"],
                            arrival_location=user_meta["arrival_location"],
                            pickup_shift=p_shift,
                            dropoff_shift=d_shift
                        )
                    else:
                        decision = np.random.random() < self.acceptance_rate
                except (IndexError, KeyError, RuntimeError):
                     decision = np.random.random() < self.acceptance_rate
            else:
                decision = np.random.random() < self.acceptance_rate
            
            accepted_mask[b] = decision

        # 2. Apply Perturbations
        final_p_shifts = pickup_shifts * accepted_mask.float()
        final_d_shifts = dropoff_shifts * accepted_mask.float()
        
        patience_penalties = (final_p_shifts.abs() + final_d_shifts.abs()) * self.patience_factor

        perturbed_request = self._apply_perturbation_batch(
            new_request, final_p_shifts, final_d_shifts
        )

        # Update current_requests (Append new request to existing history)
        # Note: Global fields are preserved from previous step
        self.current_requests = self._append_to_current(self.current_requests, perturbed_request)

        # 3. Handle Real Requests (Oracle Data - Ground Truth)
        self.real_requests = self._append_to_current(self.real_requests, new_request)

        # 4. Solve / Evaluate Cost
        new_costs = torch.zeros(self.batch_size, device=self.device)
        solver_failures = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        # IMPORTANT: Sanitize input for solver.
        # The solver expects a "fresh" problem (visited=False, current_node=Depot)
        # even though self.current_requests contains accumulated history.
        solver_input = self._get_solver_input(self.current_requests)

        if self.policy is not None:
            with torch.no_grad():
                out = self.policy(solver_input, phase='test', decode_type="greedy", return_actions=True)
                new_costs = -out["reward"] 
        else:
            # Fallback Loop for OR-Tools
            print("FALLBACK TO ORTOOLS SOLVER")
            for b in range(self.batch_size):
                single_td = solver_input[b] 
                res = darp_solver(single_td, max_vehicles=self.max_vehicles, time_limit_seconds=self.solver_time_limit)
                cost = res["total_time"]
                
                if cost == float('inf'):
                    new_costs[b] = 5000.0 
                    solver_failures[b] = True
                else:
                    new_costs[b] = cost

        # 5. Calculate Rewards
        rewards = self.previous_cost - new_costs - patience_penalties
        rewards = torch.where(solver_failures, torch.tensor(-5000.0, device=self.device), rewards)

        # Update state
        self.previous_cost = new_costs
        self.current_step += 1

        # Check termination
        terminated = torch.full((self.batch_size,), self.current_step >= self.num_customers, dtype=torch.bool, device=self.device)
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

    def _get_solver_input(self, td: TensorDict) -> TensorDict:
        """
        Prepares a TensorDict for the solver (Oracle).
        
        Crucial: Resets optimization state variables to ensure the solver 
        treats this as a fresh routing problem for the current subset of nodes, 
        rather than continuing a previous rollout.
        
        Preserves:
        - locs, time_windows, demand (The problem definition)
        - travel_time_matrix, capacity, etc. (The global constants)
        """
        solver_td = td.clone()
        
        # Reset Routing State
        if "visited" in solver_td.keys():
            solver_td["visited"] = torch.zeros_like(solver_td["visited"])
            # Some RL4CO envs assume depot (idx 0) is visited, others don't. 
            # Usually safe to leave all False for a fresh 'reset' state 
            # as the policy will mask the depot based on 'current_node'.
        
        if "used_capacity" in solver_td.keys():
            solver_td["used_capacity"] = torch.zeros_like(solver_td["used_capacity"])
            
        if "current_node" in solver_td.keys():
            # Reset to Depot (Index 0)
            solver_td["current_node"] = torch.zeros_like(solver_td["current_node"])
            
        if "i" in solver_td.keys():
            # Reset step counter
            solver_td["i"] = torch.zeros_like(solver_td["i"])
            
        return solver_td

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

        return TensorDict({
            'fixed': fixed,
            'new': new,
            'distance_matrix': d_mat,
            'step': torch.full((self.batch_size, 1), self.current_step, dtype=torch.int64, device=self.device)
        }, batch_size=[self.batch_size])

    def _init_depot_only(self, td: TensorDict) -> TensorDict:
        """
        Initialize [Batch, 1 (Depot), ...]
        Implicitly preserves fields NOT in SLICE_FIELDS (e.g., travel_time_matrix).
        """
        depot_td = td.clone()
        for key in td.keys():
            if key in self.SLICE_FIELDS:
                tensor = td[key]
                if tensor.dim() == 3: 
                    depot_td[key] = tensor[:, 0:1, :]
                elif tensor.dim() == 2: 
                    depot_td[key] = tensor[:, 0:1]
        return depot_td

    def _slice(self, td: TensorDict, step: int) -> TensorDict:
        """
        Extract P/D nodes for the whole batch at specific step.
        Implicitly preserves fields NOT in SLICE_FIELDS (e.g., travel_time_matrix).
        """
        p_idx = 2 * step + 1
        d_idx = 2 * step + 2
        indices = [p_idx, d_idx]
        
        sliced = td.clone()
        for key in td.keys():
            if key in self.SLICE_FIELDS:
                tensor = td[key]
                if tensor.dim() == 3: 
                    sliced[key] = tensor[:, indices, :]
                elif tensor.dim() == 2:
                    sliced[key] = tensor[:, indices]
        return sliced

    def _apply_perturbation_batch(self, new_request: TensorDict, p_shifts: torch.Tensor, d_shifts: torch.Tensor) -> TensorDict:
        perturbed = new_request.clone()
        tw = perturbed["time_windows"].clone()
        
        ps = p_shifts.view(-1, 1)
        ds = d_shifts.view(-1, 1)
        
        tw[:, 0, 0] += ps[:, 0]
        tw[:, 0, 1] += ps[:, 0]
        tw[:, 1, 0] += ds[:, 0]
        tw[:, 1, 1] += ds[:, 0]
        
        tw = torch.clamp(tw, min=0)
        perturbed["time_windows"] = tw
        return perturbed

    def _append_to_current(self, current: TensorDict, incoming: TensorDict) -> TensorDict:
        """Concatenates incoming nodes to current request sequence."""
        appended = current.clone()
        for key in self.SLICE_FIELDS:
            # Check if key exists in both to avoid errors if incoming missing something
            if key in current.keys() and key in incoming.keys():
                 appended[key] = torch.cat([current[key], incoming[key]], dim=1)
        return appended
    
    # --- Helpers (Masking, etc.) ---
    def _compute_mask_all_flexs(self):
        index_cols = ["traveler_id", "trip_purpose", "departure_location", "arrival_location", "departure_time_window", "arrival_time_window"]
        action_cols = ["pickup_shift_min", "dropoff_shift_min"]
        df_mask = None
        flex_types = [
            "flexible for late dropoff, but inflexible for early pickup",
            "flexible for early pickup, but inflexible for late dropoff",
            "inflexible for any schedule changes",
            "flexible for both early pickup and late dropoff"
        ]
        
        def compute_single(flex):
            df = self.traveler_decisions_df[index_cols + action_cols + [flex]].copy()
            df["indicator"] = (df[flex] == "accept").astype(int)
            wide = df.pivot_table(index=index_cols, columns=action_cols, values="indicator", fill_value=0)
            strict_ordering = [(abs(p), abs(d)) for p, d in self.ACTION_SPACE_MAP]
            wide = wide.reindex(columns=strict_ordering, fill_value=0)
            wide[flex] = wide.apply(lambda row: row.values.tolist(), axis=1)
            res = wide[[flex]].reset_index()
            if isinstance(res.columns, pd.MultiIndex): res.columns = res.columns.get_level_values(0)
            return res

        for i, f in enumerate(flex_types):
            res = compute_single(f)
            res = res.rename(columns={f: i})
            df_mask = res if df_mask is None else df_mask.merge(res, on=index_cols)
        return df_mask

    def _get_acceptance_decision(self, traveler_id, flexibility, trip_purpose, departure_location, arrival_location, pickup_shift, dropoff_shift) -> bool:
        if self.traveler_decisions_df is None: return np.random.random() < self.acceptance_rate
        p_abs = abs(pickup_shift)
        d_abs = abs(dropoff_shift)
        
        mask = (
            (self.traveler_decisions_df["traveler_id"] == traveler_id) &
            (self.traveler_decisions_df["trip_purpose"] == trip_purpose) &
            (self.traveler_decisions_df["departure_location"] == departure_location) &
            (self.traveler_decisions_df["arrival_location"] == arrival_location) &
            (self.traveler_decisions_df["pickup_shift_min"] == p_abs) &
            (self.traveler_decisions_df["dropoff_shift_min"] == d_abs)
        )
        rows = self.traveler_decisions_df[mask]
        if len(rows) == 0: return np.random.random() < self.acceptance_rate
        return rows.iloc[0][flexibility] == "accept"

    def _get_mask_from_flex(self, traveler_id, trip_purpose, departure_location, arrival_location, departure_time_window, arrival_time_window, predicted_flex_index):
        if isinstance(predicted_flex_index, torch.Tensor): predicted_flex_index = int(predicted_flex_index.item())
        selection = (
            (self.df_mask["traveler_id"] == traveler_id) &
            (self.df_mask["trip_purpose"] == trip_purpose) &
            (self.df_mask["departure_location"] == departure_location) &
            (self.df_mask["arrival_location"] == arrival_location) & 
            (self.df_mask["departure_time_window"] == departure_time_window) & 
            (self.df_mask["arrival_time_window"] == arrival_time_window)
        )
        mask_series = self.df_mask[selection][predicted_flex_index]
        return mask_series.iloc[0] if len(mask_series) > 0 else [1]*16


def test_stepwise_reward_logic():
    print("\n=== Testing Step-wise Reward Logic ===")
    
    BATCH_SIZE = 1
    NUM_CUSTOMERS = 30
    env = DVRPEnv(num_customers=NUM_CUSTOMERS, batch_size=BATCH_SIZE)
    
    obs, info = env.reset()
    prev_cost_tracker = torch.zeros(BATCH_SIZE)
    
    print(f"{'Step':<5} | {'Act':<4} | {'Old Cost':<10} | {'New Cost':<10} | {'Marginal':<10} | {'Penalty':<8} | {'Env Rew':<10} | {'Calc Rew':<10} | {'Match?'}")
    print("-" * 95)

    done = False
    step_count = 0

    while not done:
        action = torch.randint(0, 16, (BATCH_SIZE,))
        obs, reward, terminated, truncated, info = env.step(action)
        
        b_idx = 0
        act = action[b_idx].item()
        old_c = prev_cost_tracker[b_idx].item()
        new_c = info['current_cost'][b_idx].item()
        penalty = info['patience_penalty'][b_idx].item()
        env_r = reward[b_idx].item()
        solver_failed = info['solver_failed'][b_idx].item()
        
        marginal_diff = old_c - new_c
        calculated_reward = marginal_diff - penalty
        
        if solver_failed:
            calculated_reward = -5000.0
            check_str = "FAIL_OK" if env_r == -5000.0 else "FAIL_ERR"
        else:
            is_match = abs(env_r - calculated_reward) < 1e-3
            check_str = "YES" if is_match else "NO"

        print(f"{step_count:<5} | {act:<4} | {old_c:<10.2f} | {new_c:<10.2f} | {marginal_diff:<10.2f} | {penalty:<8.2f} | {env_r:<10.2f} | {calculated_reward:<10.2f} | {check_str}")
        prev_cost_tracker = info['current_cost']
        done = terminated[0].item()
        step_count += 1

    print("-" * 95)
    print("Test Complete.")


if __name__ == "__main__":
    test_stepwise_reward_logic()