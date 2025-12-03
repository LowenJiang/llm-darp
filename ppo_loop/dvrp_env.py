"""
Dynamic Vehicle Routing Problem with Time Windows (DVRP-TW) Environment
(Batched Version)
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

    SLICE_FIELDS = {
        "h3_indices", 
        "time_windows", 
        "demand", 
        "locs", 
        "action_mask", 
        "visited", 
        "user_id",
        "service_time"
    }

    def __init__(
        self,
        num_customers: int = 30,
        max_vehicles: int = 5,
        solver_time_limit: int = 1,
        acceptance_rate: float = 0.5,
        depot: Optional[Tuple[float, float]] = None,
        seed: Optional[int] = None,
        patience_factor: float = 0.002,
        batch_size: int = 1,
        num_time_bins: int = 48,  # <--- control bins
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
        
        # Time Binning Configuration
        self.num_time_bins = num_time_bins
        # Assuming 1440 minutes in a day. 
        # If num_time_bins=48, interval is 30.0
        self.time_bin_interval = 1440.0 / self.num_time_bins

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
                ckpt_path = model_path
                model = AttentionModel.load_from_checkpoint(ckpt_path, env=env, map_location="cpu", weights_only=False)
                self.policy = model.policy
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
        self.pending_requests = self.data_generator.reset(batch_size=[self.batch_size]).to(self.device)
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
        """Execute one environment step (Vectorized)."""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0).expand(self.batch_size)
        
        new_request = self._slice(self.pending_requests, self.current_step)

        # 1. Vectorized Acceptance Logic
        shifts = self.action_tensor[action] 
        pickup_shifts = shifts[:, 0]
        dropoff_shifts = shifts[:, 1]

        accepted_mask = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        trip_metadata_col = self.pending_requests.get("trip_metadata")
        pickup_idx = 2 * self.current_step + 1
        current_user_ids = self.pending_requests["user_id"][:, pickup_idx]

        for b in range(self.batch_size):
            u_id = current_user_ids[b].item()
            p_shift = pickup_shifts[b].item()
            d_shift = dropoff_shifts[b].item()
            
            decision = False
            if trip_metadata_col is not None:
                md = trip_metadata_col[b]
                if hasattr(md, "data"):
                    md = md.data

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

        self.current_requests = self._append_to_current(self.current_requests, perturbed_request)
        self.real_requests = self._append_to_current(self.real_requests, new_request)

        # 4. Solve / Evaluate Cost
        new_costs = torch.zeros(self.batch_size, device=self.device)
        solver_failures = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        solver_input = self._get_solver_input(self.current_requests)

        if self.policy is not None:
            with torch.no_grad():
                out = self.policy(solver_input, phase='test', decode_type="greedy", return_actions=True)
                new_costs = -out["reward"] 
        else:
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

        self.previous_cost = new_costs
        self.current_step += 1

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
        """Prepares a TensorDict for the solver (Oracle)."""
        solver_td = td.clone()
        if "visited" in solver_td.keys():
            solver_td["visited"] = torch.zeros_like(solver_td["visited"])
        if "used_capacity" in solver_td.keys():
            solver_td["used_capacity"] = torch.zeros_like(solver_td["used_capacity"])
        if "current_node" in solver_td.keys():
            solver_td["current_node"] = torch.zeros_like(solver_td["current_node"])
        if "i" in solver_td.keys():
            solver_td["i"] = torch.zeros_like(solver_td["i"])
        return solver_td

    def _get_observation(self) -> TensorDict:
        """
        Build batched observation for PPO.
        Converts continuous time windows into discrete bins based on num_time_bins.
        """
        fixed = torch.zeros((self.batch_size, self.num_customers, 6), device=self.device)
        
        # Calculate Bins (0-indexed)
        # e.g., if interval=30, 20->0, 350->11, 440->14
        
        if self.current_step > 0:
            p_indices = torch.arange(1, 2 * self.current_step + 1, 2, device=self.device)
            d_indices = torch.arange(2, 2 * self.current_step + 2, 2, device=self.device)
            
            p_h3 = self.current_requests["h3_indices"][:, p_indices]
            p_tw = self.current_requests["time_windows"][:, p_indices]
            d_h3 = self.current_requests["h3_indices"][:, d_indices]
            d_tw = self.current_requests["time_windows"][:, d_indices]
            
            # Apply Binning
            p_tw_binned = torch.floor(p_tw / self.time_bin_interval)
            d_tw_binned = torch.floor(d_tw / self.time_bin_interval)
            
            fixed[:, :self.current_step, 0] = p_h3
            fixed[:, :self.current_step, 1] = p_tw_binned[:, :, 0]
            fixed[:, :self.current_step, 2] = p_tw_binned[:, :, 1]
            fixed[:, :self.current_step, 3] = d_h3
            fixed[:, :self.current_step, 4] = d_tw_binned[:, :, 0]
            fixed[:, :self.current_step, 5] = d_tw_binned[:, :, 1]

        if self.current_step < self.num_customers:
            req = self._slice(self.pending_requests, self.current_step)
            new = torch.zeros((self.batch_size, 1, 6), device=self.device)
            
            # Extract raw
            p_tw_raw = req["time_windows"][:, 0, :]
            d_tw_raw = req["time_windows"][:, 1, :]
            
            # Apply Binning
            p_tw_new_binned = torch.floor(p_tw_raw / self.time_bin_interval)
            d_tw_new_binned = torch.floor(d_tw_raw / self.time_bin_interval)

            new[:, 0, 0] = req["h3_indices"][:, 0]
            new[:, 0, 1] = p_tw_new_binned[:, 0]
            new[:, 0, 2] = p_tw_new_binned[:, 1]
            new[:, 0, 3] = req["h3_indices"][:, 1]
            new[:, 0, 4] = d_tw_new_binned[:, 0]
            new[:, 0, 5] = d_tw_new_binned[:, 1]
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
        appended = current.clone()
        for key in self.SLICE_FIELDS:
            if key in current.keys() and key in incoming.keys():
                 appended[key] = torch.cat([current[key], incoming[key]], dim=1)
        return appended
    
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
        if self.traveler_decisions_df is None: 
            print("TRAVELER DECISION NOT FOUND, REVERT TO RANDOM ACCEPTANCE")
            return np.random.random() < self.acceptance_rate
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