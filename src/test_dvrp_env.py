"""
TestDVRPEnv – DVRPEnv subclass with a 49-action space.

Action space: {-30, -20, -10, 0, 10, 20, 30} x {-30, -20, -10, 0, 10, 20, 30}
for (pickup_shift, dropoff_shift), giving 49 discrete actions total.

Key additions over DVRPEnv:
  - 49-action ACTION_SPACE_MAP built from the full Cartesian product of shift steps
  - get_feasibility_mask(): per-batch boolean mask [B, 49] based on the
    serviceability constraint from comparison_save.py's random_perturbation()
  - get_encoder_input(): returns current_requests as a solver-ready TensorDict
    suitable for the oracle PDPTWAttentionPolicy encoder
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

# Ensure the src/ directory is importable
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from gymnasium import spaces
from tensordict.tensordict import TensorDict

from dvrp_env import DVRPEnv


# ---------------------------------------------------------------------------
# Action space constants
# ---------------------------------------------------------------------------

# The seven discrete shift magnitudes (minutes), matching comparison_save.py
_SHIFT_STEPS = [-30, -20, -10, 0, 10, 20, 30]

# All 49 (pickup_shift, dropoff_shift) pairs in row-major order over _SHIFT_STEPS
ACTION_SPACE_MAP: list[Tuple[int, int]] = [
    (ps, ds) for ps in _SHIFT_STEPS for ds in _SHIFT_STEPS
]

# Total number of actions
_NUM_ACTIONS = len(ACTION_SPACE_MAP)  # 49

# Clamp bounds for time window values (minutes from midnight)
_TW_MIN = 30.0
_TW_MAX = 1410.0

# Minimum gap between (clamped) dropoff early and (clamped) pickup late
_MIN_GAP_EXTRA = 15.0


class TestDVRPEnv(DVRPEnv):
    """
    DVRP environment with an expanded 49-action space.

    Inherits all logic from DVRPEnv and overrides only what is necessary
    to support the larger action space and add feasibility masking plus
    oracle encoder input helpers.

    Action Space:
        Discrete(49): indices 0..48 map to ACTION_SPACE_MAP entries.
        Index i = row * 7 + col where row indexes pickup shift and col
        indexes dropoff shift over _SHIFT_STEPS = [-30,-20,-10,0,10,20,30].

    New methods:
        get_feasibility_mask(step): [B, 49] bool tensor, True = feasible
        get_encoder_input():        solver-ready TensorDict for oracle encoder
    """

    # Override with the expanded action-space map
    ACTION_SPACE_MAP = ACTION_SPACE_MAP  # type: ignore[assignment]

    def __init__(
        self,
        num_customers: int = 30,
        max_vehicles: int = 5,
        solver_time_limit: int = 1,
        acceptance_rate: float = 0.5,
        depot=None,
        seed: Optional[int] = None,
        patience_factor: float = 0.002,
        batch_size: int = 1,
        model_path: Optional[str] = None,
        traveler_decisions_path: Optional[str] = None,
        device: str = "cpu",
        force_accept: bool = False,
        action_dim: int = 49,
    ):
        super().__init__(
            num_customers=num_customers,
            max_vehicles=max_vehicles,
            solver_time_limit=solver_time_limit,
            acceptance_rate=acceptance_rate,
            depot=depot,
            seed=seed,
            patience_factor=patience_factor,
            batch_size=batch_size,
            model_path=model_path,
            traveler_decisions_path=traveler_decisions_path,
            device=device,
            force_accept=force_accept,
        )

        # Override the action-space map and derived tensors
        self.ACTION_SPACE_MAP = ACTION_SPACE_MAP
        self.action_tensor = torch.tensor(
            self.ACTION_SPACE_MAP, dtype=torch.long, device=self.device
        )
        self.action_space = spaces.Discrete(_NUM_ACTIONS)
        self.action_dim = _NUM_ACTIONS

        # Recompute the df_mask with the 49-action ordering
        if self.traveler_decisions_df is not None:
            self.df_mask = self._compute_mask_all_flexs()

    # ------------------------------------------------------------------
    # Override _compute_mask_single_flex for 49-action ordering
    # ------------------------------------------------------------------

    def _compute_mask_single_flex(self, index_cols, action_cols, flexibility):
        import pandas as pd

        df = self.traveler_decisions_df[index_cols + action_cols + [flexibility]].copy()
        df["indicator"] = (df[flexibility] == "accept").astype(int)

        wide = df.pivot_table(
            index=index_cols,
            columns=action_cols,
            values="indicator",
            fill_value=0,
        )

        # Build 49 column keys using absolute shift magnitudes
        strict_ordering = [
            (abs(ps), abs(ds)) for ps, ds in self.ACTION_SPACE_MAP
        ]
        wide = wide.reindex(columns=strict_ordering, fill_value=0)
        wide[flexibility] = wide.apply(lambda row: row.values.tolist(), axis=1)

        result = wide[[flexibility]].reset_index()
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = result.columns.get_level_values(0)
        result.columns.name = None

        return result

    # ------------------------------------------------------------------
    # Override _get_mask_from_flex with length-49 default
    # ------------------------------------------------------------------

    def _get_mask_from_flex(
        self, traveler_id, trip_purpose, departure_location, arrival_location,
        departure_time_window, arrival_time_window, predicted_flex_index,
    ):
        if isinstance(predicted_flex_index, torch.Tensor):
            predicted_flex_index = int(predicted_flex_index.item())

        selection = (
            (self.df_mask["traveler_id"] == traveler_id)
            & (self.df_mask["trip_purpose"] == trip_purpose)
            & (self.df_mask["departure_location"] == departure_location)
            & (self.df_mask["arrival_location"] == arrival_location)
            & (self.df_mask["departure_time_window"] == departure_time_window)
            & (self.df_mask["arrival_time_window"] == arrival_time_window)
        )
        mask_series = self.df_mask[selection][predicted_flex_index]
        mask = (
            mask_series.iloc[0]
            if len(mask_series) > 0
            else [1] * _NUM_ACTIONS
        )
        return mask

    # ------------------------------------------------------------------
    # get_feasibility_mask
    # ------------------------------------------------------------------

    def get_feasibility_mask(self, step: int) -> torch.Tensor:
        """
        Compute a [batch_size, 49] boolean mask of feasible actions.

        An action (ps, ds) is feasible if after clamping to [30, 1410]:
            clamp(dropoff_early + ds) - clamp(pickup_late + ps) >= travel_time + 15
        """
        if self.pending_requests is None:
            raise RuntimeError("Call reset() before get_feasibility_mask().")

        req = self._slice(self.pending_requests, step)
        tw = req["time_windows"]          # [B, 2, 2]
        pickup_late  = tw[:, 0, 1]        # [B]
        dropoff_early = tw[:, 1, 0]       # [B]

        h3 = req["h3_indices"]            # [B, 2]
        pickup_h3   = h3[:, 0].long()
        dropoff_h3  = h3[:, 1].long()
        ttm = self.pending_requests["travel_time_matrix"]  # [B, H, H]

        pickup_rows = ttm.gather(
            1, pickup_h3.view(-1, 1, 1).expand(-1, 1, ttm.size(2))
        ).squeeze(1)
        travel_times = pickup_rows.gather(1, dropoff_h3.view(-1, 1)).squeeze(1)
        min_gap = travel_times + _MIN_GAP_EXTRA

        action_ps = self.action_tensor[:, 0].float()
        action_ds = self.action_tensor[:, 1].float()

        new_lp = torch.clamp(
            pickup_late.unsqueeze(1) + action_ps.unsqueeze(0),
            min=_TW_MIN, max=_TW_MAX,
        )
        new_ed = torch.clamp(
            dropoff_early.unsqueeze(1) + action_ds.unsqueeze(0),
            min=_TW_MIN, max=_TW_MAX,
        )
        feasible = (new_ed - new_lp) >= min_gap.unsqueeze(1)
        return feasible

    # ------------------------------------------------------------------
    # get_encoder_input
    # ------------------------------------------------------------------

    def get_encoder_input(self) -> TensorDict:
        """Return current_requests + current pending request as a solver-ready TensorDict.

        The oracle encoder (PDPTWInitEmbedding) requires at least one
        pickup-delivery pair beyond the depot to build partner indices.
        We always append the current pending request so the encoder sees
        the request being decided on, even at step 0.
        """
        if self.current_requests is None:
            raise RuntimeError("Call reset() before get_encoder_input().")

        # Append the current pending request so the encoder always has
        # at least depot + 1 pickup + 1 delivery = 3 nodes.
        if self.current_step < self.num_customers:
            pending_req = self._slice(self.pending_requests, self.current_step)
            combined = self._append_to_current(self.current_requests, pending_req)
        else:
            combined = self.current_requests

        return self._get_solver_input(combined)

    # ------------------------------------------------------------------
    # step override
    # ------------------------------------------------------------------

    def step(
        self, action: Union[int, torch.Tensor, np.ndarray],
    ) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Execute one step with the 49-action space."""
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0).expand(self.batch_size)
        return super().step(action)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env = TestDVRPEnv(num_customers=5, batch_size=2, seed=42, model_path=None)
    obs, info = env.reset()
    print(f"Obs keys: {list(obs.keys())}")
    print(f"Action dim: {env.action_dim}")

    mask = env.get_feasibility_mask(step=0)
    print(f"Feasibility mask shape: {mask.shape}, feasible: {mask.sum(dim=1)}")

    action = torch.full((2,), 24, dtype=torch.long)  # no-op (0, 0)
    obs, reward, term, trunc, info = env.step(action)
    print(f"Step reward: {reward}")

    enc_input = env.get_encoder_input()
    print(f"Encoder input keys: {list(enc_input.keys())}")
    print("Test passed!")
