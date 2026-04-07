"""
Dynamic Vehicle Routing Problem with Time Windows (DVRP-TW) Environment

A ride-sharing dispatcher receives trip requests one at a time and negotiates
time window shifts before routing.  The RL agent picks one of 16 shift
combinations (pickup earlier × dropoff later).  Travelers accept or reject
based on precomputed flexibility decisions.  Routing cost is evaluated by a
neural DARP solver or OR-Tools.

State (TensorDict):
    fixed:           [batch, num_customers, 6]   — accepted requests so far
    new:             [batch, 1, 6]               — incoming request
    distance_matrix: [batch, num_locs, num_locs] — travel times
    step:            [batch, 1]

Action: int in [0, 15]  →  (pickup_shift, dropoff_shift) from ACTION_MAP

Reward: previous_cost − new_cost − patience_penalty
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from oracle_env import PDPTWEnv
from oracle_generator import SFGenerator
from oracle_policy import PDPTWAttentionPolicy
from or_tools import darp_solver
from tensordict.tensordict import TensorDict
from dvrp_ppo_agent_gcn import PPOAgent
# Node-indexed TensorDict fields that need slicing / concatenation
NODE_FIELDS = {
    "h3_indices", "time_windows", "demand", "locs",
    "action_mask", "visited", "completed", "user_id",
}

ACTION_MAP = [
    (-30, 0), (-30, 10), (-30, 20), (-30, 30),
    (-20, 0), (-20, 10), (-20, 20), (-20, 30),
    (-10, 0), (-10, 10), (-10, 20), (-10, 30),
    (  0, 0), (  0, 10), (  0, 20), (  0, 30),
]

# Order must match embedding.py:21-26 for correct mask lookup
FLEX_ORDER_FOR_MASKS = [
    "flexible for late dropoff, but inflexible for early pickup",
    "flexible for early pickup, but inflexible for late dropoff",
    "inflexible for any schedule changes",
    "flexible for both early pickup and late dropoff",
]


def _slice_nodes(td: TensorDict, indices: list[int]) -> TensorDict:
    """Return a TensorDict with node-indexed fields sliced to `indices`."""
    out = td.clone()
    for key in NODE_FIELDS:
        if key in td.keys():
            out[key] = td[key][:, indices]
    return out


def _concat_nodes(base: TensorDict, addition: TensorDict) -> TensorDict:
    """Append node-indexed fields from `addition` onto `base` along dim 1."""
    out = base.clone()
    for key in NODE_FIELDS:
        if key in base.keys() and key in addition.keys():
            out[key] = torch.cat([base[key], addition[key]], dim=1)
    return out


def _reset_dynamic_fields(td: TensorDict) -> TensorDict:
    """Zero out mutable routing state so the solver starts fresh."""
    out = td.clone()
    for key in ("visited", "completed", "used_capacity", "current_node",
                "i", "pending_schedule", "pending_count"):
        if key in out.keys():
            out[key] = torch.zeros_like(out[key])
    return out


def _unwrap_metadata(trip_md, batch_idx: int):
    """Extract the per-traveler metadata dict for a single batch element."""
    if trip_md is None:
        return None
    md = trip_md
    if not isinstance(md, dict) and hasattr(md, "__len__"):
        md = md[batch_idx] if batch_idx < len(md) else {}
    if hasattr(md, "data"):
        md = md.data
    return md if isinstance(md, dict) else None


class DVRPEnv:
    """Batched Dynamic Vehicle Routing with Time Windows environment."""

    def __init__(
        self,
        num_customers: int = 30,
        max_vehicles: int = 5,
        solver_time_limit: int = 1,
        acceptance_rate: float = 0,
        seed: Optional[int] = None,
        patience_factor: float = 0,
        batch_size: int = 1,
        model_path: Optional[str] = None,
        traveler_decisions_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.num_customers = num_customers
        self.max_vehicles = max_vehicles
        self.solver_time_limit = solver_time_limit
        self.acceptance_rate = acceptance_rate
        self.patience_factor = patience_factor
        self.batch_size = int(batch_size)
        self.device = device
        self.action_tensor = torch.tensor(ACTION_MAP, device=device)
        self.upcoming = None
        # Data generator
        csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
        ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
        generator = SFGenerator(
            csv_path=csv_path, travel_time_matrix_path=ttm_path,
            num_customers=num_customers, seed=seed,
        )
        self.data_generator = PDPTWEnv(generator=generator)
        self.num_distinct_locations = generator.travel_time_matrix.shape[0]

        # Episode state (set properly in reset())
        self.pending_requests: Optional[TensorDict] = None
        self.current_requests: Optional[TensorDict] = None
        self.real_requests: Optional[TensorDict] = None
        self.current_step = 0
        self.previous_cost = torch.zeros(batch_size, device=device)

        # Neural routing policy (optional)
        self.policy = None
        if model_path is not None:
            path = Path(model_path)
            if path.exists():
                torch.serialization.add_safe_globals([PDPTWEnv])
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                state = ckpt.get("policy_state_dict", ckpt.get("state_dict", ckpt))
                model = PDPTWAttentionPolicy()
                model.load_state_dict(state, strict=False)
                self.policy = model.to(device).eval()
            else:
                print(f"Warning: model_path not found ({path}); running without policy.")

        # Traveler decision table
        if traveler_decisions_path is None:
            traveler_decisions_path = str(Path(__file__).with_name("traveler_decisions_augmented.csv"))
        df = pd.read_csv(traveler_decisions_path)

        # Build accept/reject lookup: (id, purpose, dep, arr, |p_shift|, |d_shift|) → {flex → bool}
        flex_cols = [
            "flexible for both early pickup and late dropoff",
            "flexible for early pickup, but inflexible for late dropoff",
            "flexible for late dropoff, but inflexible for early pickup",
            "inflexible for any schedule changes",
        ]
        self.decision_lookup = {}
        for _, row in df.iterrows():
            key = (int(row["traveler_id"]), row["trip_purpose"].strip(),
                   row["departure_location"].strip(), row["arrival_location"].strip(),
                   int(row["pickup_shift_min"]), int(row["dropoff_shift_min"]))
            self.decision_lookup[key] = row[row["flexibility"]]=="accept"
                #col: str(row[col]).strip().lower() == "accept" for col in flex_cols #! def wrong
                
            

        # Build per-flexibility action masks (precomputed for get_mask)
        index_cols = ["traveler_id", "trip_purpose", "departure_location",
                      "arrival_location", "departure_time_window", "arrival_time_window"]
        action_cols = ["pickup_shift_min", "dropoff_shift_min"]
        col_order = [(abs(p), abs(d)) for p, d in ACTION_MAP]

        self.df_mask = None
        for i, flex in enumerate(FLEX_ORDER_FOR_MASKS):
            sub = df[index_cols + action_cols + [flex]].copy()
            sub["indicator"] = (sub[flex] == "accept").astype(int)
            wide = (sub.pivot_table(index=index_cols, columns=action_cols,
                                    values="indicator", fill_value=0)
                    .reindex(columns=col_order, fill_value=0))
            wide[i] = wide.apply(lambda row: row.values.tolist(), axis=1)
            result = wide[[i]].reset_index()
            if isinstance(result.columns, pd.MultiIndex):
                result.columns = result.columns.get_level_values(0)
            result.columns.name = None
            self.df_mask = result if self.df_mask is None else self.df_mask.merge(result, on=index_cols)

        # Free the raw DataFrame — only lookup and df_mask are needed at runtime
        del df

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[TensorDict, dict]:
        if seed is not None:
            np.random.seed(seed)
        self.pending_requests = self.data_generator.reset(batch_size=[self.batch_size]).to(self.device)
        depot = _slice_nodes(self.pending_requests, [0])
        self.current_requests = depot.clone()
        self.real_requests = depot.clone() #! Not used
        self.current_step = 0
        pd_idx = [2 * self.current_step + 1, 2 * self.current_step + 2]
        self.upcoming = _slice_nodes(self.pending_requests, pd_idx)

        self.previous_cost = torch.zeros(self.batch_size, device=self.device)


        return self._get_observation(), {"total_requests": self.num_customers, "current_cost": self.previous_cost}

    def step(self, action: Union[int, torch.Tensor, np.ndarray]):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.long)
        if action.dim() == 0:
            action = action.unsqueeze(0).expand(self.batch_size)

        pd_idx = [2 * self.current_step + 1, 2 * self.current_step + 2]
        new_request = _slice_nodes(self.pending_requests, pd_idx)
        #!
        #pd_indx_next= [2 * self.current_step + 3, 2 * self.current_step + 4]
        #self.upcoming=_slice_nodes(self.pending_requests, pd_indx_next)

        shifts = self.action_tensor[action]  # [B, 2]
        pickup_shifts, dropoff_shifts = shifts[:, 0], shifts[:, 1]

        # --- Acceptance ---
        accepted = self._get_accepted(pickup_shifts, dropoff_shifts) #! Very Important, check later

        # Zero out shifts for rejections; compute patience penalty
        final_p = pickup_shifts * accepted.float()
        final_d = dropoff_shifts * accepted.float()
        patience = (final_p.abs() + final_d.abs()) * self.patience_factor

        # --- Apply perturbation inline ---
        perturbed = new_request.clone()
        
        tw = perturbed["time_windows"].clone()

        tw[:, 0, :] += final_p.view(-1, 1) #? Is this dimension correct? Isnt the second dim the node? 
        tw[:, 1, :] += final_d.view(-1, 1)
        perturbed["time_windows"] = tw.clamp(min=0)

        # --- Update route state ---
        current_perturbed = _concat_nodes(self.current_requests, perturbed)
        #current_baseline = _concat_nodes(self.current_requests, new_request)
        self.current_requests = current_perturbed
        #self.real_requests = _concat_nodes(self.real_requests, new_request) #! No need for such baseline

        # --- Evaluate routing costs ---
        new_costs = self._evaluate_costs(_reset_dynamic_fields(current_perturbed))

        # base_costs, _ = self._evaluate_costs(_reset_dynamic_fields(current_baseline))
        # rewards_baseline = self.previous_cost

        rewards = self.previous_cost - new_costs
        self.previous_cost = new_costs
        self.current_step += 1
        done = self.current_step >= self.num_customers
        state = self._get_observation() if done == False else None
        #! The _get_observation should entail current request AND new request (unperturbed)
        return (
            state,
            rewards,
            accepted.float(),
            torch.full((self.batch_size,), done, dtype=torch.bool, device=self.device),
            {"step": self.current_step, "current_cost": new_costs,
             "accepted": accepted},
        )

    # ------------------------------------------------------------------
    # Acceptance logic
    # ------------------------------------------------------------------

    def _get_accepted(self, pickup_shifts, dropoff_shifts) -> torch.Tensor:
        """Determine accept/reject for each batch element using the decision lookup."""
        accepted = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        pickup_idx = 2 * self.current_step + 1

        user_ids = (self.pending_requests["user_id"][:, pickup_idx]
                    if "user_id" in self.pending_requests.keys()
                    else torch.arange(1, self.batch_size + 1, device=self.device))

        #trip_md = self.pending_requests.get("trip_metadata", None)    
        trip_md = self.pending_requests["trip_metadata"]    
        #print(trip_md)
        for b in range(self.batch_size): 
            uid = int(user_ids[b].item()) 
            meta = _unwrap_metadata(trip_md, b)

            if meta is None:
                print("meta not fonud")
            user_meta = meta.get(uid)
            if user_meta is None or "flexibility" not in user_meta:
                print("user meta not found")

            key = (uid, user_meta.get("trip_purpose", "").strip(),
                   user_meta.get("departure_location", "").strip(),
                   user_meta.get("arrival_location", "").strip(),
                   abs(int(pickup_shifts[b].item())),
                   abs(int(dropoff_shifts[b].item())))
            
            decision = self.decision_lookup.get(key) #! Here directly query is ok

            if decision is None:
                print(f"WARNING: Flexibility '{user_meta['flexibility']}' not in CSV")
                accepted[b] = np.random.random() < self.acceptance_rate
                continue
            #! decision is a set
            accepted[b] = decision 

        return accepted

    # ------------------------------------------------------------------
    # Cost evaluation
    # ------------------------------------------------------------------

    def _evaluate_costs(self, solver_td: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
        costs = torch.zeros(self.batch_size, device=self.device)
        #failures = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        if self.policy is not None:
            with torch.no_grad():
                out = self.policy(solver_td.to(self.device), env=self.data_generator,
                                  phase="test", decode_type="greedy", return_actions=True)
            return -out["reward"] #! Note the sign

        for b in range(self.batch_size):
            result = darp_solver(solver_td[b:b+1], max_vehicles=self.max_vehicles,
                                 time_limit_seconds=self.solver_time_limit)
            c = float(result["total_time"])
            if c == float("inf"):
                failures[b] = True
            costs[b] = c

        return costs

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> TensorDict:
        B, N, s = self.batch_size, self.num_customers, self.current_step
        #! Core change
        state = torch.zeros((B, N, 6), device=self.device)

        if s > 0:
            p_idx = torch.arange(1, 2 * s + 1, 2, device=self.device)
            d_idx = torch.arange(2, 2 * s + 2, 2, device=self.device)
            state[:, :s, 0] = self.current_requests["h3_indices"][:, p_idx]
            state[:, :s, 1] = self.current_requests["time_windows"][:, p_idx, 0]
            state[:, :s, 2] = self.current_requests["time_windows"][:, p_idx, 1]
            state[:, :s, 3] = self.current_requests["h3_indices"][:, d_idx]
            state[:, :s, 4] = self.current_requests["time_windows"][:, d_idx, 0]
            state[:, :s, 5] = self.current_requests["time_windows"][:, d_idx, 1]

        req = _slice_nodes(self.pending_requests, [2 * s + 1, 2 * s + 2])
        state[:,s,0] = req["h3_indices"][:,0]
        state[:,s,1] = req["time_windows"][:,0,0]
        state[:,s,2] = req["time_windows"][:,0,1]
        state[:,s,3] = req["h3_indices"][:,1]
        state[:,s,4] = req["time_windows"][:,1,0]
        state[:,s,5] = req["time_windows"][:,1,1]

        #! This should be constructed as the state 
        #! Todo: Make sure the h3_indice becomes a valid input for state in the TripRequestEmbeddingModel
        #print("Example State: ")
        #print(state)
        return state

    # ------------------------------------------------------------------
    # Public helpers (called by training loop / PPO)
    # ------------------------------------------------------------------

    def get_current_user_id(self) -> int:
        if "user_id" in self.pending_requests.keys():
            return self.pending_requests["user_id"][0, 2 * self.current_step + 1].item()
        return self.current_step

    def get_real_requests(self) -> TensorDict:
        return self.real_requests

    def get_mask(self, traveler_id: int, predicted_flex_index) -> list:
        if isinstance(predicted_flex_index, torch.Tensor):
            predicted_flex_index = int(predicted_flex_index.item())
        meta = _unwrap_metadata(self.pending_requests.get("trip_metadata", None), 0)
        info = meta[traveler_id]
        sel = (
            (self.df_mask["traveler_id"] == traveler_id)
            & (self.df_mask["trip_purpose"] == info["trip_purpose"])
            & (self.df_mask["departure_location"] == info["departure_location"])
            & (self.df_mask["arrival_location"] == info["arrival_location"])
            & (self.df_mask["departure_time_window"] == info["departure_time_window"])
            & (self.df_mask["arrival_time_window"] == info["arrival_time_window"])
        )
        matches = self.df_mask.loc[sel, predicted_flex_index]
        return matches.iloc[0] if len(matches) > 0 else [1] * 16


# ----------------------------------------------------------------------

def test_env():
    decisions_path = Path(__file__).with_name("traveler_decisions_augmented.csv")

    device = 'cpu'
    model_path = "checkpoints/refined/best.pt"
    num_customers = 30
    num_envs=2
    env = DVRPEnv(num_customers=num_customers, batch_size=num_envs, 
                  traveler_decisions_path=str(decisions_path), model_path=model_path)
    
    tt_matrix = env.data_generator.generator.travel_time_matrix.to(device)
    agent = PPOAgent(
        travel_time_matrix=tt_matrix,
        gcn_hidden=16,
        gcn_out=16,
        time_embed_dim=8,
        time_vocab_size=50,
        transformer_embed_dim=32,
        action_dim=5,
        hidden_dim=64,
        num_heads=2,
        num_layers=2,
        device="cpu"
    )
    baseline = env
    
    obs, info = env.reset()
    #print(obs)
    #print(f"fixed: {obs['fixed'].shape}, new: {obs['new'].shape}")
    rl = []
    acc = []
    for i in range(num_customers):
        #action = torch.randint(0, 16, (env.batch_size,))
        action = agent.select_action(obs)
        obs, reward, accepted, done, info = env.step(action)
        rl.append(reward)
        acc.append(accepted)
        print(f"Step {i+1}: reward={reward.tolist()}, accepted={info['accepted'].tolist()}, action={action}")
        if done.any():
            break

    final_cost = np.array(rl).sum(axis=0)
    baseline_cost = np.array(baseline._evaluate_costs(baseline.pending_requests))
    print(final_cost)
    print(f"acceptance rate: {np.array(acc).sum(axis=0)/num_customers/num_envs}")
    print(f"Improvement Rate: {(final_cost + baseline_cost)/baseline_cost*100} %")
    


if __name__ == "__main__":
    test_env()