"""NDVRP Perturbation Environment.

Batched environment for training a PPO agent to find optimal time-window
perturbations for the Dial-a-Ride Problem. At each step the agent perturbs
one incoming customer request; routing cost is evaluated via the oracle policy.

Action space: 16 discrete = 4 pickup shifts × 4 dropoff shifts
Pickup shifts ∈ {-30, -20, -10, 0} minutes.
Dropoff shifts ∈ {0, 10, 20, 30} minutes.
"""

import torch
from tensordict import TensorDict
from typing import Dict

# --------------------------------------------------------------------------- #
# Action helpers
# --------------------------------------------------------------------------- #
P_SHIFT_VALUES = torch.tensor([-30, -20, -10, 0], dtype=torch.float32)
D_SHIFT_VALUES = torch.tensor([0, 10, 20, 30], dtype=torch.float32)
NUM_ACTIONS = len(P_SHIFT_VALUES) * len(D_SHIFT_VALUES)  # 16
# Pre-built index tables: action_id → (pickup_shift, dropoff_shift)
P_SHIFTS = P_SHIFT_VALUES.repeat_interleave(4)  # [16]
D_SHIFTS = D_SHIFT_VALUES.repeat(4)              # [16]
NO_PERTURBATION_ACTION = 12                      # index of (0, 0)


def action_to_shifts(actions: torch.Tensor, device: torch.device):
    """Map action indices [B] → (pickup_shift [B], dropoff_shift [B])."""
    return P_SHIFTS.to(device)[actions], D_SHIFTS.to(device)[actions]


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class NDVRPEnv:
    """Batched perturbation environment.

    Each episode presents ``num_customers`` requests one at a time.  The agent
    picks a perturbation action; the env evaluates the incremental routing cost
    using the oracle attention policy and returns
    ``reward = baseline_cost − ppo_cost``.
    """

    def __init__(
        self,
        generator,
        solver_policy,
        num_envs: int = 512,
        num_customers: int = 30,
        free_vehicles: int = 5,
        vehicle_penalty: float = 200.0,
        device: str = "cpu",
    ):
        self.generator = generator
        self.solver_policy = solver_policy
        self.num_envs = num_envs
        self.num_customers = num_customers
        self.free_vehicles = free_vehicles
        self.vehicle_penalty = vehicle_penalty
        self.device = torch.device(device)
        self.num_actions = NUM_ACTIONS

        # Lazy import to avoid circular deps
        from oracle_env import PDPTWEnv
        self.oracle_env = PDPTWEnv(
            free_vehicles=free_vehicles,
            vehicle_penalty=vehicle_penalty,
        )

    # ------------------------------------------------------------------ #
    # Reset / Step
    # ------------------------------------------------------------------ #
    def reset(self) -> Dict[str, torch.Tensor]:
        self.td = self.generator(self.num_envs).to(self.device)
        self.original_tw = self.td["time_windows"].clone()
        self.perturbed_tw = self.original_tw.clone()
        self.step_idx = 0
        self.stuck_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._build_obs()

    def step(self, actions: torch.Tensor):
        """Apply perturbation, evaluate costs, advance step.

        Returns (obs, reward, done, info).
        """
        t = self.step_idx
        p_node, d_node = 2 * t + 1, 2 * t + 2

        # Apply perturbation to current request
        p_shift, d_shift = action_to_shifts(actions, self.device)
        self.perturbed_tw[:, p_node] = torch.clamp(
            self.original_tw[:, p_node] + p_shift.unsqueeze(-1), min=0,
        )
        self.perturbed_tw[:, d_node] = torch.clamp(
            self.original_tw[:, d_node] + d_shift.unsqueeze(-1), min=0,
        )

        # Evaluate routing cost for both trajectories.
        # Baseline = same perturbed history for requests 1..t-1, but NO
        # perturbation on the current request t.  This isolates the marginal
        # effect of perturbing request t.
        n_nodes = 2 * (t + 1) + 1
        ppo_cost = self._eval_cost(self.perturbed_tw, n_nodes)

        baseline_tw = self.perturbed_tw.clone()
        baseline_tw[:, p_node] = self.original_tw[:, p_node]
        baseline_tw[:, d_node] = self.original_tw[:, d_node]
        baseline_cost = self._eval_cost(baseline_tw, n_nodes)

        reward = baseline_cost - ppo_cost

        self.step_idx += 1
        done = self.step_idx >= self.num_customers
        done_t = torch.full((self.num_envs,), done, dtype=torch.bool, device=self.device)
        obs = self._build_obs() if not done else {}
        info = {"ppo_cost": ppo_cost, "baseline_cost": baseline_cost}
        return obs, reward, done_t, info

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _build_obs(self) -> Dict[str, torch.Tensor]:
        t = self.step_idx
        n = 2 * (t + 1) + 1  # depot + requests 1..t+1
        return {
            "h3_indices": self.td["h3_indices"][:, :n],
            "travel_time_matrix": self.td["travel_time_matrix"],
            "locs": self.td["locs"][:, :n],
            "demand": self.td["demand"][:, :n],
            "time_windows": self.perturbed_tw[:, :n],
            "action_mask": self._compute_mask(),
            "step": torch.full((self.num_envs,), t, dtype=torch.long, device=self.device),
        }

    # ------------------------------------------------------------------ #
    # Feasibility mask (vectorised over 16 actions)
    # ------------------------------------------------------------------ #
    def _compute_mask(self) -> torch.Tensor:
        """Auto-mask actions where dropoff_early - pickup_late - trip_time < 15."""
        t = self.step_idx
        p_node, d_node = 2 * t + 1, 2 * t + 2

        pickup_tw = self.original_tw[:, p_node]    # [B, 2]
        dropoff_tw = self.original_tw[:, d_node]   # [B, 2]

        batch_idx = torch.arange(self.num_envs, device=self.device)
        p_h3 = self.td["h3_indices"][:, p_node]
        d_h3 = self.td["h3_indices"][:, d_node]
        trip_time = self.td["travel_time_matrix"][batch_idx, p_h3, d_h3]  # [B]

        p_shifts = P_SHIFTS.to(self.device)  # [16]
        d_shifts = D_SHIFTS.to(self.device)  # [16]

        new_pickup_late = torch.clamp(
            pickup_tw[:, 1:2] + p_shifts.unsqueeze(0), min=0,
        )  # [B, 49]
        new_dropoff_early = torch.clamp(
            dropoff_tw[:, 0:1] + d_shifts.unsqueeze(0), min=0,
        )  # [B, 49]
        return (new_dropoff_early - new_pickup_late - trip_time.unsqueeze(1)) >= 15.0

    # ------------------------------------------------------------------ #
    # Cost evaluation via oracle policy
    # ------------------------------------------------------------------ #
    def _eval_cost(self, tw: torch.Tensor, n_nodes: int) -> torch.Tensor:
        sub_td = TensorDict({
            "h3_indices": self.td["h3_indices"][:, :n_nodes],
            "travel_time_matrix": self.td["travel_time_matrix"],
            "time_windows": tw[:, :n_nodes],
            "demand": self.td["demand"][:, :n_nodes],
            "locs": self.td["locs"][:, :n_nodes],
            "capacity": self.td["capacity"],
        }, batch_size=[self.num_envs])

        reset_td = self.oracle_env.reset(sub_td)
        with torch.no_grad():
            out = self.solver_policy(
                reset_td, env=self.oracle_env, phase="val",
                calc_reward=True, return_actions=False,
                max_steps=300,
            )
        cost = -out["reward"]
        if out.get("truncated", False):
            self.stuck_mask |= ~out["done"]
        return cost
