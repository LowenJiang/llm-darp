"""
GNN-based Policy for DARP  (encoder + decoder integration).

Upper-layer module that combines:
- ``DARPEdgeEncoder``  (trial_encoder.py)  -- GREAT-style edge encoder
- ``DARPDecoder``      (trial_decoder.py)  -- attention-based decoder

Handles:
- Full autoregressive rollout with env.step()
- POMO batch expansion (multiple rollouts per instance)
- Greedy / sampling / evaluate decoding strategies
- Compatible with PDPTWEnv's TensorDict interface

The policy is invoked as::

    policy = DARPPolicy(...)
    outputs = policy(td, env, phase="train")
    # outputs["reward"]         : [B] or [B*P]
    # outputs["log_likelihood"] : [B] or [B*P]
    # outputs["actions"]        : [B, T] or [B*P, T]
"""

from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from trial_encoder import DARPEdgeEncoder
from trial_decoder import DARPDecoder, DecoderCache, gather_by_index

log = logging.getLogger(__name__)


# =========================================================================
# POMO helpers
# =========================================================================

def _batchify_td(td: TensorDict, pomo_size: int) -> TensorDict:
    """Expand a TensorDict for POMO: [B, ...] -> [B*P, ...].

    Each instance is replicated ``pomo_size`` times so that different
    starting nodes can be explored in parallel.

    Parameters
    ----------
    td : TensorDict  with batch_size [B]
    pomo_size : int

    Returns
    -------
    TensorDict with batch_size [B*P]
    """
    B = td.batch_size[0]
    new_data = {}
    for key in td.keys():
        val = td[key]
        if isinstance(val, torch.Tensor):
            # [B, ...] -> [B, P, ...] -> [B*P, ...]
            expanded = val.unsqueeze(1).expand(B, pomo_size, *val.shape[1:])
            new_data[key] = expanded.reshape(B * pomo_size, *val.shape[1:])
        else:
            # Non-tensor fields (e.g. trip_metadata lists)
            # Each element must be repeated pomo_size times consecutively
            if isinstance(val, list) and len(val) == B:
                replicated = []
                for item in val:
                    replicated.extend([item] * pomo_size)
                new_data[key] = replicated
            else:
                new_data[key] = val
    return TensorDict(new_data, batch_size=[B * pomo_size])


def _unbatchify(x: torch.Tensor, batch_size: int, pomo_size: int) -> torch.Tensor:
    """Reshape [B*P, ...] -> [B, P, ...]."""
    return x.view(batch_size, pomo_size, *x.shape[1:])


def _get_pomo_starting_nodes(num_nodes: int, pomo_size: int, device: torch.device) -> torch.Tensor:
    """Select POMO starting nodes (pickup nodes with odd indices).

    For DARP, valid first moves are pickup nodes (indices 1, 3, 5, ...).
    We cycle through them if pomo_size exceeds the number of pickup nodes.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes (including depot).
    pomo_size : int
    device : torch.device

    Returns
    -------
    starting_nodes : Tensor [P]
        Node indices for the first action of each POMO rollout.
    """
    pickup_nodes = torch.arange(1, num_nodes, 2, device=device)  # [1, 3, 5, ...]
    num_pickups = pickup_nodes.shape[0]
    if pomo_size <= num_pickups:
        # Select a spread of pickup nodes
        indices = torch.linspace(0, num_pickups - 1, pomo_size, device=device).long()
        return pickup_nodes[indices]
    else:
        # Cycle through pickup nodes
        repeats = (pomo_size + num_pickups - 1) // num_pickups
        repeated = pickup_nodes.repeat(repeats)
        return repeated[:pomo_size]


# =========================================================================
# Policy
# =========================================================================

class DARPPolicy(nn.Module):
    """Complete GNN policy for DARP combining GREAT encoder and attention decoder.

    Parameters
    ----------
    embed_dim : int
        Embedding dimension shared by encoder and decoder.
    num_encoder_layers : int
        Number of GREAT layers in the encoder.
    num_heads : int
        Number of attention heads in each GREAT layer.
    ff_hidden : int
        Feed-forward hidden dim in GREAT layers.
    dropout : float
        Dropout rate.
    tanh_clip : float
        Logit clipping for the decoder.
    max_time : float
        Time normalisation constant (default 1440 = 24h in minutes).
    num_h3 : int
        Number of H3 cells in the travel-time matrix.
    temperature : float
        Softmax temperature for sampling.
    train_decode_type : str
        Decoding strategy during training ("sampling").
    val_decode_type : str
        Decoding strategy during validation ("greedy").
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 5,
        num_heads: int = 8,
        ff_hidden: int = 512,
        dropout: float = 0.0,
        tanh_clip: float = 10.0,
        max_time: float = 1440.0,
        num_h3: int = 31,
        temperature: float = 1.0,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.tanh_clip = tanh_clip

        self.encoder = DARPEdgeEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            ff_hidden=ff_hidden,
            dropout=dropout,
            max_time=max_time,
            num_h3=num_h3,
        )

        self.decoder = DARPDecoder(
            embed_dim=embed_dim,
            tanh_clip=tanh_clip,
            max_time=max_time,
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def forward(
        self,
        td: TensorDict,
        env,
        phase: str = "train",
        pomo_size: int = 0,
        decode_type: Optional[str] = None,
        max_steps: int = 300,
        return_actions: bool = True,
        return_entropy: bool = False,
        calc_reward: bool = True,
        actions: Optional[torch.Tensor] = None,
        select_best: bool = False,
    ) -> dict:
        """Run the full encode-decode loop.

        Parameters
        ----------
        td : TensorDict
            Raw problem instance (will be reset by ``env.reset`` if not
            already initialised).
        env : PDPTWEnv
            Environment providing ``reset`` / ``step`` / ``get_reward``.
        phase : str
            ``"train"`` or ``"val"``/``"test"``.
        pomo_size : int
            If > 0, replicate each instance for POMO baseline.
        decode_type : str or None
            Override decoding strategy.
        max_steps : int
            Maximum number of decoding steps.
        return_actions : bool
            Whether to return the action sequence.
        return_entropy : bool
            Whether to compute and return entropy.
        calc_reward : bool
            Whether to compute reward after the episode.
        actions : Tensor or None
            If provided, evaluate these actions instead of sampling.
        select_best : bool
            If True and pomo_size > 0, only return the best rollout per
            instance (for validation).

        Returns
        -------
        dict with keys ``reward``, ``log_likelihood``, ``actions``, etc.
        """
        # Determine decode type
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type", "sampling")

        # --- Reset env if needed ---
        if "current_node" not in td.keys():
            td = env.reset(td)

        # Store original batch size before any POMO expansion
        original_B = td.batch_size[0]
        N = td["h3_indices"].shape[-1]

        # --- Encode (before POMO expansion) ---
        node_embeddings = self.encoder(td)  # [B, N, D]

        # --- POMO expansion ---
        using_pomo = pomo_size > 1
        if using_pomo:
            td = _batchify_td(td, pomo_size)
            td = env.reset(td)  # re-reset with expanded batch
            node_embeddings = (
                node_embeddings.unsqueeze(1)
                .expand(original_B, pomo_size, N, self.embed_dim)
                .reshape(original_B * pomo_size, N, self.embed_dim)
            )

        B_eff = td.batch_size[0]  # B or B*P

        # --- Pre-compute decoder cache ---
        cache = self.decoder.precompute(node_embeddings)
        self.decoder.reset(B_eff, node_embeddings.device)

        # --- POMO: force different starting nodes ---
        if using_pomo:
            starting_nodes = _get_pomo_starting_nodes(N, pomo_size, node_embeddings.device)
            # starting_nodes: [P] -- tile for all instances
            first_actions = starting_nodes.unsqueeze(0).expand(original_B, pomo_size).reshape(-1)  # [B*P]
        else:
            first_actions = None

        # --- Decode loop ---
        all_logprobs = []
        all_actions = []
        step = 0

        while not td["done"].all():
            if max_steps is not None and step >= max_steps:
                log.warning(
                    "Reached max_steps (%d) before all envs finished.", max_steps
                )
                break

            # Handle already-done envs
            done_mask = td["done"]

            # Get log-probabilities (temperature applied inside decoder before softmax)
            temp = self.temperature if decode_type == "sampling" else 1.0
            logprobs = self.decoder(td, cache, temperature=temp)  # [B_eff, N]

            # Select action
            if actions is not None:
                # Evaluate mode
                action = actions[..., step]
            elif using_pomo and step == 0 and first_actions is not None:
                # Force POMO starting node, but validate against mask
                action = first_actions.clone()
                mask = td["action_mask"]  # [B_eff, N]
                # Check which forced actions are valid
                forced_valid = mask.gather(1, action.unsqueeze(-1)).squeeze(-1)  # [B_eff]
                if not forced_valid.all():
                    # For invalid forced actions, fall back to sampling/greedy
                    fallback = self._select_action(logprobs, decode_type, mask)
                    action = torch.where(forced_valid, action, fallback)
            else:
                action = self._select_action(logprobs, decode_type, td["action_mask"])

            # Override done envs to go to depot
            if done_mask.any():
                action = action.clone()
                action[done_mask] = 0

            # Collect log-prob of chosen action
            selected_logprob = logprobs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
            all_logprobs.append(selected_logprob)
            all_actions.append(action)

            # Step environment
            td.set("action", action)
            td = env.step(td)["next"]

            # Maintain done status
            if done_mask.any():
                td.set("done", done_mask | td["done"])

            step += 1

        # --- Collect outputs ---
        logprobs_tensor = torch.stack(all_logprobs, dim=-1)  # [B_eff, T]
        actions_tensor  = torch.stack(all_actions, dim=-1)   # [B_eff, T]

        # Sum log-likelihood over steps
        log_likelihood = logprobs_tensor.sum(dim=-1)  # [B_eff]

        # --- Compute reward ---
        reward = None
        if calc_reward:
            reward = env.get_reward(td, actions_tensor)  # [B_eff]

        # --- Select best POMO rollout ---
        if using_pomo and select_best and reward is not None:
            reward_2d = _unbatchify(reward, original_B, pomo_size)  # [B, P]
            best_idx = reward_2d.argmax(dim=1)  # [B]
            reward = reward_2d[torch.arange(original_B), best_idx]
            ll_2d = _unbatchify(log_likelihood, original_B, pomo_size)
            log_likelihood = ll_2d[torch.arange(original_B), best_idx]
            act_2d = _unbatchify(actions_tensor, original_B, pomo_size)
            actions_tensor = act_2d[torch.arange(original_B), best_idx]

        # --- Build output dict ---
        out = {
            "log_likelihood": log_likelihood,
        }
        if reward is not None:
            out["reward"] = reward
        if return_actions:
            out["actions"] = actions_tensor
        if return_entropy:
            # Entropy from the last step's full distribution (rough estimate)
            lp = logprobs_tensor
            out["entropy"] = -(lp.exp() * lp).sum(dim=-1).mean()

        return out

    # -----------------------------------------------------------------
    # Action selection helpers
    # -----------------------------------------------------------------

    def _select_action(
        self,
        logprobs: torch.Tensor,
        decode_type: str,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Select an action according to the decoding strategy.

        Parameters
        ----------
        logprobs : Tensor [B, N]
        decode_type : str  ("sampling", "greedy")
        action_mask : Tensor [B, N]  bool

        Returns
        -------
        action : Tensor [B]
        """
        if decode_type == "greedy":
            return logprobs.argmax(dim=-1)
        elif decode_type == "sampling":
            probs = logprobs.exp()
            # Clamp to avoid numerical issues with multinomial
            probs = probs.clamp(min=1e-8)
            # Zero out masked actions (should already be ~0 from masking)
            probs = probs * action_mask.float()

            # Handle batches with no valid actions (e.g., done envs)
            has_valid_actions = action_mask.any(dim=-1)  # [B]
            if not has_valid_actions.all():
                # Initialize action tensor with depot (0) for invalid batches
                action = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
                # Only sample for batches with valid actions
                if has_valid_actions.any():
                    probs_valid = probs[has_valid_actions]
                    probs_valid = probs_valid / probs_valid.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    action[has_valid_actions] = torch.multinomial(probs_valid, num_samples=1).squeeze(-1)
                return action

            # Normal path when all batches have valid actions
            # Re-normalise
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")


# =========================================================================
# Standalone test
# =========================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_env import PDPTWEnv
    from oracle_generator import SFGenerator

    device = torch.device("cpu")
    print("=== DARPPolicy test ===")

    generator = SFGenerator()
    env = PDPTWEnv(generator=generator)
    td = generator(batch_size=[4])

    policy = DARPPolicy(
        embed_dim=128,
        num_encoder_layers=2,  # fewer layers for quick test
        num_heads=8,
    )
    policy.to(device)

    # --- Test greedy rollout ---
    print("\n--- Greedy rollout ---")
    state = env.reset(td)
    policy.eval()
    with torch.no_grad():
        out = policy(state, env, phase="val", decode_type="greedy", max_steps=200)
    print(f"  Reward: {out['reward'].tolist()}")
    print(f"  Actions shape: {out['actions'].shape}")
    print(f"  Log-likelihood: {out['log_likelihood'].tolist()}")

    # --- Test sampling rollout ---
    print("\n--- Sampling rollout ---")
    state = env.reset(td)
    policy.train()
    out_s = policy(state, env, phase="train", decode_type="sampling", max_steps=200)
    print(f"  Reward: {out_s['reward'].tolist()}")
    print(f"  Actions shape: {out_s['actions'].shape}")

    # --- Test gradient flow ---
    print("\n--- Gradient flow test ---")
    loss = -out_s["log_likelihood"].mean()
    loss.backward()
    num_params_with_grad = sum(
        1 for p in policy.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    total_params = sum(1 for p in policy.parameters())
    print(f"  Params with gradient: {num_params_with_grad}/{total_params}")
    assert num_params_with_grad > 0, "No gradients flowing!"

    # --- Test POMO ---
    print("\n--- POMO rollout (pomo_size=4) ---")
    policy.zero_grad()
    state = env.reset(td)
    policy.eval()
    with torch.no_grad():
        out_p = policy(
            state, env, phase="val", pomo_size=4,
            decode_type="greedy", max_steps=200, select_best=True,
        )
    print(f"  Best reward: {out_p['reward'].tolist()}")
    print(f"  Actions shape: {out_p['actions'].shape}")

    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\n  Total param count: {total_params:,}")

    print("\nAll policy tests passed.")
