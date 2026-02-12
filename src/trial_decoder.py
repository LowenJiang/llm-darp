"""
Attention-based Decoder for DARP.

Architecture overview
---------------------
The decoder produces action probabilities at each step of the autoregressive
construction.  It receives pre-computed node embeddings from the encoder and
combines several context signals into a query vector:

    query = W_last(h_current) + W_first(h_first) + W_graph(h_mean)
            + W_visited(h_visited_mean) + W_state(state_features)

Scores are computed via scaled dot-product attention between the query and
pre-computed node keys, with a travel-time distance bias and tanh clipping.
The env's ``action_mask`` is applied before the final softmax.

Design choices (from planner):
- Use env's action_mask directly -- no custom masking logic.
- Handle node retraction by always using the current ``td["visited"]``.
- Distance bias: subtract ``travel_time / sqrt(2)`` before clipping.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Utility  (self-contained copy for module independence)
# ---------------------------------------------------------------------------

def gather_by_index(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    squeeze: bool = True,
) -> torch.Tensor:
    """Gather values from *source* along *dim* using *index*."""
    if index.dim() == 1:
        index = index.unsqueeze(-1)
    if dim == 1:
        if source.dim() == index.dim():
            result = source.gather(dim, index)
        else:
            expanded_index = index.unsqueeze(-1).expand(*index.shape, source.shape[-1])
            result = source.gather(dim, expanded_index)
    elif dim == 2:
        if index.dim() == source.dim() - 1:
            index = index.unsqueeze(1).expand(-1, source.shape[1], *index.shape[1:])
        result = source.gather(dim, index)
    else:
        expanded_index = index.expand(*source.shape[:-1], index.shape[-1])
        result = source.gather(dim, expanded_index)
    if squeeze and result.shape[dim] == 1:
        result = result.squeeze(dim)
    return result


# =========================================================================
# Pre-computed cache (set once per episode from encoder output)
# =========================================================================

class DecoderCache:
    """Immutable container for pre-computed decoder tensors.

    Created once per episode (or POMO expansion) by ``DARPDecoder.precompute``.

    Attributes
    ----------
    node_embeddings : Tensor [B, N, D]
        Raw node embeddings from the encoder.
    K : Tensor [B, N, D]
        Pre-computed key projections.
    graph_emb : Tensor [B, D]
        Mean-pooled graph embedding.
    """

    __slots__ = ("node_embeddings", "K", "graph_emb")

    def __init__(
        self,
        node_embeddings: torch.Tensor,
        K: torch.Tensor,
        graph_emb: torch.Tensor,
    ) -> None:
        self.node_embeddings = node_embeddings
        self.K = K
        self.graph_emb = graph_emb

    def expand(self, pomo_size: int) -> "DecoderCache":
        """Expand the cache for POMO: [B, ...] -> [B*P, ...]."""
        B = self.node_embeddings.shape[0]

        def _expand(t: torch.Tensor) -> torch.Tensor:
            return (
                t.unsqueeze(1)
                .expand(B, pomo_size, *t.shape[1:])
                .reshape(B * pomo_size, *t.shape[1:])
            )

        return DecoderCache(
            node_embeddings=_expand(self.node_embeddings),
            K=_expand(self.K),
            graph_emb=_expand(self.graph_emb),
        )


# =========================================================================
# Decoder
# =========================================================================

class DARPDecoder(nn.Module):
    """Autoregressive decoder for the DARP policy.

    At each decoding step the decoder:

    1. Builds a context *query* from:
       - embedding of the current node
       - embedding of the first node visited in the current vehicle route
       - mean embedding of all nodes (graph context)
       - mean embedding of already-visited nodes
       - scalar state features (capacity, time, step)

    2. Scores every candidate node via ``q @ K^T / sqrt(D)`` with a
       travel-time distance bias, followed by tanh clipping.

    3. Masks infeasible nodes using ``td["action_mask"]`` and returns a
       probability distribution (softmax).

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (must match encoder output).
    tanh_clip : float
        Logit clipping constant (applied as ``C * tanh(score / C)``).
    max_time : float
        Normalisation constant for travel-time distance bias.
    state_dim : int
        Number of scalar state features (default 3: remaining_cap,
        current_time, step).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        tanh_clip: float = 10.0,
        max_time: float = 1440.0,
        state_dim: int = 3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.tanh_clip = tanh_clip
        self.max_time = max_time

        # Context projections (all project to embed_dim)
        self.W_last    = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_first   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_graph   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_visited = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_state   = nn.Linear(state_dim, embed_dim, bias=True)

        # Key projection (applied once during precompute)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias=False)

        # Track the first non-depot node of the current vehicle route
        self._first_node: Optional[torch.Tensor] = None  # [B] or [B*P]

        self._init_parameters()

    # -----------------------------------------------------------------
    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def precompute(self, node_embeddings: torch.Tensor) -> DecoderCache:
        """Pre-compute keys and graph embedding.

        Parameters
        ----------
        node_embeddings : Tensor [B, N, D]

        Returns
        -------
        DecoderCache
        """
        K = self.W_key(node_embeddings)                   # [B, N, D]
        graph_emb = node_embeddings.mean(dim=1)           # [B, D]
        return DecoderCache(
            node_embeddings=node_embeddings,
            K=K,
            graph_emb=graph_emb,
        )

    # -----------------------------------------------------------------
    def reset(self, batch_size: int, device: torch.device) -> None:
        """Reset decoder state at the start of an episode.

        Sets ``_first_node`` to zeros (depot).
        """
        self._first_node = torch.zeros(batch_size, dtype=torch.long, device=device)

    # -----------------------------------------------------------------
    def _update_first_node(
        self,
        current_node: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> None:
        """Update the tracked first-node of the current vehicle segment.

        If the vehicle just departed the depot (previous action was depot or
        this is the first real move), set first_node to the current node.
        If the vehicle returned to depot, reset first_node.

        Parameters
        ----------
        current_node : Tensor [B] or [B, 1]
            Node that was just chosen.
        previous_action : Tensor [B] or [B, 1]
            The node visited in the previous step.
        """
        cur = current_node.squeeze(-1)    # [B]
        prev = previous_action.squeeze(-1)  # [B]

        # Vehicle just left depot -> set first_node
        left_depot = (prev == 0) & (cur != 0)
        self._first_node = torch.where(left_depot, cur, self._first_node)

        # Returned to depot -> reset first_node to 0
        at_depot = cur == 0
        self._first_node = torch.where(at_depot, torch.zeros_like(self._first_node), self._first_node)

    # -----------------------------------------------------------------
    def forward(
        self,
        td: TensorDict,
        cache: DecoderCache,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute action log-probabilities for the current decoding step.

        Parameters
        ----------
        td : TensorDict
            Current environment state.  Must contain ``current_node``,
            ``action_mask``, ``visited``, ``current_time``,
            ``used_capacity``, ``vehicle_capacity``, ``i``,
            ``previous_action``, ``h3_indices``, ``travel_time_matrix``.
        cache : DecoderCache
            Pre-computed node keys and graph embedding.
        temperature : float
            Softmax temperature. Applied to scores before the softmax.
            Default 1.0 (no temperature scaling).

        Returns
        -------
        logprobs : Tensor [B, N]
            Log-probabilities over all N nodes.
        """
        node_emb = cache.node_embeddings            # [B, N, D]
        K        = cache.K                           # [B, N, D]
        B, N, D  = node_emb.shape
        device   = node_emb.device

        # --- gather context embeddings ---
        current_node   = td["current_node"].squeeze(-1)          # [B]
        previous_action = td["previous_action"].squeeze(-1)      # [B]
        visited        = td["visited"]                           # [B, N] bool
        action_mask    = td["action_mask"]                       # [B, N] bool

        # Update first-node tracker
        self._update_first_node(current_node, previous_action)

        # h_current: embedding of the current node
        h_current = gather_by_index(node_emb, current_node.unsqueeze(-1), dim=1)  # [B, D]

        # h_first: embedding of the first node of the current segment
        h_first = gather_by_index(node_emb, self._first_node.unsqueeze(-1), dim=1)  # [B, D]

        # h_graph: mean of all node embeddings (pre-computed)
        h_graph = cache.graph_emb  # [B, D]

        # h_visited: mean of visited node embeddings
        visited_float = visited.float()                           # [B, N]
        visited_count = visited_float.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, 1]
        h_visited = (node_emb * visited_float.unsqueeze(-1)).sum(dim=1) / visited_count  # [B, D]

        # state features: remaining capacity, current time (normalised), step
        remaining_cap = (td["vehicle_capacity"] - td["used_capacity"]).squeeze(-1)  # [B]
        current_time  = td["current_time"].squeeze(-1) / self.max_time              # [B]
        step_feat     = td["i"].float().squeeze(-1) / (N * 2)                       # [B]
        state_feats = torch.stack([remaining_cap, current_time, step_feat], dim=-1)  # [B, 3]

        # --- build query ---
        q = (
            self.W_last(h_current)
            + self.W_first(h_first)
            + self.W_graph(h_graph)
            + self.W_visited(h_visited)
            + self.W_state(state_feats)
        )  # [B, D]

        # --- score nodes via dot-product ---
        # score[b, n] = q[b, :] . K[b, n, :] / sqrt(D)
        score = torch.bmm(q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / math.sqrt(D)
        # [B, N]

        # --- distance bias: subtract normalised travel time to each candidate ---
        travel_bias = self._compute_travel_bias(td, current_node)  # [B, N]
        score = score - travel_bias

        # --- tanh clipping ---
        if self.tanh_clip > 0:
            score = self.tanh_clip * torch.tanh(score / self.tanh_clip)

        # --- temperature scaling (applied before softmax) ---
        if temperature != 1.0:
            score = score / temperature

        # --- mask and softmax ---
        score = score.masked_fill(~action_mask, -1e8)
        logprobs = F.log_softmax(score, dim=-1)

        return logprobs  # [B, N]

    # -----------------------------------------------------------------
    def _compute_travel_bias(
        self,
        td: TensorDict,
        current_node: torch.Tensor,
    ) -> torch.Tensor:
        """Compute normalised travel-time from current node to all candidates.

        Returns
        -------
        bias : Tensor [B, N]
            ``travel_time(current -> candidate) / sqrt(2)`` normalised by max_time.
        """
        h3_indices = td["h3_indices"]                        # [B, N]
        travel_time_matrix = td["travel_time_matrix"]        # [B, H, H]
        B = h3_indices.shape[0]

        if travel_time_matrix.dim() == 2:
            travel_time_matrix = travel_time_matrix.unsqueeze(0).expand(B, -1, -1)

        # h3 index of current node
        cur_h3 = gather_by_index(h3_indices, current_node.unsqueeze(-1), dim=1)  # [B]

        # travel_time_matrix[b, cur_h3[b], :] -> [B, H]
        row = travel_time_matrix[
            torch.arange(B, device=h3_indices.device),
            cur_h3,
        ]  # [B, H]

        # For each candidate node n, look up row[b, h3_indices[b, n]]
        cand_h3 = h3_indices                                 # [B, N]
        travel_times = row.gather(1, cand_h3)                # [B, N]

        # Normalise and scale
        return (travel_times / self.max_time) / math.sqrt(2.0)


# =========================================================================
# Standalone test
# =========================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_generator import SFGenerator
    from trial_encoder import DARPEdgeEncoder

    print("=== DARPDecoder test ===")
    generator = SFGenerator()
    td = generator(batch_size=[4])

    # Encode
    encoder = DARPEdgeEncoder(embed_dim=128, num_heads=8, num_layers=2)
    node_emb = encoder(td)
    B, N, D = node_emb.shape
    print(f"  Encoder output: {node_emb.shape}")

    # Initialise env state
    from oracle_env import PDPTWEnv
    env = PDPTWEnv(generator=generator)
    state = env.reset(td)

    # Decoder
    decoder = DARPDecoder(embed_dim=128, tanh_clip=10.0)
    cache = decoder.precompute(node_emb)
    decoder.reset(B, node_emb.device)

    logprobs = decoder(state, cache)
    print(f"  Logprobs shape: {logprobs.shape}  (expected [{B}, {N}])")
    assert logprobs.shape == (B, N), f"Shape mismatch: {logprobs.shape}"

    # Validate probability distribution
    probs = logprobs.exp()
    prob_sums = probs.sum(dim=-1)
    print(f"  Prob sums (should be ~1): {prob_sums.tolist()}")
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-4), (
        f"Probabilities don't sum to 1: {prob_sums}"
    )

    # Check that masked actions have ~0 probability
    mask = state["action_mask"]
    masked_prob = probs[~mask]
    if masked_prob.numel() > 0:
        print(f"  Max prob of masked action: {masked_prob.max().item():.2e} (should be ~0)")
        assert masked_prob.max().item() < 1e-3, "Masked actions have non-negligible probability"

    print(f"  NaN check: {torch.isnan(logprobs).any().item()}")
    print(f"  Decoder param count: {sum(p.numel() for p in decoder.parameters()):,}")

    print("\nAll decoder tests passed.")
