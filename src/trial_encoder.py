"""
GREAT-style Edge-based Encoder for DARP.

Architecture overview
---------------------
1. EdgeFeatureConstructor  -- builds dense [B, N, N, 15] edge features
2. DARPGREATLayer          -- one layer of dense asymmetric edge attention
3. DARPEdgeEncoder         -- stacks 5 GREAT layers; returns node embeddings [B, N, D]

Design rationale (from planner):
- Dense tensors rather than sparse because N = 61 is small.
- LayerNorm instead of BatchNorm for variable batch sizes.
- Edge features capture travel time, node types, pairing, time-window slack,
  demand, and raw time-window bounds.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def gather_by_index(
    source: torch.Tensor,
    index: torch.Tensor,
    dim: int = 1,
    squeeze: bool = True,
) -> torch.Tensor:
    """Gather values from *source* along *dim* using *index*.

    Mirrors the helper in oracle_env.py so the encoder is self-contained.
    """
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
    if squeeze and result.shape[-1] == 1:
        result = result.squeeze(-1)
    return result


# =========================================================================
# 1. Edge Feature Constructor
# =========================================================================

class EdgeFeatureConstructor(nn.Module):
    """Build dense edge feature tensor from a TensorDict.

    For every ordered pair (i, j) of nodes the feature vector is:

        [travel_time_ij,                     # 1
         node_type_i  (3-hot: depot/pickup/delivery),  # 3
         node_type_j  (3-hot),               # 3
         is_paired,                           # 1
         tw_slack,                            # 1
         demand_i,                            # 1
         demand_j,                            # 1
         tw_early_i,                          # 1
         tw_late_i,                           # 1
         tw_early_j,                          # 1
         tw_late_j]                           # 1
                                              ------
                                              total = 15

    All time values are normalised by ``max_time`` (default 1440 = 24 h in
    minutes).  Demands are left as-is (typically +/-1).

    Parameters
    ----------
    max_time : float
        Normalisation constant for all temporal features (minutes).
    """

    EDGE_DIM: int = 15

    def __init__(self, max_time: float = 1440.0) -> None:
        super().__init__()
        self.max_time = max_time

    # -----------------------------------------------------------------
    def forward(self, td: TensorDict) -> torch.Tensor:
        """Construct edge features.

        Parameters
        ----------
        td : TensorDict
            Must contain at least ``h3_indices``  [B, N],
            ``travel_time_matrix``  [B, H, H] or [H, H],
            ``time_windows``  [B, N, 2],  ``demand``  [B, N].

        Returns
        -------
        edges : Tensor  [B, N, N, 15]
        """
        h3_indices = td["h3_indices"]                        # [B, N]
        travel_time_matrix = td["travel_time_matrix"]        # [B, H, H] or [H, H]
        time_windows = td["time_windows"]                    # [B, N, 2]
        demand = td["demand"]                                # [B, N]

        B, N = h3_indices.shape
        device = h3_indices.device

        # --- travel time between every pair via H3 indirection --------
        if travel_time_matrix.dim() == 2:
            # shared across batch -- expand
            travel_time_matrix = travel_time_matrix.unsqueeze(0).expand(B, -1, -1)

        H = travel_time_matrix.shape[1]

        # Step 1: gather rows for source nodes -> [B, N, H]
        # rows[b, i, :] = travel_time_matrix[b, h3_indices[b, i], :]
        row_idx = h3_indices.unsqueeze(-1).expand(B, N, H)   # [B, N, H]
        rows = travel_time_matrix.gather(1, row_idx)          # [B, N, H]

        # Step 2: for each (i, j) pair, pick column h3_indices[b, j]
        # h3_j[b, i, j] = h3_indices[b, j]
        h3_j = h3_indices.unsqueeze(1).expand(B, N, N)        # [B, N, N]

        # travel_ij[b, i, j] = rows[b, i, h3_j[b, i, j]]
        travel_time_ij = rows.gather(2, h3_j)                 # [B, N, N]

        travel_time_norm = travel_time_ij / self.max_time      # [B, N, N]

        # --- node type one-hot (3): depot / pickup / delivery ---------
        node_idx = torch.arange(N, device=device)            # [N]
        is_depot   = (node_idx == 0).float()                 # [N]
        is_pickup  = ((node_idx % 2 == 1) & (node_idx != 0)).float()
        is_delivery = ((node_idx % 2 == 0) & (node_idx != 0)).float()

        node_type = torch.stack([is_depot, is_pickup, is_delivery], dim=-1)  # [N, 3]
        node_type = node_type.unsqueeze(0).expand(B, N, 3)   # [B, N, 3]

        # node_type_i[b, i, j, :] = node_type[b, i, :]
        node_type_i = node_type.unsqueeze(2).expand(B, N, N, 3)
        node_type_j = node_type.unsqueeze(1).expand(B, N, N, 3)

        # --- pairing indicator ----------------------------------------
        # is_paired = 1 iff (i is odd pickup and j == i+1) or (i is even delivery and j == i-1)
        idx_i = node_idx.unsqueeze(1).expand(N, N)           # [N, N]
        idx_j = node_idx.unsqueeze(0).expand(N, N)           # [N, N]

        pickup_paired  = (idx_i % 2 == 1) & (idx_i != 0) & (idx_j == idx_i + 1)
        delivery_paired = (idx_i % 2 == 0) & (idx_i != 0) & (idx_j == idx_i - 1)
        is_paired = (pickup_paired | delivery_paired).float()  # [N, N]
        is_paired = is_paired.unsqueeze(0).expand(B, N, N)     # [B, N, N]

        # --- time window slack ----------------------------------------
        tw_early = time_windows[..., 0]                       # [B, N]
        tw_late  = time_windows[..., 1]                       # [B, N]

        # tw_early_i[b,i,j] = tw_early[b,i];  tw_late_j[b,i,j] = tw_late[b,j]
        tw_early_i = tw_early.unsqueeze(2).expand(B, N, N)
        tw_late_j  = tw_late.unsqueeze(1).expand(B, N, N)

        # slack = max(0, tw_late_j - tw_early_i - travel_time) / max_time
        tw_slack = (tw_late_j - tw_early_i - travel_time_ij).clamp(min=0.0) / self.max_time

        # --- demand features ------------------------------------------
        demand_i = demand.unsqueeze(2).expand(B, N, N)        # [B, N, N]
        demand_j = demand.unsqueeze(1).expand(B, N, N)        # [B, N, N]

        # --- time window bounds (normalised) --------------------------
        tw_early_i_norm = tw_early_i / self.max_time
        tw_late_i  = tw_late.unsqueeze(2).expand(B, N, N)
        tw_late_i_norm  = tw_late_i / self.max_time
        tw_early_j = tw_early.unsqueeze(1).expand(B, N, N)
        tw_early_j_norm = tw_early_j / self.max_time
        tw_late_j_norm  = tw_late_j / self.max_time

        # --- assemble -------------------------------------------------
        edges = torch.stack([
            travel_time_norm,          # 0
            node_type_i[..., 0],       # 1  depot_i
            node_type_i[..., 1],       # 2  pickup_i
            node_type_i[..., 2],       # 3  delivery_i
            node_type_j[..., 0],       # 4  depot_j
            node_type_j[..., 1],       # 5  pickup_j
            node_type_j[..., 2],       # 6  delivery_j
            is_paired,                 # 7
            tw_slack,                  # 8
            demand_i,                  # 9
            demand_j,                  # 10
            tw_early_i_norm,           # 11
            tw_late_i_norm,            # 12
            tw_early_j_norm,           # 13
            tw_late_j_norm,            # 14
        ], dim=-1)                     # [B, N, N, 15]

        return edges


# =========================================================================
# 2. GREAT Layer  (dense asymmetric edge attention)
# =========================================================================

class DARPGREATLayer(nn.Module):
    """One layer of GREAT-style dense asymmetric attention on edges.

    For each target node *j* we compute two separate multi-head attentions:

    * **in-attention**:  softmax over *source* dimension (who sends to j)
    * **out-attention**: softmax over *target* dimension (where i sends)

    Both produce per-node aggregations.  The four aggregated vectors
    (in-value, in-edge, out-value, out-edge) are concatenated and projected
    to update node embeddings, which are then used to reconstruct the edge
    tensor for the next layer.

    Parameters
    ----------
    embed_dim : int
        Node / edge embedding dimension.
    num_heads : int
        Number of attention heads.
    ff_hidden : int
        Hidden dimension of the feed-forward sub-layer.
    dropout : float
        Dropout probability (applied after attention and FFN).
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        ff_hidden: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # --- in-attention projections (softmax over source i for each target j) ---
        self.W_q_in = nn.Linear(embed_dim, embed_dim)
        self.W_k_in = nn.Linear(embed_dim, embed_dim)
        self.W_v_in = nn.Linear(embed_dim, embed_dim)

        # --- out-attention projections (softmax over target j for each source i) ---
        self.W_q_out = nn.Linear(embed_dim, embed_dim)
        self.W_k_out = nn.Linear(embed_dim, embed_dim)
        self.W_v_out = nn.Linear(embed_dim, embed_dim)

        # --- combine 4 aggregations -> embed_dim node update ---
        # in-value, in-edge, out-value, out-edge each have embed_dim
        # We actually aggregate (in_val + out_val) and project once for simplicity.
        # But the canonical GREAT has 2 outputs (in, out) each of embed_dim.
        self.proj_in = nn.Linear(embed_dim, embed_dim)
        self.proj_out = nn.Linear(embed_dim, embed_dim)

        # --- feed-forward ---
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, embed_dim),
        )

        # --- norms & dropout ---
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.norm_ffn  = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # --- edge reconstruction: (node_i || node_j) -> edge ---
        self.edge_proj = nn.Linear(2 * embed_dim, embed_dim)

    # -----------------------------------------------------------------
    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one GREAT layer.

        Parameters
        ----------
        node_emb : Tensor [B, N, D]
        edge_emb : Tensor [B, N, N, D]

        Returns
        -------
        node_emb_out : Tensor [B, N, D]
        edge_emb_out : Tensor [B, N, N, D]
        """
        B, N, D = node_emb.shape
        H = self.num_heads
        d = self.head_dim

        # ---- in-attention: for each target j, attend over sources i ----
        # Q, K, V are computed from edge_emb -> [B, N, N, H, d]
        q_in = self.W_q_in(edge_emb).view(B, N, N, H, d)
        k_in = self.W_k_in(edge_emb).view(B, N, N, H, d)
        v_in = self.W_v_in(edge_emb).view(B, N, N, H, d)

        # score_in[b, i, j, h] = sum_d q_in[b,i,j,h,d] * k_in[b,i,j,h,d]
        # interpretation: edge (i->j); for target j we want softmax over source i
        alpha_in = (q_in * k_in).sum(-1) / math.sqrt(d)  # [B, N, N, H]
        alpha_in = F.softmax(alpha_in, dim=1)             # softmax over source i (dim=1)

        # aggregate: out_in[b, j, h, d] = sum_i alpha_in[b,i,j,h] * v_in[b,i,j,h,d]
        out_in = (alpha_in.unsqueeze(-1) * v_in).sum(dim=1)  # [B, N, H, d]
        out_in = out_in.reshape(B, N, D)                      # [B, N, D]

        # ---- out-attention: for each source i, attend over targets j ----
        q_out = self.W_q_out(edge_emb).view(B, N, N, H, d)
        k_out = self.W_k_out(edge_emb).view(B, N, N, H, d)
        v_out = self.W_v_out(edge_emb).view(B, N, N, H, d)

        alpha_out = (q_out * k_out).sum(-1) / math.sqrt(d)  # [B, N, N, H]
        alpha_out = F.softmax(alpha_out, dim=2)              # softmax over target j (dim=2)

        # aggregate: out_out[b, i, h, d] = sum_j alpha_out[b,i,j,h] * v_out[b,i,j,h,d]
        out_out = (alpha_out.unsqueeze(-1) * v_out).sum(dim=2)  # [B, N, H, d]
        out_out = out_out.reshape(B, N, D)                       # [B, N, D]

        # ---- combine in + out -> node update ----
        node_update = self.proj_in(out_in) + self.proj_out(out_out)
        node_update = self.dropout(node_update)
        node_emb = self.norm_attn(node_emb + node_update)

        # ---- feed-forward sub-layer ----
        ffn_out = self.ffn(node_emb)
        ffn_out = self.dropout(ffn_out)
        node_emb = self.norm_ffn(node_emb + ffn_out)

        # ---- reconstruct edges from updated nodes ----
        # edge_emb_out[b, i, j, :] = f(node_emb[b, i, :] || node_emb[b, j, :])
        ni = node_emb.unsqueeze(2).expand(B, N, N, D)  # [B, N, N, D]
        nj = node_emb.unsqueeze(1).expand(B, N, N, D)  # [B, N, N, D]
        edge_emb = self.edge_proj(torch.cat([ni, nj], dim=-1))  # [B, N, N, D]

        return node_emb, edge_emb


# =========================================================================
# 3. Full GREAT Encoder
# =========================================================================

class DARPEdgeEncoder(nn.Module):
    """Full GREAT-style encoder for DARP.

    Pipeline::

        td -> EdgeFeatureConstructor -> [B, N, N, 15]
           -> Linear project edges to D
           -> Linear project initial node features to D
           -> 5 x DARPGREATLayer
           -> node embeddings [B, N, D]

    The initial node features are taken from the diagonal of the edge
    tensor (self-loops give node-centric info such as own type, demand,
    time windows) and optionally from the H3 travel-time row.

    Parameters
    ----------
    embed_dim : int
        Internal embedding dimension (default 128).
    num_heads : int
        Number of attention heads per GREAT layer.
    num_layers : int
        Number of stacked GREAT layers (default 5).
    ff_hidden : int
        Feed-forward hidden dimension inside each GREAT layer.
    dropout : float
        Dropout rate.
    max_time : float
        Normalisation constant forwarded to ``EdgeFeatureConstructor``.
    num_h3 : int
        Number of H3 cells; used to build initial node features from the
        travel-time matrix row.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 5,
        ff_hidden: int = 512,
        dropout: float = 0.0,
        max_time: float = 1440.0,
        num_h3: int = 31,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_time = max_time

        # Edge feature constructor (no learnable parameters)
        self.edge_constructor = EdgeFeatureConstructor(max_time=max_time)

        # Project raw edge features [15] -> embed_dim
        self.edge_init_proj = nn.Linear(EdgeFeatureConstructor.EDGE_DIM, embed_dim)

        # Initial node features: travel-time row (num_h3) + demand (1) + tw (2) = num_h3 + 3
        self.node_init_proj = nn.Linear(num_h3 + 3, embed_dim)

        # Stacked GREAT layers
        self.layers = nn.ModuleList([
            DARPGREATLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_hidden=ff_hidden,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self._init_parameters()

    # -----------------------------------------------------------------
    def _init_parameters(self) -> None:
        """Xavier-uniform for all Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -----------------------------------------------------------------
    def _build_initial_node_features(self, td: TensorDict) -> torch.Tensor:
        """Build initial node features from TensorDict.

        Features per node: travel-time row from the matrix (num_h3 values),
        demand (1), tw_early (1), tw_late (1).

        Returns
        -------
        node_feats : Tensor  [B, N, num_h3 + 3]
        """
        h3_indices = td["h3_indices"]                  # [B, N]
        travel_time_matrix = td["travel_time_matrix"]  # [B, H, H] or [H, H]
        demand = td["demand"]                          # [B, N]
        time_windows = td["time_windows"]              # [B, N, 2]

        if travel_time_matrix.dim() == 2:
            travel_time_matrix = travel_time_matrix.unsqueeze(0).expand(
                h3_indices.shape[0], -1, -1
            )

        # h3_feats[b, n, :] = travel_time_matrix[b, h3_indices[b, n], :]
        h3_feats = gather_by_index(travel_time_matrix, h3_indices, dim=1, squeeze=False)
        # h3_feats: [B, N, H]

        # Normalise
        h3_feats = h3_feats / self.max_time

        return torch.cat([
            h3_feats,                          # [B, N, H]
            demand.unsqueeze(-1),              # [B, N, 1]
            time_windows / self.max_time,      # [B, N, 2]
        ], dim=-1)                             # [B, N, H+3]

    # -----------------------------------------------------------------
    def forward(self, td: TensorDict) -> torch.Tensor:
        """Encode the problem instance.

        Parameters
        ----------
        td : TensorDict
            Problem instance data (static fields only needed).

        Returns
        -------
        node_embeddings : Tensor  [B, N, embed_dim]
        """
        # 1. Construct raw edge features
        raw_edges = self.edge_constructor(td)             # [B, N, N, 15]

        # 2. Project edges to embedding dimension
        edge_emb = self.edge_init_proj(raw_edges)         # [B, N, N, D]

        # 3. Build & project initial node features
        node_feats = self._build_initial_node_features(td)  # [B, N, H+3]
        node_emb = self.node_init_proj(node_feats)          # [B, N, D]

        # 4. Run GREAT layers
        for layer in self.layers:
            node_emb, edge_emb = layer(node_emb, edge_emb)

        return node_emb  # [B, N, D]


# =========================================================================
# Standalone test
# =========================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Allow running from the src/ directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from oracle_generator import SFGenerator

    print("=== EdgeFeatureConstructor test ===")
    generator = SFGenerator()
    td = generator(batch_size=[4])

    constructor = EdgeFeatureConstructor()
    edges = constructor(td)
    B, N = td["h3_indices"].shape
    print(f"  Batch size: {B}, Num nodes: {N}")
    print(f"  Edge features shape: {edges.shape}  (expected [{B}, {N}, {N}, 15])")
    assert edges.shape == (B, N, N, 15), f"Shape mismatch: {edges.shape}"
    print(f"  Edge feature range: [{edges.min().item():.4f}, {edges.max().item():.4f}]")
    print(f"  Travel time (normalised) sample [0,0,1]: {edges[0, 0, 1, 0].item():.4f}")

    print("\n=== DARPEdgeEncoder test ===")
    encoder = DARPEdgeEncoder(embed_dim=128, num_heads=8, num_layers=2)
    node_emb = encoder(td)
    print(f"  Node embeddings shape: {node_emb.shape}  (expected [{B}, {N}, 128])")
    assert node_emb.shape == (B, N, 128), f"Shape mismatch: {node_emb.shape}"
    print(f"  Embedding range: [{node_emb.min().item():.4f}, {node_emb.max().item():.4f}]")
    print(f"  NaN check: {torch.isnan(node_emb).any().item()}")
    print(f"  Encoder param count: {sum(p.numel() for p in encoder.parameters()):,}")

    print("\nAll encoder tests passed.")
