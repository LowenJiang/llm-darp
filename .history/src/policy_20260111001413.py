"""
Self-contained PDPTW Attention Policy
Based on Kool et al. (2019) - Attention Model for Vehicle Routing
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torch import Tensor


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def gather_by_index(src: Tensor, idx: Tensor, dim: int = 1, squeeze: bool = True) -> Tensor:
    """Gather elements from src by index idx along specified dim."""
    expanded_shape = list(src.shape)
    expanded_shape[dim] = -1
    idx = idx.view(idx.shape + (1,) * (src.dim() - idx.dim())).expand(expanded_shape)
    squeeze = idx.size(dim) == 1 and squeeze
    return src.gather(dim, idx).squeeze(dim) if squeeze else src.gather(dim, idx)


def batchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """Repeat tensor along batch dimension."""
    def _batchify_single(x, repeats):
        s = x.shape
        return x.expand(repeats, *s).contiguous().view(s[0] * repeats, *s[1:])
    
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _batchify_single(x, s) if s > 0 else x
    return x


def unbatchify(x: Union[Tensor, TensorDict], shape: Union[tuple, int]) -> Union[Tensor, TensorDict]:
    """Undo batchify operation."""
    def _unbatchify_single(x, repeats):
        s = x.shape
        return x.view(repeats, s[0] // repeats, *s[1:]).permute(1, 0, *range(2, len(s) + 1))
    
    shape = [shape] if isinstance(shape, int) else shape
    for s in reversed(shape):
        x = _unbatchify_single(x, s) if s > 0 else x
    return x


def get_log_likelihood(logprobs: Tensor, actions: Tensor, mask: Optional[Tensor] = None, return_sum: bool = True) -> Tensor:
    """Compute log likelihood of actions."""
    logprobs_selected = logprobs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        logprobs_selected = logprobs_selected * mask.float()
    return logprobs_selected.sum(dim=-1) if return_sum else logprobs_selected


def calculate_entropy(logprobs: Tensor) -> Tensor:
    """Calculate entropy of log probability distribution."""
    logprobs = torch.nan_to_num(logprobs, nan=0.0)
    return -(logprobs.exp() * logprobs).sum(dim=-1).sum(dim=1)


def scaled_dot_product_attention_simple(q: Tensor, k: Tensor, v: Tensor, attn_mask: Optional[Tensor] = None, dropout_p: float = 0.0, is_causal: bool = False) -> Tensor:
    """Simple scaled dot-product attention."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores.masked_fill_(~attn_mask, float("-inf"))
        else:
            scores += attn_mask
    attn_weights = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)
    return torch.matmul(attn_weights, v)


try:
    from torch.nn.functional import scaled_dot_product_attention
except ImportError:
    scaled_dot_product_attention = scaled_dot_product_attention_simple


# =============================================================================
# DECODING STRATEGIES
# =============================================================================

class DecodingStrategy(ABC):
    """Base class for decoding strategies."""
    
    def __init__(
        self,
        temperature: float = 1.0,
        tanh_clipping: float = 0.0,
        mask_logits: bool = True,
        store_all_logp: bool = False,
        multistart: bool = False,
        num_starts: int = None,
        **kwargs,
    ):
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.store_all_logp = store_all_logp
        self.multistart = multistart
        self.num_starts = num_starts
        self.actions = []
        self.logprobs = []

    def pre_decoder_hook(self, td: TensorDict, env) -> Tuple[TensorDict, any, int]:
        """Called before decoding starts."""
        if self.multistart:
            if self.num_starts is None:
                self.num_starts = env.get_num_starts(td)
            td = batchify(td, self.num_starts)
        return td, env, self.num_starts if self.multistart else 0

    def post_decoder_hook(self, td: TensorDict, env) -> Tuple[Tensor, Tensor, TensorDict, any]:
        """Called after decoding."""
        logprobs = torch.stack(self.logprobs, dim=-1)
        actions = torch.stack(self.actions, dim=-1)
        return logprobs, actions, td, env

    def step(self, logits: Tensor, mask: Tensor, td: TensorDict, action: Optional[Tensor] = None) -> TensorDict:
        """Single decoding step."""
        logits = self._process_logits(logits, mask)
        logprobs = F.log_softmax(logits, dim=-1)
        
        if action is None:
            action = self._select_action(logprobs)
        
        selected_logprob = gather_by_index(logprobs, action, dim=-1)
        self.actions.append(action)
        self.logprobs.append(logprobs if self.store_all_logp else selected_logprob)
        
        td.set("action", action)
        return td

    def _process_logits(self, logits: Tensor, mask: Tensor) -> Tensor:
        """Apply temperature, clipping, and masking."""
        if self.temperature != 1.0:
            logits = logits / self.temperature
        if self.tanh_clipping > 0:
            logits = self.tanh_clipping * torch.tanh(logits)
        if self.mask_logits:
            logits.masked_fill_(~mask, float("-inf"))
        return logits

    @abstractmethod
    def _select_action(self, logprobs: Tensor) -> Tensor:
        raise NotImplementedError


class Greedy(DecodingStrategy):
    """Greedy: select highest probability action."""
    def _select_action(self, logprobs: Tensor) -> Tensor:
        return logprobs.argmax(dim=-1)


class Sampling(DecodingStrategy):
    """Sampling: sample from distribution."""
    def _select_action(self, logprobs: Tensor) -> Tensor:
        return torch.multinomial(logprobs.exp(), num_samples=1).squeeze(-1)


class Evaluate(DecodingStrategy):
    """Evaluate: compute logprobs for given actions."""
    def _select_action(self, logprobs: Tensor) -> Tensor:
        raise RuntimeError("Evaluate requires actions to be provided")


def get_decoding_strategy(decoding_strategy: str, **config) -> DecodingStrategy:
    """Factory for decoding strategies."""
    registry = {
        "greedy": Greedy,
        "sampling": Sampling,
        "multistart_greedy": Greedy,
        "multistart_sampling": Sampling,
        "evaluate": Evaluate,
    }
    if "multistart" in decoding_strategy:
        config["multistart"] = True
    return registry.get(decoding_strategy, Sampling)(**config)


# =============================================================================
# NN BUILDING BLOCKS
# =============================================================================

class Normalization(nn.Module):
    """Normalization layer (batch/layer/none)."""
    def __init__(self, embed_dim: int, normalization: str = "batch"):
        super().__init__()
        if normalization == "batch":
            self.norm = nn.BatchNorm1d(embed_dim, affine=True)
        elif normalization == "layer":
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.norm = nn.Identity()
        self.normalization = normalization

    def forward(self, x: Tensor) -> Tensor:
        if self.normalization == "batch":
            return self.norm(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x)


class MLP(nn.Module):
    """Simple MLP."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list = None):
        super().__init__()
        hidden_dims = hidden_dims or []
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# =============================================================================
# ATTENTION MODULES
# =============================================================================

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention."""
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True, sdpa_fn: Callable = None):
        super().__init__()
        self.num_heads = num_heads
        self.sdpa_fn = sdpa_fn or scaled_dot_product_attention
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        q, k, v = rearrange(self.Wqkv(x), "b s (three h d) -> three b h s d", three=3, h=self.num_heads).unbind(0)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1) if attn_mask.ndim == 3 else attn_mask.unsqueeze(1).unsqueeze(2)
        out = self.sdpa_fn(q, k, v, attn_mask=attn_mask)
        return self.out_proj(rearrange(out, "b h s d -> b s (h d)"))


class PointerAttention(nn.Module):
    """Pointer attention for action selection."""
    def __init__(self, embed_dim: int, num_heads: int, mask_inner: bool = True, check_nan: bool = True, sdpa_fn: Callable = None):
        super().__init__()
        self.num_heads = num_heads
        self.mask_inner = mask_inner
        self.check_nan = check_nan
        self.project_out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sdpa_fn = sdpa_fn or scaled_dot_product_attention

    def forward(self, query: Tensor, key: Tensor, value: Tensor, logit_key: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        q = rearrange(query, "... g (h s) -> ... h g s", h=self.num_heads)
        k = rearrange(key, "... g (h s) -> ... h g s", h=self.num_heads)
        v = rearrange(value, "... g (h s) -> ... h g s", h=self.num_heads)
        
        mask = None
        if self.mask_inner and attn_mask is not None:
            mask = attn_mask.unsqueeze(1) if attn_mask.ndim == 3 else attn_mask.unsqueeze(1).unsqueeze(2)
        
        heads = rearrange(self.sdpa_fn(q, k, v, attn_mask=mask), "... h n g -> ... n (h g)", h=self.num_heads)
        glimpse = self.project_out(heads)
        logits = torch.bmm(glimpse, logit_key.squeeze(-2).transpose(-2, -1)).squeeze(-2) / math.sqrt(glimpse.size(-1))
        
        if self.check_nan:
            assert not torch.isnan(logits).any(), "Logits contain NaNs"
        return logits


# =============================================================================
# ENCODER
# =============================================================================

class MultiHeadAttentionLayer(nn.Module):
    """MHA + FFN block."""
    def __init__(self, embed_dim: int, num_heads: int = 8, feedforward_hidden: int = 512, normalization: str = "batch", sdpa_fn: Callable = None):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, sdpa_fn=sdpa_fn)
        self.norm1 = Normalization(embed_dim, normalization)
        self.ffn = MLP(embed_dim, embed_dim, [feedforward_hidden] if feedforward_hidden > 0 else [])
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        x = self.norm1(x + self.attn(x, attn_mask))
        x = self.norm2(x + self.ffn(x))
        return x


class GraphAttentionNetwork(nn.Module):
    """Transformer encoder."""
    def __init__(self, num_heads: int, embed_dim: int, num_layers: int, normalization: str = "batch", feedforward_hidden: int = 512, sdpa_fn: Callable = None):
        super().__init__()
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(embed_dim, num_heads, feedforward_hidden, normalization, sdpa_fn)
            for _ in range(num_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# =============================================================================
# PDPTW EMBEDDINGS
# =============================================================================

class PDPTWInitEmbedding(nn.Module):
    """Initial embedding: H3 travel times + demand + time windows."""
    def __init__(self, embed_dim: int, num_h3: int = 31):
        super().__init__()
        self.project = nn.Linear(num_h3 + 3, embed_dim)

    def forward(self, td: TensorDict) -> Tensor:
        h3_feats = gather_by_index(td["travel_time_matrix"], td["h3_indices"], dim=1)
        demand = td["demand"].unsqueeze(-1)
        time_windows = td["time_windows"]
        return self.project(torch.cat([h3_feats, demand, time_windows], dim=-1))


class PDPTWContextEmbedding(nn.Module):
    """Context: current node + capacity + time + step."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.project_context = nn.Linear(embed_dim + 3, embed_dim)

    def forward(self, embeddings: Tensor, td: TensorDict) -> Tensor:
        cur_node = gather_by_index(embeddings, td["current_node"])
        remaining_cap = td["vehicle_capacity"] - td["used_capacity"]
        context = torch.cat([cur_node, remaining_cap, td["current_time"], td["i"]], dim=-1)
        return self.project_context(context)


class StaticEmbedding(nn.Module):
    """Static dynamic embedding (returns zeros)."""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        return 0, 0, 0


# =============================================================================
# CACHE
# =============================================================================

@dataclass
class PrecomputedCache:
    """Precomputed decoder cache."""
    node_embeddings: Tensor
    graph_context: Union[Tensor, float]
    glimpse_key: Tensor
    glimpse_val: Tensor
    logit_key: Tensor

    @property
    def fields(self):
        return tuple(getattr(self, x.name) for x in fields(self))

    def batchify(self, num_starts: int) -> "PrecomputedCache":
        return PrecomputedCache(*[
            batchify(e, num_starts) if isinstance(e, (Tensor, TensorDict)) else e
            for e in self.fields
        ])


# =============================================================================
# MAIN POLICY
# =============================================================================

class PDPTWAttentionPolicy(nn.Module):
    """
    Attention Model Policy for PDPTW.
    
    Args:
        embed_dim: Embedding dimension
        num_encoder_layers: Number of encoder layers  
        num_heads: Number of attention heads
        feedforward_hidden: FFN hidden dimension
        num_h3: Number of H3 cells
        temperature: Softmax temperature
        tanh_clipping: Logit clipping
        train_decode_type: Training decode strategy
        val_decode_type: Validation decode strategy
        test_decode_type: Test decode strategy
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        feedforward_hidden: int = 512,
        normalization: str = "batch",
        num_h3: int = 31,
        use_graph_context: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        check_nan: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_graph_context = use_graph_context
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

        # Encoder
        self.init_embedding = PDPTWInitEmbedding(embed_dim, num_h3)
        self.encoder = GraphAttentionNetwork(num_heads, embed_dim, num_encoder_layers, normalization, feedforward_hidden)

        # Decoder
        self.context_embedding = PDPTWContextEmbedding(embed_dim)
        self.dynamic_embedding = StaticEmbedding()
        self.pointer = PointerAttention(embed_dim, num_heads, check_nan=check_nan)
        self.project_node_embeddings = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.project_fixed_context = nn.Linear(embed_dim, embed_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters with Xavier for Linear layers."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_init)

    def encode(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Encode input."""
        init_h = self.init_embedding(td)
        return self.encoder(init_h), init_h

    def _precompute_cache(self, embeddings: Tensor) -> PrecomputedCache:
        """Precompute K, V, logit_key."""
        k, v, l = self.project_node_embeddings(embeddings).chunk(3, dim=-1)
        ctx = self.project_fixed_context(embeddings.mean(1)) if self.use_graph_context else 0
        return PrecomputedCache(embeddings, ctx, k, v, l)

    def _compute_query(self, cache: PrecomputedCache, td: TensorDict) -> Tensor:
        """Compute query."""
        ctx = cache.graph_context
        if td.dim() == 2 and isinstance(ctx, Tensor):
            ctx = ctx.unsqueeze(1)
        q = self.context_embedding(cache.node_embeddings, td) + ctx
        return q.unsqueeze(1) if q.ndim == 2 else q

    def _compute_kv(self, cache: PrecomputedCache, td: TensorDict) -> Tuple[Tensor, Tensor, Tensor]:
        """Get K, V, logit_key."""
        k_d, v_d, l_d = self.dynamic_embedding(td)
        return cache.glimpse_key + k_d, cache.glimpse_val + v_d, cache.logit_key + l_d

    def decode_step(self, td: TensorDict, cache: PrecomputedCache, num_starts: int = 0) -> Tuple[Tensor, Tensor]:
        """Single decode step."""
        if num_starts > 1:
            td = unbatchify(td, num_starts)

        q = self._compute_query(cache, td)
        k, v, l = self._compute_kv(cache, td)
        mask = td["action_mask"]
        logits = self.pointer(q, k, v, l, mask)

        if num_starts > 1:
            logits = rearrange(logits, "b s l -> (s b) l", s=num_starts)
            mask = rearrange(mask, "b s l -> (s b) l", s=num_starts)

        return logits, mask

    def forward(
        self,
        td: TensorDict,
        env,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions: Tensor = None,
        max_steps: int = 1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Full forward pass."""
        # Encode
        hidden, init_embeds = self.encode(td)

        # Decoding setup
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        strategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        # Pre-decode
        td, env, num_starts = strategy.pre_decoder_hook(td, env)
        cache = self._precompute_cache(hidden)
        if num_starts > 1:
            cache = cache.batchify(num_starts)

        # Decode loop
        for step in range(max_steps):
            done_mask = td["done"]
            if done_mask.all():
                break
            
            logits, mask = self.decode_step(td, cache, num_starts)
            if done_mask.any():
                mask = mask.clone()
                mask[done_mask] = False
                mask[done_mask, 0] = True

            td = strategy.step(logits, mask, td, action=actions[..., step] if actions is not None else None)
            if done_mask.any():
                action = td["action"].clone()
                action[done_mask] = 0
                td.set("action", action)

            td = env.step(td)["next"]
            if done_mask.any():
                td.set("done", done_mask | td["done"])
        else:
            # Max steps reached; mark any unfinished envs as done to avoid infinite loops
            td.set("done", torch.ones_like(td["done"], dtype=torch.bool))

        # Post-decode
        logprobs, actions, td, env = strategy.post_decoder_hook(td, env)

        # Output
        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        if logprobs.dim() == actions.dim():
            log_likelihood = logprobs
            if "mask" in td.keys():
                log_likelihood = log_likelihood * td["mask"].float()
            if return_sum_log_likelihood:
                log_likelihood = log_likelihood.sum(dim=-1)
        else:
            log_likelihood = get_log_likelihood(logprobs, actions, td.get("mask", None), return_sum_log_likelihood)

        out = {
            "reward": td["reward"],
            "log_likelihood": log_likelihood,
        }

        for k in ("feasibility", "vehicles_used", "vehicle_penalty", "waiting_penalty", "undelivered_penalty", "successful_requests"):
            if k in td.keys():
                out[k] = td[k]

        if return_actions:
            out["actions"] = actions
        if return_entropy:
            out["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            out["hidden"] = hidden
        if return_init_embeds:
            out["init_embeds"] = init_embeds

        return out


def main():
    """Simple greedy rollout test using PDPTWEnv."""
    from pathlib import Path

    from env import PDPTWEnv
    from generator import SFGenerator

    # Set up generator and environment with a small problem for quick testing
    csv_path = Path(__file__).with_name("traveler_trip_types_res_7.csv")
    ttm_path = Path(__file__).with_name("travel_time_matrix_res_7.csv")
    generator = SFGenerator(
        csv_path=csv_path,
        travel_time_matrix_path=ttm_path)
    
    env = PDPTWEnv(generator=generator)

    # Initialize policy and rollout one greedy episode
    policy = PDPTWAttentionPolicy()
    td = env.reset(batch_size=[2])

    policy.eval()
    with torch.no_grad():
        result = policy(td, env, phase="test", decode_type="greedy", max_steps=200)

    print(f"Reward: {result['reward'].tolist()}")
    print(f"Actions shape: {result['actions'].shape}")
    print(f"Log likelihood: {result['log_likelihood']}")
    print(result['actions'][1,:])


if __name__ == "__main__":
    main()
