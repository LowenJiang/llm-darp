import torch
import torch.nn as nn

from inner_loop.rl4co.utils.ops import gather_by_index
from inner_loop.rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def env_dynamic_embedding(env_name: str, config: dict) -> nn.Module:
    """Get environment dynamic embedding. The dynamic embedding is used to modify query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    Consists of a linear layer that projects the node features to the embedding space.

    Args:
        env: Environment or its name.
        config: A dictionary of configuration options for the environment.
    """
    embedding_registry = {
        "tsp": StaticEmbedding,
        "atsp": StaticEmbedding,
        "cvrp": StaticEmbedding,
        "cvrptw": StaticEmbedding,
        "ffsp": StaticEmbedding,
        "svrp": StaticEmbedding,
        "sdvrp": SDVRPDynamicEmbedding,
        "pctsp": StaticEmbedding,
        "spctsp": StaticEmbedding,
        "op": StaticEmbedding,
        "dpp": StaticEmbedding,
        "mdpp": StaticEmbedding,
        "pdp": StaticEmbedding,
        "mtsp": StaticEmbedding,
        "smtwtp": StaticEmbedding,
        "jssp": JSSPDynamicEmbedding,
        "fjsp": JSSPDynamicEmbedding,
        "mtvrp": StaticEmbedding,
        "darp": DARPDynamicEmbedding,
        "pdptw": StaticEmbedding
    }

    if env_name not in embedding_registry:
        log.warning(
            f"Unknown environment name '{env_name}'. Available dynamic embeddings: {embedding_registry.keys()}. Defaulting to StaticEmbedding."
        )
    return embedding_registry.get(env_name, StaticEmbedding)(**config)

class DARPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Dial-a-Ride Problem (DARP).
    Embed the following dynamic node features to the embedding space:
        - visited: whether each node has been visited by any vehicle
        - picked_up_by_current: whether each pickup has been picked up by the current vehicle
        - feasible_time: whether the node is feasible given current time constraints

    These features change during the rollout and are used to modify the query, key,
    and value vectors of the attention mechanism.
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(DARPDynamicEmbedding, self).__init__()
        # Project 3 binary features (visited, picked_up_by_current, time_feasible) to 3 * embed_dim
        self.projection = nn.Linear(3, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        num_loc = td["locs"].shape[-2]  # including depot
        device = td["locs"].device

        # Feature 1: Visited status (binary) - global across all vehicles
        visited = td["visited"].float()  # [..., num_loc]

        # Feature 2: Picked up status for pickups by current vehicle (binary)
        # For depot and dropoffs, this is 0; for pickups, it's whether they've been picked up by current vehicle
        picked_up_feat = torch.zeros_like(visited)
        num_pickups = (num_loc - 1) // 2
        # Interleaved: pickups at odd indices starting at 1
        if num_pickups > 0:
            picked_up = td["picked_up_by_current"].float()
            picked_up_shape = picked_up.shape
            visited_shape = visited.shape
            # Only reshape if the shapes don't match
            if picked_up_shape != visited_shape:
                picked_up = picked_up.reshape(*visited_shape[:-1], -1)[..., :visited_shape[-1]]
            picked_up_feat[..., 1::2] = picked_up[..., 1::2]
        
        # Feature 3: Time feasibility (binary)
        # Whether each node can be reached within its time window from current position
        time_feasible = self._compute_time_feasibility(td)
        
        # Stack features: [batch_size, num_loc, 3]
        dynamic_features = torch.stack([visited, picked_up_feat, time_feasible], dim=-1)
        
        # Project to embedding space: [batch_size, num_loc, 3*embed_dim]
        projected = self.projection(dynamic_features)
        
        # Split into glimpse_key, glimpse_val, and logit_key
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = projected.chunk(3, dim=-1)
        
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
    
    def _compute_time_feasibility(self, td):
        """Compute whether each node is time-feasible from current position"""
        batch_shape = td["locs"].shape[:-2]
        N = td["locs"].shape[-2]

        current_node = td["current_node"].long()
        current_node = current_node.reshape(*batch_shape, -1)[..., :1]
        current_time = td["current_time"].float().reshape(*batch_shape, -1)[..., 0]

        locs = td["locs"]
        curr_loc = locs.gather(
            -2,
            current_node.unsqueeze(-1).expand(*batch_shape, 1, locs.size(-1)),
        )  # batch_shape + (1, 2)

        dist = (locs - curr_loc).norm(p=2, dim=-1)
        
        # Compute travel time (assuming vehicle_speed is available)
        vehicle_speed = 25.0  # Default from DARPGenerator
        travel_time = torch.round(dist / vehicle_speed).long()
        
        # Compute arrival time
        arrival_time = current_time.unsqueeze(-1) + travel_time  # batch_shape + (N,)
        
        # Check if arrival time is within time window
        time_windows = td["time_windows"].reshape(*batch_shape, -1)[..., :N]
        time_feasible = arrival_time <= time_windows
        
        return time_feasible.float()

class StaticEmbedding(nn.Module):
    """Static embedding for general problems.
    This is used for problems that do not have any dynamic information, except for the
    information regarding the current action (e.g. the current node in TSP). See context embedding for more details.
    """

    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0


class SDVRPDynamicEmbedding(nn.Module):
    """Dynamic embedding for the Split Delivery Vehicle Routing Problem (SDVRP).
    Embed the following node features to the embedding space:
        - demand_with_depot: demand of the customers and the depot
    The demand with depot is used to modify the query, key and value vectors of the attention mechanism
    based on the current state of the environment (which is changing during the rollout).
    """

    def __init__(self, embed_dim, linear_bias=False):
        super(SDVRPDynamicEmbedding, self).__init__()
        self.projection = nn.Linear(1, 3 * embed_dim, bias=linear_bias)

    def forward(self, td):
        demands_with_depot = td["demand_with_depot"][..., None].clone()
        demands_with_depot[..., 0, :] = 0
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = self.projection(
            demands_with_depot
        ).chunk(3, dim=-1)
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic


class JSSPDynamicEmbedding(nn.Module):
    def __init__(self, embed_dim, linear_bias=False, scaling_factor: int = 1000) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.project_node_step = nn.Linear(2, 3 * embed_dim, bias=linear_bias)
        self.project_edge_step = nn.Linear(1, 3, bias=linear_bias)
        self.scaling_factor = scaling_factor

    def forward(self, td, cache):
        ma_emb = cache.node_embeddings["machine_embeddings"]
        bs, _, emb_dim = ma_emb.shape
        num_jobs = td["next_op"].size(1)
        # updates
        updates = ma_emb.new_zeros((bs, num_jobs, 3 * emb_dim))

        lbs = torch.clip(td["lbs"] - td["time"][:, None], 0) / self.scaling_factor
        update_feat = torch.stack((lbs, td["is_ready"]), dim=-1)
        job_update_feat = gather_by_index(update_feat, td["next_op"], dim=1)
        updates = updates + self.project_node_step(job_update_feat)

        ma_busy = td["busy_until"] > td["time"][:, None]
        # mask machines currently busy
        masked_proc_times = td["proc_times"].clone() / self.scaling_factor
        # bs, ma, ops
        masked_proc_times[ma_busy] = 0.0
        # bs, ops, ma, 3
        edge_feat = self.project_edge_step(masked_proc_times.unsqueeze(-1)).transpose(
            1, 2
        )
        job_edge_feat = gather_by_index(edge_feat, td["next_op"], dim=1)
        # bs, nodes, 3*emb
        edge_upd = torch.einsum("ijkl,ikm->ijlm", job_edge_feat, ma_emb).view(
            bs, num_jobs, 3 * emb_dim
        )
        updates = updates + edge_upd

        # (bs, nodes, emb)
        glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic = updates.chunk(
            3, dim=-1
        )
        return glimpse_key_dynamic, glimpse_val_dynamic, logit_key_dynamic
