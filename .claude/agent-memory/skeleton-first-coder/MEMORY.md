# Skeleton-First Coder Memory

## Project Structure
- `src/trial_encoder.py` - GREAT-style edge encoder (EdgeFeatureConstructor, DARPGREATLayer, DARPEdgeEncoder)
- `src/trial_decoder.py` - Attention decoder (DARPDecoder, DecoderCache)
- `src/trial_gnn_policy.py` - Policy integrating encoder+decoder (DARPPolicy)
- `src/trial_pomo_reinforce.py` - POMO REINFORCE trainer (POMOReinforce)
- `src/oracle_env.py` - PDPTWEnv (the environment, uses TensorDict)
- `src/oracle_generator.py` - SFGenerator (generates problem instances)
- `src/oracle_policy.py` - Original attention policy (reference for patterns)
- `src/oracle_reinforce.py` - Original REINFORCE trainer (has baseline classes)

## Key Patterns
- **TensorDict interface**: env uses TensorDict with keys: h3_indices[B,N], travel_time_matrix[B,H,H], time_windows[B,N,2], demand[B,N], current_node[B,1], action_mask[B,N], visited[B,N], etc.
- **Node indexing**: Node 0=depot, odd=pickup, even=delivery. Pickup i paired with delivery i+1.
- **H3 indirection**: Travel time between nodes requires h3_indices lookup into travel_time_matrix.
- **Num nodes**: N=61 (depot + 30 pickup-delivery pairs). Num H3 cells=31.
- **env.step()**: Takes td with "action" key, returns {"next": td_out}.
- **env.get_reward()**: Takes td and actions[B,T], returns -cost (negative travel time).

## Architecture Decisions
- Dense [B,N,N,15] edge features (N=61 small enough)
- LayerNorm not BatchNorm (variable batch sizes)
- GREAT layer: in-attention (softmax dim=1 over sources) + out-attention (softmax dim=2 over targets)
- Edge reconstruction after each layer: edge_proj(cat(node_i, node_j))
- Decoder query: W_last(h_cur) + W_first(h_first) + W_graph(h_mean) + W_visited(h_visited_mean) + W_state(state)
- Distance bias: travel_time/max_time/sqrt(2) subtracted from scores
- POMO starting nodes: pickup nodes (odd indices 1,3,5...)
- POMO baseline: mean reward across P rollouts per instance

## Implementation Notes
- gather_by_index utility is duplicated in each module for independence
- DecoderCache holds pre-computed keys and graph embedding
- _batchify_td expands TensorDict [B,...] -> [B*P,...] for POMO
- Decoder tracks _first_node for current vehicle segment context
- Temperature applied only during sampling, not greedy
