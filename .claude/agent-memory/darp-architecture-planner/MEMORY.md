# DARP Architecture Planner Memory

## Project Structure
- **Env**: `src/oracle_env.py` - PDPTWEnv with TensorDict state, action_mask, pending_schedule
- **Generator**: `src/oracle_generator.py` - SFGenerator, 30 customers, ~31 H3 cells, 61 nodes (depot + 30 pairs)
- **Existing Policy**: `src/oracle_policy.py` - PDPTWAttentionPolicy (standard transformer, no edge features)
- **PPO Agent**: `src/dvrp_ppo_agent_gcn.py` - GCN + Transformer encoder for meta-level DVRP
- **Meta Env**: `src/dvrp_env.py` - DVRPEnv wrapping PDPTWEnv for time-window perturbation
- **Trial files**: `src/trial_encoder.py`, `trial_decoder.py`, `trial_gnn_policy.py`, `trial_pomo_reinforce.py` (all empty)

## Reference Code (GREAT)
- `reference/great/models/great.py` - GREATEncoder, GREATLayer, GREATLayerAsymmetric, GREATLayerNodeless, FFLayer
- `reference/great/decoding/cvrp_decoder.py` - CVRPDecoder (query = graph + first + last + visited + load)
- `reference/great/decoding/tsp_decoder.py` - DecoderForLarge (similar pattern)
- `reference/great/models/cvrp_model.py` - GREATRL_CVRP (full encoder-decoder + POMO training)
- `reference/great/envs/cvrp_env.py` - CVRPEnv with POMO rollout (dataclass states, ninf_mask)

## Key Architecture Patterns
- GREAT uses **edge-level attention** via MessagePassing; edges are primary, nodes derived
- GREATLayerAsymmetric: separate in/out edge projections, 4x concat for nodes
- Asymmetric variant critical for DARP (travel_time(i->j) != travel_time(j->i))
- Final layer: TransformerConv to produce node embeddings from edge embeddings
- CVRP decoder: `score = (q * k) / sqrt(H) - distances / sqrt(2)`, then tanh_clipping
- POMO: second move selects nodes 1..P for P parallel rollouts

## DARP-Specific Constraints
- Node indexing: 0=depot, odd=pickup, even=delivery; pickup i paired with delivery i+1
- Env can RETRACT visited nodes on depot return if deliveries not completed
- action_mask computed by env (immediate reachability + one-step lookahead)
- pending_schedule tracks onboard delivery obligations (max = vehicle_capacity)
- travel_time_matrix indexed by h3_indices (indirection layer)
- Typical problem: 30 customers = 61 nodes, ~31 H3 cells

## Design Decision: Dense Edge Representation
- With 61 nodes, full graph = 61*61=3721 edges - manageable for dense representation
- Travel time matrix is 31x31 H3 cells, but node-level is 61x61 after h3_indices lookup
- Use GREATLayerAsymmetric since travel times are asymmetric

## Links
- See `architecture-plan.md` for full implementation plan
