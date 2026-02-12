# DARP Encoder-Decoder Architecture Plan

## Summary of Files
- trial_encoder.py: GREAT-style edge-based encoder adapted for DARP
- trial_decoder.py: Attention-based decoder with DARP context
- trial_gnn_policy.py: Complete policy combining encoder + decoder
- trial_pomo_reinforce.py: POMO REINFORCE training algorithm

## Key Design Decisions
1. Use GREATLayerAsymmetric for asymmetric travel times
2. Dense edge representation (61 nodes = 3721 edges, manageable)
3. Edge features: [travel_time, node_type_src, node_type_dst, tw_compatibility, demand_src, demand_dst]
4. Use env's action_mask directly (already has lookahead logic)
5. POMO starts from pickup nodes only (indices 1,3,5,...)
6. Decoder context: current_node + first_node + graph_mean + visited_mean + state_features

## Hyperparameters
- hidden_dim=128, num_layers=5, num_heads=8, tanh_clipping=10
- POMO rollouts: min(20, num_pickups)
- Edge feature initial_dim: 8
