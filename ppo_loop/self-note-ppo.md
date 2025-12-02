State Space:
    - TensorDict with keys:
        * 'fixed': Tensor [num_customers, 6] for all requests (zeros for not-yet-accepted)
        * 'new': Tensor [1, 6] for current incoming request
        * 'distance_matrix': Tensor [num_locations, num_locations]
        * 'step': Scalar tensor for current step number
Such is the observation space in dvrp_env. 
What is of interest to read from this Tensordict is: 
the fixed nodes are stored in a structuralized fixed. 

The ppo have the following structure: 

1. Projection (Linear Layer: 6-> 128)
2. ValueNetwork & PolicyNetwork
    Value netowrk outputs scaler, policy network outpus: logits over 16 actions
    Both network has the same structure as in 3., save for the output dim: 

3. Multi-head Cross Attention Mechanism
    Final Output to FFN Layer 

4. Class PPOAgent encompasses all above.


