import os
import math
import json
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from openai import OpenAI
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint

class Map:
    def __init__(self, num_locations = 10, map_scale = 1, data = None, seed = 123):
        if data is None:
            self.locations = self.generate_locations(num_locations, seed = seed)
        else:
            self.locations = self.read_locations_from_data(data)
        self.num_locations = self.locations.shape[0]
        self.distances = self.compute_pairwise_distances(self.locations) * map_scale
    
    def read_locations_from_data(data):
        """
        Compute pairwise Euclidean distances from given coordinates.

        Parameters
        ----------
        data : array-like or pd.DataFrame
            Coordinates of locations. Shape: (n, 2) or columns ['x','y']

        Returns
        -------
        distance_matrix : np.ndarray
            Pairwise distance matrix of shape (n, n)
        """
        # Convert to numpy array if input is a DataFrame
        if isinstance(data, pd.DataFrame):
            locations = data[['x', 'y']].to_numpy()
        else:
            locations = np.array(data)
        return locations
    
    def generate_locations(self, n, seed = None):
        """
        Generate n random points in a unit square [0,1] x [0,1].
        
        Returns
        -------
        locations : np.ndarray
            Array of shape (n,2) with x and y coordinates.
        """
        np.random.seed(seed)
        locations = np.random.rand(n, 2)
        return locations

    def compute_pairwise_distances(self, locations):
        """
        Compute pairwise Euclidean distances between points.
        
        Parameters
        ----------
        locations : np.ndarray of shape (n,2)
        
        Returns
        -------
        distance_matrix : np.ndarray of shape (n,n)
        """
        diff = locations[:, np.newaxis, :] - locations[np.newaxis, :, :]
        distance_matrix = np.linalg.norm(diff, axis=2)
        return distance_matrix

class Customer:
    def __init__(self, customer_id, cluster_label = 0):
        self.customer_id = customer_id
        self.cluster_label = cluster_label
        # Structure: { (origin, destination): { (pickup, dropoff): probability } }
        self.preferences = defaultdict(dict)
        self.od_preferences = defaultdict(dict)
        self.max_probs = {}
    
    def set_od_preferences(self, dist):
        self.od_preferences = dist.copy()
    
    def set_preferences(self, origin, destination, joint_dist):
        """
        Set joint distribution for a specific OD pair.

        joint_dist: dict mapping (pickup_time, dropoff_time) -> probability
        Example: { (8, 9): 0.5, (9, 10): 0.3, (10, 11): 0.2 }
        """
        self.preferences[(origin, destination)] = joint_dist.copy()
        for od in self.preferences:
            _, probs = zip(*self.preferences[od].items())
            max_prob = np.max(probs)
            self.max_probs[od] = max_prob
    
    def select_od(self):
        values, probs = zip(*self.od_preferences.items())
        idx = np.random.choice(len(probs), p = probs, size = 1)[0]
        return values[idx]
    
    def select_time(self, origin, destination):
        """Pick a (pickup, dropoff) time pair from the mode of joint distribution."""
        dist = self.preferences[(origin, destination)]
        values, probs = zip(*dist.items())
        return values[np.argmax(probs)]
    
    def __repr__(self):
        return f"Customer({self.customer_id})"
    
    def decide(self, origin, destination, pickup_time, dropoff_time):
        if (pickup_time, dropoff_time) in self.preferences[(origin, destination)]:
            prob = self.preferences[(origin, destination)][(pickup_time, dropoff_time)]
        else:
            prob = 0
        max_prob = self.max_probs[(origin, destination)]
        accepted = np.random.binomial(n = 1, p = prob / max_prob)
        return accepted

class CustomerManager:
    def __init__(self, num_customers = 100, max_shift = 2, distance_matrix = None, timeslots_per_day = 20):
        self.num_customers = num_customers
        self.max_shift = max_shift
        self.distance_matrix = distance_matrix
        self.timeslots_per_day = timeslots_per_day
        self.customer_distributions, self.customer_cluster_labels = self.generate_customer_joint_time_distributions(distance_matrix, num_customers, max_shift)
        self.customer_od_dists = self.generate_od_dists(distance_matrix)
        self.customers = []
        self.create_customers(self.customer_distributions, self.customer_od_dists, self.customer_cluster_labels)
    
    def generate_od_dists(self, distance_matrix):
        customer_od_dists = []
        for cid in range(self.num_customers):
            dct = {}
            num_ods = distance_matrix.shape[0] ** 2 - distance_matrix.shape[0]
            for o in range(distance_matrix.shape[0]):
                for d in range(distance_matrix.shape[0]):
                    if o != d:
                        dct[(o, d)] = 1 / num_ods
                    else:
                        dct[(o, d)] = 0
            customer_od_dists.append(dct)
        return customer_od_dists
            
    def generate_customer_joint_time_distributions(self, distance_matrix, num_customers, delta):
        """
        Generate joint distributions of pickup and dropoff times for multiple customers.

        Parameters
        ----------
        distance_matrix : np.ndarray
            n x n matrix of travel times.
        num_customers : int
            Number of customers.
        delta : int
            Maximum time shift for distributions.

        Returns
        -------
        customer_distributions : list of dict
            Each dict has keys (od pair) -> {
                "pickup_times": array of pickup times,
                "dropoff_times": array of dropoff times,
                "joint_probs": 2D array of shape (len(pickup_times), len(dropoff_times))
            }
        """
        n = distance_matrix.shape[0]
        customer_distributions = []
        cluster_labels = []

        # Precompute central feasible times
        central_pickups = np.zeros((num_customers, n, n))
        central_dropoffs = np.zeros((num_customers, n, n))
        for c in range(num_customers):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    travel = distance_matrix[i, j]
                    t_pickup = np.random.randint(0, self.timeslots_per_day - int(travel) - 2)
                    t_dropoff = t_pickup + int(travel) + 1
                    central_pickups[c, i, j] = t_pickup
                    central_dropoffs[c, i, j] = t_dropoff
        for c in range(num_customers):
            cust_dist = {}
            is_uniform = c < num_customers // 2
            cluster_labels.append(is_uniform)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    t_pickup = central_pickups[c, i, j]
                    t_dropoff = central_dropoffs[c, i, j]
                    travel = distance_matrix[i, j]

                    # All possible pickup times ±delta
                    pickup_times = np.arange(t_pickup - delta, t_pickup + 1)
                    pickup_times = pickup_times[pickup_times >= 0]

                    # Dropoff times depend on pickup + travel
                    dropoff_times = np.arange(t_dropoff, t_dropoff + delta + 1)
                    dropoff_times = dropoff_times[dropoff_times < self.timeslots_per_day]

                    # Initialize joint probability matrix
                    joint_probs = np.zeros((len(pickup_times), len(dropoff_times)))

                    for pi, pu in enumerate(pickup_times):
                        # Dropoff must be >= pickup + travel
                        feasible_dropoffs = dropoff_times >= pu + travel
                        if np.any(feasible_dropoffs):
                            if is_uniform:
                                #probs = np.ones(feasible_dropoffs.sum()) / feasible_dropoffs.sum()
                                probs = 1 - 0.001 * ((dropoff_times[feasible_dropoffs] - t_dropoff)**2 + (pu - t_pickup) ** 2)
                            else:
                                # Peaked around central dropoff
                                probs = np.exp(-2 * ((dropoff_times[feasible_dropoffs] - t_dropoff)**2 + (pu - t_pickup) ** 2))
                            joint_probs[pi, feasible_dropoffs] = probs

                    # Normalize entire joint to sum to 1
                    if joint_probs.sum() > 0:
                        joint_probs /= joint_probs.sum()

                    cust_dist[(i, j)] = {
                        "pickup_times": pickup_times,
                        "dropoff_times": dropoff_times,
                        "joint_probs": joint_probs
                    }
                    
            new_cust_dist = {}
            for od, info in cust_dist.items():
                pickup_times = info["pickup_times"]
                dropoff_times = info["dropoff_times"]
                joint_probs = info["joint_probs"]

                prob_dict = {}
                for i, pu in enumerate(pickup_times):
                    for j, do in enumerate(dropoff_times):
                        p = joint_probs[i, j]
                        if p > 0:
                            prob_dict[(int(pu), int(do))] = p

                new_cust_dist[od] = prob_dict
            customer_distributions.append(new_cust_dist)

        return customer_distributions, cluster_labels
    
    def create_customers(self, customer_distributions, customer_od_dists, customer_cluster_labels):
        """
        Create customers with individual preference distributions.

        Parameters
        ----------
        customer_distributions : list of dict
            Each element corresponds to one customer and is a dict mapping
            (origin, destination) -> joint distribution dict.
            Example:
            [
                {("A","B"): {(8,9):0.5, (9,10):0.5}},  # Customer 0
                {("A","B"): {(8,9):0.2, (9,10):0.3, (10,11):0.5}}  # Customer 1
            ]
        od_pairs : list of (origin, destination)
        """
        for i, cust_dist in enumerate(customer_distributions):
            c = Customer(customer_id=i, cluster_label = customer_cluster_labels[i])
            for key in cust_dist:
                c.set_preferences(key[0], key[1], cust_dist[key])
            c.set_od_preferences(customer_od_dists[i])
            self.customers.append(c)

class MDP:
    def __init__(self, map, customer_manager, daily_active_customers = 100, timeslots_per_day = 20, num_vehicles = 1, vehicle_capacity = 4):
        self.map = map
        self.customer_manager = customer_manager
        self.num_regions = self.map.num_locations
        self.distances = self.map.distances
        self.total_customers = customer_manager.num_customers
        self.daily_active_customers = daily_active_customers
        self.timeslots_per_day = timeslots_per_day
        self.num_vehicles = num_vehicles
        self.vehicle_capacities = [vehicle_capacity] * num_vehicles
        
        ## Set up initial state
        self.setup()
        self.state_len = self.get_state_len()
        self.reset()
    
    def reset(self):
        self.state = torch.zeros(self.state_len)
        self.ts = 0
        self.today_active_customers = np.random.choice(self.total_customers, self.daily_active_customers, replace = False)
        self.passenger_requests = []
        self.current_vrp_cost = 0
        ## Generate new arrival
        self.current_trip_type = self.generate_arrival()
    
    def setup(self):
        ## Setup states
        triptype_idx = 0
        self.triptype_attr_to_idx = {}
        for origin in range(self.num_regions):
            for destination in range(self.num_regions):
                for pickup_time in range(self.timeslots_per_day - 1):
                    for dropoff_time in range(pickup_time + 1, self.timeslots_per_day):
                        self.triptype_attr_to_idx[(origin, destination, pickup_time, dropoff_time)] = triptype_idx
                        triptype_idx += 1
        ## Setup chat history
        self.chat_history = {}
        for cid in range(self.total_customers):
            self.chat_history[cid] = []
    
    def get_state_len(self):
        return len(self.triptype_attr_to_idx)
    
    def get_state(self):
        return self.state.clone()
    
    def get_local_state(self):
        triptype_idx = self.triptype_index_by_attributes(**self.current_trip_type)
        local_state = torch.zeros(self.state_len)
        local_state[triptype_idx] += 1
        return local_state.clone()
    
    def get_chat_history(self, cid = None):
        if cid is None:
            return self.chat_history.copy()
        return self.chat_history[cid].copy()
    
    def get_current_cid(self):
        return self.today_active_customers[self.ts]
    
    def triptype_index_by_attributes(self, origin, destination, pickup_time, dropoff_time):
        return self.triptype_attr_to_idx[(origin, destination, pickup_time, dropoff_time)]
    
    ## Return a dictionay containing origin, destination, pickup time, dropoff time
    def generate_arrival(self):
        if self.ts < self.daily_active_customers:
            cid = self.today_active_customers[self.ts]
            curr_customer = self.customer_manager.customers[cid]
            origin, destination = curr_customer.select_od()
            pickup_time, dropoff_time = curr_customer.select_time(origin, destination)
            return {"origin": origin, "destination": destination, "pickup_time": int(pickup_time), "dropoff_time": int(dropoff_time)}
        return None
    
    def customer_acceptance(self, customer_id, current_trip_type, msg):
        curr_customer = self.customer_manager.customers[customer_id]
        accepted = curr_customer.decide(**current_trip_type)
        reply = self.generate_reply_from_decision(accepted)
        return reply
    
    def generate_reply_from_decision(self, accepted):
        if accepted:
            msg = "Yes. I accept the change."
        else:
            msg = "No. I want to keep my original schedule."
        return msg
    
    def generate_message_from_action(self, action):
        return f"Can you move your earliest pickup time by {action[0]} and latest dropoff time by {action[1]}?"
    
    def infer_acceptance_from_reply(self, reply):
        if "yes" in reply.lower():
            return True
        return False
    
    ## Action is a 2-d vector indicating the shifts of pickup & drop-off times
    ##  e.g. [-1, 2] indicates shifting pickup time before by 1 timeslot and shifting dropoff time after by 2 timeslots
    def next(self, action):
        ## Obtain proposed schedule update and acceptance of the request from action
        ## There are 4 types:
        ##  1) No update, accept: Accept the request with user-proposed schedule
        ##  2) Update, accept: Propose an updated schedule, and accept the request no matter the user agrees with the change or not
        ##  3) Update, decline: Propose an updated schedule, accept the request if user agrees and otherwise decline the request
        ##  4) No update, decline: Decline the request directly
        schedule_update, acceptance = action
        
        proposed_trip_type = {
            "origin": self.current_trip_type["origin"],
            "destination": self.current_trip_type["destination"],
            "pickup_time": max(0, self.current_trip_type["pickup_time"] + int(schedule_update[0])),
            "dropoff_time": min(self.current_trip_type["dropoff_time"] + int(schedule_update[1]), self.timeslots_per_day - 1)
        }
        
        ## Determine acceptance/rejection
        customer_id = self.today_active_customers[self.ts]
        msg = self.generate_message_from_action(schedule_update)
        reply = self.customer_acceptance(customer_id, proposed_trip_type, msg)
        user_accepted = self.infer_acceptance_from_reply(reply)
        ## Update chat history
        self.chat_history[customer_id] += [msg, str(reply)]
        
        ## Update state
        if user_accepted:
            self.current_trip_type = proposed_trip_type
        triptype_idx = self.triptype_index_by_attributes(**self.current_trip_type)
        self.state[triptype_idx] += 1
        self.passenger_requests.append({
            "origin": self.current_trip_type["origin"],
            "destination": self.current_trip_type["destination"],
            "pickup_tw": (self.current_trip_type["pickup_time"], self.timeslots_per_day - 1),
            "dropoff_tw": (0, self.current_trip_type["dropoff_time"])
        })
        self.ts += 1
        
        ## Compute reward
        ## First, try to negotiate with a proposed time change (alternative time, accept/reject)
        ##  If person accept the proposed time change: inconvenience cost
        ##  If person reject the proposed time change: 0
        ## Then, decide if accept this request or not
        ##  If accepts: reward of providing the service - incremental in travel cost (i.e. vrp cost after n-th person confirms - vrp cost of past n-1 persons) - inconvenience cost (if negotiation happens)
        ##  If declines the request: 0
        proposed_change = 1 - (schedule_update[0] == 0) * (schedule_update[1] == 0)
        reward_from_service = 10
        ## For now, we assume the incremental travel cost is the distance from origin to destination
        new_vrp_cost = 0#self.vrp_cost()
        incremental_travel_cost = self.distances[self.current_trip_type["origin"], self.current_trip_type["destination"]] #new_vrp_cost - self.current_vrp_cost
        inconvenience_cost = 1
        accept_request = True
        if not acceptance:
            if not proposed_change:
                accept_request = False
            else:
                accept_request = user_accepted
        reward = accept_request * (reward_from_service - incremental_travel_cost - inconvenience_cost * proposed_change * user_accepted)
        
        self.current_vrp_cost = new_vrp_cost
        ## Generate new arrival
        self.current_trip_type = self.generate_arrival()
        return reward
    
    def vrp_cost(self):
        """
        Solve multi-vehicle VRP for passenger transport with pickup & dropoff time windows.

        Parameters
        ----------
        distance_matrix : 2D list
            Travel times/distances between all locations.
        passengers : list of dict
            Each passenger defined as:
            {
                "origin": int,
                "destination": int,
                "pickup_tw": (start, end),
                "dropoff_tw": (start, end)
            }
        vehicle_capacities : list of int
            Capacity of each vehicle (# of passengers).
        num_vehicles : int
            Number of vehicles.
        depot : int
            Index of the depot.

        Returns
        -------
        routes : list of list of int
            Routes for each vehicle (list of node indices).
        total_cost : int
            Total travel cost.
        arrival_times : dict
            Arrival times at each node.
        loads : dict
            Vehicle load at each node.
        """
        distance_matrix = self.distances
        passengers = self.passenger_requests
        vehicle_capacities = self.vehicle_capacities
        num_vehicles = self.num_vehicles
        depot = 0
        
        n = len(distance_matrix)

        # Nodes: depot + pickup/dropoff for each passenger
        num_nodes = 1 + 2 * len(passengers)  # node 0 = depot
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)
        
        # Map node index -> travel time
        node_distance = [[0]*num_nodes for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                # node 0 = depot, nodes 1..num_nodes-1 = passengers
                from_loc = 0 if i==0 else passengers[(i-1)//2]["origin" if (i-1)%2==0 else "destination"]
                to_loc = 0 if j==0 else passengers[(j-1)//2]["origin" if (j-1)%2==0 else "destination"]
                node_distance[i][j] = distance_matrix[from_loc][to_loc]

        # Distance callback
        def distance_callback(from_index, to_index):
            return node_distance[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

        transit_idx = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        # Capacity dimension
        demands = [0] * num_nodes
        for pid in range(len(passengers)):
            origin_idx, dest_idx = 2 * pid + 1, 2 * pid + 2
            demands[origin_idx] = 1   # pickup adds one passenger
            demands[dest_idx] = -1  # dropoff removes one passenger

        def demand_callback(from_index):
            return demands[manager.IndexToNode(from_index)]

        demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0, vehicle_capacities, True, "Capacity"
        )
        cap_dim = routing.GetDimensionOrDie("Capacity")

        # Time dimension
        routing.AddDimension(
            transit_idx,
            self.timeslots_per_day,   # slack
            self.timeslots_per_day,   # horizon
            False,  # don't force time 0 at depot
            "Time"
        )
        time_dim = routing.GetDimensionOrDie("Time")

        # Time windows
        # Default: depot [0, horizon]
        time_dim.CumulVar(manager.NodeToIndex(depot)).SetRange(0, self.timeslots_per_day)
        for i in range(len(passengers)):
            p = passengers[i]
            origin_idx = manager.NodeToIndex(1 + 2 * i) #manager.NodeToIndex(p["origin"])
            dest_idx = manager.NodeToIndex(2 + 2 * i) #manager.NodeToIndex(p["destination"])
            travel_time = int(math.ceil(self.distances[(p["origin"], p["destination"])]))
            # Adjust pickup latest
            pickup_latest_feasible = min(p["pickup_tw"][1], p["dropoff_tw"][1] - travel_time)
            p["pickup_tw"] = (p["pickup_tw"][0], pickup_latest_feasible)
            # Adjust dropoff earliest
            dropoff_earliest_feasible = max(p["dropoff_tw"][0], p["pickup_tw"][0] + travel_time)
            p["dropoff_tw"] = (dropoff_earliest_feasible, p["dropoff_tw"][1])
            time_dim.CumulVar(origin_idx).SetRange(*p["pickup_tw"])
            time_dim.CumulVar(dest_idx).SetRange(*p["dropoff_tw"])

        # Pickup & delivery constraints
        for i in range(len(passengers)):
            origin_idx = manager.NodeToIndex(1 + 2 * i) #manager.NodeToIndex(p["origin"])
            dest_idx = manager.NodeToIndex(2 + 2 * i) #manager.NodeToIndex(p["destination"])
            routing.AddPickupAndDelivery(origin_idx, dest_idx)
            routing.solver().Add(routing.VehicleVar(origin_idx) == routing.VehicleVar(dest_idx))
            routing.solver().Add(time_dim.CumulVar(origin_idx) <= time_dim.CumulVar(dest_idx))

        # Solver parameters
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.FromSeconds(10)

        solution = routing.SolveWithParameters(search_params)
        if not solution:
            return None #return None, None, None, None

        # Extract routes
        routes, arrival_times, loads = [], {}, {}
        total_cost = solution.ObjectiveValue()
        for v in range(num_vehicles):
            index = routing.Start(v)
            route = []
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                arrival_times[node] = solution.Value(time_dim.CumulVar(index))
                loads[node] = solution.Value(cap_dim.CumulVar(index))
                index = solution.Value(routing.NextVar(index))
            node = manager.IndexToNode(index)
            route.append(node)
            arrival_times[node] = solution.Value(time_dim.CumulVar(index))
            loads[node] = solution.Value(cap_dim.CumulVar(index))
            routes.append(route)

        return -total_cost #return routes, total_cost, arrival_times, loads

class Actor(nn.Module):
    def __init__(self, state_dim, schedule_dim, num_clusters, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim * 2 + num_clusters, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.schedule_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, schedule_dim),
            nn.Softmax(dim = -1)
        )
        
        self.acceptance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim = -1)
        )

    def forward(self, state, local_state, customer_cluster):
        if len(state.size()) == 1:
            state = state.view((1, -1))
            local_state = local_state.view((1, -1))
            customer_cluster = customer_cluster.view((1, -1))
        x_actor = torch.concat((state, local_state, customer_cluster), dim = 1)
        x = self.shared(x_actor)
        schedule_update = self.schedule_head(x)
        acceptance = self.acceptance_head(x)
        return schedule_update, acceptance

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        if len(state.size()) == 1:
            state = state.view((1, -1))
        x = self.model(state)
        return x.squeeze(-1)

class LLMPPO:
    def __init__(self, state_dim, action_dim, num_clusters, value_epochs = 200, policy_epochs = 3, batch_size = 64, gamma = 1, lam = 1, clip_epsilon = 0.2, vf_coeff = 0.1, ent_coeff = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_clusters = num_clusters
        self.actor = Actor(self.state_dim, self.action_dim, self.num_clusters)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic = Critic(self.state_dim)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.value_epochs = value_epochs
        self.policy_epochs = policy_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon_init = clip_epsilon
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.clip_epsilon = self.clip_epsilon_init
#        self.setup_chatgpt()
    
    def setup_chatgpt(self):
        # Initialize the client (make sure OPENAI_API_KEY is set in your environment)
        self.llm_client = OpenAI()
    
    def classify_customer(self, customer_chat_history):
        # Send a chat completion request
#        prompt = f"Determine the customer cluster based on the following chat history: {customer_chat_history}"
#        response = self.llm_client.chat.completions.create(
#            model="gpt-4o-mini",   # You can also use "gpt-4.1", "gpt-4o", or "gpt-3.5-turbo"
#            messages=[
#                {"role": "system", "content": "You are a helpful assistant."},
#                {"role": "user", "content": prompt}
#            ]
#        )
#
#        # Extract and print the response
#        reply = response.choices[0].message["content"]
        return torch.zeros(self.num_clusters)
    
    def query(self, state, local_state, customer_cluster):
        #return self.model(state, customer_cluster)
        probs_schedule, probs_acceptance = self.actor(state, local_state, customer_cluster)
        values = self.critic(state)
        return probs_schedule, probs_acceptance, values
    
    def compute_gae(self, rewards, values, dones):
        """
        Compute GAE advantages and returns.
        rewards: [T]
        values: [T+1] (bootstrap with last value)
        dones: [T]
        """
        rewards = rewards.flatten()
        values = values.flatten()
        dones = dones.flatten()
        T = len(rewards)
        advantages = torch.zeros(T)
        gae = 0
        for t in reversed(range(T)):
            mask = 1 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * mask - values[t]
            gae = delta + self.gamma * self.lam * gae * mask
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns
    
    def train(self, states, local_states, schedules, acceptances, old_log_probs, returns, customer_clusters, advantages, policy_itr=0):
        """
        Run multiple epochs of PPO updates on rollout batch.
        Train policy and value networks separately.
        """
        dataset_size = states.size(0)
        policy_losses, value_losses, entropies = [], [], []

        for _ in tqdm(range(self.value_epochs), leave=False):
            perm = torch.randperm(dataset_size)
            epoch_value_loss = 0
            num_batches = 0

            # -------------------------------
            # Phase 1: Train the Critic first
            # -------------------------------
            for i in range(0, dataset_size, self.batch_size):
                idx = perm[i:i + self.batch_size]

                batch_states = states[idx]
                batch_returns = returns[idx]

                # Forward critic only
                values = self.critic(batch_states)
                value_loss = (batch_returns - values).pow(2).mean()

                # Critic update
                self.optimizer_critic.zero_grad()
                value_loss.backward()
                self.optimizer_critic.step()

                epoch_value_loss += value_loss.item()
                num_batches += 1

            value_losses.append(epoch_value_loss / max(1, num_batches))
        
        for _ in tqdm(range(self.policy_epochs), leave=False):
            perm = torch.randperm(dataset_size)
            epoch_policy_loss = 0
            num_batches = 0

            # -------------------------------
            # Phase 2: Train the Actor
            # -------------------------------
            for i in range(0, dataset_size, self.batch_size):
                idx = perm[i:i + self.batch_size]

                batch_states = states[idx]
                batch_local_states = local_states[idx]
                batch_customer_clusters = customer_clusters[idx]
                batch_schedules = schedules[idx]
                batch_acceptances = acceptances[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]

                # Forward actor
                probs_schedule, probs_acceptance = self.actor(batch_states, batch_local_states, batch_customer_clusters)
                dist_schedule, dist_acceptance = Categorical(probs_schedule), Categorical(probs_acceptance)
                new_log_probs = dist_schedule.log_prob(batch_schedules) + dist_acceptance.log_prob(batch_acceptances)

                # Policy ratio
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                unclipped = ratios * batch_advantages
                self.clip_epsilon = self.clip_epsilon_init * (0.99 ** policy_itr)
                clipped = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Actor update
                self.optimizer_actor.zero_grad()
                policy_loss.backward()
                self.optimizer_actor.step()

                epoch_policy_loss += policy_loss.item()
                num_batches += 1

            # Average per epoch
            policy_losses.append(epoch_policy_loss / max(1, num_batches))

        return policy_losses, value_losses
    
    def plot_training_curves(self, policy_losses, value_losses, policy_itr=0):
        epochs = range(1, len(policy_losses) + 1)

        plt.figure(figsize=(12, 6))

        # Policy loss
        plt.subplot(1, 2, 1)
        plt.plot(policy_losses, label="Policy Loss", color="blue")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Policy Loss")
        plt.legend()

        # Value loss
        plt.subplot(1, 2, 2)
        plt.plot(value_losses, label="Value Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Value Function Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"TrainingPlots/itr={policy_itr}.png")
        plt.clf()
        plt.close()

def sim_and_train(map_scale = 1, num_days = 100, timeslots_per_day = 20, max_shift = 2, num_locations = 5, num_customers = 50, daily_active_customers = 20, num_clusters = 1, num_vehicles = 2, vehicle_capacity = 4, training_freq = 1):
    map = Map(num_locations = num_locations, map_scale = map_scale)
    customer_manager = CustomerManager(num_customers = num_customers, distance_matrix = map.distances, max_shift = max_shift, timeslots_per_day = timeslots_per_day)
    mdp = MDP(map, customer_manager, daily_active_customers = daily_active_customers, timeslots_per_day = timeslots_per_day, num_vehicles = num_vehicles, vehicle_capacity = vehicle_capacity)
    state_len = mdp.get_state_len()
    schedule_dim_single = max_shift + 1 #max_shift * 2 + 1
    schedule_dim = schedule_dim_single ** 2
    model = LLMPPO(state_len, schedule_dim, num_clusters = num_clusters)
    total_reward_lst = []
    avg_daily_reward_lst = []
    all_states = torch.zeros((num_days * daily_active_customers, state_len))
    all_local_states = torch.zeros((num_days * daily_active_customers, state_len))
    all_schedules = torch.zeros((num_days * daily_active_customers, 1))
    all_acceptances = torch.zeros((num_days * daily_active_customers, 1))
    all_rewards = torch.zeros((num_days * daily_active_customers, 1))
    all_customer_clusters = torch.zeros((num_days * daily_active_customers, num_clusters))
    all_log_probs = torch.zeros((num_days * daily_active_customers, 1))
    all_dones = torch.zeros((num_days * daily_active_customers, 1))
    all_values = torch.zeros((num_days * daily_active_customers + 1, 1))
    policy_itr = 0
    for day in tqdm(range(num_days)):
        mdp.reset()
        total_reward = 0
        for customer in range(daily_active_customers):
            cid = mdp.get_current_cid()
            state = mdp.get_state()
            local_state = mdp.get_local_state()
            customer_chat_history = mdp.get_chat_history(cid)
            ## Use ground-truth for now
            #customer_cluster = model.classify_customer(customer_chat_history)
            customer_cluster = torch.zeros(num_clusters)
            customer_cluster[int(customer_manager.customers[cid].cluster_label)] = 1
            with torch.no_grad():
                probs_schedule, probs_acceptance, value = model.query(state, local_state, customer_cluster)
            dist_schedule, dist_acceptance = Categorical(probs_schedule), Categorical(probs_acceptance)
            proposed_schedule, acceptance = dist_schedule.sample(), dist_acceptance.sample()
            ## Convert action to tuple
            converted_schedule = [int(proposed_schedule[0] // schedule_dim_single), int(proposed_schedule[0] % schedule_dim_single)]
            ## Can only shift earliest pickup window before and shift latest dropoff window after
            converted_schedule = [-converted_schedule[0], converted_schedule[1]] #[int(x - max_shift) for x in converted_action]
            action = (converted_schedule, acceptance)
            reward = mdp.next(action)
            idx = day * daily_active_customers + customer
            all_states[idx,:] += state
            all_local_states[idx,:] += local_state
            all_schedules[idx,0] += proposed_schedule.squeeze(0)
            all_acceptances[idx,0] += acceptance.squeeze(0)
            all_rewards[idx,0] += reward
            all_customer_clusters[idx,:] += customer_cluster
            all_log_probs[idx,0] += dist_schedule.log_prob(proposed_schedule)[0] + dist_acceptance.log_prob(acceptance)[0]
            all_values[idx,0] += value.squeeze(0)
            all_dones[idx,0] += customer == daily_active_customers - 1
            total_reward += reward
        total_reward_lst.append(float(total_reward))
        if day > 0 and day % training_freq == 0:
            avg_reward = float(np.mean(total_reward_lst[(day - training_freq):]))
            avg_daily_reward_lst.append(avg_reward)
            state = mdp.get_state()
            with torch.no_grad():
                _, _, value = model.query(state, local_state, torch.zeros(num_clusters))
            all_values[idx+1,0] += value.squeeze(0)
            train_begin = (day - training_freq) * daily_active_customers
            train_end = day * daily_active_customers
            advantages, returns = model.compute_gae(all_rewards[train_begin:train_end], all_values[train_begin:(train_end+1)], all_dones[train_begin:train_end])
            # Normalize advantages (helps training)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            policy_losses, value_losses = model.train(all_states[train_begin:train_end], all_local_states[train_begin:train_end], all_schedules[train_begin:train_end], all_acceptances[train_begin:train_end], all_log_probs[train_begin:train_end], returns, all_customer_clusters[train_begin:train_end], advantages, policy_itr)
            model.plot_training_curves(policy_losses, value_losses, policy_itr)
            policy_itr += 1
    chat_history = mdp.get_chat_history()
    return total_reward_lst, avg_daily_reward_lst, chat_history

total_reward_lst, avg_daily_reward_lst, chat_history = sim_and_train(map_scale = 5, num_days = 12000, timeslots_per_day = 20, max_shift = 1, num_locations = 5, num_customers = 50, daily_active_customers = 50, num_clusters = 2, num_vehicles = 2, vehicle_capacity = 4, training_freq = 300)
with open("chat_histories.txt", "w") as f:
    json.dump(chat_history, f, indent=2)
plt.plot(avg_daily_reward_lst)
plt.xlabel("Policy Iteration")
plt.ylabel("Avg. Daily Reward")
plt.savefig("reward_curve.png")
plt.clf()
plt.close()
