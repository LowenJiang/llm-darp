from typing import Callable, Union

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.ops import get_distance
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DARPGenerator(Generator):
    """Data generator for the Dial-a-Ride Problem (DARP) environment.
    
    Generates pickup and dropoff locations with time windows and demands.
    Each DARP instance consists of a pickup-dropoff pair, where the pickup has positive demand
    and the dropoff has negative demand of the same magnitude.
    
    Args:
        num_loc: number of locations (must be even), corresponding to num_loc/2 DARP instances
        num_agents: number of vehicles/agents
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        vehicle_capacity: capacity of each vehicle
        min_demand: minimum demand value (for pickups)
        max_demand: maximum demand value (for pickups)
        demand_distribution: distribution for the demand
        vehicle_speed: speed of the vehicle (used to convert distance to time)
        min_pickup_tw: minimum time window for pickup nodes
        max_pickup_tw: maximum time window for pickup nodes
        
    Returns:
        A TensorDict with the following keys:
            locs [batch_size, num_loc, 2]: locations of each pickup/dropoff node
            depot [batch_size, 2]: location of the depot
            time_windows [batch_size, num_loc]: time window (integer 1-48) for each node
            demand [batch_size, num_loc]: demand of each node (positive for pickup, negative for dropoff)
            capacity [batch_size, num_agents]: capacity of each vehicle
    """
    
    def __init__(
        self,
        num_loc: int = 20,
        num_agents: int = 5,
        min_loc: float = 0.0,
        max_loc: float = 100.0,
        loc_distribution: Union[int, float, str, type, Callable] = Uniform,
        depot_distribution: Union[int, float, str, type, Callable] = Uniform,
        vehicle_capacity: int = 5,
        min_demand: int = 1,
        max_demand: int = 3,
        demand_distribution: Union[int, float, str, type, Callable] = Uniform,
        vehicle_speed: float = 25.0,
        min_pickup_tw: int = 10,
        max_pickup_tw: int = 30,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.num_agents = num_agents
        self.min_loc = min_loc
        self.max_loc = max_loc
        self.vehicle_capacity = vehicle_capacity
        self.min_demand = min_demand
        self.max_demand = max_demand
        self.vehicle_speed = vehicle_speed
        self.min_pickup_tw = min_pickup_tw
        self.max_pickup_tw = max_pickup_tw
        
        # Number of locations must be even (pickup-dropoff pairs)
        if num_loc % 2 != 0:
            log.warning(
                "Number of locations must be even. Adding 1 to the number of locations."
            )
            self.num_loc += 1
        
        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )
        
        # Depot distribution
        if kwargs.get("depot_sampler", None) is not None:
            self.depot_sampler = kwargs["depot_sampler"]
        else:
            self.depot_sampler = get_sampler(
                "depot", depot_distribution, min_loc, max_loc, **kwargs
            )
        
        # Demand distribution
        if kwargs.get("demand_sampler", None) is not None:
            self.demand_sampler = kwargs["demand_sampler"]
        else:
            self.demand_sampler = get_sampler(
                "demand", demand_distribution, min_demand - 1, max_demand - 1, **kwargs
            )
    
    def _generate(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        
        # Sample depot location
        depot = self.depot_sampler.sample((*batch_size, 2))
        
        # Sample locations for all nodes (pickups and dropoffs)
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        
        # Sample demands for pickup nodes
        demand_values = self.demand_sampler.sample((*batch_size, self.num_loc // 2))
        demand_values = (demand_values.int() + 1).float()  # Convert to integers in [min_demand, max_demand]
        
        # Create demand tensor: positive for pickups, negative for dropoffs
        demand = torch.zeros(*batch_size, self.num_loc)
        for i in range(self.num_loc // 2):
            demand[..., 2 * i] = demand_values[..., i]      # Pickup: positive
            demand[..., 2 * i + 1] = -demand_values[..., i]  # Dropoff: negative
        
        # Generate time windows
        time_windows = torch.zeros(*batch_size, self.num_loc, dtype=torch.long)
        
        for i in range(self.num_loc // 2):
            pickup_idx = 2 * i
            dropoff_idx = 2 * i + 1
            
            # Sample pickup time window uniformly from [min_pickup_tw, max_pickup_tw]
            pickup_tw = torch.randint(
                self.min_pickup_tw, 
                self.max_pickup_tw + 1, 
                size=batch_size,
                dtype=torch.long
            )
            time_windows[..., pickup_idx] = pickup_tw
            
            # Calculate distance between pickup and dropoff
            pickup_locs = locs[..., pickup_idx, :]
            dropoff_locs = locs[..., dropoff_idx, :]
            dist = torch.norm(pickup_locs - dropoff_locs, p=2, dim=-1)
            
            # Convert distance to time using vehicle speed
            travel_time = dist / self.vehicle_speed
            
            # Ensure minimum travel time of 1
            travel_time = torch.max(travel_time, torch.ones_like(travel_time))
            
            # Sample dropoff time window uniformly from 
            # [pickup_tw + travel_time, pickup_tw + 2 * travel_time]
            min_dropoff_tw = pickup_tw + torch.ceil(travel_time).long()
            max_dropoff_tw = pickup_tw + torch.ceil(2 * travel_time).long()
            
            # Handle case where min == max
            max_dropoff_tw = torch.max(max_dropoff_tw, min_dropoff_tw + 1)
            
            # Sample dropoff time window
            dropoff_tw = torch.zeros_like(pickup_tw)
            for b_idx in range(dropoff_tw.numel()):
                flat_idx = b_idx
                dropoff_tw.view(-1)[flat_idx] = torch.randint(
                    min_dropoff_tw.view(-1)[flat_idx].item(),
                    max_dropoff_tw.view(-1)[flat_idx].item() + 1,
                    size=(1,)
                ).item()
            
            time_windows[..., dropoff_idx] = dropoff_tw
        
        # Create capacity tensor
        capacity = torch.full((*batch_size, self.num_agents), self.vehicle_capacity, dtype=torch.long)
        
        return TensorDict(
            {
                "locs": locs,
                "depot": depot,
                "time_windows": time_windows,
                "demand": demand,
                "capacity": capacity,
            },
            batch_size=batch_size,
        )