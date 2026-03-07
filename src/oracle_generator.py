"""
SF_simulator description:
simulate based on synthetic SF travelr_trip_types.csv.
-----------------------
Input: (default values specified)
num_customers = 30  # How many customers we are sampling each round
perturbation = 30 # Randomly take 10min-wide perturbations up to 30: 10, 20, 30; to shift earlier for pickup node and to delay for dropoff node
-----------------------
Simulation:
Depot is set as default to be 37°47N, 122°25W
each round simulate num_customers requests. Request i (in range(1, num_customers+1)) corresponds to traveler_id == i
Each simulation round for request i, uniformly sample from the rows for traveler_id == i
Collect the columns: "origin_h3", "destination_h3", "departure_time_window", "arrival_time_window"

Node order:
node 0: depot
node 2i-1: pickup_node for request i
node 2i  : dropoff_node for request i
-----------------------
Note on the data type:
For the origin_h3, destination_h3: (H3 cell indices)
data = pd.read_csv("traveler_trip_types.csv")
data["origin_h3"][0] : '89283082d97ffff'

For the departure_time_window, arrival_time_window:
data["departure_time_window"][0].split('-')[0] = '08:00' : str
data["departure_time_window"][0].split('-')[0] = '08:30' : str
-----------------------
Methods:
_generate(self, batch_size)
-----------------------
Output:
return TensorDict({
    "h3_indices": # shape[batch_size, num_customers*2] - H3 cell indices for each node
    "travel_time_matrix": # shape[num_h3_cells, num_h3_cells] - travel times in minutes
    "capacity":  # vehicle capacity, default = 4
    "time_windows":  # shape[batch_size, num_customers*2, 2]; store time as min; i.e. str('8:00') => 480, str('8:30') => 510
    "demand": randomized_demand, here all customers treated as =1
}
-----------------------
Create a main func to see the output
"""

from __future__ import annotations
import h3
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import logging
import pandas as pd
import torch
from tensordict.tensordict import TensorDict

log = logging.getLogger(__name__)

Location = Tuple[float, float]
TimeWindow = Tuple[int, int]
H3Index = str


class Generator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, batch_size) -> TensorDict:
        batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        return self._generate(batch_size)

    def _generate(self, batch_size, **kwargs) -> TensorDict:
        raise NotImplementedError("Generator subclasses must override `_generate`.")


class SFGenerator(Generator):
    """Generator that replays synthetic San Francisco ride requests from a CSV file."""

    def __init__(
        self,
        csv_path: Optional[str | Path] = None,
        travel_time_matrix_path: Optional[str | Path] = None,
        num_customers: int = 30,
        perturbation: int = 30, # turn to 30 for enrichment; turn to 0 for off
        vehicle_capacity: int = 8,
        demand_per_customer: int = 1,
        depot_h3: Optional[str] = None,
        shuffle_pairs: bool = False,
        seed: Optional[int] = None,
        device: torch.device | str = "cpu",
        pickup_earliest_min: Optional[int] = None,
        dropoff_latest_min: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            csv_path: Optional custom path to ``traveler_trip_types.csv``.
            travel_time_matrix_path: Optional custom path to ``travel_time_matrix_h3.csv``.
            num_customers: Number of customer requests sampled each round.
            perturbation: Maximum shift (minutes) applied in 10-minute steps.
            vehicle_capacity: Vehicle capacity reported in the TensorDict.
            demand_per_customer: Demand magnitude for each pickup node.
            depot_h3: Optional H3 index for the depot; defaults to a central SF H3 cell.
            shuffle_pairs: Whether to shuffle pickup/dropoff pairs in the output.
            seed: Optional deterministic seed for sampling.
            pickup_earliest_min: If set, only keep CSV trips whose pickup window
                start >= this value (minutes after midnight).
            dropoff_latest_min: If set, only keep CSV trips whose dropoff window
                end <= this value (minutes after midnight).
        """
        super().__init__(**kwargs)

        # Normalize device to ensure it has an index
        self.device = torch.device(device) if isinstance(device, str) else device
        if self.device.type in ("mps", "cuda") and self.device.index is None:
            self.device = torch.device(f"{self.device.type}:0")

        self.num_customers = num_customers
        self.perturbation = max(0, perturbation)
        self.vehicle_capacity = vehicle_capacity
        self.pickup_earliest_min = pickup_earliest_min
        self.dropoff_latest_min = dropoff_latest_min
        self.demand_per_customer = demand_per_customer
        self.shuffle_pairs = shuffle_pairs
        self.csv_path = (
            Path(csv_path) if csv_path is not None else Path(__file__).with_name("traveler_trip_types_res_7.csv")
        )
        self.travel_time_matrix_path = (
            Path(travel_time_matrix_path) if travel_time_matrix_path is not None
            else Path(__file__).with_name("travel_time_matrix_res_7.csv")
        )
        self.day_minutes = 24 * 60
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(int(seed))
        else:
            self.generator.seed()

        # Load travel time matrix and create H3 to index mapping
        self._h3_to_idx, self._idx_to_h3, self.travel_time_matrix = self._load_travel_time_matrix()

        # Set depot H3 index (default to a central SF H3 cell that should be in the matrix)
        # Using H3 cell that covers downtown SF area
        if depot_h3 is None:
            # Find a suitable depot H3 from the matrix (use first available or a known central one)
            # Default to a common H3 cell in SF - 89283082877ffff is near downtown
            depot_h3 = h3.cell_to_parent("89283082877ffff", 7)

        if depot_h3 not in self._h3_to_idx:
            available_h3 = list(self._h3_to_idx.keys())[:5]
            raise ValueError(
                f"Depot H3 index '{depot_h3}' not found in travel time matrix. "
                f"Available H3 indices include: {available_h3}..."
            )
        self.depot_h3 = depot_h3
        self.depot_h3_idx = self._h3_to_idx[depot_h3]
        
        self._h3_to_gps = {} # To store GPS coordinates for rendering

        # Precompute H3 cell centroids [num_cells, 2] — (lat, lng)
        # All nodes sharing an H3 cell get identical spatial embeddings,
        # matching the granularity of the travel-time cost function.
        self._h3_centroids = self._build_centroid_table()

        # CPU copy of travel-time matrix for fast perturbation feasibility checks
        self._travel_time_cpu = self.travel_time_matrix.cpu()

        self._trips_by_traveler = self._load_trips()
        expected_ids = set(range(1, self.num_customers + 1))
        if not expected_ids.issubset(self._trips_by_traveler):
            missing = sorted(expected_ids - self._trips_by_traveler.keys())
            raise ValueError(
                f"Requested traveler IDs {missing} are missing from {self.csv_path}. "
                "Ensure the CSV contains rows for the requested IDs."
            )
        self._traveler_ids = sorted(expected_ids)
        self._precompute_trip_arrays()

    def _flexibility_to_index(self, flexibility_str: str) -> int:
        """Convert flexibility string to numeric index."""
        flexibility_map = {
            "flexible for both early pickup and late dropoff": 0,
            "flexible for early pickup, but inflexible for late dropoff": 1,
            "flexible for late dropoff, but inflexible for early pickup": 2,
            "inflexible for any schedule changes": 3,
        }
        return flexibility_map.get(flexibility_str, 3)  # Default to inflexible if unknown

    def _generate(self, batch_size) -> TensorDict:
        batch_shape = self._normalize_batch_size(batch_size)
        flat_batch = math.prod(batch_shape)
        num_nodes = self.num_customers * 2

        # Allocate on CPU for fast batch filling; bulk-transfer to device after loop
        h3_indices = torch.zeros(flat_batch, num_nodes, dtype=torch.long)
        locs = torch.zeros(flat_batch, num_nodes, 2, dtype=torch.float32)
        time_windows = torch.zeros(flat_batch, num_nodes, 2, dtype=torch.float32)
        demand = torch.zeros(flat_batch, num_nodes, dtype=torch.float32)
        user_id = torch.zeros(flat_batch, num_nodes, dtype=torch.long)
        flexibility = torch.zeros(flat_batch, num_nodes, dtype=torch.float32)
        trip_metadata_batch = [{} for _ in range(flat_batch)]

        # Vectorized: iterate over 30 customers, batch-sample all instances at once
        for cust_idx, traveler_id in enumerate(self._traveler_ids):
            trips = self._trips_by_traveler[traveler_id]
            arr = self._trip_arrays[traveler_id]
            n_trips = len(trips)

            # Batch-sample trip indices for all instances
            if n_trips == 1:
                trip_indices = torch.zeros(flat_batch, dtype=torch.long)
            else:
                trip_indices = torch.randint(
                    n_trips, (flat_batch,), generator=self.generator
                )

            # Gather sampled trip fields via batch indexing (all CPU)
            sampled_origin_h3 = arr["origin_h3_idx"][trip_indices]
            sampled_dest_h3 = arr["dest_h3_idx"][trip_indices]
            sampled_pickup_tw = arr["pickup_tw"][trip_indices]    # [flat_batch, 2]
            sampled_dropoff_tw = arr["dropoff_tw"][trip_indices]  # [flat_batch, 2]
            sampled_origin_gps = arr["origin_gps"][trip_indices]  # [flat_batch, 2]
            sampled_dest_gps = arr["dest_gps"][trip_indices]      # [flat_batch, 2]
            sampled_flex = arr["flex_idx"][trip_indices]           # [flat_batch]

            # Apply perturbation via rejection sampling (fully vectorized)
            # Sample random deltas, keep only those that preserve the
            # feasibility constraint: dropoff_early - pickup_late >= travel_time + 15
            if self.perturbation > 0:
                travel_time = self._travel_time_cpu[sampled_origin_h3, sampled_dest_h3]
                min_gap = travel_time + 15.0

                # Start with no shift applied
                pickup_deltas = torch.zeros(flat_batch)
                dropoff_deltas = torch.zeros(flat_batch)
                unsettled = torch.ones(flat_batch, dtype=torch.bool)

                for _attempt in range(10):
                    if not unsettled.any():
                        break
                    n_open = int(unsettled.sum().item())
                    # Sample candidate deltas for unsettled instances
                    cand_p = self._sample_perturbation_batch(n_open)
                    cand_d = self._sample_perturbation_batch(n_open)
                    # Compute shifted windows
                    cand_pickup_tw = self._shift_window_signed(
                        sampled_pickup_tw[unsettled], cand_p
                    )
                    cand_dropoff_tw = self._shift_window_signed(
                        sampled_dropoff_tw[unsettled], cand_d
                    )
                    # Check feasibility: gap >= travel_time + 15
                    gap = cand_dropoff_tw[:, 0] - cand_pickup_tw[:, 1]
                    feasible = gap >= min_gap[unsettled]
                    # Accept feasible candidates
                    accept_idx = unsettled.nonzero(as_tuple=True)[0][feasible]
                    pickup_deltas[accept_idx] = cand_p[feasible]
                    dropoff_deltas[accept_idx] = cand_d[feasible]
                    unsettled[accept_idx] = False

                # unsettled instances keep delta=0 (original windows are feasible)
                sampled_pickup_tw = self._shift_window_signed(sampled_pickup_tw, pickup_deltas)
                sampled_dropoff_tw = self._shift_window_signed(sampled_dropoff_tw, dropoff_deltas)

            pickup_idx = cust_idx * 2
            dropoff_idx = pickup_idx + 1

            h3_indices[:, pickup_idx] = sampled_origin_h3
            h3_indices[:, dropoff_idx] = sampled_dest_h3
            locs[:, pickup_idx] = sampled_origin_gps
            locs[:, dropoff_idx] = sampled_dest_gps
            time_windows[:, pickup_idx] = sampled_pickup_tw
            time_windows[:, dropoff_idx] = sampled_dropoff_tw
            demand[:, pickup_idx] = self.demand_per_customer
            demand[:, dropoff_idx] = -self.demand_per_customer
            user_id[:, pickup_idx] = traveler_id
            user_id[:, dropoff_idx] = traveler_id
            flexibility[:, pickup_idx] = sampled_flex
            flexibility[:, dropoff_idx] = sampled_flex

            # Build trip metadata (Python dicts — fast even in flat_batch loop)
            indices_list = trip_indices.tolist()
            for i, idx in enumerate(indices_list):
                trip = trips[idx]
                trip_metadata_batch[i][traveler_id] = {
                    "flexibility": trip["flexibility"],
                    "trip_purpose": trip["trip_purpose"],
                    "departure_location": trip["departure_location"],
                    "arrival_location": trip["arrival_location"],
                    "departure_time_window": trip["departure_time_window"],
                    "arrival_time_window": trip["arrival_time_window"],
                }

        # Bulk transfer to device
        h3_indices = h3_indices.to(self.device)
        locs = locs.to(self.device)
        time_windows = time_windows.to(self.device)
        demand = demand.to(self.device)
        user_id = user_id.to(self.device)
        flexibility = flexibility.to(self.device)

        if self.shuffle_pairs and self.num_customers > 1:
            h3_indices, time_windows, demand, locs, user_id, flexibility = self._shuffle_pairs(
                h3_indices, time_windows, demand, locs, user_id, flexibility
            )

        # Add depot time window
        depot_tw = torch.zeros(flat_batch, 1, 2, dtype=time_windows.dtype, device=self.device)
        depot_tw[:, 0, 1] = float(self.day_minutes)
        time_windows = torch.cat([depot_tw, time_windows], dim=1)

        # Add depot H3 index at position 0
        depot_h3_indices = torch.full((flat_batch, 1), self.depot_h3_idx, dtype=torch.long, device=self.device)
        h3_indices = torch.cat([depot_h3_indices, h3_indices], dim=1)

        # Add depot GPS coordinates at position 0 (H3 centroid)
        depot_locs = self._h3_centroids[self.depot_h3_idx].view(1, 1, 2).expand(flat_batch, 1, 2)
        locs = torch.cat([depot_locs, locs], dim=1)

        # Add depot user_id (0 for depot)
        depot_user_id = torch.zeros(flat_batch, 1, dtype=torch.long, device=self.device)
        user_id = torch.cat([depot_user_id, user_id], dim=1)

        # Add depot demand (0 for depot)
        depot_demand = torch.zeros(flat_batch, 1, dtype=torch.float32, device=self.device)
        demand = torch.cat([depot_demand, demand], dim=1)

        # Add depot flexibility (0 for depot)
        depot_flexibility = torch.zeros(flat_batch, 1, dtype=torch.float32, device=self.device)
        flexibility = torch.cat([depot_flexibility, flexibility], dim=1)

        capacity = torch.full((flat_batch,), float(self.vehicle_capacity), dtype=torch.float32, device=self.device)

        # Reshape for batch dimensions
        h3_indices = h3_indices.view(*batch_shape, num_nodes + 1)
        locs = locs.view(*batch_shape, num_nodes + 1, 2)
        time_windows = time_windows.view(*batch_shape, num_nodes + 1, 2)
        demand = demand.view(*batch_shape, num_nodes + 1)
        capacity = capacity.view(*batch_shape)
        user_id = user_id.view(*batch_shape, num_nodes + 1) # Note: user_id [0] corresponds to depot
        flexibility = flexibility.view(*batch_shape, num_nodes + 1)

        # Expand travel time matrix to batch size
        # Use expand to avoid copying data
        travel_time_matrix = self.travel_time_matrix.expand(*batch_shape, *self.travel_time_matrix.shape)

        return TensorDict(
            {
                "user_id": user_id,
                "locs": locs,
                "h3_indices": h3_indices,
                "travel_time_matrix": travel_time_matrix,
                "capacity": capacity,
                "time_windows": time_windows,
                "demand": demand,
                "flexibility": flexibility,
                "trip_metadata": trip_metadata_batch,  # List of dicts mapping traveler_id -> metadata
            },
            batch_size=batch_shape,
        )

    def _load_travel_time_matrix(self) -> Tuple[Dict[str, int], Dict[int, str], torch.Tensor]:
        """
        Load the travel time matrix from CSV and create H3 to index mappings.

        Returns:
            h3_to_idx: Dictionary mapping H3 string to integer index
            idx_to_h3: Dictionary mapping integer index to H3 string
            travel_time_matrix: Tensor of shape [num_h3_cells, num_h3_cells] with travel times in minutes
        """
        if not self.travel_time_matrix_path.exists():
            raise FileNotFoundError(f"Travel time matrix file not found at {self.travel_time_matrix_path}")

        df = pd.read_csv(self.travel_time_matrix_path, index_col=0)

        # Get H3 indices from column names
        h3_cells = list(df.columns)

        # Create mappings
        h3_to_idx = {h3: idx for idx, h3 in enumerate(h3_cells)}
        idx_to_h3 = {idx: h3 for idx, h3 in enumerate(h3_cells)}

        # Convert to tensor and convert seconds to minutes
        travel_time_matrix = torch.tensor(df.values, dtype=torch.float32, device=self.device) / 60.0

        log.info(f"Loaded travel time matrix with {len(h3_cells)} H3 cells")

        return h3_to_idx, idx_to_h3, travel_time_matrix

    def _build_centroid_table(self) -> torch.Tensor:
        """Build a [num_h3_cells, 2] tensor mapping cell index → (lat, lng) centroid."""
        num_cells = len(self._idx_to_h3)
        centroids = torch.zeros(num_cells, 2, dtype=torch.float32, device=self.device)
        for idx, h3_str in self._idx_to_h3.items():
            lat, lng = h3.cell_to_latlng(h3_str)
            centroids[idx, 0] = lat
            centroids[idx, 1] = lng
        return centroids

    def _load_trips(self) -> Dict[int, List[Dict[str, Tuple[int, int] | H3Index]]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_columns = {
            "traveler_id", "origin_h3", "destination_h3",
            "departure_time_window", "arrival_time_window",
            "flexibility", "trip_purpose", "departure_location", "arrival_location"
        }
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns {missing_cols} in {self.csv_path}")

        trips: Dict[int, List[Dict[str, Tuple[int, int] | H3Index]]] = {}
        for traveler_id, group in df.groupby("traveler_id"):
            processed: List[Dict[str, Tuple[int, int] | H3Index]] = []
            for _, row in group.iterrows():
                try:
                    origin_h3 = str(row["origin_h3"]).strip()
                    destination_h3 = str(row["destination_h3"]).strip()
                    pickup_tw = self._parse_time_window(row["departure_time_window"])
                    dropoff_tw = self._parse_time_window(row["arrival_time_window"])

                    # Parse [lat, lon]
                    origin_gps_str = str(row["origin_gps"]).strip()
                    destination_gps_str = str(row["destination_gps"]).strip()
                    
                    def parse_gps(s):
                        s = s.strip("[]")
                        lat, lon = map(float, s.split(","))
                        return lat, lon
                    
                    origin_gps = parse_gps(origin_gps_str)
                    destination_gps = parse_gps(destination_gps_str)

                    # Filter by time-window bounds (prune pool before sampling)
                    if self.pickup_earliest_min is not None and pickup_tw[0] < self.pickup_earliest_min:
                        continue
                    if self.dropoff_latest_min is not None and dropoff_tw[1] > self.dropoff_latest_min:
                        continue

                    # Validate H3 indices exist in the travel time matrix
                    if origin_h3 not in self._h3_to_idx:
                        log.warning(f"Origin H3 {origin_h3} not found in travel time matrix, skipping trip")
                        continue
                    if destination_h3 not in self._h3_to_idx:
                        log.warning(f"Destination H3 {destination_h3} not found in travel time matrix, skipping trip")
                        continue

                    # Update h3 to gps cache
                    if origin_h3 not in self._h3_to_gps:
                        self._h3_to_gps[origin_h3] = origin_gps
                    if destination_h3 not in self._h3_to_gps:
                        self._h3_to_gps[destination_h3] = destination_gps

                    # Parse trip metadata
                    flexibility = str(row["flexibility"]).strip()
                    trip_purpose = str(row["trip_purpose"]).strip()
                    departure_location = str(row["departure_location"]).strip()
                    arrival_location = str(row["arrival_location"]).strip()

                except (TypeError, ValueError) as e:
                    log.warning(f"Error parsing trip: {e}")
                    continue

                processed.append(
                    {
                        "origin_h3": origin_h3,
                        "destination_h3": destination_h3,
                        "pickup_tw": pickup_tw,
                        "dropoff_tw": dropoff_tw,
                        "origin_gps": origin_gps,
                        "destination_gps": destination_gps,
                        "flexibility": flexibility,
                        "trip_purpose": trip_purpose,
                        "departure_location": departure_location,
                        "arrival_location": arrival_location,
                        "departure_time_window": str(row["departure_time_window"]).strip(),
                        "arrival_time_window": str(row["arrival_time_window"]).strip(),
                    }
                )

            if processed:
                trips[int(traveler_id)] = processed

        if not trips:
            raise ValueError(f"No valid trips parsed from {self.csv_path}")

        return trips

    def _sample_trip(self, traveler_id: int) -> Dict[str, H3Index | TimeWindow]:
        trips = self._trips_by_traveler[traveler_id]
        if len(trips) == 1:
            return trips[0]

        # Randomly sample a trip; which includes all triptype information
        idx = self._randint(len(trips))
        return trips[idx]

    def _shuffle_pairs(
        self,
        h3_indices: torch.Tensor,
        time_windows: torch.Tensor,
        demand: torch.Tensor,
        locs: torch.Tensor,
        user_id: torch.Tensor,
        flexibility: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_instances = h3_indices.shape[0]
        randomized_h3_indices = torch.zeros_like(h3_indices)
        randomized_time_windows = torch.zeros_like(time_windows)
        randomized_demand = torch.zeros_like(demand)
        randomized_locs = torch.zeros_like(locs)
        randomized_user_id = torch.zeros_like(user_id)
        randomized_flexibility = torch.zeros_like(flexibility)

        for instance_idx in range(num_instances):
            order = torch.randperm(self.num_customers, generator=self.generator)
            for new_pair_idx, old_pair_idx in enumerate(order.tolist()):
                src_pickup = old_pair_idx * 2
                src_dropoff = src_pickup + 1
                dst_pickup = new_pair_idx * 2
                dst_dropoff = dst_pickup + 1

                randomized_h3_indices[instance_idx, dst_pickup] = h3_indices[instance_idx, src_pickup]
                randomized_h3_indices[instance_idx, dst_dropoff] = h3_indices[instance_idx, src_dropoff]
                randomized_time_windows[instance_idx, dst_pickup] = time_windows[instance_idx, src_pickup]
                randomized_time_windows[instance_idx, dst_dropoff] = time_windows[instance_idx, src_dropoff]
                randomized_demand[instance_idx, dst_pickup] = demand[instance_idx, src_pickup]
                randomized_demand[instance_idx, dst_dropoff] = demand[instance_idx, src_dropoff]

                randomized_locs[instance_idx, dst_pickup] = locs[instance_idx, src_pickup]
                randomized_locs[instance_idx, dst_dropoff] = locs[instance_idx, src_dropoff]

                randomized_user_id[instance_idx, dst_pickup] = user_id[instance_idx, src_pickup]
                randomized_user_id[instance_idx, dst_dropoff] = user_id[instance_idx, src_dropoff]

                randomized_flexibility[instance_idx, dst_pickup] = flexibility[instance_idx, src_pickup]
                randomized_flexibility[instance_idx, dst_dropoff] = flexibility[instance_idx, src_dropoff]

        return randomized_h3_indices, randomized_time_windows, randomized_demand, randomized_locs, randomized_user_id, randomized_flexibility

    def _sample_perturbation_step(self) -> int:
        if self.perturbation < 10:
            return 0
        steps = torch.arange(10, self.perturbation + 1, 10)
        idx = self._randint(len(steps))
        return int(steps[idx].item())

    def _shift_window(self, window: TimeWindow, direction: str, delta: int) -> TimeWindow:
        if delta == 0:
            return window
        start, end = window
        duration = max(0, end - start)

        if direction == "earlier":
            new_start = max(0, start - delta)
            new_end = min(self.day_minutes, new_start + duration)
        elif direction == "later":
            new_end = min(self.day_minutes, end + delta)
            new_start = max(0, new_end - duration)
        else:
            raise ValueError(f"Unknown direction '{direction}' for window shift.")

        if new_end <= new_start:
            new_end = min(self.day_minutes, new_start + max(duration, 10))
        return int(new_start), int(new_end)

    def _precompute_trip_arrays(self):
        """Pre-build CPU tensors for each traveler's trips for fast batch sampling."""
        # H3 centroids are on self.device; move to CPU for batch construction
        centroids_cpu = self._h3_centroids.cpu()
        self._trip_arrays = {}
        for traveler_id in self._traveler_ids:
            trips = self._trips_by_traveler[traveler_id]
            origin_h3_idx = torch.tensor(
                [self._h3_to_idx[t["origin_h3"]] for t in trips], dtype=torch.long
            )
            dest_h3_idx = torch.tensor(
                [self._h3_to_idx[t["destination_h3"]] for t in trips], dtype=torch.long
            )
            self._trip_arrays[traveler_id] = {
                "origin_h3_idx": origin_h3_idx,
                "dest_h3_idx": dest_h3_idx,
                "pickup_tw": torch.tensor(
                    [t["pickup_tw"] for t in trips], dtype=torch.float32
                ),  # [n_trips, 2]
                "dropoff_tw": torch.tensor(
                    [t["dropoff_tw"] for t in trips], dtype=torch.float32
                ),  # [n_trips, 2]
                # Use H3 cell centroids instead of raw GPS — matches cost granularity
                "origin_gps": centroids_cpu[origin_h3_idx],   # [n_trips, 2]
                "dest_gps": centroids_cpu[dest_h3_idx],       # [n_trips, 2]
                "flex_idx": torch.tensor(
                    [self._flexibility_to_index(t["flexibility"]) for t in trips],
                    dtype=torch.float32,
                ),  # [n_trips]
            }

    def _sample_perturbation_batch(self, n: int) -> torch.Tensor:
        """Sample signed perturbation deltas uniform in [-perturbation, +perturbation] (CPU)."""
        if self.perturbation <= 0:
            return torch.zeros(n)
        return (torch.rand(n, generator=self.generator) * 2 - 1) * self.perturbation

    def _shift_window_signed(
        self,
        windows: torch.Tensor,
        deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Shift time windows by signed deltas, preserving window duration.

        Args:
            windows: [N, 2] time windows (start, end)
            deltas: [N] signed perturbation (positive = later, negative = earlier)

        Returns:
            [N, 2] shifted windows clamped to [0, day_minutes]
        """
        duration = torch.clamp(windows[:, 1] - windows[:, 0], min=0)
        new_starts = torch.clamp(windows[:, 0] + deltas, min=0,
                                 max=float(self.day_minutes))
        new_ends = torch.clamp(new_starts + duration, max=float(self.day_minutes))
        # Ensure duration is preserved even when clamped at day end
        new_starts = torch.clamp(new_ends - duration, min=0)
        return torch.stack([new_starts, new_ends], dim=-1)

    def _randint(self, high: int) -> int:
        if high <= 1:
            return 0
        return int(torch.randint(high, (1,), generator=self.generator).item())

    @staticmethod
    def _parse_time_window(value: str) -> TimeWindow:
        if not isinstance(value, str) or "-" not in value:
            raise ValueError("Time window must be a 'HH:MM-HH:MM' string.")

        start_str, end_str = value.split("-")
        return SFGenerator._hm_to_minutes(start_str), SFGenerator._hm_to_minutes(end_str)

    @staticmethod
    def _hm_to_minutes(value: str) -> int:
        hour_str, minute_str = value.strip().split(":")
        return int(hour_str) * 60 + int(minute_str)

    @staticmethod
    def _normalize_batch_size(batch_size) -> Tuple[int, ...]:
        if isinstance(batch_size, torch.Size):
            batch_size = tuple(batch_size)
        elif isinstance(batch_size, int):
            batch_size = (batch_size,)
        elif isinstance(batch_size, (list, tuple)):
            batch_size = tuple(int(x) for x in batch_size) if batch_size else (1,)
        else:
            raise TypeError(f"Unsupported batch_size type: {type(batch_size)}")

        batch_size = tuple(max(1, b) for b in batch_size)
        return batch_size if batch_size else (1,)

    @staticmethod
    def _minutes_to_hhmm(value: int) -> str:
        hour = value // 60
        minute = value % 60
        return f"{hour:02d}:{minute:02d}"


def main():
    generator = SFGenerator()
    td = generator(2)
    print(td)


if __name__ == "__main__":
    main()
