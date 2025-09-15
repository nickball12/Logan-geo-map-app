import pandas as pd
import numpy as np
from datetime import datetime
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import concurrent.futures
from station_model import Station
from station_manager import StationManager
import logging
import pickle
import re
import sys
import threading
import io
import contextlib

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- YOUR STARTING ADDRESS ---
STARTING_LOCATION = {
    'Name': 'My Home Base',
    'Full Address': '5300 S 300 W, Murray, UT',
    'Inspection Time (min)': 0,
    'Coordinates': '40.6542, -111.8955'
}

# Geocoding cache to store previously geocoded addresses
GEOCODE_CACHE_FILE = 'cache/geocode_cache.pkl'
geocode_cache = {}

# Load geocoding cache from file if it exists
try:
    if os.path.exists(GEOCODE_CACHE_FILE):
        with open(GEOCODE_CACHE_FILE, 'rb') as f:
            geocode_cache = pickle.load(f)
        logger.info(f"Loaded {len(geocode_cache)} cached geocoded addresses")
except Exception as e:
    logger.error(f"Error loading geocode cache: {e}")
    geocode_cache = {}

def get_coordinates_local(address, use_cache=True):
    """
    Geocode an address to get its coordinates
    Returns string format: "latitude, longitude"
    
    Args:
        address (str): The address to geocode
        use_cache (bool): Whether to use and update the cache
    """
    global geocode_cache
    
    if not address or pd.isna(address):
        logger.warning(f"Invalid address: {address}")
        return None
    
    # Clean up the address - remove extra spaces and normalize format
    address = address.strip()
    
    # Check cache first if enabled
    if use_cache and address in geocode_cache:
        logger.info(f"Using cached coordinates for: {address}")
        return geocode_cache[address]
    
    try:
        # Check if ZIP code exists in the address
        zip_code = None
        # Extract ZIP code if present
        import re
        zip_match = re.search(r'(?:\s|,\s*)(\d{5}(?:-\d{4})?)', address)
        if zip_match:
            zip_code = zip_match.group(1)
            
        # Try to add Utah state and USA if not present for better geocoding
        if 'UT' not in address and 'Utah' not in address:
            if zip_code:
                search_address = f"{address}, USA"  # ZIP code implies state
            else:
                search_address = f"{address}, UT, USA"
        elif 'USA' not in address:
            search_address = f"{address}, USA"
        else:
            search_address = address
            
        logger.info(f"Geocoding address: {search_address}")
        geolocator = Nominatim(user_agent="station_router", timeout=20)  # Increased timeout
        location = geolocator.geocode(search_address, exactly_one=True)
        
        if location:
            result = f"{location.latitude}, {location.longitude}"
            if use_cache:
                # Save to cache
                geocode_cache[address] = result
                # Save cache to file periodically (every 10 new entries)
                if len(geocode_cache) % 10 == 0:
                    save_geocode_cache()
            return result
        else:
            # If first attempt fails, try a more general search
            if ',' in address:
                parts = address.split(',')
                
                # Extract street address and city if possible
                street_address = parts[0].strip()
                
                # Try geocoding with just street and city + UT
                if len(parts) > 1:
                    city_part = parts[1].strip()
                    # First try with existing city
                    simplified_address = f"{street_address}, {city_part}, UT, USA"
                    logger.info(f"Trying simplified address: {simplified_address}")
                    location = geolocator.geocode(simplified_address, exactly_one=True)
                    
                    if location:
                        result = f"{location.latitude}, {location.longitude}"
                        if use_cache:
                            geocode_cache[address] = result
                            if len(geocode_cache) % 10 == 0:
                                save_geocode_cache()
                        return result
                
                # If that fails, try just the street address with UT, USA
                simplified_address = f"{street_address}, UT, USA"
                logger.info(f"Trying just street address: {simplified_address}")
                location = geolocator.geocode(simplified_address, exactly_one=True)
                
                if location:
                    result = f"{location.latitude}, {location.longitude}"
                    if use_cache:
                        geocode_cache[address] = result
                        if len(geocode_cache) % 10 == 0:
                            save_geocode_cache()
                    return result
                
                # If zip code exists, try with just zip code in Utah
                if zip_code:
                    zip_only_address = f"UT {zip_code}, USA"
                    logger.info(f"Trying ZIP only: {zip_only_address}")
                    location = geolocator.geocode(zip_only_address, exactly_one=True)
                    
                    if location:
                        result = f"{location.latitude}, {location.longitude}"
                        if use_cache:
                            geocode_cache[address] = result
                            if len(geocode_cache) % 10 == 0:
                                save_geocode_cache()
                        return result
            
            logger.warning(f"Could not geocode address: {address}")
            # Return a default coordinate in Utah if geocoding fails
            default_coords = "40.7608, -111.8910"  # Default to Salt Lake City
            if use_cache:
                geocode_cache[address] = default_coords
            return default_coords
    except Exception as e:
        logger.error(f"Error geocoding {address}: {e}")
        # Return a default coordinate in Utah if geocoding fails
        default_coords = "40.7608, -111.8910"  # Default to Salt Lake City
        if use_cache:
            geocode_cache[address] = default_coords
        return default_coords

def save_geocode_cache():
    """Save the geocoding cache to a file"""
    global geocode_cache
    try:
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(GEOCODE_CACHE_FILE), exist_ok=True)
        with open(GEOCODE_CACHE_FILE, 'wb') as f:
            pickle.dump(geocode_cache, f)
        logger.info(f"Saved {len(geocode_cache)} geocoded addresses to cache")
    except Exception as e:
        logger.error(f"Error saving geocode cache: {e}")

def calculate_travel_time(coord1, coord2):
    """
    Calculate travel time in minutes between two coordinates
    using straight-line distance and an average speed of 30 mph
    """
    try:
        # Extract lat and long from coordinate strings
        lat1, lon1 = map(float, coord1.split(','))
        lat2, lon2 = map(float, coord2.split(','))
        
        # Calculate straight-line distance in miles
        distance = geodesic((lat1, lon1), (lat2, lon2)).miles
        
        # Calculate travel time using average speed of 30 mph
        # Add a 10% buffer for traffic and delays
        time_in_minutes = (distance / 30) * 60 * 1.1
        
        return int(time_in_minutes)
    except Exception as e:
        logger.error(f"Error calculating travel time: {e}")
        # Default to 30 minutes if calculation fails
        return 30

def calculate_drive_times(stations, progress_callback=None, max_routes=None, cancellation_event=None, use_cache=True):
    """
    Calculate drive times between all stations
    Returns a pandas DataFrame with travel times in minutes
    
    Args:
        stations: List of Station objects
        progress_callback: Function to call for progress updates
        max_routes: Maximum number of routes to calculate (None for all)
        cancellation_event: Event to signal cancellation
        use_cache: Whether to use cached geocoding data
    """
    logger.info(f"Calculating travel times for {len(stations)} stations")
    
    # Extract coordinates from stations
    coordinates = []
    logger.info(f"Processing geocoding for {len(stations)} stations")
    if progress_callback:
        progress_callback(0, len(stations), f"Geocoding stations (0/{len(stations)})", False, 'geocoding', 'Geocoding')
    
    cached_count = 0
    for i, station in enumerate(stations):
        if cancellation_event and cancellation_event.is_set():
            logger.info("Geocoding cancelled")
            return None
            
        if station.coordinates:
            coordinates.append(station.coordinates)
        else:
            # Check if this address is already in the cache
            is_cached = use_cache and station.full_address in geocode_cache
            if is_cached:
                cached_count += 1
                
            # Geocode the address if coordinates are not set
            station.coordinates = get_coordinates_local(station.full_address, use_cache=use_cache)
            coordinates.append(station.coordinates)
        
        # Update progress every 5 stations
        if progress_callback and i % 5 == 0:
            progress_callback(i+1, len(stations), 
                             f"Geocoding stations ({i+1}/{len(stations)}) - {cached_count} from cache", 
                             False, 'geocoding', 'Geocoding')
    
    # Add starting location coordinates to the beginning
    coordinates.insert(0, STARTING_LOCATION['Coordinates'])
    
    # Create a matrix to store travel times
    n = len(coordinates)
    travel_times = np.zeros((n, n))
    
    # Calculate total number of route calculations
    total_calculations = n * (n - 1)
    completed = 0
    
    # Apply max_routes limit if specified
    if max_routes and max_routes > 0 and total_calculations > max_routes:
        logger.warning(f"Limiting route calculations to {max_routes} (out of {total_calculations})")
        sample_factor = max_routes / total_calculations
    else:
        sample_factor = 1.0
    
    # Use a thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list to store future results
        future_to_route = {}
        
        # Submit all route calculations to the thread pool
        for i in range(n):
            for j in range(n):
                if i != j:  # Don't calculate travel time from a location to itself
                    # Apply sampling if needed to stay under max_routes
                    if sample_factor < 1.0 and np.random.random() > sample_factor:
                        # Skip this calculation and use an estimate later
                        continue
                        
                    # Check for cancellation
                    if cancellation_event and cancellation_event.is_set():
                        logger.info("Route calculation cancelled")
                        return None
                        
                    # Submit the task to calculate travel time
                    future = executor.submit(calculate_travel_time, coordinates[i], coordinates[j])
                    future_to_route[(i, j)] = future
        
        # Process results as they complete
        for i, j in list(future_to_route.keys()):
            try:
                # Check for cancellation
                if cancellation_event and cancellation_event.is_set():
                    logger.info("Route calculation cancelled")
                    return None
                    
                # Get the travel time from the future
                travel_time = future_to_route[(i, j)].result()
                travel_times[i, j] = travel_time
                
                # Update progress
                completed += 1
                if progress_callback and completed % 10 == 0:
                    progress_callback(completed, total_calculations, f"Calculated {completed} of {total_calculations} routes")
                    
            except Exception as e:
                logger.error(f"Error calculating route {i} to {j}: {e}")
                # Use a default travel time in case of error
                travel_times[i, j] = 30
    
    # Fill in any missing values using estimates
    if sample_factor < 1.0:
        logger.info("Filling in missing travel times with estimates")
        for i in range(n):
            for j in range(n):
                if i != j and travel_times[i, j] == 0:
                    # Use the average travel time for this origin
                    i_times = [t for t in travel_times[i, :] if t > 0]
                    if i_times:
                        travel_times[i, j] = sum(i_times) / len(i_times)
                    else:
                        travel_times[i, j] = 30  # Default to 30 minutes
    
    # Return the travel times as a DataFrame
    return pd.DataFrame(travel_times)

def create_data_model(stations, travel_times_df, hours, max_stations=None, max_pumps=None, time_between_stops=0):
    """
    Create a data model for the routing problem
    """
    data = {}
    data['travel_times'] = travel_times_df.values.astype(int).tolist()
    
    # Create inspection times list (depot has 0, all stations have their times)
    inspection_times = [0]  # Depot has 0 inspection time
    for station in stations:
        inspection_times.append(station.inspection_time_min)
    data['inspection_times'] = inspection_times
    
    data['time_between_stops'] = time_between_stops
    data['num_vehicles'] = 1
    data['depot'] = 0
    data['time_limit'] = int(hours * 60)
    
    # Add priority data for each location
    priorities = [0]  # Depot has 0 priority
    for station in stations:
        # Calculate priority based on status
        priorities.append(station.priority_score)
    
    # Normalize priorities to range 0-1
    max_priority = max(priorities) if max(priorities) > 0 else 1
    data['normalized_priority'] = [p / max_priority for p in priorities]
    
    # Add pump data
    pumps_to_inspect = [0]  # Depot has 0 pumps to inspect
    total_pumps = [0]  # Depot has 0 total pumps
    for station in stations:
        # For out of service pumps, they count as pumps to inspect
        if station.out_of_service_pumps > 0:
            pumps_to_inspect.append(station.out_of_service_pumps)
        else:
            # For regular stations, default to 2 pumps per station (or user-specified)
            pumps_per_station = 2  # Default value
            pumps_to_inspect.append(min(pumps_per_station, station.num_pumps))
        
        total_pumps.append(station.num_pumps)
    
    data['pumps_to_inspect'] = pumps_to_inspect
    data['total_pumps'] = total_pumps
    
    # Print some debug information
    logger.info(f"\nRoute Optimization Parameters:")
    logger.info(f"- Number of locations: {len(data['travel_times'])}")
    logger.info(f"- Time limit: {data['time_limit']} minutes ({hours} hours)")
    logger.info(f"- Time between stops: {data['time_between_stops']} minutes")
    logger.info(f"- Total pumps across all stations: {sum(data['total_pumps'])}")
    logger.info(f"- Pumps to inspect across all stations: {sum(data['pumps_to_inspect'])}")
    logger.info(f"- Max pumps per day: {max_pumps if max_pumps else 'unlimited'}")
    logger.info(f"- Total inspection time: {sum(data['inspection_times'])} minutes")
    
    return data

def solve_route(data, progress_callback=None, cancellation_event=None, max_pumps=None, max_stations=None):
    """
    Solve the route optimization problem using OR-Tools
    """
    logger.info("Starting route optimization...")
    
    if progress_callback:
        progress_callback(1, 5, "Initializing route solver...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(1, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None
    
    # Create the routing model
    manager = pywrapcp.RoutingIndexManager(len(data['travel_times']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    if progress_callback:
        progress_callback(2, 5, "Setting up transit callback and constraints...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(2, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None

    # Define a callback that returns the travel time between two points
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Add the time between stops to each travel segment (except when returning to depot)
        additional_time = data['time_between_stops'] if to_node != data['depot'] else 0
        return data['travel_times'][from_node][to_node] + data['inspection_times'][from_node] + additional_time
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time dimension - allows us to track the cumulative time
    routing.AddDimension(transit_callback_index, 0, data['time_limit'], True, 'Time')
    
    # Add pump tracking dimension if max_pumps is specified
    if max_pumps and 'pumps_to_inspect' in data:
        def pumps_callback(from_index, to_index):
            # Only count pumps when arriving at a location (not when leaving)
            to_node = manager.IndexToNode(to_index)
            # Don't count pumps when returning to depot
            if to_node == data['depot']:
                return 0
            # Return the number of pumps to inspect at this location
            return data['pumps_to_inspect'][to_node]
            
        pumps_callback_index = routing.RegisterTransitCallback(pumps_callback)
        # Add dimension for tracking pump inspections with daily limit
        routing.AddDimension(
            pumps_callback_index,  # Transit callback
            0,                     # Slack (no slack needed)
            max_pumps,             # Maximum pumps per day
            True,                  # Start cumul to zero
            'Pumps'                # Name of the dimension
        )
        logger.info(f"Added pump inspection dimension with max {max_pumps} pumps per day")
        
        # Get the pump dimension
        pump_dimension = routing.GetDimensionOrDie('Pumps')
        
        # Set a cost coefficient to prioritize filling up the pump capacity
        # This incentivizes the solver to include stops with pumps
        pump_dimension.SetGlobalSpanCostCoefficient(100)
        logger.info("Set priority to maximize pump inspections")
    else:
        logger.info("No pump limit set, or no pump data available")
    
    # Make nodes droppable (except depot) with penalties based on priority
    # This allows the solver to skip some stops if the time constraint can't fit them all
    base_penalty = 10000  # Base penalty for all nodes
    for node in range(1, len(data['travel_times'])):
        # Apply higher penalties for locations with higher priority
        # This makes them less likely to be skipped
        priority_factor = 1 + (data['normalized_priority'][node] * 2)  # Scale factor based on priority
        penalty = int(base_penalty * priority_factor)
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
        
        # Debug the prioritization
        if data['normalized_priority'][node] > 0:
            logger.debug(f"Node {node}: priority factor: {priority_factor:.2f}, penalty: {penalty}")
    
    # Set the objective to minimize the total time/distance
    time_dimension = routing.GetDimensionOrDie('Time')
    # Use a moderate span cost coefficient to balance between route length and number of stops
    time_dimension.SetGlobalSpanCostCoefficient(10)  # Moderate value to balance optimization priorities

    if progress_callback:
        progress_callback(3, 5, "Configuring search parameters...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(3, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None

    # Set up search parameters for much faster optimization
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Use AUTOMATIC to let OR-Tools choose the best strategy quickly
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC)
    
    # Use faster local search
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
    
    # Much shorter time limit to prevent web timeouts
    search_parameters.time_limit.seconds = 60  # Just 1 minute for web responsiveness
    
    # Reduce iterations significantly for speed
    search_parameters.log_search = False  # Disable logging for speed
    search_parameters.solution_limit = 20  # Much lower limit for quick results

    # Use standard propagation for better compatibility
    search_parameters.guided_local_search_lambda_coefficient = 0.1  # Less aggressive local search
    
    if progress_callback:
        progress_callback(4, 5, "Running optimization algorithm (this may take a moment)...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(4, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None
    
    # Create a simpler progress monitor that doesn't rely on stderr capture
    class OptimizationMonitor:
        def __init__(self, progress_callback, cancellation_event, time_limit_seconds=60):
            self.progress_callback = progress_callback
            self.cancellation_event = cancellation_event
            self.start_time = time.time()
            self.time_limit = time_limit_seconds
            self.last_progress = 0
            
        def update_progress(self):
            if self.progress_callback:
                elapsed = int(time.time() - self.start_time)
                # Estimate progress based on time elapsed (capped at 95% until completion)
                time_progress = min(95, int((elapsed / self.time_limit) * 100))
                
                # Update more frequently for shorter optimization
                if time_progress > self.last_progress + 5 or elapsed % 5 == 0:
                    message = f"Optimizing route... ~{time_progress}% complete ({elapsed}s elapsed)"
                    self.progress_callback(4, 5, message, False, 'solving', 'Route Optimization')
                    self.last_progress = time_progress
                
                return elapsed < self.time_limit
    
    monitor = OptimizationMonitor(progress_callback, cancellation_event, search_parameters.time_limit.seconds)
    
    # Start a thread to periodically update progress during optimization
    stop_monitor = threading.Event()
    
    def monitor_optimization():
        while not stop_monitor.is_set():
            time.sleep(3)  # Update every 3 seconds to reduce overhead
            if not stop_monitor.is_set():
                if not monitor.update_progress():
                    # Time limit approaching, stop monitoring
                    break
    
    monitor_thread = threading.Thread(target=monitor_optimization, daemon=True)
    monitor_thread.start()
    
    try:
        # Solve the route with a timeout wrapper
        logger.info(f"Starting route optimization with {search_parameters.time_limit.seconds}s time limit")
        solution = routing.SolveWithParameters(search_parameters)
        logger.info("Route optimization completed")
    finally:
        # Stop the monitoring thread
        stop_monitor.set()
        monitor_thread.join(timeout=1)  # Wait up to 1 second for thread to finish
    
    # If no solution was found, try with a different strategy
    if not solution:
        logger.warning("First solution strategy failed, trying SAVINGS algorithm...")
        if progress_callback:
            progress_callback(4, 5, "First attempt failed, trying alternative algorithm...", False, 'solving', 'Route Optimization')
        
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            if progress_callback:
                progress_callback(4, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
            return None, None, None
        
        # Try a different strategy
        search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.SAVINGS)
        solution = routing.SolveWithParameters(search_parameters)
        
        # If still no solution, try one more strategy
        if not solution:
            logger.warning("SAVINGS algorithm failed, trying PARALLEL_CHEAPEST_INSERTION...")
            if progress_callback:
                progress_callback(4, 5, "Second attempt failed, trying final algorithm...", False, 'solving', 'Route Optimization')
            
            # Check for cancellation
            if cancellation_event and cancellation_event.is_set():
                if progress_callback:
                    progress_callback(4, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
                return None, None, None
            
            # Try one more strategy that's often good for time-constrained problems
            search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION)
            solution = routing.SolveWithParameters(search_parameters)
    
    if progress_callback:
        if solution:
            # Simple completion message
            progress_callback(5, 5, "Optimization complete, solution found!", False, 'solving', 'Route Optimization')
        else:
            progress_callback(5, 5, "Optimization complete, but no solution found.", False, 'solving', 'Route Optimization')
    
    return solution, manager, routing

def extract_route_plan(solution, manager, routing, stations):
    """
    Extract the optimized route plan from the solution
    """
    if not solution:
        logger.warning("No solution found!")
        return []
        
    route_plan = []
    index = routing.Start(0)
    
    # Add the depot to the route plan
    depot = {
        'id': 'depot',
        'name': STARTING_LOCATION['Name'],
        'address': STARTING_LOCATION['Full Address'],
        'coordinates': STARTING_LOCATION['Coordinates'],
        'is_depot': True,
        'inspection_time': 0,
        'status': []
    }
    route_plan.append(depot)
    
    # Extract each stop in the route
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        
        # Skip the depot node (already added above)
        if node_index > 0:
            station = stations[node_index - 1]  # -1 because stations list doesn't include depot
            
            # Prepare status flags for this station
            status = []
            priority_reason = ""
            
            if station.has_complaint:
                status.append('complaint')
                priority_reason = "Customer complaint"
            elif station.needs_reinspection:
                status.append('reinspection')
                priority_reason = f"Scheduled reinspection: {station.reinspection_reason}" if station.reinspection_reason else "Scheduled reinspection"
            elif station.out_of_service_pumps > 0:
                status.append('out_of_service')
                priority_reason = f"{station.out_of_service_pumps} pump(s) out of service"
            elif station.days_since_inspection and station.days_since_inspection > 180:
                priority_reason = f"Last inspection: {station.days_since_inspection} days ago"
            
            # Add the station to the route plan
            station_info = {
                'id': station.business_id,
                'name': station.name,
                'address': station.address,
                'city': station.city,
                'state': station.state,
                'zip': station.zip_code,
                'county': station.county,
                'full_address': station.full_address,
                'coordinates': station.coordinates,
                'is_depot': False,
                'inspection_time': station.inspection_time_min,
                'num_pumps': station.num_pumps if station.num_pumps > 0 else 2,  # Default to 2 if not specified
                'pumps_to_inspect': 0,  # Will be calculated below
                'priority_score': station.priority_score,
                'reinspection': station.needs_reinspection,
                'complaint': station.has_complaint,
                'out_of_service': station.out_of_service_pumps > 0,
                'priority_reason': priority_reason,
                'last_visited': station.last_inspection_date.strftime('%Y-%m-%d') if station.last_inspection_date else None,
                'days_since_inspection': station.days_since_inspection,
                'out_of_service_pumps': station.out_of_service_pumps,
                'out_of_service_details': station.out_of_service_details,
                'days_out_of_service': station.days_out_of_service
            }
            
            # Calculate pumps to inspect for this station
            if station.out_of_service_pumps > 0:
                station_info['pumps_to_inspect'] = station.out_of_service_pumps
            else:
                # Default to 2 pumps per station, or the actual number if less
                pumps_per_station = 2  # Default value
                station_info['pumps_to_inspect'] = min(pumps_per_station, station.num_pumps) if station.num_pumps > 0 else 2
            
            route_plan.append(station_info)
        
        # Move to the next stop
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        
        # Calculate travel time to the next stop
        if not routing.IsEnd(index):
            next_node_index = manager.IndexToNode(index)
            # Add travel time
            time_var = routing.GetDimensionOrDie("Time").CumulVar(index)
            route_plan[-1]['travel_time'] = solution.Min(time_var) - (route_plan[-1]['inspection_time'] if route_plan[-1].get('inspection_time') else 0)
            
            # Calculate arrival time (assuming 8 AM start)
            start_time = datetime.strptime("08:00", "%H:%M")
            arrival_time = start_time + pd.Timedelta(minutes=solution.Min(time_var))
            route_plan[-1]['arrival_time'] = arrival_time.strftime("%H:%M")
            
            # Calculate distance
            if route_plan[-1].get('coordinates') and next_node_index < len(stations):
                next_station = stations[next_node_index - 1] if next_node_index > 0 else {'coordinates': STARTING_LOCATION['Coordinates']}
                next_coordinates = next_station['coordinates'] if isinstance(next_station, dict) else next_station.coordinates
                
                if route_plan[-1].get('coordinates') and next_coordinates:
                    try:
                        coords1 = [float(x.strip()) for x in route_plan[-1]['coordinates'].split(',')]
                        coords2 = [float(x.strip()) for x in next_coordinates.split(',')]
                        distance = geodesic((coords1[0], coords1[1]), (coords2[0], coords2[1])).miles
                        route_plan[-1]['distance'] = distance
                    except Exception as e:
                        logger.warning(f"Error calculating distance: {e}")
                        route_plan[-1]['distance'] = 0
    
    # Add summary statistics to the first entry (depot)
    time_dimension = routing.GetDimensionOrDie('Time')
    total_time = time_dimension.CumulVar(routing.End(0)).Max()
    
    route_plan[0]['total_time'] = total_time
    route_plan[0]['stops_count'] = len(route_plan) - 1  # Exclude depot
    
    # Count total pumps to inspect
    total_pumps = sum(stop.get('pumps_to_inspect', 0) for stop in route_plan)
    route_plan[0]['total_pumps'] = total_pumps
    
    return route_plan

def plan_optimal_route(excel_file, hours=8, max_stations=None, max_pumps=20, time_between_stops=10, 
                       progress_callback=None, cancellation_event=None, use_cache=True,
                       include_normal=True, include_complaints=True, include_reinspections=True,
                       include_out_of_service=True):
    """
    Main function to plan an optimal route from the Excel data
    
    Args:
        excel_file: Path to the Excel file with station data
        hours: Working hours for the route
        max_stations: Maximum number of stations to include (None for all)
        max_pumps: Maximum number of pumps to inspect per day
        time_between_stops: Time in minutes to add between stops
        progress_callback: Function to call for progress updates
        cancellation_event: Event to signal cancellation
        use_cache: Whether to use cached geocoding data
        include_normal: Whether to include normal stations
        include_complaints: Whether to include stations with complaints
        include_reinspections: Whether to include stations needing reinspection
        include_out_of_service: Whether to include stations with out-of-service pumps
        
    Returns:
        List of dictionaries with route plan
    """
    # Load stations from Excel
    station_manager = StationManager()
    if not station_manager.load_from_excel(excel_file):
        logger.error("Failed to load stations from Excel")
        return None
    
    # Get all stations first
    all_stations = station_manager.get_all_stations()
    
    # Filter stations based on their type flags
    filtered_stations = []
    for station in all_stations:
        # Normal stations (no special flags)
        is_normal = (not station.has_complaint and 
                    not station.needs_reinspection and 
                    not station.has_out_of_service_pump)
        
        if ((include_normal and is_normal) or
            (include_complaints and station.has_complaint) or
            (include_reinspections and station.needs_reinspection) or
            (include_out_of_service and station.has_out_of_service_pump)):
            filtered_stations.append(station)
    
    # Use filtered stations and sort by priority
    stations = sorted(filtered_stations, key=lambda x: x.priority_score, reverse=True)
    logger.info(f"Filtered to {len(stations)} stations based on type filters")
        
    logger.info(f"Using {len(stations)} stations, sorted by priority")
    
    # Skip if no stations match the criteria
    if not stations:
        logger.warning("No stations match the selected status filters")
        return None
    
    # Limit stations if max_stations is specified
    if max_stations and max_stations > 0 and len(stations) > max_stations:
        logger.info(f"Limiting to {max_stations} highest priority stations")
        stations = stations[:max_stations]
    
    # Calculate travel times
    travel_times_df = calculate_drive_times(stations, progress_callback, cancellation_event=cancellation_event, use_cache=use_cache)
    if travel_times_df is None:
        logger.error("Travel time calculation was cancelled")
        return None
    
    # Create data model
    data_model = create_data_model(stations, travel_times_df, hours, max_stations, max_pumps, time_between_stops)
    
    # Solve the route
    solution, manager, routing = solve_route(data_model, progress_callback, cancellation_event, max_pumps, max_stations)
    
    # Save geocode cache to ensure all geocoded addresses are persisted
    save_geocode_cache()
    
    # Extract the route plan
    if solution:
        route_plan = extract_route_plan(solution, manager, routing, stations)
        
        # Calculate and report pump utilization if max_pumps was specified
        if route_plan and max_pumps and max_pumps > 0:
            total_pumps_in_route = route_plan[0].get('total_pumps', 0)
            pump_utilization = min(100, int((total_pumps_in_route / max_pumps) * 100))
            
            # Update the final progress with pump utilization information
            if progress_callback:
                final_msg = f"Route complete! {total_pumps_in_route}/{max_pumps} pumps ({pump_utilization}% of daily limit)"
                progress_callback(5, 5, final_msg, True, 'complete', 'Route Planning Complete')
        
        return route_plan
    else:
        logger.warning("No solution found")
        return None

# For testing
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python route_planner.py <excel_file> [hours] [max_pumps]")
        sys.exit(1)
    
    excel_file = sys.argv[1]
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else 8
    max_pumps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    route_plan = plan_optimal_route(excel_file, hours, max_pumps=max_pumps)
    
    if route_plan:
        print("\n--- Optimized Route ---")
        print(f"Starting from: {route_plan[0]['name']} ({route_plan[0]['address']})")
        print(f"Total time: {route_plan[0]['total_time']} minutes")
        print(f"Stops: {route_plan[0]['stops_count']}")
        print(f"Total pumps to inspect: {route_plan[0]['total_pumps']}")
        
        print("\nRoute Sequence:")
        for i, stop in enumerate(route_plan[1:], 1):
            status_str = ", ".join(stop['status']) if stop['status'] else "Regular Inspection"
            print(f"{i}. {stop['name']} - {stop['address']} - {stop['pumps_to_inspect']} pumps - {status_str}")
    else:
        print("Failed to generate route plan")

def geocode_stations_only(station_manager, progress_callback=None, cancellation_event=None):
    """
    Geocode all stations in the station manager - Stage 2 operation
    
    Args:
        station_manager: StationManager instance with loaded stations
        progress_callback: Function to report progress
        cancellation_event: Event to check for cancellation
        
    Returns:
        int: Number of stations successfully geocoded
    """
    stations = station_manager.get_all_stations()
    geocoded_count = 0
    
    if progress_callback:
        progress_callback(0, len(stations), 'Starting geocoding process...', False, 'geocoding', 'Geocoding')
    
    for i, station in enumerate(stations):
        # Check for cancellation
        if cancellation_event and cancellation_event.is_set():
            break
            
        try:
            # Only geocode if coordinates are missing
            if not station.coordinates or station.coordinates == 'None':
                coords = get_coordinates_local(station.full_address, use_cache=True)
                if coords:
                    station.coordinates = f"{coords[0]}, {coords[1]}"
                    geocoded_count += 1
                else:
                    logger.warning(f"Failed to geocode: {station.full_address}")
            
            if progress_callback:
                progress_callback(i + 1, len(stations), 
                                f'Geocoded {i + 1}/{len(stations)} stations', 
                                False, 'geocoding', 'Geocoding')
        except Exception as e:
            logger.error(f"Error geocoding station {station.full_address}: {e}")
    
    # Save updated geocode cache
    save_geocode_cache()
    
    if progress_callback:
        progress_callback(len(stations), len(stations), 
                        f'Geocoding complete! {geocoded_count} stations geocoded', 
                        False, 'geocoding', 'Geocoding Complete')
    
    return geocoded_count

def plan_route_from_geocoded_stations(stations, hours=8, max_stations=None, max_pumps=20, 
                                    time_between_stops=10, progress_callback=None, 
                                    cancellation_event=None, include_normal=True, 
                                    include_complaints=True, include_reinspections=True, 
                                    include_out_of_service=True):
    """
    Plan route using pre-geocoded stations - Stage 3 operation
    
    Args:
        stations: List of Station objects with coordinates
        hours: Work hours available
        max_stations: Maximum stations to visit
        max_pumps: Maximum pumps per day
        time_between_stops: Travel time between stops
        progress_callback: Function to report progress
        cancellation_event: Event to check for cancellation
        include_*: Station type filters
        
    Returns:
        List of route stops or None if failed
    """
    if progress_callback:
        progress_callback(1, 5, 'Filtering stations by type...', False, 'planning', 'Route Planning')
    
    # Filter stations by type
    filtered_stations = []
    for station in stations:
        # Check station type and include based on filters
        include_station = False
        
        if include_normal and not any([station.needs_reinspection, station.has_complaint, station.has_out_of_service_pump]):
            include_station = True
        if include_complaints and station.has_complaint:
            include_station = True
        if include_reinspections and station.needs_reinspection:
            include_station = True
        if include_out_of_service and station.has_out_of_service_pump:
            include_station = True
            
        if include_station:
            filtered_stations.append(station)
    
    if not filtered_stations:
        logger.warning("No stations match the selected filters")
        return None
    
    logger.info(f"Filtered to {len(filtered_stations)} stations based on type filters")
    
    if progress_callback:
        progress_callback(2, 5, f'Using {len(filtered_stations)} stations, sorting by priority...', 
                        False, 'planning', 'Route Planning')
    
    # Sort stations by priority
    filtered_stations.sort(key=lambda x: x.priority_score, reverse=True)
    
    if progress_callback:
        progress_callback(3, 5, f'Calculating travel times for {len(filtered_stations)} stations...', 
                        False, 'planning', 'Route Planning')
    
    # Calculate travel times and create distance matrix
    travel_times_df = calculate_drive_times(filtered_stations, progress_callback, cancellation_event=cancellation_event)
    
    if cancellation_event and cancellation_event.is_set():
        return None
        
    if travel_times_df is None:
        logger.error("Failed to calculate travel times")
        return None
    
    if cancellation_event and cancellation_event.is_set():
        return None
    
    if progress_callback:
        progress_callback(4, 5, 'Running optimization algorithm...', False, 'planning', 'Route Planning')
    
    # Create data model
    data_model = create_data_model(filtered_stations, travel_times_df, hours, max_stations, max_pumps, time_between_stops)
    
    # Solve the routing problem
    solution, manager, routing = solve_route(
        data_model, 
        progress_callback, 
        cancellation_event,
        max_pumps,
        max_stations
    )
    
    if progress_callback:
        if solution:
            progress_callback(5, 5, 'Extracting route plan...', False, 'planning', 'Route Planning Complete')
        else:
            progress_callback(5, 5, 'Route planning failed', True, 'planning', 'Route Planning Failed')
            return None
    
    # Extract the route plan from the solution
    if solution:
        route_plan = extract_route_plan(solution, manager, routing, filtered_stations)
        return route_plan
    else:
        logger.warning("No solution found")
        return None
