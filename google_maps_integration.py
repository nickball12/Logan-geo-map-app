"""
Google Maps API Integration for Better Route Optimization

This module provides functions to use Google Maps API for calculating driving times
between locations, which generally offers more accurate real-world driving times 
compared to OSRM, especially in areas with complex road networks.

To use this module:
1. Get a Google Maps API key from https://developers.google.com/maps/documentation/distance-matrix/
2. Add the API key to your application
3. Update the route_planner.py to use these functions instead of OSRM

Note: Google Maps API has usage limits and may require payment for high-volume use.
"""

import requests
import time
import json
import os
from datetime import datetime

def get_google_maps_drive_time(coord1, coord2, api_key):
    """
    Calculate drive time between two coordinates using Google Maps Distance Matrix API.
    
    Args:
        coord1 (str): Source coordinates as "lat,lng"
        coord2 (str): Destination coordinates as "lat,lng"
        api_key (str): Google Maps API key
        
    Returns:
        float: Driving time in minutes
    """
    try:
        lat1, lon1 = map(float, coord1.split(','))
        lat2, lon2 = map(float, coord2.split(','))
        
        # Build the API URL
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"
        params = {
            "origins": f"{lat1},{lon1}",
            "destinations": f"{lat2},{lon2}",
            "mode": "driving",
            "departure_time": "now",  # Use current time for traffic info if available
            "key": api_key
        }
        
        # Make the API request
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Check if the response has valid data
        if (data.get('status') == 'OK' and 
            data.get('rows') and 
            data['rows'][0].get('elements') and 
            data['rows'][0]['elements'][0].get('status') == 'OK'):
            
            # Get duration value in seconds and convert to minutes
            duration_seconds = data['rows'][0]['elements'][0]['duration']['value']
            duration_minutes = duration_seconds / 60
            
            # If traffic info is available, use that instead
            if 'duration_in_traffic' in data['rows'][0]['elements'][0]:
                traffic_seconds = data['rows'][0]['elements'][0]['duration_in_traffic']['value']
                duration_minutes = traffic_seconds / 60
            
            return duration_minutes
        else:
            print(f"Google Maps API returned unexpected data: {data}")
            # Fall back to the current fallback method
            from route_planner import get_better_distance_estimate
            lat1, lon1 = map(float, coord1.split(','))
            lat2, lon2 = map(float, coord2.split(','))
            return get_better_distance_estimate(lat1, lon1, lat2, lon2)
            
    except Exception as e:
        print(f"Error in Google Maps API call: {e}")
        # Fall back to the current fallback method
        try:
            from route_planner import get_better_distance_estimate
            lat1, lon1 = map(float, coord1.split(','))
            lat2, lon2 = map(float, coord2.split(','))
            return get_better_distance_estimate(lat1, lon1, lat2, lon2)
        except Exception as e2:
            print(f"Error in fallback calculation: {e2}")
            return 30  # Default to 30 minutes if all else fails

def calculate_google_drive_times(station_data, api_key, use_cache=True, cache_file="google_drive_times.json", progress_callback=None):
    """
    Calculate all pairwise driving times between stations using Google Maps API.
    
    Args:
        station_data (DataFrame): DataFrame containing station data with Coordinates column
        api_key (str): Google Maps API key
        use_cache (bool): Whether to use cached times if available
        cache_file (str): File to store/retrieve cached times
        progress_callback (callable): Function to call with progress updates
        
    Returns:
        DataFrame: Matrix of drive times between all stations
    """
    import pandas as pd
    import numpy as np
    
    # Check if we have valid coordinates for all stations
    station_data = station_data.copy()
    station_data = station_data.dropna(subset=['Coordinates'])
    
    num_stations = len(station_data)
    station_indices = list(range(num_stations))
    
    # Initialize drive times matrix
    drive_times = pd.DataFrame(
        index=station_indices,
        columns=station_indices,
        data=np.zeros((num_stations, num_stations))
    )
    
    # Check if cache exists and should be used
    cache_exists = os.path.exists(cache_file)
    cache_data = {}
    
    if use_cache and cache_exists:
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            print(f"Loaded {len(cache_data)} cached drive times")
        except Exception as e:
            print(f"Error loading cache: {e}")
            cache_data = {}
    
    # Calculate number of API calls needed
    total_pairs = num_stations * (num_stations - 1)
    api_calls_needed = 0
    
    # Count missing pairs
    for i in range(num_stations):
        for j in range(num_stations):
            if i != j:  # Skip self-to-self
                coord1 = station_data.iloc[i]['Coordinates']
                coord2 = station_data.iloc[j]['Coordinates']
                cache_key = f"{coord1}|{coord2}"
                
                if cache_key not in cache_data:
                    api_calls_needed += 1
    
    print(f"Need to make {api_calls_needed} API calls out of {total_pairs} pairs")
    
    # Update on the number of API calls needed
    if progress_callback:
        progress_callback(0, api_calls_needed or 1, 
                         f"Calculating drive times ({api_calls_needed} API calls needed)", 
                         api_calls_needed == 0)
    
    # If all times are already cached, just build the matrix
    if api_calls_needed == 0 and cache_data:
        calls_made = 0
        for i in range(num_stations):
            for j in range(num_stations):
                if i == j:
                    # No travel time to self, just the inspection time
                    drive_times.iloc[i, j] = 0
                else:
                    coord1 = station_data.iloc[i]['Coordinates']
                    coord2 = station_data.iloc[j]['Coordinates']
                    cache_key = f"{coord1}|{coord2}"
                    
                    if cache_key in cache_data:
                        drive_times.iloc[i, j] = cache_data[cache_key]
                    else:
                        # This shouldn't happen if we counted correctly
                        print(f"Warning: Missing cache entry for {cache_key}")
                        drive_times.iloc[i, j] = 30  # Default
        
        if progress_callback:
            progress_callback(api_calls_needed, api_calls_needed, 
                             "All drive times retrieved from cache", True)
        
        return drive_times
    
    # Make the necessary API calls
    calls_made = 0
    
    for i in range(num_stations):
        for j in range(num_stations):
            if i == j:
                # No travel time to self, just the inspection time
                drive_times.iloc[i, j] = 0
            else:
                coord1 = station_data.iloc[i]['Coordinates']
                coord2 = station_data.iloc[j]['Coordinates']
                cache_key = f"{coord1}|{coord2}"
                
                if cache_key in cache_data:
                    # Use cached value
                    drive_times.iloc[i, j] = cache_data[cache_key]
                else:
                    # Call the API
                    drive_time = get_google_maps_drive_time(coord1, coord2, api_key)
                    drive_times.iloc[i, j] = drive_time
                    cache_data[cache_key] = drive_time
                    
                    # Increment counter and update progress
                    calls_made += 1
                    
                    if progress_callback and calls_made % 5 == 0:
                        progress_callback(calls_made, api_calls_needed, 
                                         f"Calculated {calls_made}/{api_calls_needed} drive times", 
                                         calls_made == api_calls_needed)
                    
                    # Respect Google's rate limits (optional, adjust as needed)
                    time.sleep(0.2)  # 5 queries per second max
    
    # Save updated cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        print(f"Saved {len(cache_data)} drive times to cache")
    except Exception as e:
        print(f"Error saving cache: {e}")
    
    if progress_callback:
        progress_callback(api_calls_needed, api_calls_needed, 
                         "All drive times calculated and cached", True)
    
    return drive_times

# Example usage in your main application:
"""
# In your main code, you would use it like this:

# 1. Import the module
from google_maps_integration import calculate_google_drive_times

# 2. Set your API key
google_maps_api_key = "YOUR_API_KEY_HERE"

# 3. Replace the OSRM call with Google Maps
travel_times_df = calculate_google_drive_times(station_data, google_maps_api_key, 
                                               progress_callback=update_progress)

# 4. Then continue with your existing code
data = create_data_model(station_data, travel_times_df, hours=10)
solution, manager, routing = solve_route(data)
"""
