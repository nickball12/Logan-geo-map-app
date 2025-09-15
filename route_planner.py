import pandas as pd
from datetime import datetime
import numpy as np
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import requests
import concurrent.futures
import pickle

# --- YOUR STARTING ADDRESS ---
STARTING_LOCATION = {
    'Name': 'My Home Base',
    'Full Address': '5300 S 300 W, Murray, UT',
    'Inspection Time (min)': 0,
    'Coordinates': '40.6542, -111.8955'
}


def get_coordinates_local(address):
    if not address or pd.isna(address):
        print(f"Invalid address: {address}")
        return None
    
    try:
        # Clean up the address - remove extra spaces and normalize format
        address = address.strip()
        
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
            
        print(f"Geocoding address: {search_address}")
        geolocator = Nominatim(user_agent="multi_sheet_router", timeout=20)  # Increased timeout
        location = geolocator.geocode(search_address, exactly_one=True)
        
        if location:
            return f"{location.latitude}, {location.longitude}"
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
                    print(f"Trying simplified address: {simplified_address}")
                    location = geolocator.geocode(simplified_address, exactly_one=True)
                    
                    if location:
                        return f"{location.latitude}, {location.longitude}"
                
                # If that fails, try just the street address with UT, USA
                simplified_address = f"{street_address}, UT, USA"
                print(f"Trying just street address: {simplified_address}")
                location = geolocator.geocode(simplified_address, exactly_one=True)
                
                if location:
                    return f"{location.latitude}, {location.longitude}"
                
                # If zip code exists, try with just zip code in Utah
                if zip_code:
                    zip_only_address = f"UT {zip_code}, USA"
                    print(f"Trying ZIP only: {zip_only_address}")
                    location = geolocator.geocode(zip_only_address, exactly_one=True)
                    
                    if location:
                        return f"{location.latitude}, {location.longitude}"
            
            print(f"Could not geocode address: {address}")
            # Return a default coordinate in Utah if geocoding fails
            return "40.7608, -111.8910"  # Default to Salt Lake City
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        # Return a default coordinate in Utah if geocoding fails
        return "40.7608, -111.8910"  # Default to Salt Lake City

def excel_column_to_index(column_letter):
    """Convert Excel column letter to zero-based index."""
    if not column_letter:
        return 0
    column_letter = column_letter.upper()
    result = 0
    for char in column_letter:
        result = result * 26 + (ord(char) - ord('A') + 1)
    return result - 1

def read_and_process_address_data(file_name, address_col, city_col, skip_rows, sheet_name=0, 
                              zip_col=None, state_col=None, last_visited_col=None,
                              pumps_col=None, pumps_per_station=2):
    try:
        print(f"Processing file: {file_name}")
        print(f"Address column: {address_col}, City column: {city_col}")
        print(f"ZIP column: {zip_col}, State column: {state_col}")
        print(f"Last visited column: {last_visited_col}")
        print(f"Pumps column: {pumps_col}, Pumps per station: {pumps_per_station}")
        print(f"Skip rows: {skip_rows}, Sheet name: {sheet_name}")
        
        if file_name.lower().endswith('.xlsx'):
            # For Excel files, handle column letters (A, B, C, etc.)
            if isinstance(address_col, str) and address_col.isalpha():
                address_col_index = excel_column_to_index(address_col)
            else:
                address_col_index = int(address_col)
                
            if isinstance(city_col, str) and city_col.isalpha():
                city_col_index = excel_column_to_index(city_col)
            else:
                city_col_index = int(city_col)
            
            # Process ZIP code column if provided
            zip_col_index = None
            if zip_col and zip_col.strip():
                if isinstance(zip_col, str) and zip_col.isalpha():
                    zip_col_index = excel_column_to_index(zip_col)
                else:
                    try:
                        zip_col_index = int(zip_col)
                    except ValueError:
                        print(f"Invalid ZIP column format: {zip_col}. Ignoring.")
            
            # Process state column if provided
            state_col_index = None
            if state_col and state_col.strip():
                if isinstance(state_col, str) and state_col.isalpha():
                    state_col_index = excel_column_to_index(state_col)
                else:
                    try:
                        state_col_index = int(state_col)
                    except ValueError:
                        print(f"Invalid state column format: {state_col}. Ignoring.")
            
            # Process last visited date column if provided
            last_visited_col_index = None
            if last_visited_col and last_visited_col.strip():
                if isinstance(last_visited_col, str) and last_visited_col.isalpha():
                    last_visited_col_index = excel_column_to_index(last_visited_col)
                else:
                    try:
                        last_visited_col_index = int(last_visited_col)
                    except ValueError:
                        print(f"Invalid last visited column format: {last_visited_col}. Ignoring.")
            
            # Process pumps column if provided
            pumps_col_index = None
            if pumps_col and pumps_col.strip():
                if isinstance(pumps_col, str) and pumps_col.isalpha():
                    pumps_col_index = excel_column_to_index(pumps_col)
                else:
                    try:
                        pumps_col_index = int(pumps_col)
                    except ValueError:
                        print(f"Invalid pumps column format: {pumps_col}. Ignoring.")
            
            print(f"Reading Excel file with pandas. Address index: {address_col_index}, City index: {city_col_index}")
            print(f"ZIP index: {zip_col_index}, State index: {state_col_index}")
            print(f"Last visited index: {last_visited_col_index}, Pumps index: {pumps_col_index}")
            # Read the Excel file with specified sheet
            df = pd.read_excel(file_name, sheet_name=sheet_name, skiprows=skip_rows, header=0, engine='openpyxl')
            print(f"Excel file read successfully. Columns: {list(df.columns)}")
            
            # Use explicitly provided state and zip columns first
            state_col_index = state_col_index if state_col_index is not None else None
            zip_col_index = zip_col_index if zip_col_index is not None else None
            
            # If no explicit columns provided, attempt to detect them
            if state_col_index is None or zip_col_index is None:
                # Look for state and zip columns
                for col_idx, col_name in enumerate(df.columns):
                    col_str = str(col_name).lower()
                    # Look for column names likely to contain state info
                    if state_col_index is None and col_str in ['state', 'st', 'province']:
                        state_col_index = col_idx
                    # Look for column names likely to contain zip/postal code info
                    elif zip_col_index is None and col_str in ['zip', 'zipcode', 'postal', 'postal code', 'zip code']:
                        zip_col_index = col_idx
        else:
            # For CSV files, use numeric indices
            address_col_index = int(address_col)
            city_col_index = int(city_col)
            
            # Process ZIP code column if provided
            zip_col_index = None
            if zip_col and zip_col.strip():
                try:
                    zip_col_index = int(zip_col)
                except ValueError:
                    print(f"Invalid ZIP column format for CSV: {zip_col}. Ignoring.")
            
            # Process state column if provided
            state_col_index = None
            if state_col and state_col.strip():
                try:
                    state_col_index = int(state_col)
                except ValueError:
                    print(f"Invalid state column format for CSV: {state_col}. Ignoring.")
            
            # Process last visited date column if provided
            last_visited_col_index = None
            if last_visited_col and last_visited_col.strip():
                try:
                    last_visited_col_index = int(last_visited_col)
                except ValueError:
                    print(f"Invalid last visited column format for CSV: {last_visited_col}. Ignoring.")
            
            # Process pumps column if provided
            pumps_col_index = None
            if pumps_col and pumps_col.strip():
                try:
                    pumps_col_index = int(pumps_col)
                except ValueError:
                    print(f"Invalid pumps column format for CSV: {pumps_col}. Ignoring.")
            
            df = pd.read_csv(file_name, skiprows=skip_rows, header=0)
            print(f"CSV file read successfully. Columns: {list(df.columns)}")
            
            # If no explicit columns provided, attempt to detect them
            if state_col_index is None or zip_col_index is None:
                for col_idx, col_name in enumerate(df.columns):
                    col_str = str(col_name).lower()
                    if state_col_index is None and col_str in ['state', 'st', 'province']:
                        state_col_index = col_idx
                    elif zip_col_index is None and col_str in ['zip', 'zipcode', 'postal', 'postal code', 'zip code']:
                        zip_col_index = col_idx

        # Get the actual column names from their index
        address_col_name = df.columns[address_col_index]
        city_col_name = df.columns[city_col_index]
        
        # Create a copy with the address and city columns
        station_data = df[[address_col_name, city_col_name]].copy()
        station_data = station_data.dropna(subset=[address_col_name, city_col_name])
        
        # Add last visited date if available
        if last_visited_col_index is not None and last_visited_col_index < len(df.columns):
            last_visited_col_name = df.columns[last_visited_col_index]
            station_data['Last Visited'] = df[last_visited_col_name]
            # Convert to datetime and handle conversion errors
            try:
                station_data['Last Visited'] = pd.to_datetime(station_data['Last Visited'], errors='coerce')
                # Fill missing/invalid dates with a very old date to prioritize them
                station_data['Last Visited'].fillna(pd.Timestamp('1900-01-01'), inplace=True)
                # Calculate days since last visit
                station_data['Days Since Visit'] = (pd.Timestamp.now() - station_data['Last Visited']).dt.days
                # For sorting purposes, any negative values (future dates) should be set to 0
                station_data['Days Since Visit'] = station_data['Days Since Visit'].clip(lower=0)
                print("Last visited dates processed successfully")
            except Exception as e:
                print(f"Error processing last visited dates: {e}")
                # Create a default column with 0 days (no priority)
                station_data['Days Since Visit'] = 0
        else:
            # If no last visited column, set all to same priority
            station_data['Last Visited'] = pd.NaT
            station_data['Days Since Visit'] = 0
            
        # Add pumps information if available
        if pumps_col_index is not None and pumps_col_index < len(df.columns):
            pumps_col_name = df.columns[pumps_col_index]
            station_data['Total Pumps'] = df[pumps_col_name]
            # Convert to numeric and handle conversion errors
            try:
                station_data['Total Pumps'] = pd.to_numeric(station_data['Total Pumps'], errors='coerce')
                # Fill missing/invalid values with default (1 pump)
                station_data['Total Pumps'].fillna(1, inplace=True)
                # Make sure they're integers
                station_data['Total Pumps'] = station_data['Total Pumps'].astype(int)
                # Cap at 1 if negative
                station_data['Total Pumps'] = station_data['Total Pumps'].clip(lower=1)
                print("Pump counts processed successfully")
            except Exception as e:
                print(f"Error processing pump counts: {e}")
                # Create a default column with 1 pump
                station_data['Total Pumps'] = 1
        else:
            # If no pumps column, set default of 1 pump per station
            station_data['Total Pumps'] = 1
            
        # Calculate number of pumps to inspect at each station
        # Default is the user-specified pumps_per_station or total pumps, whichever is smaller
        station_data['Pumps To Inspect'] = station_data['Total Pumps'].apply(
            lambda x: min(x, pumps_per_station)
        )
        
        # Build full address string with state and zip if available
        if state_col_index is not None and state_col_index < len(df.columns):
            state_col_name = df.columns[state_col_index]
            station_data['State'] = df[state_col_name]
        else:
            # Default to Utah if no state column
            station_data['State'] = 'UT'
            
        if zip_col_index is not None and zip_col_index < len(df.columns):
            zip_col_name = df.columns[zip_col_index]
            # Convert zip codes to string format without decimal
            station_data['Zip'] = df[zip_col_name].astype(str).replace(r'\.0$', '', regex=True)
            # Build full address with all components
            station_data['Full Address'] = station_data.apply(
                lambda row: f"{row[address_col_name]}, {row[city_col_name]}, {row['State']} {row.get('Zip', '')}".strip(),
                axis=1
            )
        else:
            # Build full address without zip
            station_data['Full Address'] = station_data.apply(
                lambda row: f"{row[address_col_name]}, {row[city_col_name]}, {row['State']}".strip(),
                axis=1
            )
        
        station_data['Name'] = station_data['Full Address']
        station_data['Inspection Time (min)'] = 30

        # Add starting location to the dataframe
        start_df = pd.DataFrame([STARTING_LOCATION])
        station_data = pd.concat([start_df, station_data], ignore_index=True)
        
        return station_data.drop_duplicates(subset=['Full Address']).reset_index(drop=True)

    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return None
    except IndexError:
        print(f"Error: Column index out of range for {file_name}. Please check your column indices.")
        return None
    except Exception as e:
        print(f"An error occurred with {file_name}: {e}")
        return None


def get_osrm_drive_time(coord1, coord2):
    try:
        lat1, lon1 = map(float, coord1.split(','))
        lat2, lon2 = map(float, coord2.split(','))
        
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        response = requests.get(url, timeout=10)  # Add explicit timeout
        data = response.json()
        
        if data.get('code') == 'Ok' and 'routes' in data and len(data['routes']) > 0:
            return data['routes'][0]['duration'] / 60
        else:
            print(f"OSRM API returned unexpected data: {data}")
            # Improved fallback to distance-based estimate
            return get_better_distance_estimate(lat1, lon1, lat2, lon2)
    except Exception as e:
        print(f"Error in OSRM API call: {e}")
        try:
            lat1, lon1 = map(float, coord1.split(','))
            lat2, lon2 = map(float, coord2.split(','))
            # Use better fallback calculation
            return get_better_distance_estimate(lat1, lon1, lat2, lon2)
        except Exception as e2:
            print(f"Error in fallback calculation: {e2}")
            return 30  # Default to 30 minutes if all else fails

def get_better_distance_estimate(lat1, lon1, lat2, lon2):
    """
    Improved distance-based time estimation that accounts for:
    - Actual geodesic distance
    - Typical road network inefficiencies
    - Variable speeds for different distances
    """
    # Calculate direct distance in miles
    distance_miles = geodesic((lat1, lon1), (lat2, lon2)).miles
    
    # Add road network inefficiency factor - roads aren't straight lines
    # Longer distances tend to use highways which are more direct
    if distance_miles < 5:
        # Very short distances have more turns, traffic lights, etc.
        distance_miles *= 1.4  # 40% longer than direct
        avg_speed = 25  # mph (local roads)
    elif distance_miles < 15:
        # Medium distances mix local roads and some highways
        distance_miles *= 1.3  # 30% longer than direct
        avg_speed = 35  # mph (mix of roads)
    else:
        # Longer distances primarily use highways
        distance_miles *= 1.2  # 20% longer than direct
        avg_speed = 55  # mph (highways)
    
    # Calculate time in minutes
    time_minutes = (distance_miles / avg_speed) * 60
    
    # Add fixed time for departure/arrival (traffic lights, finding parking, etc.)
    time_minutes += 5
    
    return time_minutes


def calculate_drive_times_local(station_data, use_cache=True, progress_callback=None, max_routes=500, cancellation_event=None):
    """
    Calculate travel times between all stations with enhanced caching and optimization
    
    Args:
        station_data: DataFrame containing station information with coordinates
        use_cache: Whether to use the travel time cache (default: True)
        progress_callback: Optional callback function for progress updates
        max_routes: Maximum number of routes to calculate (default: 500)
        cancellation_event: Optional event to signal cancellation
        
    Returns:
        DataFrame with travel times between all stations
    """
    num_stations = len(station_data)
    total_possible_pairs = num_stations * (num_stations - 1)
    
    print(f"\n=== Route Calculation Summary ===")
    print(f"Total stations: {num_stations}")
    print(f"Total possible routes: {total_possible_pairs}")
    print(f"Maximum routes set to: {max_routes}")
    print(f"Reduction target: {100 - (max_routes / total_possible_pairs * 100):.2f}% fewer calculations")
    print(f"===============================\n")
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(0, 1, "Operation cancelled", True, 'routing', 'Route Calculation Cancelled', True)
        return None
    
    travel_times_df = pd.DataFrame(np.zeros((num_stations, num_stations)))
    
    # Create a cache file for travel times
    cache_file = 'travel_time_cache.pkl'
    travel_time_cache = {}
    
    # Load existing cache if available
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                travel_time_cache = pickle.load(f)
            print(f"Loaded {len(travel_time_cache)} cached travel times")
        except Exception as e:
            print(f"Error loading travel time cache: {e}")
            travel_time_cache = {}

    # Ensure all stations have coordinates
    station_data['Coordinates'] = station_data.apply(
        lambda row: get_coordinates_local(row['Full Address']) if pd.isna(row.get('Coordinates')) else row.get('Coordinates'),
        axis=1
    )
    
    # Ensure all stations have coordinates (use default if geocoding failed)
    station_data['Coordinates'] = station_data['Coordinates'].fillna("40.7608, -111.8910")  # Default to Salt Lake City
    
    # Group stations by city and zip to prioritize nearby locations
    if 'City' not in station_data.columns:
        # Try to extract city from Full Address
        station_data['City'] = station_data['Full Address'].apply(
            lambda addr: addr.split(',')[1].strip() if ',' in addr and len(addr.split(',')) > 1 else "Unknown"
        )
    
    if 'Zip' not in station_data.columns:
        # Try to extract zip from Full Address using regex
        import re
        station_data['Zip'] = station_data['Full Address'].apply(
            lambda addr: re.search(r'(\d{5}(?:-\d{4})?)', addr).group(1) 
            if re.search(r'(\d{5}(?:-\d{4})?)', addr) else "Unknown"
        )
        
    # Extract road names for Utah's grid-based system
    def extract_road_name(address):
        if not address or not isinstance(address, str):
            return "Unknown"
        
        # First, get the first part of the address (before the first comma)
        if ',' in address:
            street_part = address.split(',')[0].strip()
        else:
            street_part = address.strip()
            
        # Utah road patterns to match
        import re
        
        # 1. Number + Direction (Utah's grid system - very common)
        # Match patterns like "300 S", "400 W", "100 North", "200 East"
        grid_match = re.search(r'(\d+\s+(?:[NSEW]|[Nn]orth|[Ss]outh|[Ee]ast|[Ww]est))', street_part)
        if grid_match:
            return grid_match.group(1).upper()
            
        # 2. Named roads with common suffixes
        road_suffixes = ['St', 'Street', 'Ave', 'Avenue', 'Blvd', 'Boulevard', 'Rd', 'Road', 
                         'Ln', 'Lane', 'Dr', 'Drive', 'Way', 'Place', 'Pl', 'Ct', 'Court', 'Circle', 'Cir']
        for suffix in road_suffixes:
            pattern = fr'([\w\s]+\s+{suffix})'
            named_road_match = re.search(pattern, street_part, re.IGNORECASE)
            if named_road_match:
                return named_road_match.group(1).strip()
                
        # 3. For Utah's common highways (e.g., State St, Redwood Rd)
        common_roads = ['State St', 'Redwood Rd', 'Main St', 'Bangerter', 'Highland Dr']
        for road in common_roads:
            if road.lower() in street_part.lower():
                return road
                
        # 4. If no match, try to get any string with a number followed by words
        generic_match = re.search(r'(\d+\s+[\w\s]+)', street_part)
        if generic_match:
            return generic_match.group(1).strip()
            
        # If all else fails, return the first part of the address
        return street_part[:30] if len(street_part) > 30 else street_part
        
    # Add road name column
    station_data['Road'] = station_data['Full Address'].apply(extract_road_name)
    print("Road names extracted, found the following examples:")
    # Print some sample road names to verify extraction
    road_samples = station_data['Road'].sample(min(5, len(station_data))).tolist()
    for road in road_samples:
        print(f"  - {road}")
    
    # Helper function to calculate straight-line distance between coordinates
    def calculate_distance(coord1, coord2):
        if not coord1 or not coord2:
            return float('inf')
        try:
            lat1, lon1 = map(float, coord1.split(','))
            lat2, lon2 = map(float, coord2.split(','))
            return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5
        except:
            return float('inf')
    
    # Create a list of location pairs to process
    location_pairs = []
    
    # First, add pairs within the same city
    cities = station_data['City'].unique()
    print(f"Found {len(cities)} unique cities")
    city_pair_count = 0
    
    for city in cities:
        city_indices = station_data[station_data['City'] == city].index.tolist()
        city_count = len(city_indices)
        potential_city_pairs = city_count * (city_count - 1)
        city_pair_count += potential_city_pairs
        
        for i in city_indices:
            for j in city_indices:
                if i != j:
                    location_pairs.append((i, j, 1))  # Priority 1 (highest) for same city
    
    print(f"Added {city_pair_count} same-city pairs")
    
    # Then add pairs within the same zip code that weren't already added
    zips = station_data['Zip'].unique()
    added_pairs = set((i, j) for i, j, _ in location_pairs)
    zip_pair_count = 0
    
    print(f"Found {len(zips)} unique ZIP codes")
    for zip_code in zips:
        zip_indices = station_data[station_data['Zip'] == zip_code].index.tolist()
        zip_count = len(zip_indices)
        potential_zip_pairs = zip_count * (zip_count - 1)
        
        new_zip_pairs = 0
        for i in zip_indices:
            for j in zip_indices:
                if i != j and (i, j) not in added_pairs:
                    location_pairs.append((i, j, 2))  # Priority 2 for same zip
                    added_pairs.add((i, j))
                    new_zip_pairs += 1
        
        zip_pair_count += new_zip_pairs
    
    print(f"Added {zip_pair_count} same-ZIP pairs")
    
    # Add pairs that are on the same road (using Utah's grid system logic)
    roads = station_data['Road'].unique()
    road_pair_count = 0
    
    print(f"\n=== Road-Based Prioritization ===")
    print(f"Found {len(roads)} unique roads")
    
    # Print some example roads for debugging
    sample_roads = roads[:min(5, len(roads))]
    print("Sample road names found:")
    for road in sample_roads:
        road_count = len(station_data[station_data['Road'] == road])
        print(f"  - '{road}' ({road_count} locations)")
    
    for road in roads:
        if road == "Unknown":
            continue
            
        road_indices = station_data[station_data['Road'] == road].index.tolist()
        road_count = len(road_indices)
        
        # Skip roads with only one location
        if road_count <= 1:
            continue
            
        # For roads with many locations, print info
        if road_count > 3:
            print(f"Processing road '{road}' with {road_count} locations")
        
        new_road_pairs = 0
        for i in road_indices:
            for j in road_indices:
                if i != j and (i, j) not in added_pairs:
                    # Priority 3 for same road
                    location_pairs.append((i, j, 3))
                    added_pairs.add((i, j))
                    new_road_pairs += 1
        
        road_pair_count += new_road_pairs
    
    print(f"Added {road_pair_count} same-road pairs")
    print(f"===========================\n")
    
    # Calculate distances for remaining pairs and sort by distance
    remaining_pairs = []
    for i in range(num_stations):
        for j in range(num_stations):
            if i != j and (i, j) not in added_pairs:
                coord1 = station_data.iloc[i]['Coordinates']
                coord2 = station_data.iloc[j]['Coordinates']
                distance = calculate_distance(coord1, coord2)
                remaining_pairs.append((i, j, distance))
    
    print(f"Found {len(remaining_pairs)} remaining pairs by distance")
    
    # Sort remaining pairs by distance
    remaining_pairs.sort(key=lambda x: x[2])
    
    # Add the closest remaining pairs up to the limit
    remaining_limit = max(0, max_routes - len(location_pairs))
    location_pairs.extend(remaining_pairs[:remaining_limit])
    
    print(f"Added {min(remaining_limit, len(remaining_pairs))} closest remaining pairs")
    
    # Sort all pairs by priority
    location_pairs.sort(key=lambda x: x[2] if isinstance(x[2], (int, float)) else float('inf'))
    
    # Limit the total number of pairs to calculate
    location_pairs = location_pairs[:max_routes]
    
    total_possible_pairs = num_stations * (num_stations-1)
    print(f"Calculating travel times for {len(location_pairs)} location pairs (out of {total_possible_pairs} possible pairs)")
    print(f"Reduction: {100 - (len(location_pairs) / total_possible_pairs * 100):.2f}% fewer calculations")
    
    # Sequential implementation for better progress tracking
    total_calculations = len(location_pairs)
    completed = 0
    
    # Update progress if callback provided
    if progress_callback:
        progress_callback(0, total_calculations, 
                         f"Starting travel time calculations for {total_calculations} routes " +
                         f"(reduced from {total_possible_pairs} total possible routes)")
    
    # Set default large travel time for all pairs
    for i in range(num_stations):
        for j in range(num_stations):
            if i != j:
                travel_times_df.iloc[i, j] = 999  # Default large value
    
    # Calculate travel times only for the selected pairs
    for idx, (i, j, _) in enumerate(location_pairs):
        if i == j:
            travel_times_df.iloc[i, j] = 0  # No travel time to same location
            continue
            
        coord1 = station_data.iloc[i]['Coordinates']
        coord2 = station_data.iloc[j]['Coordinates']
        
        # Skip if coordinates are missing
        if not coord1 or not coord2:
            travel_times_df.iloc[i, j] = 30  # Default value
            completed += 1
            continue
            
        # Create a unique key for this pair of coordinates
        cache_key = f"{coord1}_to_{coord2}"
        
        # Check if we have this in cache
        if use_cache and cache_key in travel_time_cache:
            travel_times_df.iloc[i, j] = travel_time_cache[cache_key]
        else:
            try:
                travel_time_minutes = get_osrm_drive_time(coord1, coord2)
                travel_times_df.iloc[i, j] = travel_time_minutes
                
                # Add to cache
                if use_cache:
                    travel_time_cache[cache_key] = travel_time_minutes
            except Exception as e:
                print(f"Error calculating drive time from {coord1} to {coord2}: {e}")
                travel_times_df.iloc[i, j] = 30  # Default if API fails
        
        # Update progress
        completed += 1
        if progress_callback and (completed % 5 == 0 or completed == total_calculations):  # Update every 5 calculations
            percent_complete = (completed / total_calculations) * 100
            progress_callback(completed, total_calculations, 
                             f"Calculating travel times: {completed}/{total_calculations} routes " +
                             f"({percent_complete:.1f}% complete)")
    
    # Save cache for future use
    if use_cache:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(travel_time_cache, f)
            print(f"Saved {len(travel_time_cache)} travel times to cache")
        except Exception as e:
            print(f"Error saving travel time cache: {e}")
    
    if progress_callback:
        progress_callback(total_calculations, total_calculations, 
                         f"Travel time calculations complete! Processed {total_calculations} routes " +
                         f"(saved {total_possible_pairs - total_calculations} calculations)")
    
    return np.ceil(travel_times_df.fillna(999)).astype(int)


def create_data_model(station_data, travel_times, hours, time_between_stops=0):
    data = {}
    data['travel_times'] = travel_times.values.astype(int).tolist()
    data['inspection_times'] = station_data['Inspection Time (min)'].values.astype(int).tolist()
    data['time_between_stops'] = time_between_stops
    data['num_vehicles'] = 1
    data['depot'] = 0
    data['time_limit'] = int(hours * 60)
    
    # Add days since last visit if available (for prioritization)
    if 'Days Since Visit' in station_data.columns:
        data['days_since_visit'] = station_data['Days Since Visit'].fillna(0).astype(int).tolist()
        # Normalize days since visit for use in prioritization
        max_days = max(data['days_since_visit']) if max(data['days_since_visit']) > 0 else 1
        data['normalized_priority'] = [day / max_days for day in data['days_since_visit']]
    else:
        # Default all locations to same priority if no visit data
        data['days_since_visit'] = [0] * len(data['travel_times'])
        data['normalized_priority'] = [0] * len(data['travel_times'])
    
    # Add pumps information if available
    if 'Total Pumps' in station_data.columns and 'Pumps to Inspect' in station_data.columns:
        data['total_pumps'] = station_data['Total Pumps'].fillna(0).astype(int).tolist()
        data['pumps_to_inspect'] = station_data['Pumps to Inspect'].fillna(0).astype(int).tolist()
        print("Using pump data from dataset")
    else:
        # Default to 0 pump per station if no pump data
        data['total_pumps'] = [0] * len(data['travel_times'])
        data['pumps_to_inspect'] = [0] * len(data['travel_times'])
        print("No pump data found in dataset, defaulting to 0")
    
    # Print some debug information
    print(f"\nRoute Optimization Parameters:")
    print(f"- Number of locations: {len(data['travel_times'])}")
    print(f"- Time limit: {data['time_limit']} minutes ({hours} hours)")
    print(f"- Time between stops: {data['time_between_stops']} minutes")
    print(f"- Using last visited dates for prioritization: {'Yes' if 'Days Since Visit' in station_data.columns else 'No'}")
    print(f"- Total pumps across all stations: {sum(data['total_pumps'])}")
    print(f"- Pumps to inspect across all stations: {sum(data['pumps_to_inspect'])}")
    print(f"- Total inspection time: {sum(data['inspection_times'])} minutes")
    
    # Calculate whether it's even possible to visit all locations
    total_min_travel_time = 0
    for i in range(1, len(data['travel_times'])):
        min_time_to_i = min([data['travel_times'][j][i] for j in range(len(data['travel_times'])) if j != i])
        total_min_travel_time += min_time_to_i
    
    min_return_time = min([data['travel_times'][i][0] for i in range(1, len(data['travel_times']))]) if len(data['travel_times']) > 1 else 0
    total_min_travel_time += min_return_time
    
    # Add time between stops for all stops except the last one (return to depot)
    total_time_between_stops = data['time_between_stops'] * (len(data['travel_times']) - 2) if len(data['travel_times']) > 1 else 0
    total_inspection_time = sum(data['inspection_times'])
    total_min_time = total_min_travel_time + total_inspection_time + total_time_between_stops
    
    print(f"- Minimum estimated travel time: {total_min_travel_time} minutes")
    print(f"- Total inspection time: {total_inspection_time} minutes")
    print(f"- Total time between stops: {total_time_between_stops} minutes")
    print(f"- Minimum total time needed: {total_min_time} minutes")
    
    # Adjust time limit if it's too restrictive
    if total_min_time > data['time_limit'] and hours < 24:
        # Increase time limit to allow at least half of the stops to be included
        min_feasible_time = max(data['time_limit'], total_min_time * 0.7)
        data['time_limit'] = int(min_feasible_time)
        print(f"- ADJUSTED time limit to: {data['time_limit']} minutes to make problem more feasible")
    
    print(f"- Final time limit: {data['time_limit']} minutes")
    print(f"- Is problem likely solvable: {'Yes' if total_min_time <= data['time_limit'] else 'No - time limit too low'}")
    
    return data


def solve_route(data, progress_callback=None, cancellation_event=None, max_pumps=None):
    print("Starting route optimization...")
    
    if progress_callback:
        progress_callback(1, 5, "Initializing route solver...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(1, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None
    
    manager = pywrapcp.RoutingIndexManager(len(data['travel_times']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    if progress_callback:
        progress_callback(2, 5, "Setting up transit callback and constraints...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(2, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Add the time between stops to each travel segment (except when returning to depot)
        additional_time = data['time_between_stops'] if to_node != data['depot'] else 0
        return data['travel_times'][from_node][to_node] + data['inspection_times'][from_node] + additional_time
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add time dimension - MODIFIED to allow drops with penalties
    routing.AddDimension(transit_callback_index, 0, data['time_limit'], True, 'Time')
    
    # If we have pump data and a max_pumps limit, add a dimension to track pump inspections
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
        print(f"Added pump inspection dimension with max {max_pumps} pumps per day")
        
        # Get the pump dimension
        pump_dimension = routing.GetDimensionOrDie('Pumps')
        
        # Set a cost coefficient to prioritize filling up the pump capacity
        # This incentivizes the solver to include stops with pumps
        pump_dimension.SetGlobalSpanCostCoefficient(100)
        print("Set priority to maximize pump inspections")
    else:
        print("No pump limit set, or no pump data available")
    
    # Important change: Make nodes droppable (except depot) with penalties
    # This allows the solver to skip some stops while maximizing the number of stops visited
    # Use penalty based on days since last visit to prioritize locations that haven't been visited recently
    base_penalty = 10000  # Base penalty for all nodes
    for node in range(1, len(data['travel_times'])):
        # Apply higher penalties for locations that haven't been visited in a long time
        # This makes them less likely to be skipped
        visit_priority_factor = 1 + (data['normalized_priority'][node] * 0.5)  # Scale factor based on days since visit
        penalty = int(base_penalty * visit_priority_factor)
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
        
        # Debug the prioritization
        if 'days_since_visit' in data and data['days_since_visit'][node] > 0:
            print(f"Node {node}: {data['days_since_visit'][node]} days since visit, priority factor: {visit_priority_factor:.2f}, penalty: {penalty}")
    
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

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Use PATH_CHEAPEST_ARC for a more reliable initial solution
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    
    # Improve local search to find better solutions
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    
    # Add time limit to prevent getting stuck for too long
    search_parameters.time_limit.seconds = 180  # 3 minutes is typically sufficient
    
    # Try harder to find a feasible solution
    search_parameters.log_search = True  # Enable logging for debugging
    search_parameters.solution_limit = 300  # Try to find up to 300 solutions

    # Use standard propagation for better compatibility
    search_parameters.guided_local_search_lambda_coefficient = 0.1  # Less aggressive local search
    
    if progress_callback:
        progress_callback(4, 5, "Running optimization algorithm (this may take a moment)...", False, 'solving', 'Route Optimization')
    
    # Check for cancellation
    if cancellation_event and cancellation_event.is_set():
        if progress_callback:
            progress_callback(4, 5, "Operation cancelled", True, 'solving', 'Route Optimization Cancelled', True)
        return None, None, None
    
    # Solve the route
    solution = routing.SolveWithParameters(search_parameters)
    
    # If no solution was found, try with a different strategy
    if not solution:
        print("First solution strategy failed, trying SAVINGS algorithm...")
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
            print("SAVINGS algorithm failed, trying PARALLEL_CHEAPEST_INSERTION...")
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
            progress_callback(5, 5, "Optimization complete, solution found!", False, 'solving', 'Route Optimization')
        else:
            progress_callback(5, 5, "Optimization complete, but no solution found.", False, 'solving', 'Route Optimization')
    
    print("Route optimization completed.")
    return solution, manager, routing


def print_solution(manager, routing, solution, station_data, file_name):
    if not solution:
        print("No solution found!")
        return
        
    index = routing.Start(0)
    route_time, route_inspection_time, route_travel_time = 0, 0, 0
    stops_count = 0
    total_pumps_inspected = 0
    
    route_plan = []
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        route_plan.append(station_data.iloc[node_index])
        stops_count += 1
        
        # Count pumps if the data exists
        if 'Pumps to Inspect' in station_data.columns:
            pumps_at_stop = station_data.iloc[node_index]['Pumps to Inspect']
            if not pd.isna(pumps_at_stop):
                total_pumps_inspected += pumps_at_stop
        
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        travel_time = routing.GetArcCostForVehicle(previous_index, index, 0)
        
        route_travel_time += travel_time
        route_inspection_time += station_data.iloc[node_index]['Inspection Time (min)']
        route_time += travel_time + station_data.iloc[node_index]['Inspection Time (min)']

    print(f"\n--- Optimized Route for: {file_name} ---")
    for i, stop in enumerate(route_plan):
        if i == 0:
            print(f"Start at: {stop['Name']} ({stop['Full Address']})")
        else:
            pump_info = ""
            if 'Pumps to Inspect' in stop and not pd.isna(stop['Pumps to Inspect']):
                pump_info = f" - {int(stop['Pumps to Inspect'])} pumps to inspect"
            if 'Total Pumps' in stop and not pd.isna(stop['Total Pumps']):
                pump_info += f" (of {int(stop['Total Pumps'])} total)"
            print(f"{i}. Visit: {stop['Name']}{pump_info}")
            
    print(f"\nReturn trip to: {station_data.iloc[manager.IndexToNode(routing.End(0))]['Name']}")
    
    print("\n--- Totals ---")
    print(f"Time on route: {int(route_time)} min ({int(route_time/60)} hrs, {int(route_time % 60)} min)")
    print(f"Travel time: {int(route_travel_time)} min")
    print(f"Inspection time: {int(route_inspection_time)} min")
    print(f"Stations to visit: {len(route_plan) - 1}")
    
    # Add pump inspection summary if applicable
    if 'Pumps to Inspect' in station_data.columns:
        total_available_pumps = station_data['Pumps to Inspect'].sum()
        remaining_pumps = total_available_pumps - total_pumps_inspected
        print(f"Total pumps to inspect: {int(total_pumps_inspected)} (of {int(total_available_pumps)} available)")
        print(f"Remaining pumps: {int(remaining_pumps)}")
    
    print(f"Skipped stops: {len(station_data) - stops_count}")
    
    return route_plan


if __name__ == "__main__":
    files_to_process = {
        "your_station_data1.xlsx - Salt Lake, Utah, Tooele.csv": 3,
        "your_station_data1.xlsx - Reinspections.csv": 3,
        "your_station_data1.xlsx - Complaints.csv": 3,
        "your_station_data1.xlsx - Out of Service Pumps.csv": 4
    }
    
    for file_name, skip_rows in files_to_process.items():
        print(f"\nProcessing {file_name}...")
        station_data = read_and_process_address_data(file_name, address_col_index=18, city_col_index=19, skip_rows=skip_rows)

        if station_data is not None and not station_data.empty:
            travel_times_df = calculate_drive_times_local(station_data)
            
            # Increase the time limit to allow more stops to be included
            data = create_data_model(station_data, travel_times_df, hours=16)
            solution, manager, routing = solve_route(data)
            
            print_solution(manager, routing, solution, station_data, file_name)
        else:
            print(f"Could not process {file_name}. It might be empty or have incorrect formatting.")