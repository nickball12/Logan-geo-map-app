import pandas as pd
from datetime import datetime
import numpy as np
import time
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os

# Define your custom column names for easier use
COLUMN_MAPPING = {
    'business name': 'Name',
    'address': 'Address',
    'city': 'City',
    'state': 'State',
    'zip': 'Zip',
    'total active devices': 'Active Devices',
    'last regular inspection date': 'Last Inspection Date',
    'violation status': 'Reinspection Status',
    'violation date': 'Violation Date',
    'Insp No.': 'Inspection Number',
    'Coordinates': 'Coordinates'
}

# --- YOUR STARTING ADDRESS ---
STARTING_LOCATION = {
    'Name': 'My Home Base',
    'Full Address': '5300 S 300 W, Murray, UT',
    'Active Devices': 0,
    'Last Inspection Date': pd.to_datetime('2000-01-01'),
    'Reinspection': 'No',
    'Inspection Time (min)': 0
}

def create_data_model(station_data, travel_times, hours):
    """Creates the data model for the a routing problem."""
    data = {}
    
    # Convert hours to minutes
    total_time_limit_minutes = hours * 60
    
    # Filter stations by priority
    station_data['Priority Score'] = station_data['Days Since Inspection']
    station_data.loc[station_data['Reinspection'] == 'Yes', 'Priority Score'] *= 0.8
    
    # Replace any NaN values with 0 before casting to int
    station_data['Priority Score'] = station_data['Priority Score'].fillna(0)
    station_data = station_data.sort_values(by='Priority Score', ascending=False)
    
    # Fill any NaN values with 0 before converting to a list
    travel_times = travel_times.fillna(0)
    data['travel_times'] = travel_times.values.astype(int).tolist()
    data['inspection_times'] = station_data['Inspection Time (min)'].values.astype(int).tolist()
    data['demands'] = station_data['Priority Score'].values.astype(int).tolist()

    data['num_vehicles'] = 1
    data['depot'] = 0
    data['time_limit'] = int(total_time_limit_minutes)
    
    return data

def solve_route(data):
    """
    Solves the routing problem with a time limit and capacity constraint.
    """
    manager = pywrapcp.RoutingIndexManager(len(data['travel_times']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        return data['travel_times'][from_index][to_index]
    
    time_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

    def demand_callback(from_index):
        return data['demands'][from_index]
        
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [100000],
        True,
        'Capacity'
    )
    
    def total_time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['travel_times'][from_node][to_node]
        inspection_time = data['inspection_times'][from_node]
        return travel_time + inspection_time

    total_time_callback_index = routing.RegisterTransitCallback(total_time_callback)
    
    routing.AddDimension(
        total_time_callback_index,
        100000,
        data['time_limit'],
        False,
        'Time'
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        
    solution = routing.SolveWithParameters(search_parameters)
    
    return solution, manager, routing

def print_solution(manager, routing, solution, station_data):
    """Prints the final solution."""
    index = routing.Start(0)
    route_time = 0
    route_inspection_time = 0
    route_travel_time = 0
    
    route_plan = []
    
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        station_info = station_data.iloc[node_index]
        route_plan.append(station_info)
        
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        
        travel_time = routing.GetArcCostForVehicle(previous_index, index, 0)
        
        route_travel_time += travel_time
        route_inspection_time += station_info['Inspection Time (min)']
        route_time += travel_time + station_info['Inspection Time (min)']

    print("\n--- Optimized Daily Plan ---")
    for i, stop in enumerate(route_plan):
        if stop['Name'] == 'My Home Base':
            print(f"Start at: {stop['Name']} ({stop['Full Address']})")
        else:
            print(f"{i}. Visit: {stop['Name']} ({stop['Full Address']})")
            print(f"   Inspection Time: {int(stop['Inspection Time (min)']):>2} min")
            print(f"   Days since last inspection: {int(stop['Days Since Inspection'])} days")
            if stop['Reinspection'] == 'Yes':
                print("   **This is a reinspection.**")
    
    last_stop_index = manager.IndexToNode(index)
    return_time = routing.GetArcCostForVehicle(last_stop_index, routing.End(0), 0)
    route_travel_time += return_time
    route_time += return_time

    print(f"\nReturn trip to: {station_data.iloc[manager.IndexToNode(routing.End(0))]['Name']}")
    
    print("\n--- Estimated Totals ---")
    print(f"Total time on route: {int(route_time)} minutes ({int(route_time/60)} hours, {int(route_time % 60)} minutes)")
    print(f"Total travel time: {int(route_travel_time)} minutes")
    print(f"Total inspection time: {int(route_inspection_time)} minutes")
    print(f"Total stations to visit: {len(route_plan) - 1}")

def get_coordinates_local(address):
    """
    Returns the coordinates (latitude, longitude) for a given address
    by using a free geocoding service. Includes a delay to respect rate limits.
    """
    geolocator = Nominatim(user_agent="my_inspector_app")
    
    time.sleep(1)
    
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return (location.latitude, location.longitude)
        else:
            print(f"Warning: Could not geocode address: {address}")
            return None
    except Exception as e:
        print(f"Error getting coordinates for {address}: {e}")
        return None

def read_and_process_station_data(file_path):
    """
    Reads station data from an Excel file, renames columns, and processes the data.
    """
    try:
        df = pd.read_excel(file_path)
        print("Excel file loaded successfully.")

        df.rename(columns=COLUMN_MAPPING, inplace=True)
        
        df = df[df['Inspection Number'] != 'NEW']
        
        df.dropna(subset=['Name', 'Active Devices'], inplace=True)
        df['Active Devices'] = pd.to_numeric(df['Active Devices'], errors='coerce').fillna(0).astype(int)
        df.dropna(subset=['Active Devices'], inplace=True)
        
        df['Last Inspection Date'] = pd.to_datetime(df['Last Inspection Date'], errors='coerce')
        
        today = datetime.now().date()
        df['Violation Date'] = pd.to_datetime(df['Violation Date'], errors='coerce')
        df['Days Since Violation'] = (pd.to_datetime(today) - df['Violation Date']).dt.days

        df['Reinspection'] = df['Reinspection Status'].apply(
            lambda x: 'Yes' if x == 'V' else 'No'
        )
        
        df.loc[(df['Reinspection'] == 'Yes') & (df['Days Since Violation'] < 14), 'Reinspection'] = 'No'

        df['Last Inspection Date'] = df['Last Inspection Date'].fillna(pd.to_datetime('2000-01-01'))
        df['Days Since Inspection'] = (pd.to_datetime(today) - df['Last Inspection Date']).dt.days.astype(int)

        df['Inspection Time (min)'] = df.apply(
            lambda row: 15 if row['Reinspection'] == 'Yes' else row['Active Devices'] * 6,
            axis=1
        )
        
        df['Zip'] = df['Zip'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        df['Full Address'] = df['Address'].fillna('') + ', ' + df['City'].fillna('') + ', ' + df['State'].fillna('') + ' ' + df['Zip'].fillna('').astype(str)

        starting_df = pd.DataFrame([STARTING_LOCATION])
        df = pd.concat([starting_df, df], ignore_index=True)
        
        print("Data processed with priority scores and estimated times.")
        return df

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except KeyError as e:
        print(f"Error: A required column was not found. Please check your column headers. Missing column: {e}")
        return None

def calculate_drive_times_local(station_data):
    """
    Calculates drive times using local, free libraries.
    """
    # Force recalculation every time to prevent corrupted CSV data from being used.
    if os.path.exists("travel_times.csv"):
        os.remove("travel_times.csv")

    print("\nFetching coordinates for all locations. This will be the longest step...")

    # Geocode any addresses that do not have coordinates
    station_data['Coordinates'] = station_data.apply(lambda row: get_coordinates_local(row['Full Address']) if pd.isna(row['Coordinates']) else row['Coordinates'], axis=1)
    
    station_data.reset_index(drop=True, inplace=True)

    un_geocoded_stations = station_data[station_data['Coordinates'].isna()]
    if not un_geocoded_stations.empty:
        print("\n--- The following addresses could not be geocoded and will be skipped: ---")
        print(un_geocoded_stations[['Name', 'Full Address']])
    
    station_data.dropna(subset=['Coordinates'], inplace=True)
    
    locations = list(station_data['Coordinates'])
    num_locations = len(locations)
    
    travel_times_df = pd.DataFrame(
        np.zeros((num_locations, num_locations)), 
        index=station_data.index, 
        columns=station_data.index
    )
    
    for i in range(num_locations):
        for j in range(i, num_locations):
            if i == j:
                travel_times_df.iloc[i, j] = 0
            else:
                try:
                    distance = geodesic(locations[i], locations[j]).miles
                    travel_time_minutes = (distance / 40) * 60
                    travel_times_df.iloc[i, j] = travel_time_minutes
                    travel_times_df.iloc[j, i] = travel_time_minutes
                except Exception as e:
                    print(f"Error calculating distance: {e}")
                    travel_times_df.iloc[i, j] = np.nan
                    travel_times_df.iloc[j, i] = np.nan
    
    # --- The Fix: Round up to the nearest integer and convert to list ---
    travel_times_df = np.ceil(travel_times_df.fillna(0)).astype(int)
    
    return travel_times_df

if __name__ == "__main__":
    file_name = "your_station_data.xlsx"
    
    try:
        import ortools
    except ImportError:
        print("The 'ortools' library is required to solve the routing problem.")
        print("Please run 'pip install ortools' or 'py -m pip install ortools' in your terminal.")
        exit()
        
    station_data = read_and_process_station_data(file_name)

    if station_data is not None:
        travel_times_df = calculate_drive_times_local(station_data)
        travel_times_df.to_csv("travel_times.csv")
        
        hours_to_work = float(input("Enter the number of hours you plan to work today: "))
        
        data = create_data_model(station_data, travel_times_df, int(hours_to_work))
        
        solution, manager, routing = solve_route(data)
        
        if solution:
            print_solution(manager, routing, solution, station_data)
        else:
            print("\nNo solution found that fits within the time limit. Try a longer workday.")