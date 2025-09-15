"""
Test script for the new station model and route planner
"""
import os
import pandas as pd
from station_model import Station
from station_manager import StationManager
from new_route_planner import plan_optimal_route

def main():
    """Test the new station model and route planner"""
    print("Testing new Station Model...")
    
    # Test with the example Excel file
    file_path = "your_station_data.xlsx"
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    
    print(f"Loading data from {file_path}...")
    
    # Create a station manager and load the data
    manager = StationManager()
    manager.load_from_excel(file_path, skip_rows=3)
    
    print(f"Loaded {len(manager.stations)} stations")
    
    # Print some information about the first few stations
    for i, station in enumerate(list(manager.stations.values())[:5]):
        print(f"\nStation {i+1}:")
        print(f"  Name: {station.name}")
        print(f"  Address: {station.address}, {station.city}, {station.state} {station.zip}")
        print(f"  Coordinates: {station.lat}, {station.lng}")
        print(f"  Pumps: {station.pumps}")
        print(f"  Reinspection: {station.reinspection}")
        print(f"  Complaint: {station.complaint}")
        print(f"  Out of Service: {station.out_of_service}")
        print(f"  Last Visited: {station.last_visited}")
    
    # Test the route planning
    print("\nPlanning a route for 8 hours with max 20 pumps...")
    
    def progress_callback(current, total, address='', done=False, phase=None, phase_name=None, cancelled=False):
        """Print progress updates"""
        print(f"Progress: {phase_name} - {current}/{total} - {address}")
    
    route_plan = plan_optimal_route(
        file_path,
        hours=8,
        max_stations=None,
        max_pumps=20,
        time_between_stops=15,
        progress_callback=progress_callback
    )
    
    if route_plan:
        print(f"\nRoute plan created with {len(route_plan) - 1} stops")
        print(f"Total travel time: {route_plan[0]['total_time']/60:.1f} hours")
        print(f"Total pumps: {route_plan[0]['total_pumps']}")
        
        # Print the route
        print("\nRoute:")
        for i, stop in enumerate(route_plan):
            if i == 0:
                print(f"Start: {stop.get('name', 'Starting Point')}")
            else:
                status = []
                if stop.get('reinspection'):
                    status.append("Reinspection")
                if stop.get('complaint'):
                    status.append("Complaint")
                if stop.get('out_of_service'):
                    status.append("Out of Service")
                
                status_str = f" ({', '.join(status)})" if status else ""
                print(f"Stop {i}: {stop.get('name')}{status_str} - {stop.get('travel_time', 0):.0f} min travel")
    else:
        print("Failed to create a route plan")

if __name__ == "__main__":
    main()
