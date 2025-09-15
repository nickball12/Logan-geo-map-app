import os
import json
from pprint import pprint
import sys

# Define the path to the latest route file
output_dir = "output"
latest_file = os.path.join(output_dir, "latest_route.json")

def diagnose_route_file(filepath):
    """Check and print details of a route file"""
    print(f"\n=== Analyzing route file: {filepath} ===")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File does not exist: {filepath}")
        return
    
    try:
        with open(filepath, 'r') as f:
            route_data = json.load(f)
        
        # Basic file statistics
        print(f"File size: {os.path.getsize(filepath)} bytes")
        print(f"Generated at: {route_data.get('generated_at', 'Unknown')}")
        
        # Route summary
        stops_count = route_data.get('stops_count', 0)
        total_stops = route_data.get('total_stops', 0)
        skipped_stops = route_data.get('skipped_stops', 0)
        
        print(f"Stops in route: {stops_count}")
        print(f"Total stops available: {total_stops}")
        print(f"Skipped stops: {skipped_stops} ({route_data.get('skipped_percentage', 0)}%)")
        print(f"Total route time: {route_data.get('total_time_minutes', 0)} minutes")
        
        # Analyze route plan
        route_plan = route_data.get('route_plan', [])
        if not route_plan:
            print("\nWARNING: Route plan is empty!")
        else:
            print(f"\nRoute contains {len(route_plan)} stops:")
            for i, stop in enumerate(route_plan):
                name = stop.get('Name', 'Unknown')
                address = stop.get('Full Address', 'No address')
                insp_time = stop.get('Inspection Time (min)', 0)
                print(f"{i+1}. {name} - {address} (Inspection time: {insp_time} min)")
                
            # Check if we only have the starting location
            if len(route_plan) == 1 and route_plan[0]['Name'] == 'My Home Base':
                print("\nISSUE DETECTED: Route only contains the starting location!")
                print("This suggests the route optimizer is dropping all other stops.")
                print("Possible causes:")
                print("1. Disjunction penalty is too low (needs to be much higher)")
                print("2. Time limit is too restrictive for any stops to be included")
                print("3. Travel time calculations might be incorrect")
            
    except json.JSONDecodeError:
        print(f"ERROR: File is not valid JSON: {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to analyze file: {e}")

if __name__ == "__main__":
    # Use file specified as argument or default to latest_route.json
    if len(sys.argv) > 1:
        file_to_check = sys.argv[1]
    else:
        file_to_check = latest_file
    
    diagnose_route_file(file_to_check)
