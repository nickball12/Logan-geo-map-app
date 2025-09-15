import requests
import json
from pprint import pprint
import os

def test_view_saved_route():
    """Test the view_saved_route endpoint"""
    print("Testing /view_saved_route endpoint...")
    
    # Check if the route file exists directly
    output_dir = "output"
    latest_file = os.path.join(output_dir, "latest_route.json")
    
    if os.path.exists(latest_file):
        print(f"Route file exists: {latest_file}")
        print(f"File size: {os.path.getsize(latest_file)} bytes")
        print(f"File permissions: {'Readable' if os.access(latest_file, os.R_OK) else 'Not readable'}")
        
        # Try to read the file directly
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                print(f"File contains {len(data.get('route_plan', []))} stops")
        except Exception as e:
            print(f"Error reading file directly: {e}")
    else:
        print(f"Route file does not exist: {latest_file}")
    
    try:
        # Make request to the endpoint
        response = requests.get('http://localhost:5001/view_saved_route')
        
        # Check response status
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            # Try to parse response as JSON
            try:
                data = response.json()
                
                # Print basic info
                print(f"\nRoute generated at: {data.get('generated_at', 'Unknown')}")
                print(f"Stops: {data.get('stops_count', 0)} of {data.get('total_stops', 0)}")
                print(f"Skipped: {data.get('skipped_percentage', 0)}%")
                
                # Check if route_plan exists and is valid
                route_plan = data.get('route_plan', [])
                if route_plan and isinstance(route_plan, list):
                    print(f"\nRoute plan contains {len(route_plan)} stops")
                    print("First 3 stops:")
                    for i, stop in enumerate(route_plan[:3]):
                        print(f"{i+1}. {stop.get('Name', 'Unknown')} - {stop.get('Full Address', 'No address')}")
                else:
                    print("\nERROR: Invalid or empty route_plan in response")
                    print("route_plan value:", route_plan)
            except json.JSONDecodeError:
                print("\nERROR: Response is not valid JSON")
                print("Raw response:", response.text[:200])  # Show first 200 chars
        else:
            print("\nERROR: Received non-200 response")
            print("Response content:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to server. Is the Flask app running?")
        print("Make sure the Flask app is running on port 5003")
    except Exception as e:
        print(f"\nERROR: {str(e)}")

if __name__ == "__main__":
    test_view_saved_route()
