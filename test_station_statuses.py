import os
from station_manager import StationManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test loading station data and check for statuses."""
    excel_path = os.path.join('uploads', 'your_station_data1.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return
    
    # Create a station manager and load data
    print("Creating StationManager and loading data...")
    station_manager = StationManager()
    
    # Try to load data from the Excel file
    success = station_manager.load_from_excel(excel_path)
    
    if not success:
        print("Failed to load data from Excel file")
        return
    
    # Get all stations
    all_stations = station_manager.get_all_stations()
    print(f"Loaded {len(all_stations)} stations")
    
    # Count stations with different statuses
    reinspection_count = 0
    complaint_count = 0
    out_of_service_count = 0
    normal_count = 0
    
    for station in all_stations:
        if station.needs_reinspection:
            reinspection_count += 1
        elif station.has_complaint:
            complaint_count += 1
        elif station.out_of_service_pumps > 0:
            out_of_service_count += 1
        else:
            normal_count += 1
    
    print("\nStation Status Summary:")
    print(f"Total Stations: {len(all_stations)}")
    print(f"Stations Needing Reinspection: {reinspection_count}")
    print(f"Stations with Complaints: {complaint_count}")
    print(f"Stations with Out of Service Pumps: {out_of_service_count}")
    print(f"Normal Stations: {normal_count}")
    
    # Check reinspection data specifically
    reinspection_stations = station_manager.get_stations_by_status(reinspection=True)
    if reinspection_stations:
        print(f"\nFound {len(reinspection_stations)} stations that need reinspection:")
        for i, station in enumerate(reinspection_stations[:5], 1):  # Show first 5
            print(f"  {i}. {station.name} (ID: {station.business_id})")
            print(f"     Reason: {station.reinspection_reason or 'Not specified'}")
        
        if len(reinspection_stations) > 5:
            print(f"  ... and {len(reinspection_stations) - 5} more")
    else:
        print("\nNo stations need reinspection.")
    
    # Check complaint data
    complaint_stations = station_manager.get_stations_by_status(complaint=True)
    if complaint_stations:
        print(f"\nFound {len(complaint_stations)} stations with complaints:")
        for i, station in enumerate(complaint_stations[:5], 1):  # Show first 5
            print(f"  {i}. {station.name} (ID: {station.business_id})")
            print(f"     Details: {station.complaint_details or 'Not specified'}")
        
        if len(complaint_stations) > 5:
            print(f"  ... and {len(complaint_stations) - 5} more")
    else:
        print("\nNo stations have complaints.")
    
    # Check out of service data
    oos_stations = station_manager.get_stations_by_status(out_of_service=True)
    if oos_stations:
        print(f"\nFound {len(oos_stations)} stations with out of service pumps:")
        for i, station in enumerate(oos_stations[:5], 1):  # Show first 5
            print(f"  {i}. {station.name} (ID: {station.business_id})")
            print(f"     Pumps OOS: {station.out_of_service_pumps}")
            print(f"     Details: {station.out_of_service_details or 'Not specified'}")
        
        if len(oos_stations) > 5:
            print(f"  ... and {len(oos_stations) - 5} more")
    else:
        print("\nNo stations have out of service pumps.")

if __name__ == "__main__":
    main()
