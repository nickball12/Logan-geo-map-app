import os
import json
import pickle
import logging
from station_model import Station
from station_manager import StationManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = "data"
STATIONS_FILE = os.path.join(DATA_DIR, "stations.pkl")
STATIONS_JSON = os.path.join(DATA_DIR, "stations.json")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.json")

def ensure_data_dir():
    """Ensure the data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")

def save_stations(station_manager):
    """Save the station manager data to disk"""
    ensure_data_dir()
    
    # Save the station data using pickle (preserves Python objects)
    try:
        with open(STATIONS_FILE, 'wb') as f:
            pickle.dump(station_manager.stations, f)
        logger.info(f"Saved {len(station_manager.stations)} stations to {STATIONS_FILE}")
        
        # Also save metadata about the source file
        metadata = {
            'source_file': station_manager.excel_file,
            'station_count': len(station_manager.stations),
            'timestamp': str(station_manager.today)
        }
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {METADATA_FILE}")
        
        # Create a JSON representation for external use (if needed)
        stations_json = []
        for station in station_manager.get_all_stations():
            station_dict = station.__dict__.copy()
            # Convert datetime objects to strings for JSON serialization
            if station_dict.get('last_inspection_date'):
                station_dict['last_inspection_date'] = str(station_dict['last_inspection_date'])
            if station_dict.get('complaint_date'):
                station_dict['complaint_date'] = str(station_dict['complaint_date'])
            stations_json.append(station_dict)
            
        with open(STATIONS_JSON, 'w') as f:
            json.dump(stations_json, f, indent=2)
        logger.info(f"Saved stations JSON to {STATIONS_JSON}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving stations: {str(e)}", exc_info=True)
        return False

def load_stations():
    """Load station data from disk"""
    if not os.path.exists(STATIONS_FILE):
        logger.warning(f"Stations file not found: {STATIONS_FILE}")
        return None
    
    try:
        # Load station data from pickle file
        with open(STATIONS_FILE, 'rb') as f:
            stations_data = pickle.load(f)
            
        # Create a new StationManager and populate it with the loaded data
        station_manager = StationManager()
        station_manager.stations = stations_data
        
        # Load metadata
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
                station_manager.excel_file = metadata.get('source_file')
                
        logger.info(f"Loaded {len(station_manager.stations)} stations from {STATIONS_FILE}")
        return station_manager
    except Exception as e:
        logger.error(f"Error loading stations: {str(e)}", exc_info=True)
        return None

def update_station(business_id, field, value):
    """Update a specific field of a station and save the changes"""
    station_manager = load_stations()
    if not station_manager:
        logger.error("Could not load stations for updating")
        return False
    
    # Get the station by ID
    station = station_manager.get_station_by_id(business_id)
    if not station:
        logger.error(f"Station not found with ID: {business_id}")
        return False
    
    try:
        # Update the field if it exists
        if hasattr(station, field):
            setattr(station, field, value)
            logger.info(f"Updated station {business_id}, field {field} to {value}")
            
            # Save the updated data
            save_stations(station_manager)
            return True
        else:
            logger.error(f"Field {field} not found in station model")
            return False
    except Exception as e:
        logger.error(f"Error updating station: {str(e)}", exc_info=True)
        return False

def get_metadata():
    """Get metadata about the saved station data"""
    if not os.path.exists(METADATA_FILE):
        return None
    
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}", exc_info=True)
        return None

def has_saved_data():
    """Check if saved station data exists"""
    return os.path.exists(STATIONS_FILE)
