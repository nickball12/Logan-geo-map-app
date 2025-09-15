from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import pandas as pd
import numpy as np
import os
import traceback
import json
import time
import pickle
from werkzeug.utils import secure_filename
import threading
from queue import Queue
import concurrent.futures
from threading import Lock, Timer, Event
from flask_session import Session  # Import Flask-Session
import tempfile
import uuid
import logging

import atexit

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from new station model and route planner
from station_model import Station
from station_manager import StationManager
from new_route_planner import plan_optimal_route, STARTING_LOCATION
from data_store import save_stations, load_stations, update_station, get_metadata, has_saved_data
from route_helpers import plan_route_from_stations

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'  # Add a data folder for saved station data
app.secret_key = 'your_secret_key_here'  # Needed for session

# Configure Flask-Session to use filesystem instead of cookies
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_session_' + str(uuid.uuid4()))
app.config['SESSION_PERMANENT'] = False
Session(app)  # Initialize Flask-Session

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATA_FOLDER'], exist_ok=True)  # Ensure data directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)  # Ensure session directory exists

# Function to clean up resources at application exit
def cleanup_resources():
    global progress_timer
    logger.info("Performing application cleanup...")
    if progress_timer is not None:
        logger.info("Cancelling timer thread")
        progress_timer.cancel()
        
    # Clean up any other resources here
    logger.info("Cleanup complete")

# Register the cleanup function to run at exit
atexit.register(cleanup_resources)

# Global variables for progress tracking
progress_queue = Queue()
# Add a lock for thread-safe progress updates
progress_lock = Lock()
progress_data = {
    'current': 0,
    'total': 0,
    'address': '',
    'done': False,
    'phase': 'init',  # Added phase tracking: 'init', 'data', 'geocoding', 'routing'
    'phase_name': 'Initializing'  # Human-readable phase name
}
progress_timer = None  # Timer for detecting stalled operations
cancellation_event = Event()  # Event to signal cancellation

# Function to check for stalled operations
def check_timeout():
    global progress_data, progress_queue
    with progress_lock:
        if not progress_data.get('done', False):
            timeout_data = progress_data.copy()
            timeout_data['address'] = f"{timeout_data['address']} (Operation taking longer than expected, but still working...)"
            progress_queue.put(timeout_data)

def update_progress(current, total, address='', done=False, phase=None, phase_name=None, cancelled=False):
    """Update progress tracking information"""
    global progress_data, progress_timer
    with progress_lock:
        # Create a new dict for the updated progress data
        updated_data = {
            'current': current,
            'total': total,
            'address': address,
            'done': done,
            'cancelled': cancelled
        }
        
        # Update phase information if provided
        if phase is not None:
            updated_data['phase'] = phase
            updated_data['phase_name'] = phase_name or phase.capitalize()
        
        # Update the global progress data
        progress_data.update(updated_data)
        # Put a copy of the data in the queue to avoid reference issues
        progress_queue.put(updated_data.copy())
        print(f"Progress update: [{progress_data['phase']}] {current}/{total} - {address} {'(Cancelled)' if cancelled else ''}")
        
        # Reset timeout timer
        if progress_timer is not None:
            progress_timer.cancel()
        
        # Start a new timer if not done and not cancelled
        if not done and not cancelled and phase == 'solving':
            # Set a timeout timer for route solving (15 seconds)
            progress_timer = Timer(15.0, check_timeout)
            # Don't use daemon threads to avoid shutdown issues
            progress_timer.daemon = False
            progress_timer.start()

@app.route('/progress')
def progress():
    """Server-sent events endpoint for progress updates"""
    def generate():
        last_data = None
        while True:
            try:
                # Check if process was cancelled
                if cancellation_event.is_set():
                    cancel_data = {
                        'phase_name': 'Cancelled',
                        'progress': 'Operation cancelled by user',
                        'cancelled': True
                    }
                    yield f"data: {json.dumps(cancel_data)}\n\n"
                    break
                
                # Try to get the latest data from the queue with a timeout
                try:
                    latest_data = progress_queue.get(timeout=0.5)
                    # If we got new data, update it and send it
                    yield f"data: {json.dumps(latest_data)}\n\n"
                    last_data = latest_data
                    # If the process is done or cancelled, break out of the loop
                    if latest_data.get('done', False) or latest_data.get('cancelled', False):
                        break
                except:
                    # If the queue is empty, send the current progress data
                    # Only send if it's different from the last sent data
                    current_data = progress_data.copy()
                    if current_data != last_data:
                        yield f"data: {json.dumps(current_data)}\n\n"
                        last_data = current_data
                    time.sleep(0.5)
            except GeneratorExit:
                # Client disconnected
                break
            except Exception as e:
                print(f"Error in progress SSE: {e}")
                time.sleep(1)
                
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    })

@app.route('/cancel', methods=['POST'])
def cancel_processing():
    """Endpoint to handle cancellation requests"""
    global cancellation_event
    # Set the cancellation event
    cancellation_event.set()
    # Update progress to indicate cancellation
    update_progress(
        progress_data.get('current', 0),
        progress_data.get('total', 0),
        'Operation cancelled by user',
        True,
        'cancelled',
        'Cancelled',
        True
    )
    return jsonify({'status': 'cancelling', 'message': 'Cancellation request received'})

@app.route('/', methods=['GET', 'POST'])
def index():
    error = session.pop('error', None)
    status = session.pop('status', None)
    route_plan = session.pop('route_plan', None)
    
    # Check if we have saved station data
    saved_data_exists = has_saved_data()
    
    # Reset progress tracking and cancellation event
    global cancellation_event
    cancellation_event.clear()
    update_progress(0, 0, 'Ready to process', False, 'init', 'Ready')
    
    if request.method == 'POST':
        try:
            # Reset cancellation event at the start of processing
            cancellation_event.clear()
            
            # Extract form data
            hours_to_work = float(request.form['hours'])
            time_between_stops = int(request.form.get('time_between_stops', 0))
            max_stations = int(request.form.get('max_stations', 0))
            max_pumps_per_day = int(request.form.get('max_pumps_per_day', 20))
            skip_rows = int(request.form.get('skip_rows', 3))
            
            # Check if we should use cached results
            use_cache = 'use_cache' in request.form
            
            # Check if we should use saved data
            use_saved_data = 'use_saved_data' in request.form
            
            if use_saved_data and has_saved_data():
                status = "Using saved station data..."
                
                # Load station data from saved file
                station_manager = load_stations()
                
                if not station_manager or not station_manager.stations:
                    error = "Failed to load saved station data"
                    return redirect(url_for('index'))
                
                # Get all stations
                all_stations = station_manager.get_all_stations()
                
                # Check status filters
                include_statuses = []
                if 'include_normal' in request.form:
                    include_statuses.append('normal')
                if 'include_complaints' in request.form:
                    include_statuses.append('complaint')
                if 'include_reinspections' in request.form:
                    include_statuses.append('reinspection')
                if 'include_out_of_service' in request.form:
                    include_statuses.append('out_of_service')
                
                # If no status filters selected, include all
                if not include_statuses:
                    include_statuses = ['normal', 'complaint', 'reinspection', 'out_of_service']
                
                # Plan route using saved stations
                update_progress(0, 5, "Planning route from saved data", False, 'data', 'Route Planning')
                
                # We need to create a function to plan routes from stations
                route_plan = plan_route_from_stations(
                    all_stations,
                    hours=hours_to_work,
                    max_stations=max_stations if max_stations > 0 else None,
                    max_pumps=max_pumps_per_day,
                    time_between_stops=time_between_stops,
                    progress_callback=update_progress,
                    cancellation_event=cancellation_event,
                    use_cache=use_cache,
                    include_statuses=include_statuses
                )
                
                # Check if operation was cancelled
                if cancellation_event.is_set():
                    # Operation was cancelled
                    error = "Operation cancelled by user"
                    session['error'] = error
                    session['status'] = status
                    session['route_plan'] = None
                    return redirect(url_for('index'))
                    
                if route_plan:
                    status += f" Route generated with {len(route_plan) - 1} stops."
                    
                    # Save route plan to a file for easy access
                    save_route_to_file(route_plan)
                    
                    update_progress(5, 5, 
                                  f"Route with {len(route_plan) - 1} stops generated successfully", 
                                  True, 'solving', 'Route Optimization Complete')
                else:
                    error = "Failed to generate any route within the time constraint. Please try increasing work hours."
                    update_progress(5, 5, "Route optimization failed. Please try increasing work hours.", 
                                  True, 'solving', 'Route Optimization Failed')
                
                # Store results in session for redirect
                session['error'] = error
                session['status'] = status
                session['route_plan'] = route_plan if route_plan else None
                session['time_between_stops'] = time_between_stops
                return redirect(url_for('index'))
            else:
                # Process uploaded file
                status = "Processing uploaded file..."
                
                # Check if a file was uploaded
                if 'file' not in request.files:
                    error = "No file uploaded. Please upload an Excel file."
                    return redirect(url_for('index'))
                    
                file = request.files['file']
                
                # Get sheet name if provided, otherwise use the first sheet
                sheet_name = request.form.get('sheet_name', '0')
                if sheet_name.isdigit():
                    sheet_name = int(sheet_name)
            use_cache = 'use_cache' in request.form
            
            # Get sheet name if provided, otherwise use the first sheet
            sheet_name = request.form.get('sheet_name', '0')
            if sheet_name.isdigit():
                sheet_name = int(sheet_name)
            
            # Update progress for file processing phase
            update_progress(0, 5, "Preparing to process file", False, 'data', 'Data Preparation')
            
            # Save the uploaded file
            data_file_path = os.path.join("uploads", file.filename)
            os.makedirs("uploads", exist_ok=True)
            file.save(data_file_path)
            
            # Store the file path in the session for later use
            session['latest_file'] = data_file_path
            
            update_progress(1, 5, f"Saved file: {file.filename}", False, 'data', 'Data Preparation')
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join("cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate cache filename based on input parameters
            cache_key = f"{file.filename}_{skip_rows}_{sheet_name}"
            cache_file = os.path.join(cache_dir, f"{cache_key.replace('/', '_')}.pkl")
            update_progress(2, 5, "Checking cache", False, 'data', 'Data Preparation')
            
            # Check status filters
            include_statuses = []
            if 'include_normal' in request.form:
                include_statuses.append('normal')
            if 'include_complaints' in request.form:
                include_statuses.append('complaint')
            if 'include_reinspections' in request.form:
                include_statuses.append('reinspection')
            if 'include_out_of_service' in request.form:
                include_statuses.append('out_of_service')
            
            # If no status filters selected, include all
            if not include_statuses:
                include_statuses = ['normal', 'complaint', 'reinspection', 'out_of_service']
            
            # Plan the optimal route using new route planner
            update_progress(3, 5, "Planning optimal route", False, 'data', 'Route Planning')
            
            route_plan = plan_optimal_route(
                data_file_path,
                hours=hours_to_work,
                max_stations=max_stations if max_stations > 0 else None,
                max_pumps=max_pumps_per_day,
                time_between_stops=time_between_stops,
                progress_callback=update_progress,
                cancellation_event=cancellation_event,
                use_cache=use_cache,
                include_statuses=include_statuses
            )
            
            # Check if operation was cancelled
            if cancellation_event.is_set():
                # Operation was cancelled
                error = "Operation cancelled by user"
                session['error'] = error
                session['status'] = status
                session['route_plan'] = None
                return redirect(url_for('index'))
                
            if route_plan:
                status += f" Route generated with {len(route_plan) - 1} stops."
                
                # Save route plan to a file for easy access
                save_route_to_file(route_plan)
                
                update_progress(5, 5, 
                              f"Route with {len(route_plan) - 1} stops generated successfully", 
                              True, 'solving', 'Route Optimization Complete')
            else:
                error = "Failed to generate any route within the time constraint. Please try increasing work hours."
                update_progress(5, 5, "Route optimization failed. Please try increasing work hours.", 
                              True, 'solving', 'Route Optimization Failed')
            
            # Store results in session for redirect
            session['error'] = error
            session['status'] = status
            session['route_plan'] = route_plan if route_plan else None
            session['time_between_stops'] = time_between_stops
            return redirect(url_for('index'))
            
        except Exception as e:
            # Check if this was a cancellation
            if cancellation_event.is_set() or "Operation cancelled" in str(e):
                error = "Operation cancelled by user"
                status = "Processing cancelled"
            else:
                error = f"Error processing data: {str(e)}"
                logger.error(f"Exception: {e}", exc_info=True)
            
            session['error'] = error
            session['status'] = status
            session['route_plan'] = None
            return redirect(url_for('index'))
    
    return render_template('new_index.html', 
                          error=error, 
                          status=status, 
                          route_plan=route_plan,
                          time_between_stops=session.get('time_between_stops', 0),
                          saved_data_exists=saved_data_exists)

@app.route('/addresses')
def addresses():
    """Show all extracted addresses with duplicate detection"""
    error = request.args.get('error')
    status = request.args.get('status')
    
    # First check if we have saved data
    if has_saved_data():
        try:
            # Load station data from saved file
            station_manager = load_stations()
            
            if station_manager and station_manager.stations:
                # Get all stations
                all_stations = station_manager.get_all_stations()
                
                # Detect duplicate addresses
                address_count = {}
                for station in all_stations:
                    # Use the full address as the key for duplicate detection
                    if station.full_address in address_count:
                        address_count[station.full_address] += 1
                    else:
                        address_count[station.full_address] = 1
                
                # Mark duplicates in the station objects
                for station in all_stations:
                    station.is_duplicate = address_count[station.full_address] > 1
                
                # Calculate statistics
                unique_count = sum(1 for count in address_count.values() if count == 1)
                duplicate_count = sum(1 for count in address_count.values() if count > 1)
                
                return render_template('addresses.html', 
                                      addresses=all_stations, 
                                      unique_count=unique_count, 
                                      duplicate_count=duplicate_count,
                                      error=error,
                                      status=status,
                                      editable=True)
            else:
                # No valid saved data, try loading from file
                pass
        except Exception as e:
            logger.error(f"Error loading saved data: {e}", exc_info=True)
            error = f"Error loading saved data: {str(e)}"
    
    # If no saved data or error loading saved data, check for uploaded file
    if not session.get('latest_file'):
        return render_template('addresses.html', 
                              addresses=[], 
                              unique_count=0, 
                              duplicate_count=0,
                              error=error or "No data available. Please upload a file first.",
                              editable=False)
    
    try:
        # Load station data from the most recently processed file
        station_manager = StationManager()
        file_path = session.get('latest_file')
        
        if not os.path.exists(file_path):
            return render_template('addresses.html', 
                                  addresses=[], 
                                  unique_count=0, 
                                  duplicate_count=0,
                                  error=error or f"File not found: {file_path}",
                                  editable=False)
        
        # Load all stations, not just high priority ones
        if not station_manager.load_from_excel(file_path):
            return render_template('addresses.html', 
                                  addresses=[], 
                                  unique_count=0, 
                                  duplicate_count=0,
                                  error=error or "Failed to load data from the Excel file",
                                  editable=False)
        
        # Get all stations
        all_stations = station_manager.get_all_stations()
        
        # Save stations to persistent storage
        save_stations(station_manager)
        
        # Detect duplicate addresses
        address_count = {}
        for station in all_stations:
            # Use the full address as the key for duplicate detection
            if station.full_address in address_count:
                address_count[station.full_address] += 1
            else:
                address_count[station.full_address] = 1
        
        # Mark duplicates in the station objects
        for station in all_stations:
            station.is_duplicate = address_count[station.full_address] > 1
        
        # Calculate statistics
        unique_count = sum(1 for count in address_count.values() if count == 1)
        duplicate_count = sum(1 for count in address_count.values() if count > 1)
        
        return render_template('addresses.html', 
                              addresses=all_stations, 
                              unique_count=unique_count, 
                              duplicate_count=duplicate_count,
                              error=error,
                              status=status,
                              editable=True)
    
    except Exception as e:
        logger.error(f"Error displaying addresses: {e}", exc_info=True)
        return render_template('addresses.html', 
                              addresses=[], 
                              unique_count=0, 
                              duplicate_count=0,
                              error=error or f"Error: {str(e)}",
                              editable=False)

@app.route('/refresh_status')
def refresh_status():
    """Refresh station status from all sheets in the Excel file"""
    # Check if a file has been uploaded and processed
    if not session.get('latest_file'):
        return redirect(url_for('addresses', error="No data available. Please upload a file first."))
    
    try:
        file_path = session.get('latest_file')
        
        if not os.path.exists(file_path):
            return redirect(url_for('addresses', error=f"File not found: {file_path}"))
        
        # Create a new station manager to load all sheets explicitly
        station_manager = StationManager()
        
        # Explicitly load the main sheet first
        if not station_manager.load_from_excel(file_path):
            return redirect(url_for('addresses', error="Failed to load data from the Excel file"))
        
        # Now explicitly load additional sheets
        excel = pd.ExcelFile(file_path)
        sheet_names = excel.sheet_names
        
        # Track which sheets were processed
        processed_sheets = []
        
        # Process reinspection sheet if available
        if 'Reinspections' in sheet_names:
            station_manager._load_reinspection_data(excel, 'Reinspections', 3)
            processed_sheets.append('Reinspections')
        
        # Process complaints sheet if available
        if 'Complaints' in sheet_names:
            station_manager._load_complaint_data(excel, 'Complaints', 3)
            processed_sheets.append('Complaints')
        
        # Process out of service sheet if available
        if 'Out of Service Pumps' in sheet_names:
            station_manager._load_out_of_service_data(excel, 'Out of Service Pumps', 3)
            processed_sheets.append('Out of Service Pumps')
        
        # Recalculate priority scores
        for station in station_manager.stations.values():
            station.calculate_priority()
            
        # Save the updated data
        save_stations(station_manager)
        
        # Redirect back to addresses page with success message
        return redirect(url_for('addresses', status=f"Refreshed status from {len(processed_sheets)} sheets"))
        for station in all_stations:
            if station.full_address in address_count:
                address_count[station.full_address] += 1
            else:
                address_count[station.full_address] = 1
        
        # Mark duplicates
        for station in all_stations:
            station.is_duplicate = address_count[station.full_address] > 1
        
        # Calculate statistics
        unique_count = sum(1 for count in address_count.values() if count == 1)
        duplicate_count = sum(1 for count in address_count.values() if count > 1)
        
        # Count status indicators
        reinspection_count = sum(1 for s in all_stations if s.needs_reinspection)
        complaint_count = sum(1 for s in all_stations if s.has_complaint)
        out_of_service_count = sum(1 for s in all_stations if s.out_of_service_pumps > 0)
        
        success_message = f"Status refreshed from sheets: {', '.join(processed_sheets)}. "
        status_message = f"Found {reinspection_count} reinspections, {complaint_count} complaints, and {out_of_service_count} out of service stations."
        
        return render_template('addresses.html', 
                              addresses=all_stations, 
                              unique_count=unique_count, 
                              duplicate_count=duplicate_count,
                              success=success_message,
                              status_message=status_message)
    
    except Exception as e:
        logger.error(f"Error refreshing status: {e}", exc_info=True)
        return redirect(url_for('addresses', error=f"Error refreshing status: {str(e)}"))

def save_route_to_file(route_plan):
    """Save the route plan to a JSON file for easy access"""
    if not route_plan:
        logger.warning("save_route_to_file called with empty route_plan")
        return
    
    logger.info(f"Saving route plan with {len(route_plan) - 1} stops")
    
    try:
        # Create output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Format timestamp for filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(output_dir, f"route_plan_{timestamp}.json")
        
        # Prepare data for saving
        # Convert complex objects to JSON-serializable formats
        def clean_for_json(obj):
            if isinstance(obj, pd.Series):
                return {k: clean_for_json(v) for k, v in obj.to_dict().items()}
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (dict)):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [clean_for_json(i) for i in obj]
            else:
                return obj
                
        # Clean route plan data for JSON serialization
        clean_route_plan = [clean_for_json(stop) for stop in route_plan]
        
        # Create the data structure to save
        route_data = {
            "timestamp": timestamp,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "route_plan": clean_route_plan
        }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(route_data, f, indent=2)
            
        # Also save as latest.json for easy access
        latest_file = os.path.join(output_dir, "latest_route.json")
        with open(latest_file, 'w') as f:
            json.dump(route_data, f, indent=2)
            
        logger.info(f"Route saved to {output_file} and {latest_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error saving route to file: {e}", exc_info=True)
        return None
        return output_file
    except Exception as e:
        print(f"Error saving route to file: {e}")
        traceback.print_exc()
        return None

@app.route('/view_saved_route')
def view_saved_route():
    """View the saved route plan"""
    try:
        # Try to find the latest route file
        output_dir = "output"
        latest_file = os.path.join(output_dir, "latest_route.json")
        
        print(f"Looking for route file at: {latest_file}")
        
        # Check if the output directory exists
        if not os.path.exists(output_dir):
            print(f"Output directory not found: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            return jsonify({"error": "No saved routes found"}), 404
            
        # Check if the latest route file exists
        if not os.path.exists(latest_file):
            print(f"Latest route file not found: {latest_file}")
            
            # Try to find any route files
            route_files = [f for f in os.listdir(output_dir) if f.startswith("route_plan_")]
            
            if not route_files:
                return jsonify({"error": "No saved routes found"}), 404
                
            # Use the most recent route file
            route_files.sort(reverse=True)
            latest_file = os.path.join(output_dir, route_files[0])
            print(f"Using alternative route file: {latest_file}")
        
        # Verify file is readable and has proper permissions
        if not os.access(latest_file, os.R_OK):
            print(f"Permission error: Cannot read file {latest_file}")
            return jsonify({"error": f"Permission denied: Cannot read route file"}), 403
        
        # Check file size
        file_size = os.path.getsize(latest_file)
        print(f"Route file size: {file_size} bytes")
        
        if file_size == 0:
            print(f"Route file is empty: {latest_file}")
            return jsonify({"error": "Route file is empty"}), 500
            
        # Read the route file    
        with open(latest_file, 'r') as f:
            try:
                route_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSON parse error in {latest_file}: {e}")
                return jsonify({"error": f"Invalid JSON in route file: {str(e)}"}), 500
            
        # Validate the structure of the route data
        if not isinstance(route_data, dict):
            print(f"Route data is not a dictionary: {type(route_data)}")
            return jsonify({"error": "Invalid route data format"}), 500
            
        if 'route_plan' not in route_data or not isinstance(route_data['route_plan'], list):
            print(f"Route data missing 'route_plan' list")
            return jsonify({"error": "Invalid route data: missing route_plan list"}), 500
            
        # Convert any NaN values to null for valid JSON
        import math
        def replace_nan_with_null(obj):
            if isinstance(obj, dict):
                return {k: replace_nan_with_null(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan_with_null(item) for item in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            else:
                return obj
                
        # Apply the NaN replacement to the route data
        route_data = replace_nan_with_null(route_data)
            
        print(f"Successfully loaded route data from {latest_file} with {len(route_data['route_plan'])} stops")
        return jsonify(route_data)
    except Exception as e:
        print(f"Error loading saved route: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error loading saved route: {str(e)}"}), 500

@app.route('/list_saved_routes')
def list_saved_routes():
    """List all saved route plans"""
    try:
        output_dir = "output"
        if not os.path.exists(output_dir):
            return jsonify({"routes": []})
            
        # Find all route plan files
        route_files = [f for f in os.listdir(output_dir) if f.startswith("route_plan_")]
        
        # Sort by date (newest first)
        route_files.sort(reverse=True)
        
        # Create a list of route metadata
        routes = []
        for filename in route_files:
            # Extract date and time from filename
            try:
                date_str = filename.split('_')[2].split('.')[0]
                date_formatted = f"{date_str[0:8]} {date_str[9:11]}:{date_str[11:13]}:{date_str[13:15]}"
            except:
                date_formatted = "Unknown date"
                
            routes.append({
                "filename": filename,
                "created_at": date_formatted,
                "display_name": f"Route Plan - {date_formatted}"
            })
            
        return jsonify({"routes": routes})
    except Exception as e:
        print(f"Error listing saved routes: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/load_saved_route/<filename>')
def load_saved_route(filename):
    """Load a specific saved route plan"""
    try:
        output_dir = "output"
        file_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": f"Route file not found: {filename}"}), 404
            
        # Verify file is readable
        if not os.access(file_path, os.R_OK):
            return jsonify({"error": f"Permission denied: Cannot read route file"}), 403
            
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return jsonify({"error": "Route file is empty"}), 500
            
        # Read the route file
        with open(file_path, 'r') as f:
            try:
                route_data = json.load(f)
            except json.JSONDecodeError as e:
                return jsonify({"error": f"Invalid JSON in route file: {str(e)}"}), 500
                
        # Validate the structure of the route data
        if not isinstance(route_data, dict):
            return jsonify({"error": "Invalid route data format"}), 500
            
        if 'route_plan' not in route_data or not isinstance(route_data['route_plan'], list):
            return jsonify({"error": "Invalid route data: missing route_plan list"}), 500
            
        # Convert any NaN values to null for valid JSON
        import math
        def replace_nan_with_null(obj):
            if isinstance(obj, dict):
                return {k: replace_nan_with_null(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_nan_with_null(item) for item in obj]
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            else:
                return obj
                
        # Apply the NaN replacement to the route data
        route_data = replace_nan_with_null(route_data)
            
        print(f"Successfully loaded route data from {file_path} with {len(route_data['route_plan'])} stops")
        return jsonify(route_data)
    except Exception as e:
        print(f"Error loading saved route: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error loading saved route: {str(e)}"}), 500

def read_and_process_address_data_with_progress(file_name, address_col, city_col, skip_rows, sheet_name=0, 
                                     zip_col=None, state_col=None, last_visited_col=None, 
                                     pumps_col=None, pumps_per_station=2):
    """Wrapper around read_and_process_address_data that updates progress"""
    from route_planner import excel_column_to_index, STARTING_LOCATION, get_coordinates_local
    
    # Use the original function to read the data from file
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
            
            print(f"Reading Excel file with pandas. Address index: {address_col_index}, City index: {city_col_index}")
            print(f"ZIP index: {zip_col_index}, State index: {state_col_index}")
            
            # Read the Excel file with specified sheet
            df = pd.read_excel(file_name, sheet_name=sheet_name, skiprows=skip_rows, header=0, engine='openpyxl')
            print(f"Excel file read successfully. Columns: {list(df.columns)}")
            
            # Clean the dataframe - remove completely blank rows and rows with only whitespace
            df = df.replace(r'^\s*$', np.nan, regex=True)  # Replace whitespace-only strings with NaN
            df = df.dropna(how='all')  # Drop rows where all columns are NaN
            print(f"Removed blank rows. Remaining rows: {len(df)}")
            
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
            
            df = pd.read_csv(file_name, skiprows=skip_rows, header=0)
            print(f"CSV file read successfully. Columns: {list(df.columns)}")
            
            # Clean the dataframe - remove completely blank rows and rows with only whitespace
            # Check if all columns in a row are either NaN or just whitespace strings
            df = df.replace(r'^\s*$', np.nan, regex=True)  # Replace whitespace-only strings with NaN
            df = df.dropna(how='all')  # Drop rows where all columns are NaN
            print(f"Removed blank rows. Remaining rows: {len(df)}")
            
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
        
        # Clean the data: replace empty strings with NaN and remove rows with missing address or city
        station_data = station_data.replace(r'^\s*$', np.nan, regex=True)
        station_data = station_data.dropna(subset=[address_col_name, city_col_name])
        print(f"After removing rows with missing address/city: {len(station_data)} rows remaining")
        
        # Build full address string with state and zip if available
        if state_col_index is not None and state_col_index < len(df.columns):
            state_col_name = df.columns[state_col_index]
            station_data['State'] = df[state_col_name]
        else:
            # Default to Utah if no state column
            station_data['State'] = 'UT'
            
        if zip_col_index is not None and zip_col_index < len(df.columns):
            zip_col_name = df.columns[zip_col_index]
            
            # Get zip codes and convert to string format without decimal
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
        
        # Process last visited date if column is provided
        if last_visited_col and last_visited_col.strip():
            try:
                # Convert Excel column letter to index if needed
                last_visited_col_index = None
                if isinstance(last_visited_col, str) and last_visited_col.isalpha():
                    last_visited_col_index = excel_column_to_index(last_visited_col)
                else:
                    try:
                        last_visited_col_index = int(last_visited_col)
                    except ValueError:
                        print(f"Invalid last visited column format: {last_visited_col}. Ignoring.")
                
                if last_visited_col_index is not None and last_visited_col_index < len(df.columns):
                    last_visited_col_name = df.columns[last_visited_col_index]
                    station_data['Last Visited'] = df[last_visited_col_name]
                    print(f"Added last visited dates from column: {last_visited_col_name}")
            except Exception as e:
                print(f"Error processing last visited dates: {e}")
        
        # Process pump data if column is provided
        if pumps_col and pumps_col.strip():
            try:
                # Convert Excel column letter to index if needed
                pumps_col_index = None
                if isinstance(pumps_col, str) and pumps_col.isalpha():
                    pumps_col_index = excel_column_to_index(pumps_col)
                else:
                    try:
                        pumps_col_index = int(pumps_col)
                    except ValueError:
                        print(f"Invalid pumps column format: {pumps_col}. Ignoring.")
                
                if pumps_col_index is not None and pumps_col_index < len(df.columns):
                    pumps_col_name = df.columns[pumps_col_index]
                    # Add total pumps column
                    station_data['Total Pumps'] = df[pumps_col_name]
                    # Calculate pumps to inspect based on the specified number per station
                    station_data['Pumps to Inspect'] = station_data['Total Pumps'].apply(
                        lambda x: min(pumps_per_station, x) if pd.notnull(x) else 0
                    )
                    print(f"Added pump data from column: {pumps_col_name}, inspecting {pumps_per_station} per station")
            except Exception as e:
                print(f"Error processing pump data: {e}")

        # Add starting location to the dataframe
        start_df = pd.DataFrame([STARTING_LOCATION])
        station_data = pd.concat([start_df, station_data], ignore_index=True)
        
        # Process and geocode addresses with progress updates
        total_addresses = len(station_data)
        update_progress(0, total_addresses, "Starting geocoding", False, 'geocoding', 'Geocoding Addresses')
        
        # Create a new column for coordinates
        station_data['Coordinates'] = None
        
        # Handle the first row (starting location) separately
        if 'Coordinates' in STARTING_LOCATION:
            station_data.at[0, 'Coordinates'] = STARTING_LOCATION['Coordinates']
        
        # Function to process a single address with geocoding
        def process_address(idx, address):
            # Check for cancellation
            if cancellation_event.is_set():
                return idx, None
                
            # Skip the first row (starting location) since we already handled it
            if idx == 0:
                return idx, STARTING_LOCATION.get('Coordinates')
                
            # Geocode the address
            coordinates = get_coordinates_local(address)
            
            # Update progress directly - don't rely on progress_data for counting
            current = idx  # Use the index as the progress counter
            # Update progress with geocoding phase
            update_progress(current, total_addresses-1, address, False, 'geocoding', 'Geocoding Addresses')
            
            # Small delay to prevent rate limiting
            time.sleep(0.1)
            
            return idx, coordinates
        
        # Use ThreadPoolExecutor for parallel processing
        addresses_to_process = [(i, row['Full Address']) for i, row in station_data.iterrows() if i > 0]
        processed_results = []
        
        # Set the number of worker threads based on system capabilities
        # Using too many can trigger rate limiting on geocoding services
        max_workers = min(5, len(addresses_to_process))  # Reduce to 5 concurrent threads for better progress tracking
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all addresses for processing
            future_to_address = {
                executor.submit(process_address, idx, address): (idx, address) 
                for idx, address in addresses_to_process
            }
            
            # Process results as they complete
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_address):
                # Check for cancellation
                if cancellation_event.is_set():
                    # Cancel all pending futures
                    for f in future_to_address:
                        f.cancel()
                    # Use wait=True to ensure threads complete before shutdown
                    executor.shutdown(wait=True)
                    update_progress(completed_count, len(addresses_to_process), 
                                  "Operation cancelled", True, 'geocoding', 'Geocoding Cancelled', True)
                    raise Exception("Operation cancelled by user")
                
                try:
                    idx, coordinates = future.result()
                    if coordinates:  # Skip cancelled results
                        station_data.at[idx, 'Coordinates'] = coordinates
                        processed_results.append((idx, coordinates))
                    
                    # Update progress counter
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == len(addresses_to_process):
                        # Send periodic updates to avoid overwhelming the event stream
                        update_progress(completed_count, len(addresses_to_process), 
                                      f"Processed {completed_count} of {len(addresses_to_process)} addresses", 
                                      completed_count == len(addresses_to_process))
                except Exception as e:
                    print(f"Error processing address: {e}")
        
        # Final update to mark completion
        update_progress(len(addresses_to_process), len(addresses_to_process), "Geocoding complete", True, 'geocoding', 'Geocoding Complete')
        
        return station_data.drop_duplicates(subset=['Full Address']).reset_index(drop=True)

    except Exception as e:
        print(f"Error in read_and_process_address_data_with_progress: {e}")
        traceback.print_exc()
        update_progress(0, 0, f"Error: {str(e)}", True, 'error', 'Error Occurred')
        raise

@app.route('/update_station', methods=['POST'])
def update_station_route():
    """Update a station field"""
    try:
        data = request.json
        business_id = data.get('business_id')
        field = data.get('field')
        value = data.get('value')
        
        if not business_id or not field:
            return jsonify({'success': False, 'error': 'Missing business_id or field'})
        
        # Update the station in the data store
        success = update_station(business_id, field, value)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to update station'})
    
    except Exception as e:
        logger.error(f"Error updating station: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})

@app.route('/reset_app', methods=['GET', 'POST'])
def reset_app():
    """Reset the application data (clear uploads, cache, and saved data)"""
    try:
        # Clear session data
        session.clear()
        
        # Get confirmation from form if POST method
        confirmed = request.form.get('confirm') if request.method == 'POST' else None
        
        if request.method == 'GET' or not confirmed:
            # Show confirmation page
            return render_template('reset_confirmation.html')
        
        # Clear uploaded files
        upload_dir = app.config['UPLOAD_FOLDER']
        for file in os.listdir(upload_dir):
            if file != '.gitkeep':  # Keep .gitkeep to preserve the directory in git
                file_path = os.path.join(upload_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Deleted uploaded file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        # Clear geocoding cache
        cache_dir = 'cache'
        for file in os.listdir(cache_dir):
            if file != '.gitkeep':
                file_path = os.path.join(cache_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Deleted cache file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        # Clear saved routes in output directory
        output_dir = 'output'
        for file in os.listdir(output_dir):
            if file != '.gitkeep':
                file_path = os.path.join(output_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Deleted route file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        # Clear saved station data
        data_dir = app.config['DATA_FOLDER']
        for file in os.listdir(data_dir):
            if file != '.gitkeep':
                file_path = os.path.join(data_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                        logger.info(f"Deleted data file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting {file_path}: {e}")
        
        # Reset geocode cache in memory
        import new_route_planner
        new_route_planner.geocode_cache = {}
        
        return redirect(url_for('index', status="Application data has been reset successfully"))
        
    except Exception as e:
        logger.error(f"Error resetting app: {e}", exc_info=True)
        return redirect(url_for('index', error=f"Error resetting application data: {str(e)}"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
