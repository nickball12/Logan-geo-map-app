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
from flask_session import Session
import tempfile
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from new station model and route planner
from station_model import Station
from station_manager import StationManager
from new_route_planner import plan_optimal_route, STARTING_LOCATION

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key_here'  # Needed for session

# Configure Flask-Session to use filesystem instead of cookies
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = os.path.join(tempfile.gettempdir(), 'flask_session_' + str(uuid.uuid4()))
app.config['SESSION_PERMANENT'] = False
Session(app)  # Initialize Flask-Session

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Global variables for progress tracking
progress_queue = Queue()
progress_lock = Lock()
progress_data = {
    'current': 0,
    'total': 0,
    'address': '',
    'done': False,
    'phase': 'init',
    'phase_name': 'Initializing'
}
progress_timer = None
cancellation_event = Event()

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
        logger.info(f"Progress update: [{progress_data['phase']}] {current}/{total} - {address} {'(Cancelled)' if cancelled else ''}")
        
        # Reset timeout timer
        if progress_timer is not None:
            progress_timer.cancel()
        
        # Start a new timer if not done and not cancelled
        if not done and not cancelled and phase == 'solving':
            # Set a timeout timer for route solving (15 seconds)
            progress_timer = Timer(15.0, check_timeout)
            progress_timer.daemon = True
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
                logger.error(f"Error in progress SSE: {e}")
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
    
    # Reset progress tracking and cancellation event
    global cancellation_event
    cancellation_event.clear()
    update_progress(0, 0, 'Ready to process', False, 'init', 'Ready')
    
    if request.method == 'POST':
        try:
            # Reset cancellation event at the start of processing
            cancellation_event.clear()
            
            status = "Processing uploaded file..."
            
            # Extract form data
            hours_to_work = float(request.form['hours'])
            time_between_stops = int(request.form.get('time_between_stops', 0))
            max_stations = int(request.form.get('max_stations', 0))
            max_pumps_per_day = int(request.form.get('max_pumps_per_day', 20))
            skip_rows = int(request.form.get('skip_rows', 3))
            file = request.files['file']
            
            # Check if we should use cached results
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
            update_progress(1, 5, f"Saved file: {file.filename}", False, 'data', 'Data Preparation')
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join("cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Generate cache filename based on input parameters
            cache_key = f"{file.filename}_{skip_rows}_{sheet_name}"
            cache_file = os.path.join(cache_dir, f"{cache_key.replace('/', '_')}.pkl")
            update_progress(2, 5, "Checking cache", False, 'data', 'Data Preparation')
            
            # Plan the optimal route using new route planner
            update_progress(3, 5, "Planning optimal route", False, 'data', 'Route Planning')
            
            route_plan = plan_optimal_route(
                data_file_path,
                hours=hours_to_work,
                max_stations=max_stations if max_stations > 0 else None,
                max_pumps=max_pumps_per_day,
                time_between_stops=time_between_stops,
                progress_callback=update_progress,
                cancellation_event=cancellation_event
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
    
    return render_template('index.html', error=error, status=status, route_plan=route_plan,
                          time_between_stops=session.get('time_between_stops', 0))

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
            elif isinstance(obj, datetime):
                return obj.isoformat()
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

@app.route('/view_saved_route')
def view_saved_route():
    """View the saved route plan"""
    try:
        # Try to find the latest route file
        output_dir = "output"
        latest_file = os.path.join(output_dir, "latest_route.json")
        
        logger.info(f"Looking for route file at: {latest_file}")
        
        # Check if the output directory exists
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory not found: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            return jsonify({"error": "No saved routes found"}), 404
            
        # Check if the latest route file exists
        if not os.path.exists(latest_file):
            logger.warning(f"Latest route file not found: {latest_file}")
            
            # Try to find any route files
            route_files = [f for f in os.listdir(output_dir) if f.startswith("route_plan_")]
            
            if not route_files:
                return jsonify({"error": "No saved routes found"}), 404
                
            # Use the most recent route file
            route_files.sort(reverse=True)
            latest_file = os.path.join(output_dir, route_files[0])
            logger.info(f"Using alternative route file: {latest_file}")
        
        # Verify file is readable and has proper permissions
        if not os.access(latest_file, os.R_OK):
            logger.error(f"Permission error: Cannot read file {latest_file}")
            return jsonify({"error": f"Permission denied: Cannot read route file"}), 403
        
        # Check file size
        file_size = os.path.getsize(latest_file)
        logger.info(f"Route file size: {file_size} bytes")
        
        if file_size == 0:
            logger.error(f"Route file is empty: {latest_file}")
            return jsonify({"error": "Route file is empty"}), 500
            
        # Read the route file    
        with open(latest_file, 'r') as f:
            try:
                route_data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error in {latest_file}: {e}")
                return jsonify({"error": f"Invalid JSON in route file: {str(e)}"}), 500
            
        # Validate the structure of the route data
        if not isinstance(route_data, dict):
            logger.error(f"Route data is not a dictionary: {type(route_data)}")
            return jsonify({"error": "Invalid route data format"}), 500
            
        if 'route_plan' not in route_data or not isinstance(route_data['route_plan'], list):
            logger.error(f"Route data missing 'route_plan' list")
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
            
        logger.info(f"Successfully loaded route data from {latest_file} with {len(route_data['route_plan'])} stops")
        return jsonify(route_data)
    except Exception as e:
        logger.error(f"Error loading saved route: {e}", exc_info=True)
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
            try:
                # Extract date and time from filename
                date_str = filename.split('_')[2].split('.')[0]
                date_formatted = f"{date_str[0:8]} {date_str[9:11]}:{date_str[11:13]}:{date_str[13:15]}"
                
                # Read the file to get summary information
                with open(os.path.join(output_dir, filename), 'r') as f:
                    route_data = json.load(f)
                    
                    # Calculate basic statistics about the route
                    stops_count = len(route_data['route_plan']) - 1  # Exclude depot
                    
                    # Get the first stop info (depot) which has summary data
                    depot_info = route_data['route_plan'][0]
                    total_time = depot_info.get('total_time', 0)
                    total_pumps = depot_info.get('total_pumps', 0)
                    
                    routes.append({
                        'filename': filename,
                        'date': date_formatted,
                        'generated_at': route_data.get('generated_at', 'Unknown'),
                        'stops_count': stops_count,
                        'total_time': total_time,
                        'total_pumps': total_pumps
                    })
            except Exception as e:
                logger.error(f"Error processing route file {filename}: {e}")
                # Include a minimal entry for this file
                routes.append({
                    'filename': filename,
                    'date': 'Unknown',
                    'error': str(e)
                })
                
        return jsonify({"routes": routes})
    except Exception as e:
        logger.error(f"Error listing saved routes: {e}")
        return jsonify({"error": str(e), "routes": []})

@app.route('/schedule_view')
def schedule_view():
    """View the schedule as a calendar"""
    try:
        # Load the latest route
        output_dir = "output"
        latest_file = os.path.join(output_dir, "latest_route.json")
        
        if not os.path.exists(latest_file):
            return render_template('schedule.html', error="No route plan found", route_plan=None)
        
        with open(latest_file, 'r') as f:
            route_data = json.load(f)
        
        # Pass the route plan to the template
        return render_template('schedule.html', route_plan=route_data['route_plan'])
    except Exception as e:
        logger.error(f"Error viewing schedule: {e}", exc_info=True)
        return render_template('schedule.html', error=str(e), route_plan=None)

if __name__ == "__main__":
    from datetime import datetime
    app.run(debug=True, port=5001)
