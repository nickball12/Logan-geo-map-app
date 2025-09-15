# Station Route Planner

A Flask web application for optimizing routes for station inspections.

## New Object-Oriented Architecture

The application has been restructured to use an object-oriented approach, which provides a more efficient workflow and better correlates data across different Excel sheets.

### Key Components

1. **Station Model** (`station_model.py`)
   - Core data structure representing a station with all relevant properties
   - Handles properties like pumps, reinspections, complaints, and out-of-service flags
   - Contains geocoding information for location-based routing

2. **Station Manager** (`station_manager.py`)
   - Manages the collection of station objects
   - Handles loading data from Excel sheets
   - Correlates information across sheets (Main, Reinspection, Complaints, Out-of-Service)

3. **Route Planner** (`new_route_planner.py`)
   - Optimizes routes based on station properties and constraints
   - Uses OrTools for the optimization algorithm
   - Handles time windows, pump limits, and prioritization based on station properties

4. **Flask Application** (`app_new_model.py`)
   - Web interface for the route planning system
   - Handles file uploads, form processing, and route visualization
   - Provides real-time progress updates via Server-Sent Events

## Prioritization Logic

Stations are prioritized in the following order:
1. Complaints (highest priority)
2. Reinspections
3. Out-of-Service Pumps
4. Last Visited Date (older visits first)
5. Location (minimize travel time)

## Running the Application

### With Original Model
```bash
python app.py
```

### With New Object-Oriented Model
```bash
python app_new_model.py
```

## Testing the New Model
```bash
python test_new_model.py
```

## Configuration

The application expects an Excel file with the following structure:
- Main sheet: All stations with basic information
- Additional sheets: Reinspection, Complaints, Out-of-Service Pumps

## Features

- **Excel Data Integration**: Reads all sheets at once and correlates data
- **Intelligent Prioritization**: Prioritizes stations based on multiple factors
- **Efficient Routing**: Minimizes travel time while respecting constraints
- **Daily Pump Limits**: Controls how many pumps can be inspected per day
- **Visualization**: Displays the route with station details and status tags
- **Progress Tracking**: Provides real-time updates during processing
- **Caching**: Caches geocoding results for faster processing
