def plan_route_from_stations(stations, hours=8, max_stations=None, max_pumps=20,
                            time_between_stops=0, progress_callback=None, cancellation_event=None,
                            use_cache=True, include_normal=True, include_complaints=True,
                            include_reinspections=True, include_out_of_service=True):
    """Plan an optimal route directly from station objects, filtering by station types"""
    from new_route_planner import (calculate_drive_times, create_data_model, 
                                  solve_route, extract_route_plan, 
                                  save_geocode_cache, STARTING_LOCATION)
    import logging
    
    logger = logging.getLogger(__name__)
    
    # First, filter out skipped stations
    stations = [station for station in stations if not station.skipped]
    
    # Filter stations based on their type flags
    filtered_stations = []
    for station in stations:
        # Normal stations (no special flags)
        is_normal = (not station.has_complaint and 
                    not station.needs_reinspection and 
                    not station.has_out_of_service_pump)
        
        if ((include_normal and is_normal) or
            (include_complaints and station.has_complaint) or
            (include_reinspections and station.needs_reinspection) or
            (include_out_of_service and station.has_out_of_service_pump)):
            filtered_stations.append(station)
    
    stations = filtered_stations
    logger.info(f"Filtered to {len(stations)} stations based on type filters")
    
    # Skip if no stations match the criteria
    if not stations:
        logger.warning("No stations match the selected status filters")
        return None
    
    # Sort stations by priority
    stations.sort(key=lambda x: x.priority_score, reverse=True)
    
    # If max_stations is set, limit the number of stations
    if max_stations and max_stations > 0 and len(stations) > max_stations:
        stations = stations[:max_stations]
    
    # Convert hours to minutes
    total_minutes = hours * 60
    
    # Calculate travel times
    travel_times_df = calculate_drive_times(stations, progress_callback, 
                                           cancellation_event=cancellation_event, 
                                           use_cache=use_cache)
    
    # Check if operation was cancelled
    if cancellation_event and cancellation_event.is_set():
        return None
    
    if travel_times_df is None:
        logger.error("Travel time calculation failed")
        return None
    
    # Create data model
    data_model = create_data_model(stations, travel_times_df, hours, 
                                  max_stations, max_pumps, time_between_stops)
    
    # Solve the route
    solution, manager, routing = solve_route(data_model, progress_callback, 
                                            cancellation_event, max_pumps, max_stations)
    
    # Save geocode cache
    save_geocode_cache()
    
    # Check if operation was cancelled
    if cancellation_event and cancellation_event.is_set():
        return None
    
    # Extract the route plan
    if solution:
        route_plan = extract_route_plan(solution, manager, routing, stations)
        return route_plan
    else:
        logger.warning("No solution found")
        return None
