import os
import pandas as pd
from station_manager import StationManager
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test last inspection date column mapping and data loading."""
    excel_path = os.path.join('uploads', 'your_station_data1.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return
    
    # First, let's look at the Excel file directly to find the last inspection date column
    print("Examining Excel file directly...")
    try:
        df = pd.read_excel(excel_path, sheet_name=0, skiprows=3)
        
        # Print the first few column names to see what we're working with
        print("\nColumn names in Excel sheet:")
        for i, col in enumerate(df.columns):
            print(f"  Column {i+1}: '{col}'")
        
        # Try to identify last inspection date column by looking for relevant keywords
        possible_date_cols = []
        for i, col in enumerate(df.columns):
            col_str = str(col).lower()
            if 'last' in col_str and ('insp' in col_str or 'inspection' in col_str or 'date' in col_str):
                possible_date_cols.append((i, col))
        
        print("\nPossible last inspection date columns:")
        for i, col in possible_date_cols:
            print(f"  Column {i+1}: '{col}'")
            
            # Let's check the data in these columns
            print(f"  Sample values in column '{col}':")
            sample_values = df[col].dropna().head(5).tolist()
            for j, val in enumerate(sample_values):
                print(f"    Value {j+1}: {val} (type: {type(val).__name__})")
                
    except Exception as e:
        print(f"Error examining Excel directly: {e}")
    
    # Now let's use the StationManager to load the data and check how it processes the dates
    print("\nLoading data using StationManager...")
    station_manager = StationManager()
    success = station_manager.load_from_excel(excel_path)
    
    if not success:
        print("Failed to load data from Excel file")
        return
    
    # Get all stations
    all_stations = station_manager.get_all_stations()
    print(f"Loaded {len(all_stations)} stations")
    
    # Count stations with last inspection dates
    stations_with_date = 0
    for station in all_stations:
        if station.last_inspection_date is not None:
            stations_with_date += 1
    
    print(f"\nStations with last inspection date: {stations_with_date} out of {len(all_stations)} ({stations_with_date/len(all_stations)*100:.1f}%)")
    
    # Print some sample last inspection dates
    print("\nSample stations with inspection dates:")
    count = 0
    for station in all_stations:
        if station.last_inspection_date is not None:
            print(f"  {station.name} (ID: {station.business_id})")
            print(f"    Last Inspection Date: {station.last_inspection_date}")
            print(f"    Days Since Inspection: {station.days_since_inspection}")
            count += 1
            if count >= 5:
                break
    
    if count == 0:
        print("  No stations found with inspection dates.")
    
    # Let's also examine the column mapping for the current Excel file
    
    # Create a StationManager instance
    sm = StationManager()
    
    # Load the Excel file and get the first sheet
    excel = pd.ExcelFile(excel_path)
    sheet_name = excel.sheet_names[0]
    df = pd.read_excel(excel, sheet_name=sheet_name, skiprows=3)
    
    # Use the _identify_columns method to get the column mapping
    col_map = sm._identify_columns(df)
    
    # Print the results
    print("\nColumn Mapping Results:")
    for purpose, column in col_map.items():
        print(f"  {purpose}: {column}")
    
    # Focus on last_inspection_date specifically
    last_insp_col = col_map.get('last_inspection_date')
    if last_insp_col:
        print(f"\nLast Inspection Date column: '{last_insp_col}'")
        print("Sample values:")
        sample_values = df[last_insp_col].dropna().head(5).tolist()
        for i, val in enumerate(sample_values, 1):
            print(f"  Value {i}: {val} (type: {type(val).__name__})")
    else:
        print("\nNo Last Inspection Date column found in mapping!")
        
        # Let's add a fix by explicitly mapping a column for last inspection date
        print("\nTrying to find a column with date values that might be the last inspection date:")
        
        date_cols = []
        for i, col in enumerate(df.columns):
            # Check first 5 non-null values in the column
            values = df[col].dropna().head(5).tolist()
            
            # If any of these values are datetime objects, this might be a date column
            for val in values:
                if isinstance(val, (datetime, pd.Timestamp)):
                    date_cols.append((i, col))
                    break
        
        for i, col in date_cols:
            print(f"  Found possible date column {i+1}: '{col}'")
            print("  Sample values:")
            sample_values = df[col].dropna().head(3).tolist()
            for j, val in enumerate(sample_values, 1):
                print(f"    Value {j}: {val} (type: {type(val).__name__})")

if __name__ == "__main__":
    main()
