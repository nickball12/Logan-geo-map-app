"""
Helper script to generate test data with a mix of regular and priority stations
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_test_data():
    """Generate test data with a mix of regular and priority stations"""
    # Create directory for output
    os.makedirs("test_data", exist_ok=True)
    
    # Base date for random dates
    base_date = datetime.now() - timedelta(days=365)
    
    # Generate main station data (50 stations)
    num_stations = 50
    
    # Business IDs
    business_ids = [f"BUS{i:04d}" for i in range(1, num_stations + 1)]
    
    # Station names
    station_names = [f"Gas Station {i}" for i in range(1, num_stations + 1)]
    
    # Addresses (made up)
    streets = ["Main St", "Oak Ave", "Cedar Ln", "Pine St", "Maple Dr", 
               "Washington Ave", "Lincoln Rd", "Jefferson Blvd", "Franklin St", "Roosevelt Ave"]
    
    addresses = []
    for i in range(num_stations):
        street_num = random.randint(100, 9999)
        street = random.choice(streets)
        addresses.append(f"{street_num} {street}")
    
    # Cities
    cities = ["Logan", "Providence", "Smithfield", "Hyrum", "Newton", 
              "Millville", "Nibley", "North Logan", "Richmond", "Wellsville"]
    
    city_list = []
    for i in range(num_stations):
        city_list.append(random.choice(cities))
    
    # State (always UT)
    states = ["UT"] * num_stations
    
    # ZIP Codes
    zip_codes = []
    for i in range(num_stations):
        # Logan area zip codes
        zip_codes.append(random.choice(["84321", "84322", "84323", "84325", "84326", "84327", "84328", "84329", "84330", "84331"]))
    
    # Number of pumps (random between 1 and 8)
    num_pumps = [random.randint(1, 8) for _ in range(num_stations)]
    
    # Last inspection date (random in the past year)
    last_inspection_dates = []
    for i in range(num_stations):
        days_ago = random.randint(0, 365)
        last_inspection_dates.append(base_date + timedelta(days=days_ago))
    
    # Create the main dataframe
    main_df = pd.DataFrame({
        "Business ID #": business_ids,
        "Business Name": station_names,
        "Address": addresses,
        "City": city_list,
        "State": states,
        "ZIP Code": zip_codes,
        "Number of Pumps": num_pumps,
        "Last Inspection Date": last_inspection_dates
    })
    
    # Create reinspection sheet (about 10% of stations)
    reinspection_ids = random.sample(business_ids, num_stations // 10)
    reinspection_reasons = [
        "Failed pressure test", 
        "Leaking fuel line", 
        "Failed vapor recovery", 
        "Pump calibration error",
        "Faulty check valve"
    ]
    
    reinspection_df = pd.DataFrame({
        "Business ID #": reinspection_ids,
        "Business Name": [main_df.loc[main_df["Business ID #"] == id, "Business Name"].values[0] for id in reinspection_ids],
        "Address": [main_df.loc[main_df["Business ID #"] == id, "Address"].values[0] for id in reinspection_ids],
        "City": [main_df.loc[main_df["Business ID #"] == id, "City"].values[0] for id in reinspection_ids],
        "State": [main_df.loc[main_df["Business ID #"] == id, "State"].values[0] for id in reinspection_ids],
        "ZIP Code": [main_df.loc[main_df["Business ID #"] == id, "ZIP Code"].values[0] for id in reinspection_ids],
        "Reinspection Reason": [random.choice(reinspection_reasons) for _ in range(len(reinspection_ids))],
        "Scheduled Date": [datetime.now() + timedelta(days=random.randint(1, 30)) for _ in range(len(reinspection_ids))]
    })
    
    # Create complaint sheet (about 5% of stations)
    complaint_ids = random.sample([id for id in business_ids if id not in reinspection_ids], num_stations // 20)
    complaint_types = [
        "Pump not dispensing correctly",
        "Customer overcharged",
        "Pump display malfunction",
        "Fuel quality concerns",
        "Water in fuel"
    ]
    
    complaint_df = pd.DataFrame({
        "Business ID #": complaint_ids,
        "Business Name": [main_df.loc[main_df["Business ID #"] == id, "Business Name"].values[0] for id in complaint_ids],
        "Address": [main_df.loc[main_df["Business ID #"] == id, "Address"].values[0] for id in complaint_ids],
        "City": [main_df.loc[main_df["Business ID #"] == id, "City"].values[0] for id in complaint_ids],
        "State": [main_df.loc[main_df["Business ID #"] == id, "State"].values[0] for id in complaint_ids],
        "ZIP Code": [main_df.loc[main_df["Business ID #"] == id, "ZIP Code"].values[0] for id in complaint_ids],
        "Complaint Type": [random.choice(complaint_types) for _ in range(len(complaint_ids))],
        "Complaint Date": [datetime.now() - timedelta(days=random.randint(1, 30)) for _ in range(len(complaint_ids))]
    })
    
    # Create out-of-service pumps sheet (about 8% of stations)
    oos_ids = random.sample([id for id in business_ids if id not in reinspection_ids and id not in complaint_ids], num_stations // 12)
    oos_reasons = [
        "Failed pressure test",
        "Leaking hose",
        "Display malfunction",
        "Credit card reader broken",
        "Physical damage"
    ]
    
    # For each station, calculate number of pumps out of service (1 to max pumps at station)
    oos_count = []
    for id in oos_ids:
        total_pumps = main_df.loc[main_df["Business ID #"] == id, "Number of Pumps"].values[0]
        oos_count.append(random.randint(1, min(total_pumps, 3)))
    
    oos_df = pd.DataFrame({
        "Business ID #": oos_ids,
        "Business Name": [main_df.loc[main_df["Business ID #"] == id, "Business Name"].values[0] for id in oos_ids],
        "Address": [main_df.loc[main_df["Business ID #"] == id, "Address"].values[0] for id in oos_ids],
        "City": [main_df.loc[main_df["Business ID #"] == id, "City"].values[0] for id in oos_ids],
        "State": [main_df.loc[main_df["Business ID #"] == id, "State"].values[0] for id in oos_ids],
        "ZIP Code": [main_df.loc[main_df["Business ID #"] == id, "ZIP Code"].values[0] for id in oos_ids],
        "Pumps Out of Service": oos_count,
        "Out of Service Reason": [random.choice(oos_reasons) for _ in range(len(oos_ids))],
        "Date Reported": [datetime.now() - timedelta(days=random.randint(1, 90)) for _ in range(len(oos_ids))]
    })
    
    # Create Excel file with all sheets
    with pd.ExcelWriter("test_data/test_station_data.xlsx") as writer:
        main_df.to_excel(writer, sheet_name="Main", index=False)
        reinspection_df.to_excel(writer, sheet_name="Reinspections", index=False)
        complaint_df.to_excel(writer, sheet_name="Complaints", index=False)
        oos_df.to_excel(writer, sheet_name="Out of Service Pumps", index=False)
        
    print(f"Generated test data with {num_stations} stations:")
    print(f"  - {len(reinspection_ids)} stations needing reinspection")
    print(f"  - {len(complaint_ids)} stations with complaints")
    print(f"  - {len(oos_ids)} stations with out-of-service pumps")
    print("Data saved to test_data/test_station_data.xlsx")

if __name__ == "__main__":
    generate_test_data()
