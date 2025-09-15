import pandas as pd
import numpy as np
from pathlib import Path

def analyze_sheet(df, sheet_name):
    print(f"\n=== Analyzing {sheet_name} ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 non-empty rows with their indices:")
    non_empty = df.dropna(how='all').head()
    print(non_empty)
    
    print("\nColumn names found:")
    print(df.columns.tolist())
    
    print("\nSample of unique business IDs:")
    if 'Business ID' in df.columns:
        print(df['Business ID'].dropna().unique()[:5])
    
    print("\nSample of unique addresses:")
    address_cols = [col for col in df.columns if 'Address' in col or 'address' in col]
    if address_cols:
        print(df[address_cols[0]].dropna().unique()[:5])
    
    print("\nEmpty cell analysis:")
    print(df.isnull().sum())

def diagnose_excel(file_path):
    print(f"Reading Excel file: {file_path}")
    
    # Sheet 1: Salt Lake, Utah, Tooele (header on row 3)
    print("\nReading Sheet 1 (Main Stations)")
    main_df = pd.read_excel(file_path, sheet_name=0, header=2)
    analyze_sheet(main_df, "Sheet 1 (Main Stations)")
    
    # Sheet 2: Reinspections (header on row 4)
    print("\nReading Sheet 2 (Reinspections)")
    reinsp_df = pd.read_excel(file_path, sheet_name=1, header=3)
    analyze_sheet(reinsp_df, "Sheet 2 (Reinspections)")
    
    # Sheet 3: Complaints (header on row 4)
    print("\nReading Sheet 3 (Complaints)")
    complaints_df = pd.read_excel(file_path, sheet_name=2, header=3)
    analyze_sheet(complaints_df, "Sheet 3 (Complaints)")
    
    # Sheet 4: Out of Service Pumps (header on row 4)
    print("\nReading Sheet 4 (Out of Service)")
    oos_df = pd.read_excel(file_path, sheet_name=3, header=3)
    analyze_sheet(oos_df, "Sheet 4 (Out of Service)")
    
    # Cross-reference analysis
    print("\n=== Cross-Reference Analysis ===")
    
    # Get all business IDs and addresses from each sheet
    def get_ids_and_addresses(df, sheet_name):
        ids = set(df['Business ID'].dropna()) if 'Business ID' in df.columns else set()
        addr_cols = [col for col in df.columns if 'Address' in col or 'address' in col]
        addresses = set(df[addr_cols[0]].dropna()) if addr_cols else set()
        return ids, addresses
    
    main_ids, main_addrs = get_ids_and_addresses(main_df, "Main")
    reinsp_ids, reinsp_addrs = get_ids_and_addresses(reinsp_df, "Reinspections")
    compl_ids, compl_addrs = get_ids_and_addresses(complaints_df, "Complaints")
    oos_ids, oos_addrs = get_ids_and_addresses(oos_df, "Out of Service")
    
    print("\nBusiness ID overlap analysis:")
    print(f"Main stations has {len(main_ids)} unique business IDs")
    print(f"Reinspections has {len(reinsp_ids)} unique business IDs")
    print(f"Complaints has {len(compl_ids)} unique business IDs")
    print(f"Out of Service has {len(oos_ids)} unique business IDs")
    
    print("\nReinspection matches:")
    print(f"Business IDs found in main sheet: {len(reinsp_ids & main_ids)}/{len(reinsp_ids)}")
    print(f"Addresses found in main sheet: {len(reinsp_addrs & main_addrs)}/{len(reinsp_addrs)}")
    
    print("\nComplaints matches:")
    print(f"Business IDs found in main sheet: {len(compl_ids & main_ids)}/{len(compl_ids)}")
    print(f"Addresses found in main sheet: {len(compl_addrs & main_addrs)}/{len(compl_addrs)}")
    
    print("\nOut of Service matches:")
    print(f"Business IDs found in main sheet: {len(oos_ids & main_ids)}/{len(oos_ids)}")
    print(f"Addresses found in main sheet: {len(oos_addrs & main_addrs)}/{len(oos_addrs)}")

if __name__ == "__main__":
    excel_path = Path("uploads/your_station_data1.xlsx")
    if not excel_path.exists():
        print(f"Error: File not found at {excel_path}")
    else:
        diagnose_excel(excel_path)