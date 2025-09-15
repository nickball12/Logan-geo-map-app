import pandas as pd
import os

def main():
    """Diagnose business ID formatting issues between main sheet and secondary sheets."""
    excel_path = os.path.join('uploads', 'your_station_data1.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return
    
    # Load Excel sheets
    print("Loading Excel sheets...")
    try:
        main_sheet = pd.read_excel(excel_path, sheet_name=0)
        reinspection_sheet = pd.read_excel(excel_path, sheet_name='Reinspections')
        complaint_sheet = pd.read_excel(excel_path, sheet_name='Complaints')
        out_of_service_sheet = pd.read_excel(excel_path, sheet_name='Out of Service Pumps')
    except Exception as e:
        print(f"Error loading Excel sheets: {e}")
        return
    
    # Get business ID column name
    main_business_id_col = None
    for col in main_sheet.columns:
        if 'business' in str(col).lower() and 'id' in str(col).lower():
            main_business_id_col = col
            break
    
    if not main_business_id_col:
        print("Could not find business ID column in main sheet")
        return
    
    # Analyze business IDs
    print(f"Business ID column in main sheet: {main_business_id_col}")
    
    # Print sample business IDs from main sheet
    print("\nSample business IDs from main sheet:")
    main_ids = main_sheet[main_business_id_col].dropna().astype(str).tolist()[:10]
    for i, bid in enumerate(main_ids, 1):
        print(f"  {i}. '{bid}' (type: {type(bid).__name__})")
    
    # Get reinspection business ID column
    try:
        ri_business_id_col = None
        for col in reinspection_sheet.columns:
            if 'business' in str(col).lower() and 'id' in str(col).lower():
                ri_business_id_col = col
                break
        
        if ri_business_id_col:
            print("\nSample business IDs from reinspection sheet:")
            ri_ids = reinspection_sheet[ri_business_id_col].dropna().astype(str).tolist()[:10]
            for i, bid in enumerate(ri_ids, 1):
                print(f"  {i}. '{bid}' (type: {type(bid).__name__})")
    except Exception as e:
        print(f"Error processing reinspection sheet: {e}")
    
    # Get complaint business ID column
    try:
        complaint_business_id_col = None
        for col in complaint_sheet.columns:
            if 'business' in str(col).lower() and 'id' in str(col).lower():
                complaint_business_id_col = col
                break
        
        if complaint_business_id_col:
            print("\nSample business IDs from complaint sheet:")
            complaint_ids = complaint_sheet[complaint_business_id_col].dropna().astype(str).tolist()[:10]
            for i, bid in enumerate(complaint_ids, 1):
                print(f"  {i}. '{bid}' (type: {type(bid).__name__})")
    except Exception as e:
        print(f"Error processing complaint sheet: {e}")
    
    # Get out of service business ID column
    try:
        oos_business_id_col = None
        for col in out_of_service_sheet.columns:
            if 'business' in str(col).lower() and 'id' in str(col).lower():
                oos_business_id_col = col
                break
        
        if oos_business_id_col:
            print("\nSample business IDs from out of service sheet:")
            oos_ids = out_of_service_sheet[oos_business_id_col].dropna().astype(str).tolist()[:10]
            for i, bid in enumerate(oos_ids, 1):
                print(f"  {i}. '{bid}' (type: {type(bid).__name__})")
    except Exception as e:
        print(f"Error processing out of service sheet: {e}")
    
    # Now check for exact matches between main and secondary sheets
    try:
        if ri_business_id_col:
            ri_ids_set = set(reinspection_sheet[ri_business_id_col].dropna().astype(str).tolist())
            main_ids_set = set(main_sheet[main_business_id_col].dropna().astype(str).tolist())
            matches = ri_ids_set.intersection(main_ids_set)
            match_percentage = len(matches)/len(ri_ids_set)*100 if ri_ids_set else 0
            print(f"\nReinspection matches with main sheet: {len(matches)} out of {len(ri_ids_set)} ({match_percentage:.2f}%)")
    except Exception as e:
        print(f"Error checking reinspection matches: {e}")
    
    # Try with standardized IDs (remove decimal point)
    try:
        if ri_business_id_col:
            ri_ids_std = [str(id).split('.')[0] for id in reinspection_sheet[ri_business_id_col].dropna().tolist()]
            main_ids_std = [str(id).split('.')[0] for id in main_sheet[main_business_id_col].dropna().tolist()]
            ri_ids_std_set = set(ri_ids_std)
            main_ids_std_set = set(main_ids_std)
            std_matches = ri_ids_std_set.intersection(main_ids_std_set)
            std_match_percentage = len(std_matches)/len(ri_ids_std_set)*100 if ri_ids_std_set else 0
            print(f"Reinspection matches with standardized IDs: {len(std_matches)} out of {len(ri_ids_std_set)} ({std_match_percentage:.2f}%)")
    except Exception as e:
        print(f"Error checking standardized reinspection matches: {e}")
    
    # Show example of failing match
    try:
        if ri_business_id_col and ri_ids_set and len(matches) < len(ri_ids_set):
            mismatch_id = next(iter(ri_ids_set - matches))
            mismatch_std = str(mismatch_id).split('.')[0]
            print(f"\nExample mismatch: '{mismatch_id}' from reinspection sheet")
            print(f"After standardization: '{mismatch_std}'")
            
            if mismatch_std in main_ids_std_set:
                print(f"Found match in main sheet after standardization!")
                for main_id in main_ids:
                    if str(main_id).split('.')[0] == mismatch_std:
                        print(f"Original main sheet ID: '{main_id}'")
                        break
            else:
                print(f"No match found in main sheet even after standardization")
    except Exception as e:
        print(f"Error showing example mismatch: {e}")

if __name__ == "__main__":
    main()
