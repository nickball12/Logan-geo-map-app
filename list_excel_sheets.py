import pandas as pd
import os

def main():
    """List all sheet names in the Excel file."""
    excel_path = os.path.join('uploads', 'your_station_data1.xlsx')
    
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return
    
    # Load Excel sheets
    print("Loading Excel file...")
    try:
        xls = pd.ExcelFile(excel_path)
        sheet_names = xls.sheet_names
        print(f"Found {len(sheet_names)} sheets:")
        for i, name in enumerate(sheet_names, 1):
            print(f"  {i}. '{name}'")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

if __name__ == "__main__":
    main()
