import pandas as pd
import os

excel_file = 'uploads/your_station_data1.xlsx'
print(f"Examining Excel file with more rows: {excel_file}")

try:
    # Try to read with different skip rows to find the actual data
    for sheet in ['Salt Lake, Utah, Tooele', 'Reinspections', 'Complaints', 'Out of Service Pumps']:
        print(f"\n--- Sheet: {sheet} ---")
        # Try different skip row values to find the actual data
        for skip_rows in [2, 3, 4]:
            try:
                print(f"With skiprows={skip_rows}:")
                df = pd.read_excel(excel_file, sheet_name=sheet, skiprows=skip_rows, nrows=5)
                # Only print if we have actual data (not just headers)
                if not df.empty and len(df.columns) > 3:
                    print(df)
                    print("Columns:", df.columns.tolist())
                    
                    # Look for columns that might contain station identifiers
                    id_columns = []
                    for col in df.columns:
                        if any(s in str(col).lower() for s in ['name', 'business', 'id', 'station', 'address']):
                            id_columns.append(col)
                    
                    if id_columns:
                        print("Potential ID columns:", id_columns)
                        for col in id_columns:
                            print(f"\nValues in {col}:")
                            print(df[col].tolist())
            except Exception as e:
                print(f"Error with skiprows={skip_rows}: {str(e)}")
except Exception as e:
    print(f"Error: {str(e)}")
