import pandas as pd
import os

print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir("."))

excel_file = 'uploads/your_station_data1.xlsx'
print(f"Examining Excel file: {excel_file}")

try:
    excel = pd.ExcelFile(excel_file)
    print("Sheet names:", excel.sheet_names)
    
    for sheet in excel.sheet_names:
        print(f"\n--- Sheet: {sheet} ---")
        try:
            # Try to read the first few rows, skipping different header rows
            for skip_rows in range(0, 6):
                try:
                    df = pd.read_excel(excel, sheet_name=sheet, skiprows=skip_rows, nrows=3)
                    print(f"With skiprows={skip_rows}:")
                    print(df.head(2))
                    print("Columns:", df.columns.tolist())
                    if not df.empty and len(df.columns) > 3:
                        break  # Found good data, no need to try more skip_rows
                except Exception as e:
                    print(f"Error with skiprows={skip_rows}: {str(e)}")
        except Exception as e:
            print(f"Error reading sheet {sheet}: {str(e)}")
except Exception as e:
    print(f"Error opening Excel file: {str(e)}")
