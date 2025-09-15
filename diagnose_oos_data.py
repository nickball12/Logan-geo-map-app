import pandas as pd
import os
import numpy as np
from pathlib import Path

def analyze_oos_sheet(file_path):
    print(f"\nAnalyzing Out of Service sheet in {file_path}")
    
    # Try different header rows to see which works best
    for skip_rows in range(3, 6):
        print(f"\nTrying with skip_rows={skip_rows}")
        try:
            # Read the OOS sheet (index 3)
            df = pd.read_excel(file_path, sheet_name=3, skiprows=skip_rows)
            print("\nFirst few rows of data:")
            print(df.head())
            
            # Show column names
            print("\nColumn names:")
            for i, col in enumerate(df.columns):
                print(f"Column {i} ({chr(65+i)}): {col}")
            
            # Look for business ID column
            id_cols = df.iloc[:, :10].apply(lambda x: x.astype(str).str.contains('Business|ID|#', case=False).any())
            id_col_indices = id_cols[id_cols].index
            print("\nPossible Business ID columns:", id_col_indices.tolist())
            
            # Look for OOS related columns
            oos_cols = df.apply(lambda x: x.astype(str).str.contains('out.*service|oos|pump', case=False).any())
            oos_col_indices = oos_cols[oos_cols].index
            print("\nPossible Out of Service columns:", oos_col_indices.tolist())
            
            if len(oos_col_indices) > 0:
                print("\nSample values from potential OOS columns:")
                for col in oos_col_indices:
                    print(f"\nColumn {col}:")
                    print(df[col].dropna().head())
                    
                # Count non-empty values in OOS columns
                print("\nNon-empty value counts in OOS columns:")
                for col in oos_col_indices:
                    non_empty = df[col].dropna().shape[0]
                    print(f"Column {col}: {non_empty} non-empty values")
            
            # Analyze numeric columns that might contain pump numbers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            print("\nNumeric columns that might contain pump numbers:")
            for col in numeric_cols:
                non_zero = (df[col] > 0).sum()
                if non_zero > 0:
                    print(f"Column {col}: {non_zero} non-zero values")
                    print("Sample values:", df[col][df[col] > 0].head().tolist())
            
            break  # If we got here, the read was successful
        except Exception as e:
            print(f"Error with skip_rows={skip_rows}: {str(e)}")
            continue

if __name__ == "__main__":
    excel_path = Path("uploads/your_station_data1.xlsx")
    if not excel_path.exists():
        print(f"Error: File not found at {excel_path}")
    else:
        analyze_oos_sheet(excel_path)