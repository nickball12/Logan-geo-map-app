import os
import sys
import logging
import pandas as pd
from station_manager import StationManager

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_column_mapping')

def test_column_mapping():
    """Test the column mapping to verify column R is used for name"""
    try:
        # Initialize the station manager
        manager = StationManager()
        
        # Get the list of Excel files in the uploads folder
        excel_files = [f for f in os.listdir('uploads') if f.endswith('.xlsx')]
        
        if not excel_files:
            logger.error("No Excel files found in uploads directory")
            return False
            
        # Use the first Excel file for testing
        excel_file = os.path.join('uploads', excel_files[0])
        logger.info(f"Testing with Excel file: {excel_file}")
        
        # Read the Excel file
        df = pd.read_excel(excel_file)
        
        # Get the column mapping
        col_map = manager._identify_columns(df)
        
        # Print all column names and their index for reference
        logger.info("Excel columns:")
        for i, col in enumerate(df.columns):
            col_letter = ""
            index = i + 1
            while index > 0:
                index, remainder = divmod(index - 1, 26)
                col_letter = chr(65 + remainder) + col_letter
            logger.info(f"  {i}: {col} (Column {col_letter})")
        
        # Check if 'name' is in the column mapping
        if 'name' in col_map:
            name_col = col_map['name']
            name_idx = df.columns.get_loc(name_col)
            col_letter = ""
            index = name_idx + 1
            while index > 0:
                index, remainder = divmod(index - 1, 26)
                col_letter = chr(65 + remainder) + col_letter
                
            logger.info(f"'name' is mapped to column: {name_col} (index {name_idx}, column {col_letter})")
            
            # Check if it's column R (index 17)
            if col_letter == 'R':
                logger.info("SUCCESS: 'name' is correctly mapped to column R")
            else:
                logger.warning(f"FAIL: 'name' is mapped to column {col_letter}, not R")
        else:
            logger.error("'name' not found in column mapping")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error testing column mapping: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    result = test_column_mapping()
    sys.exit(0 if result else 1)
