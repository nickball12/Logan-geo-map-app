import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('diagnose_business_ids')

def diagnose_business_ids():
    """Check if business IDs in additional sheets exist in the main sheet"""
    excel_file = 'uploads/your_station_data1.xlsx'
    
    if not os.path.exists(excel_file):
        logger.error(f"Excel file not found: {excel_file}")
        return
    
    try:
        # Read the Excel file
        excel = pd.ExcelFile(excel_file)
        sheet_names = excel.sheet_names
        logger.info(f"Found sheets: {sheet_names}")
        
        # Read the main sheet (first sheet)
        main_sheet_name = sheet_names[0]
        main_df = pd.read_excel(excel, sheet_name=main_sheet_name, skiprows=3)
        
        # Find the business ID column in the main sheet
        business_id_col = None
        for col in main_df.columns:
            if 'business id' in str(col).lower() or 'id #' in str(col).lower():
                business_id_col = col
                break
        
        if not business_id_col:
            # Try to use the 5th column (index 4) which is typically the business ID
            if len(main_df.columns) > 4:
                business_id_col = main_df.columns[4]
            else:
                logger.error("Could not find business ID column in main sheet")
                return
        
        logger.info(f"Using business ID column '{business_id_col}' in main sheet")
        
        # Get all business IDs from the main sheet
        main_business_ids = set()
        for bid in main_df[business_id_col]:
            if pd.notna(bid) and bid != 'Business ID #':
                main_business_ids.add(str(bid))
        
        logger.info(f"Found {len(main_business_ids)} business IDs in main sheet")
        
        # Check each additional sheet
        for sheet_name in sheet_names[1:]:
            try:
                df = pd.read_excel(excel, sheet_name=sheet_name, skiprows=3)
                
                # Find business ID column in this sheet
                sheet_business_id_col = None
                for col in df.columns:
                    if 'business id' in str(col).lower() or 'id #' in str(col).lower():
                        sheet_business_id_col = col
                        break
                
                if not sheet_business_id_col:
                    # Try to use the 5th column (index 4)
                    if len(df.columns) > 4:
                        sheet_business_id_col = df.columns[4]
                    else:
                        logger.error(f"Could not find business ID column in sheet: {sheet_name}")
                        continue
                
                logger.info(f"Using business ID column '{sheet_business_id_col}' in {sheet_name} sheet")
                
                # Get business IDs from this sheet
                sheet_business_ids = set()
                sheet_ids_not_in_main = set()
                
                for bid in df[sheet_business_id_col]:
                    if pd.notna(bid) and bid != 'Business ID #':
                        bid_str = str(bid)
                        sheet_business_ids.add(bid_str)
                        if bid_str not in main_business_ids:
                            sheet_ids_not_in_main.add(bid_str)
                
                logger.info(f"Sheet: {sheet_name}")
                logger.info(f"  Total business IDs: {len(sheet_business_ids)}")
                logger.info(f"  Business IDs not in main sheet: {len(sheet_ids_not_in_main)} ({(len(sheet_ids_not_in_main) / len(sheet_business_ids) * 100):.1f}%)")
                
                if sheet_ids_not_in_main:
                    logger.info(f"  First 5 missing IDs: {list(sheet_ids_not_in_main)[:5]}")
                    
                    # Show actual rows for some missing business IDs
                    logger.info("Sample rows with missing business IDs:")
                    for i, missing_id in enumerate(list(sheet_ids_not_in_main)[:3]):
                        sample_row = df[df[sheet_business_id_col].astype(str) == missing_id].iloc[0] if not df[df[sheet_business_id_col].astype(str) == missing_id].empty else None
                        if sample_row is not None:
                            logger.info(f"  Row {i+1}: {dict(sample_row)}")
            
            except Exception as e:
                logger.error(f"Error processing sheet {sheet_name}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    diagnose_business_ids()
