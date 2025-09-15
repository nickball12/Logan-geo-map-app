import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
from station_model import Station
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StationManager:
    """
    Manages the collection of stations and handles loading data from Excel sheets.
    """
    def __init__(self):
        self.stations = {}  # Dictionary of Station objects keyed by business_id
        self.excel_file = None
        self.today = datetime.now().date()
        
    def load_from_excel(self, excel_file_path, skip_rows=3):
        """
        Load station data from the Excel file, processing all sheets
        and correlating the data into unified Station objects.
        """
        self.excel_file = excel_file_path
        logger.info(f"Loading data from {excel_file_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(excel_file_path):
                logger.error(f"Excel file not found: {excel_file_path}")
                return False
                
            # Load the Excel file
            excel = pd.ExcelFile(excel_file_path)
            sheet_names = excel.sheet_names
            logger.info(f"Found sheets: {sheet_names}")
            
            # First, load the main station data from the first sheet
            self._load_main_station_data(excel, sheet_names[0], skip_rows)
            
            # Then process other sheets to enhance station data
            if len(sheet_names) > 1 and 'Reinspections' in sheet_names:
                self._load_reinspection_data(excel, 'Reinspections', skip_rows)
                
            if len(sheet_names) > 2 and 'Complaints' in sheet_names:
                self._load_complaint_data(excel, 'Complaints', skip_rows)
                
            if len(sheet_names) > 3 and 'Out of Service Pumps' in sheet_names:
                self._load_out_of_service_data(excel, 'Out of Service Pumps', skip_rows)
                
            # Calculate priority scores for all stations
            for station in self.stations.values():
                station.calculate_priority()
                
            logger.info(f"Successfully loaded {len(self.stations)} stations")
            return True
            
        except Exception as e:
            logger.error(f"Error loading Excel data: {str(e)}", exc_info=True)
            return False
    
    def _load_main_station_data(self, excel, sheet_name, skip_rows):
        """Load the main station data from the first sheet"""
        logger.info(f"Loading main station data from sheet: {sheet_name}")
        
        try:
            # Read the sheet
            df = pd.read_excel(excel, sheet_name=sheet_name, skiprows=skip_rows)
            
            # Remove completely empty rows
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all')
            
            # Identify the column names - could vary by Excel file format
            # Business ID is typically column 4 (index 3)
            # Name column is typically column 17 or 18
            # Address is typically column 18 or 19
            # Searching for column names that might contain identifiers
            col_map = self._identify_columns(df)
            
            # Process each row into a Station object
            for _, row in df.iterrows():
                try:
                    # Skip rows without a business ID
                    if pd.isna(row.get(col_map.get('business_id'))) or row.get(col_map.get('business_id')) == 'Business ID #':
                        continue
                        
                    # Create a unique identifier for the station (business ID)
                    # Standardize the business ID by removing decimal points and converting to string
                    business_id = str(row.get(col_map.get('business_id'))).split('.')[0]
                    
                    # Create a new Station object if this is the first time we're seeing this ID
                    if business_id not in self.stations:
                        station = Station(
                            business_id=business_id,
                            name=row.get(col_map.get('name')),
                            address=row.get(col_map.get('address')),
                            city=row.get(col_map.get('city')),
                            state=row.get(col_map.get('state')),
                            zip_code=row.get(col_map.get('zip')),
                            county=row.get(col_map.get('county'))
                        )
                        station.update_full_address()
                        
                        # Get number of pumps
                        pumps_col = col_map.get('num_pumps')
                        if pumps_col and not pd.isna(row.get(pumps_col)):
                            try:
                                station.num_pumps = int(row.get(pumps_col))
                            except (ValueError, TypeError):
                                # If we can't convert to int, default to 0
                                station.num_pumps = 0
                        
                        # Get last inspection date
                        last_insp_col = col_map.get('last_inspection_date')
                        if last_insp_col and not pd.isna(row.get(last_insp_col)):
                            insp_date = row.get(last_insp_col)
                            # Convert to datetime if not already
                            if isinstance(insp_date, str):
                                try:
                                    from datetime import datetime
                                    # Try common date formats
                                    for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y%m%d']:
                                        try:
                                            insp_date = datetime.strptime(insp_date, fmt)
                                            break
                                        except ValueError:
                                            continue
                                except Exception:
                                    insp_date = None
                            station.last_inspection_date = insp_date if isinstance(insp_date, datetime) else None
                            
                            # Calculate days since inspection
                            if isinstance(station.last_inspection_date, datetime):
                                days_diff = (self.today - station.last_inspection_date.date()).days
                                station.days_since_inspection = max(0, days_diff)
                        
                        self.stations[business_id] = station
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    
            logger.info(f"Loaded {len(self.stations)} stations from main sheet")
        except Exception as e:
            logger.error(f"Error loading main station data: {str(e)}", exc_info=True)
    
    def _load_reinspection_data(self, excel, sheet_name, skip_rows):
        """Load reinspection data and update relevant stations"""
        logger.info(f"Loading reinspection data from sheet: {sheet_name}")
        
        try:
            # Read the sheet
            df = pd.read_excel(excel, sheet_name=sheet_name, skiprows=skip_rows)
            
            # Remove completely empty rows
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all')
            
            # Identify columns
            col_map = self._identify_columns(df)
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Skip rows without a business ID
                    if pd.isna(row.get(col_map.get('business_id'))) or row.get(col_map.get('business_id')) == 'Business ID #':
                        continue
                        
                    # Standardize the business ID by removing decimal points and converting to string
                    business_id = str(row.get(col_map.get('business_id'))).split('.')[0]
                    
                    # Skip if the station is not in our main data
                    if business_id not in self.stations:
                        continue
                        
                    # Mark as needing reinspection
                    self.stations[business_id].needs_reinspection = True
                    
                    # Get the reason for reinspection from notes column
                    notes_col = col_map.get('notes')
                    if notes_col and not pd.isna(row.get(notes_col)):
                        self.stations[business_id].reinspection_reason = row.get(notes_col)
                    
                except Exception as e:
                    logger.warning(f"Error processing reinspection row: {e}")
                    
            logger.info(f"Processed reinspection data")
        except Exception as e:
            logger.error(f"Error loading reinspection data: {str(e)}", exc_info=True)
    
    def _load_complaint_data(self, excel, sheet_name, skip_rows):
        """Load complaint data and update relevant stations"""
        logger.info(f"Loading complaint data from sheet: {sheet_name}")
        
        try:
            # Read the sheet
            df = pd.read_excel(excel, sheet_name=sheet_name, skiprows=skip_rows)
            
            # Remove completely empty rows
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all')
            
            # Identify columns
            col_map = self._identify_columns(df)
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Skip rows without a business ID
                    if pd.isna(row.get(col_map.get('business_id'))) or row.get(col_map.get('business_id')) == 'Business ID #':
                        continue
                        
                    # Standardize the business ID by removing decimal points and converting to string
                    business_id = str(row.get(col_map.get('business_id'))).split('.')[0]
                    
                    # Skip if the station is not in our main data
                    if business_id not in self.stations:
                        continue
                        
                    # Mark as having a complaint
                    self.stations[business_id].has_complaint = True
                    
                    # Get complaint details
                    notes_col = col_map.get('notes')
                    if notes_col and not pd.isna(row.get(notes_col)):
                        self.stations[business_id].complaint_details = row.get(notes_col)
                    
                    # Get complaint date
                    viol_date_col = col_map.get('viol_date')
                    if viol_date_col and not pd.isna(row.get(viol_date_col)):
                        self.stations[business_id].complaint_date = row.get(viol_date_col)
                    
                except Exception as e:
                    logger.warning(f"Error processing complaint row: {e}")
                    
            logger.info(f"Processed complaint data")
        except Exception as e:
            logger.error(f"Error loading complaint data: {str(e)}", exc_info=True)
    
    def _load_out_of_service_data(self, excel, sheet_name, skip_rows):
        """Load out of service data and update relevant stations"""
        logger.info(f"Loading out of service data from sheet: {sheet_name}")
        
        try:
            # Read the sheet - use skip_rows=4 specifically for OOS sheet
            df = pd.read_excel(excel, sheet_name=sheet_name, skiprows=4)
            
            # Remove completely empty rows
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all')
            
            # Identify columns
            col_map = self._identify_columns(df)
            
            # Log column mapping for debugging
            logger.debug(f"OOS sheet column mapping: {col_map}")
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Skip rows without a business ID
                    business_id_col = col_map.get('business_id')
                    if not business_id_col or pd.isna(row.get(business_id_col)):
                        # Try using the explicit column name if mapping failed
                        business_id_col = 'Business ID #'
                        if pd.isna(row.get(business_id_col)):
                            continue
                    
                    # Standardize the business ID by removing decimal points and converting to string
                    business_id = str(row.get(business_id_col)).split('.')[0]
                    
                    # Skip if the station is not in our main data
                    if business_id not in self.stations:
                        logger.debug(f"Business ID {business_id} not found in main data")
                        continue
                        
                    # Get out of service pump count (numeric)
                    oos_pumps_col = col_map.get('oos_pumps')
                    if oos_pumps_col and not pd.isna(row.get(oos_pumps_col)):
                        try:
                            num_oos = int(float(row.get(oos_pumps_col)))
                            if num_oos > 0:
                                self.stations[business_id].out_of_service_pumps = num_oos
                                logger.debug(f"Set {num_oos} OOS pumps for station {business_id}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error converting OOS pumps for station {business_id}: {e}")
                    
                    # Get OOS details (text description)
                    oos_details_col = col_map.get('oos_details')
                    if oos_details_col and not pd.isna(row.get(oos_details_col)):
                        details = str(row.get(oos_details_col))
                        self.stations[business_id].out_of_service_details = details
                        logger.debug(f"Set OOS details for station {business_id}: {details}")
                    
                    # Get days out of service
                    days_oos_col = col_map.get('days_oos', 'Elapsed Days OOS')  # Fallback to explicit name
                    if days_oos_col and not pd.isna(row.get(days_oos_col)):
                        try:
                            days = int(float(row.get(days_oos_col)))
                            self.stations[business_id].days_out_of_service = days
                            logger.debug(f"Set {days} OOS days for station {business_id}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error converting OOS days for station {business_id}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Error processing OOS row: {e}")
                    
            logger.info(f"Processed out of service data")
        except Exception as e:
            logger.error(f"Error loading OOS data: {str(e)}", exc_info=True)
    
    def _identify_columns(self, df):
        """
        Identify column indices for relevant data
        Returns a dictionary mapping column purposes to actual column names
        """
        col_map = {}
        
        # Dictionary of keywords to look for in column names
        column_keywords = {
            'business_id': ['business id', 'id #'],
            'name': ['business name', 'name'],
            'address': ['address'],
            'city': ['city'],
            'state': ['state'],
            'zip': ['zip'],
            'county': ['county'],
            'num_pumps': ['no. of pumps', 'number of pumps'],
            'last_inspection_date': ['last inspection', 'last insp', 'regular inspect'],
            'notes': ['notes'],
            'viol_date': ['viol date', 'violation date'],
            'oos_pumps': ['oos pumps  112', 'oos pumps'],  # Updated to match exact column name
            'oos_details': ['out of service pumps'],  # This matches the details column
            'days_oos': ['elapsed days oos', 'elapsed days o']  # Handle truncated column name
        }
        
        # Helper function to convert column index to Excel column letter
        def col_index_to_letter(index):
            letters = ""
            while index > 0:
                index, remainder = divmod(index - 1, 26)
                letters = chr(65 + remainder) + letters
            return letters
            
        # Search for columns that match our keywords
        for purpose, keywords in column_keywords.items():
            for col in df.columns:
                col_str = str(col).lower()
                if any(keyword.lower() in col_str for keyword in keywords):
                    col_map[purpose] = col
                    break
        
        # Explicitly map column 'R' to the name field
        # Find the index of column 'R' (18th column, index 17)
        col_r_index = None
        for i, col in enumerate(df.columns):
            if col_index_to_letter(i+1) == 'R':
                col_r_index = i
                break
                
        if col_r_index is not None:
            col_map['name'] = df.columns[col_r_index]
            logger.debug(f"Explicitly mapping column 'R' ({df.columns[col_r_index]}) to name field")
        
        # Special case for common column positions if we couldn't find by name
        if 'business_id' not in col_map and len(df.columns) > 4:
            col_map['business_id'] = df.columns[4]  # Usually 5th column
        
        # If name wasn't set by column 'R', use the default fallback
        if 'name' not in col_map and len(df.columns) > 17:
            col_map['name'] = df.columns[17]  # Usually 18th column
            
        # Explicitly map column B (index 1) to last inspection date
        if 'last_inspection_date' not in col_map and len(df.columns) > 1:
            col_map['last_inspection_date'] = df.columns[1]  # Column B (index 1)
            logger.debug(f"Explicitly mapping column B ({df.columns[1]}) to last_inspection_date field")
            
        if 'address' not in col_map and len(df.columns) > 18:
            col_map['address'] = df.columns[18]  # Usually 19th column
            
        if 'city' not in col_map and len(df.columns) > 19:
            col_map['city'] = df.columns[19]  # Usually 20th column
            
        if 'state' not in col_map and len(df.columns) > 20:
            col_map['state'] = df.columns[20]  # Usually 21st column
            
        if 'zip' not in col_map and len(df.columns) > 21:
            col_map['zip'] = df.columns[21]  # Usually 22nd column
            
        if 'county' not in col_map and len(df.columns) > 22:
            col_map['county'] = df.columns[22]  # Usually 23rd column
            
        if 'notes' not in col_map and len(df.columns) > 29:
            col_map['notes'] = df.columns[29]  # Usually 30th column
            
        # Explicitly map the last inspection date column (based on user preference - column B)
        if 'last_inspection_date' not in col_map and len(df.columns) > 1:
            col_map['last_inspection_date'] = df.columns[1]  # Column B (index 1)
            logger.debug(f"Explicitly mapping column B ({df.columns[1]}) to last_inspection_date field")
            
        logger.debug(f"Column mapping: {col_map}")
        return col_map
    
    def get_all_stations(self):
        """Return a list of all stations"""
        return list(self.stations.values())
    
    def get_station_by_id(self, business_id):
        """Get a station by its business ID"""
        # Standardize the business ID by removing decimal points
        business_id = str(business_id).split('.')[0]
        return self.stations.get(business_id)
    
    def get_stations_by_status(self, reinspection=False, complaint=False, out_of_service=False):
        """Get stations filtered by their status"""
        result = []
        
        for station in self.stations.values():
            if reinspection and station.needs_reinspection:
                result.append(station)
            elif complaint and station.has_complaint:
                result.append(station)
            elif out_of_service and station.out_of_service_pumps > 0:
                result.append(station)
                
        return result
    
    def get_high_priority_stations(self, limit=None):
        """Get stations sorted by priority score (highest first)"""
        stations = list(self.stations.values())
        stations.sort(key=lambda s: s.priority_score, reverse=True)
        
        if limit:
            return stations[:limit]
        return stations
    
    def __len__(self):
        """Return the number of stations"""
        return len(self.stations)
        
    def save_to_file(self, filepath):
        """
        Save the processed station data to a file
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save the stations dictionary to a pickle file
            with open(filepath, 'wb') as f:
                pickle.dump(self.stations, f)
            
            logger.info(f"Successfully saved {len(self.stations)} stations to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving station data: {str(e)}", exc_info=True)
            return False
    
    def load_from_file(self, filepath):
        """
        Load station data from a saved file
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Station data file not found: {filepath}")
                return False
                
            # Load the stations dictionary from the pickle file
            with open(filepath, 'rb') as f:
                self.stations = pickle.load(f)
            
            # Reset today's date to ensure days_since_inspection is current
            self.today = datetime.now().date()
            
            # Update days_since_inspection for all stations
            for station in self.stations.values():
                if station.last_inspection_date and isinstance(station.last_inspection_date, datetime):
                    days_diff = (self.today - station.last_inspection_date.date()).days
                    station.days_since_inspection = max(0, days_diff)
            
            logger.info(f"Successfully loaded {len(self.stations)} stations from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading station data: {str(e)}", exc_info=True)
            return False
            
    def to_dataframe(self):
        """
        Convert the stations to a pandas DataFrame for easy editing
        """
        data = []
        for station in self.stations.values():
            # Extract the key attributes of each station
            station_data = {
                'business_id': station.business_id,
                'name': station.name,
                'address': station.address,
                'city': station.city,
                'state': station.state,
                'zip_code': station.zip_code,
                'county': station.county,
                'num_pumps': station.num_pumps,
                'last_inspection_date': station.last_inspection_date,
                'days_since_inspection': station.days_since_inspection,
                'needs_reinspection': station.needs_reinspection,
                'reinspection_reason': station.reinspection_reason,
                'has_complaint': station.has_complaint,
                'complaint_details': station.complaint_details,
                'out_of_service_pumps': station.out_of_service_pumps,
                'out_of_service_details': station.out_of_service_details,
                'priority_score': station.priority_score
            }
            data.append(station_data)
        
        # Create DataFrame from the list of dictionaries
        df = pd.DataFrame(data)
        return df
    
    def update_from_dataframe(self, df):
        """
        Update stations from a DataFrame (used after editing)
        """
        # Track how many stations were updated
        updated_count = 0
        
        # Process each row in the DataFrame
        for _, row in df.iterrows():
            business_id = str(row['business_id']).split('.')[0]
            
            # Check if this station exists
            if business_id in self.stations:
                station = self.stations[business_id]
                
                # Update basic attributes
                station.name = row['name']
                station.address = row['address']
                station.city = row['city']
                station.state = row['state']
                station.zip_code = row['zip_code']
                station.county = row['county']
                station.num_pumps = row['num_pumps']
                
                # Update status attributes
                station.needs_reinspection = row['needs_reinspection']
                station.reinspection_reason = row['reinspection_reason']
                station.has_complaint = row['has_complaint']
                station.complaint_details = row['complaint_details']
                station.out_of_service_pumps = row['out_of_service_pumps']
                station.out_of_service_details = row['out_of_service_details']
                
                # Recalculate priority
                station.calculate_priority()
                
                updated_count += 1
        
        logger.info(f"Updated {updated_count} stations from DataFrame")
        return updated_count
