class Station:
    """
    Represents a station with all its properties and statuses
    across different sheets in the Excel file.
    """
    def __init__(self, business_id=None, name=None, address=None, city=None, state=None, zip_code=None, county=None):
        # Basic identification
        self.business_id = business_id
        self.name = name
        self.address = address
        self.city = city
        self.state = state
        self.zip_code = zip_code
        self.county = county
        
        # Full address for geocoding
        self.full_address = None
        self.coordinates = None
        
        # Station details
        self.num_pumps = 2  # Default to 2 pumps if not specified
        self.inspection_time_min = 30  # Default inspection time in minutes
        
        # Last inspection data
        self.last_inspection_date = None
        self.days_since_inspection = None
        
        # Station statuses (each can be independently set)
        self.needs_reinspection = False
        self.reinspection_reason = None
        
        self.has_complaint = False
        self.complaint_details = None
        self.complaint_date = None
        
        self.has_out_of_service_pump = False
        self.out_of_service_pumps = 0
        self.out_of_service_details = None
        self.days_out_of_service = 0
        
        # Priority factors (for optimization)
        self.priority_score = 0
        
        # Additional tracking fields
        self.skipped = False
        self.notes = ""
        
    def update_full_address(self):
        """Generate the full address for geocoding"""
        address_parts = []
        if self.address:
            address_parts.append(self.address)
        if self.city:
            address_parts.append(self.city)
        if self.state:
            address_parts.append(self.state)
        if self.zip_code:
            address_parts.append(str(self.zip_code))
            
        self.full_address = ", ".join(address_parts)
        return self.full_address
    
    def calculate_priority(self):
        """
        Calculate a priority score for this station based on:
        - Days since last inspection
        - Whether it has complaints
        - Whether it has out-of-service pumps
        - Whether it needs reinspection
        
        Higher score = higher priority
        """
        score = 0
        
        # Base score from days since inspection (if available)
        if self.days_since_inspection:
            # Prioritize stations not inspected for a long time
            score += min(self.days_since_inspection / 30, 10)  # Cap at 10 points (about 10 months)
        
        # Boost for stations with issues
        if self.needs_reinspection:
            score += 10  # High priority for reinspections
        
        if self.has_complaint:
            score += 8   # High priority for complaints
            
        if self.out_of_service_pumps > 0:
            # Priority increases with number of OOS pumps and days out of service
            oos_factor = min(self.days_out_of_service / 30, 12)  # Cap at 12 points (about 1 year)
            score += (2 * self.out_of_service_pumps) + oos_factor
            
        self.priority_score = score
        return score
    
    def __str__(self):
        """String representation of the station"""
        status = []
        if self.needs_reinspection:
            status.append("NEEDS REINSPECTION")
        if self.has_complaint:
            status.append("HAS COMPLAINT")
        if self.out_of_service_pumps > 0:
            status.append(f"{self.out_of_service_pumps} PUMPS OUT OF SERVICE")
            
        status_str = " | ".join(status) if status else "OK"
        
        return f"{self.name} ({self.business_id}) - {self.full_address} - Status: {status_str}"
