function filterAddresses() {
    console.log('Filtering addresses...'); // Debug
    
    // Safely get element values with defaults
    const searchBox = document.getElementById('searchBox');
    const duplicateFilterEl = document.getElementById('duplicateFilter');
    const searchText = searchBox ? searchBox.value.toLowerCase() : '';
    const duplicateFilter = duplicateFilterEl ? duplicateFilterEl.value : 'all';
    const rows = document.querySelectorAll('#addressTable tbody tr');
    
    // Get filter states with null checks
    const filterNormal = document.getElementById('filter-normal');
    const filterReinspection = document.getElementById('filter-reinspection');
    const filterComplaint = document.getElementById('filter-complaint');
    const filterOutOfService = document.getElementById('filter-out-of-service');
    
    const showNormal = filterNormal ? filterNormal.checked : true;
    const showReinspection = filterReinspection ? filterReinspection.checked : true;
    const showComplaint = filterComplaint ? filterComplaint.checked : true;
    const showOutOfService = filterOutOfService ? filterOutOfService.checked : true;
    
    console.log('Filter states:', { showNormal, showReinspection, showComplaint, showOutOfService }); // Debug

    let visibleCount = 0;
    rows.forEach(row => {
        const addressText = row.textContent.toLowerCase();
        const isDuplicate = row.getAttribute('data-duplicate') === 'true';
        let showRow = addressText.includes(searchText);

        // Apply duplicate filter
        if (duplicateFilter === 'duplicates' && !isDuplicate) {
            showRow = false;
        } else if (duplicateFilter === 'unique' && isDuplicate) {
            showRow = false;
        }

        // Apply status filter
        if (showRow) {
            const statusControls = row.querySelector('.edit-controls.status-controls');
            console.log('Status controls found:', !!statusControls); // Debug
            
            if (statusControls) {
                const hasReinspection = statusControls.querySelector('.edit-needs-reinspection').checked;
                const hasComplaint = statusControls.querySelector('.edit-has-complaint').checked;
                const hasOutOfService = statusControls.querySelector('.edit-has-out-of-service').checked;
                const isNormal = !hasReinspection && !hasComplaint && !hasOutOfService;

                console.log('Station statuses:', { hasReinspection, hasComplaint, hasOutOfService, isNormal }); // Debug

                // Show the row if any enabled filter matches the station's status
                showRow = (isNormal && showNormal) ||
                         (hasReinspection && showReinspection) ||
                         (hasComplaint && showComplaint) ||
                         (hasOutOfService && showOutOfService);
            }
        }

        row.style.display = showRow ? '' : 'none';
        if (showRow) visibleCount++;
    });
    
    console.log('Visible rows:', visibleCount); // Debug
}

// Function to handle status changes when editing a station
function handleStationStatusChange(container) {
    const specialStatusCheckboxes = [
        container.querySelector('.edit-needs-reinspection'),
        container.querySelector('.edit-has-complaint'),
        container.querySelector('.edit-has-out-of-service')
    ].filter(Boolean); // Filter out null values

    // If any special status is checked, normal is implicitly false
    const hasSpecialStatus = specialStatusCheckboxes.some(cb => cb.checked);
    
    // Update button state to show changes are pending
    const saveButton = container.closest('tr').querySelector('.edit-button');
    if (saveButton) {
        saveButton.classList.add('pending');
    }
}

// Initialize the filter controls and add event listeners
function initializeFilters() {
    console.log('Initializing filters...'); // Debug

    // Wait for the table to be available
    const table = document.getElementById('addressTable');
    if (!table) {
        console.log('Table not found, retrying in 100ms...'); // Debug
        setTimeout(initializeFilters, 100);
        return;
    }

    console.log('Table found, setting up filters...'); // Debug

    // Initialize all status filters as checked
    const statusCheckboxes = document.querySelectorAll('.status-filter-group input[type="checkbox"]');
    statusCheckboxes.forEach(checkbox => {
        if (checkbox) {
            checkbox.checked = true;
            // Add change event listener to each status filter checkbox
            checkbox.addEventListener('change', filterAddresses);
        }
    });

    // Add event listeners for other filters
    const searchBox = document.getElementById('searchBox');
    const duplicateFilter = document.getElementById('duplicateFilter');

    if (searchBox) {
        searchBox.addEventListener('input', filterAddresses);
    }

    if (duplicateFilter) {
        duplicateFilter.addEventListener('change', filterAddresses);
    }

    // Initial filter application only if all elements are present
    if (searchBox && duplicateFilter && statusCheckboxes.length > 0) {
        console.log('All elements found, applying initial filter...'); // Debug
        filterAddresses();
    }
}

// Start initialization when DOM is ready
document.addEventListener('DOMContentLoaded', initializeFilters);