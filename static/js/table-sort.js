let currentSort = {
    column: null,
    direction: 'asc'
};

function initTableSort() {
    const table = document.getElementById('addressTable');
    if (!table) return;

    const headers = table.querySelectorAll('th.sortable');
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const column = header.getAttribute('data-sort');
            sortTable(column, header);
        });
    });
}

function sortTable(column, header) {
    const table = document.getElementById('addressTable');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));

    // Reset all headers
    table.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('sorted-asc', 'sorted-desc');
    });

    // Update sort direction
    if (currentSort.column === column) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = column;
        currentSort.direction = 'asc';
    }

    // Add sort indicator to header
    header.classList.add(currentSort.direction === 'asc' ? 'sorted-asc' : 'sorted-desc');

    // Sort rows
    rows.sort((a, b) => {
        let valueA, valueB;

        if (column === 'status') {
            valueA = getStatusValue(a);
            valueB = getStatusValue(b);
        } else if (column === 'priority') {
            valueA = parseFloat(a.querySelector('.edit-priority-score').value) || 0;
            valueB = parseFloat(b.querySelector('.edit-priority-score').value) || 0;
        } else if (column === 'inspection') {
            valueA = a.querySelector('.edit-inspection-date').value || '0000-00-00';
            valueB = b.querySelector('.edit-inspection-date').value || '0000-00-00';
        } else {
            // Default text sorting for other columns
            valueA = a.children[getColumnIndex(column)].textContent.trim().toLowerCase();
            valueB = b.children[getColumnIndex(column)].textContent.trim().toLowerCase();
        }

        // Compare values
        if (valueA === valueB) return 0;
        if (currentSort.direction === 'asc') {
            return valueA < valueB ? -1 : 1;
        } else {
            return valueA > valueB ? -1 : 1;
        }
    });

    // Re-append rows in sorted order
    rows.forEach(row => tbody.appendChild(row));
}

function getStatusValue(row) {
    const statusCell = row.querySelector('.status-controls');
    if (!statusCell) return 0;

    let value = 0;
    if (statusCell.querySelector('.edit-needs-reinspection').checked) value += 4;
    if (statusCell.querySelector('.edit-has-complaint').checked) value += 2;
    if (statusCell.querySelector('.edit-has-out-of-service').checked) value += 1;
    return value;
}

function getColumnIndex(column) {
    const columnMap = {
        'business-id': 0,
        'name': 1,
        'address': 2,
        'city': 3,
        'state': 4,
        'zip': 5,
        'county': 6
    };
    return columnMap[column] || 0;
}

// Initialize sorting when the document loads
document.addEventListener('DOMContentLoaded', initTableSort);