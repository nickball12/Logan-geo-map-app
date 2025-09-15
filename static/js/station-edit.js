function editStation(businessId) {
    const row = document.querySelector(`tr[data-business-id="${businessId}"]`);
    const lastInspectionDate = row.querySelector('.edit-inspection-date').value;
    const priorityScore = row.querySelector('.edit-priority-score').value;
    const needsReinspection = row.querySelector('.edit-needs-reinspection').checked;
    const hasComplaint = row.querySelector('.edit-has-complaint').checked;
    const hasOutOfService = row.querySelector('.edit-has-out-of-service').checked;
    const skipped = row.querySelector('.edit-skipped').checked;
    const notes = row.querySelector('.edit-notes').value;

    fetch('/edit_station', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            business_id: businessId,
            last_inspection_date: lastInspectionDate,
            priority_score: priorityScore,
            needs_reinspection: needsReinspection,
            has_complaint: hasComplaint,
            has_out_of_service: hasOutOfService,
            skipped: skipped,
            notes: notes
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Station updated successfully');
        } else {
            alert('Error updating station: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error updating station');
    });
}