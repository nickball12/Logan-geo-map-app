@app.route('/update_station', methods=['POST'])
def update_station():
    """Update a station field"""
    try:
        data = request.json
        business_id = data.get('business_id')
        field = data.get('field')
        value = data.get('value')
        
        if not business_id or not field:
            return jsonify({'success': False, 'error': 'Missing business_id or field'})
        
        # Update the station in the data store
        success = data_store.update_station(business_id, field, value)
        
        if success:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Failed to update station'})
    
    except Exception as e:
        logger.error(f"Error updating station: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})
