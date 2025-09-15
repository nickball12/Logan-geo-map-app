import unittest
import os
import json
import pandas as pd
import app as flask_app
import shutil

class TestRouteSaving(unittest.TestCase):
    
    def setUp(self):
        # Create test output directory
        self.output_dir = "test_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Sample route data
        self.route_plan = [
            {
                'Name': 'Start Location',
                'Full Address': '123 Main St, Logan, UT 84321',
                'Coordinates': '41.7452, -111.8097',
                'Inspection Time (min)': 30
            },
            {
                'Name': 'Stop 1',
                'Full Address': '456 Center St, Logan, UT 84321',
                'Coordinates': '41.7398, -111.8337',
                'Inspection Time (min)': 20
            },
            {
                'Name': 'Stop 2',
                'Full Address': '789 North St, Logan, UT 84321',
                'Coordinates': '41.7550, -111.8150',
                'Inspection Time (min)': 15
            }
        ]
        
        # Convert to pandas Series for each item
        self.pd_route_plan = []
        for item in self.route_plan:
            self.pd_route_plan.append(pd.Series(item))
    
    def test_save_route_to_file(self):
        # Call the function with our test data
        file_path = flask_app.save_route_to_file(self.pd_route_plan, 65, 2, 5)
        
        # Check that file was created
        self.assertIsNotNone(file_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Check that the latest_route.json file was also created
        latest_file = os.path.join("output", "latest_route.json")
        self.assertTrue(os.path.exists(latest_file))
        
        # Verify file contents
        with open(latest_file, 'r') as f:
            data = json.load(f)
            
            # Check basic structure
            self.assertTrue('route_plan' in data)
            self.assertTrue('total_time_minutes' in data)
            self.assertTrue('skipped_stops' in data)
            self.assertTrue('total_stops' in data)
            
            # Check values
            self.assertEqual(len(data['route_plan']), 3)
            self.assertEqual(data['total_time_minutes'], 65)
            self.assertEqual(data['skipped_stops'], 2)
            self.assertEqual(data['total_stops'], 5)
            self.assertEqual(data['skipped_percentage'], 40.0)
            
            # Check first stop's data
            self.assertEqual(data['route_plan'][0]['Name'], 'Start Location')
            self.assertEqual(data['route_plan'][0]['Full Address'], '123 Main St, Logan, UT 84321')
    
    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        
        # Also clean up the real output directory
        output_dir = "output"
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.startswith("route_plan_") or f == "latest_route.json":
                    os.remove(os.path.join(output_dir, f))

if __name__ == '__main__':
    unittest.main()
