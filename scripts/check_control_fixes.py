#!/usr/bin/env python3

import json
import os

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enhanced_file = os.path.join(script_dir, "cis_benchmark_enhanced.json")
    
    with open(enhanced_file, 'r') as f:
        controls = json.load(f)
    
    # Check control 1.16
    control_1_16 = next((c for c in controls if c["control_id"] == "1.16"), None)
    print(f"Control 1.16 Checkov Policies: {control_1_16.get('checkov_policies', [])}")
    
    # Check control 5.2
    control_5_2 = next((c for c in controls if c["control_id"] == "5.2"), None)
    print(f"Control 5.2 Checkov Policies: {control_5_2.get('checkov_policies', [])}")
    
    # Check control 5.5
    control_5_5 = next((c for c in controls if c["control_id"] == "5.5"), None)
    print(f"Control 5.5 Checkov Policies: {control_5_5.get('checkov_policies', [])}")
    
    # Check control 3.5 resource types
    control_3_5 = next((c for c in controls if c["control_id"] == "3.5"), None)
    print(f"Control 3.5 Resource Types: {control_3_5.get('resource_type', [])}")
    print(f"Control 3.5 Relevant Attributes: {control_3_5.get('relevant_attributes', [])}")

if __name__ == "__main__":
    main() 