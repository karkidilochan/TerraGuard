#!/usr/bin/env python3

import json
import os
import re
from collections import Counter
from typing import Dict, List, Any

def load_enhanced_benchmark(file_path: str) -> List[Dict[str, Any]]:
    """Load the enhanced CIS benchmark from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_resource_types(controls: List[Dict[str, Any]]) -> None:
    """Validate resource type mappings for each control"""
    print("\n===== Resource Type Validation =====")
    
    # Count resource types across all controls
    resource_counter = Counter()
    for control in controls:
        resource_types = control.get("resource_type", [])
        for rt in resource_types:
            resource_counter[rt] += 1
    
    # Print the most common resource types
    print(f"Found {len(resource_counter)} unique resource types")
    print("\nTop 10 most common resource types:")
    for rt, count in resource_counter.most_common(10):
        print(f"  {rt}: {count} controls")
    
    # Check for controls without resource types
    missing_resources = [c["control_id"] for c in controls if len(c.get("resource_type", [])) == 0]
    if missing_resources:
        print(f"\nWarning: {len(missing_resources)} controls have no resource types:")
        for control_id in missing_resources[:5]:  # Show only first 5
            print(f"  - {control_id}")
        if len(missing_resources) > 5:
            print(f"  ... and {len(missing_resources) - 5} more")

def validate_attributes(controls: List[Dict[str, Any]]) -> None:
    """Validate relevant attributes for each control"""
    print("\n===== Attribute Validation =====")
    
    # Count controls with attributes
    with_attrs = [c for c in controls if len(c.get("relevant_attributes", [])) > 0]
    print(f"{len(with_attrs)} of {len(controls)} controls have specified relevant attributes")
    
    # Show examples of controls with attributes
    if with_attrs:
        print("\nExamples of controls with attributes:")
        for control in with_attrs[:5]:  # Show only first 5
            print(f"  {control['control_id']} ({control['title']}): {control['relevant_attributes']}")

def validate_checkov_policies(controls: List[Dict[str, Any]]) -> None:
    """Validate Checkov policy mappings"""
    print("\n===== Checkov Policy Validation =====")
    
    # Count controls with Checkov policies
    with_policies = [c for c in controls if len(c.get("checkov_policies", [])) > 0]
    print(f"{len(with_policies)} of {len(controls)} controls have Checkov policy mappings")
    
    # Show examples of controls with Checkov policies
    if with_policies:
        print("\nExamples of controls with Checkov policies:")
        for control in with_policies[:5]:  # Show only first 5
            print(f"  {control['control_id']} ({control['title']}): {control['checkov_policies']}")
    
    # Count the most common Checkov policies
    policy_counter = Counter()
    for control in controls:
        policies = control.get("checkov_policies", [])
        for policy in policies:
            policy_counter[policy] += 1
    
    if policy_counter:
        print("\nMost common Checkov policies:")
        for policy, count in policy_counter.most_common(5):
            print(f"  {policy}: {count} controls")

def check_consistency(controls: List[Dict[str, Any]]) -> None:
    """Check consistency between control sections and resource types"""
    print("\n===== Consistency Validation =====")
    
    section_patterns = {
        r"^1\.": "IAM",
        r"^2\.": "Storage",
        r"^3\.|^4\.": "Logging/Monitoring",
        r"^5\.": "Networking"
    }
    
    expected_resources = {
        "IAM": ["aws_iam_", "aws_accessanalyzer_"],
        "Storage": ["aws_s3_"],
        "Logging/Monitoring": ["aws_cloudtrail", "aws_cloudwatch_", "aws_config_"],
        "Networking": ["aws_vpc", "aws_security_group", "aws_network_acl", "aws_route"]
    }
    
    inconsistencies = []
    
    for control in controls:
        control_id = control["control_id"]
        resource_types = control.get("resource_type", [])
        
        # Determine expected section
        section = None
        for pattern, sec_name in section_patterns.items():
            if re.match(pattern, control_id):
                section = sec_name
                break
        
        if section and resource_types:
            # Check if resource types match expected patterns for the section
            matches_section = False
            for resource in resource_types:
                for pattern in expected_resources.get(section, []):
                    if pattern in resource:
                        matches_section = True
                        break
                if matches_section:
                    break
            
            if not matches_section:
                inconsistencies.append((control_id, section, resource_types))
    
    if inconsistencies:
        print(f"Found {len(inconsistencies)} potential inconsistencies between control sections and resource types:")
        for control_id, section, resources in inconsistencies[:10]:  # Show only first 10
            print(f"  Control {control_id} (Section: {section}) has unexpected resources: {resources}")
        if len(inconsistencies) > 10:
            print(f"  ... and {len(inconsistencies) - 10} more")
    else:
        print("No major inconsistencies found between control sections and resource types")

def verify_known_checkov_mappings(controls: List[Dict[str, Any]]) -> None:
    """Verify Checkov policy mappings against known CIS controls from AWS documentation"""
    print("\n===== Checkov to CIS Mapping Verification =====")
    
    # Known mappings from Checkov to CIS Benchmark controls
    known_mappings = {
        # IAM control mappings
        "1.1": [],  # Maintain current contact details
        "1.4": ["CKV_AWS_1"],  # IAM root user access key
        "1.5": ["CKV_AWS_41"],  # MFA for root user
        "1.8": ["CKV_AWS_10"],  # IAM password policy length
        "1.9": ["CKV_AWS_13"],  # IAM password reuse prevention
        "1.10": ["CKV_AWS_41"],  # MFA for IAM users
        "1.16": ["CKV_AWS_62", "CKV_AWS_63"],  # No full admin privileges
        "1.20": ["CKV_AWS_85"],  # IAM Access Analyzer enabled
        
        # S3 control mappings
        "2.1.1": ["CKV_AWS_53", "CKV_AWS_54", "CKV_AWS_55", "CKV_AWS_56"],  # S3 public access
        "2.1.2": ["CKV_AWS_19"],  # S3 encryption
        "2.2": ["CKV_AWS_21"],  # S3 versioning
        
        # CloudTrail control mappings
        "4.1": ["CKV_AWS_67", "CKV_AWS_153"],  # CloudTrail enabled
        "4.2": ["CKV_AWS_35"],  # CloudTrail encryption
        
        # Networking control mappings
        "5.2": ["CKV_AWS_24"],  # Security group SSH restrictions
        "5.3": ["CKV_AWS_24"],  # Security group remote access
        "5.4": ["CKV_AWS_25"],  # IPv6 remote access restrictions
        "5.5": ["CKV2_AWS_12"]  # Default security group restrictions
    }
    
    # Verify mappings in the enhanced benchmark
    verification_results = []
    for control in controls:
        control_id = control["control_id"]
        # Find base control ID without subparts (e.g., "1.2.3" -> "1.2")
        base_id = re.sub(r'(\d+\.\d+).*', r'\1', control_id)
        
        # Get expected and actual Checkov policies
        expected_policies = known_mappings.get(control_id, known_mappings.get(base_id, []))
        actual_policies = control.get("checkov_policies", [])
        
        # Compare expected vs. actual
        if expected_policies:
            # Missing policies
            missing = [p for p in expected_policies if p not in actual_policies]
            
            # Extra policies
            extra = [p for p in actual_policies if p not in expected_policies]
            
            if missing or extra:
                verification_results.append({
                    "control_id": control_id,
                    "missing": missing,
                    "extra": extra,
                    "expected": expected_policies,
                    "actual": actual_policies
                })
    
    if verification_results:
        print(f"Found {len(verification_results)} discrepancies in Checkov mappings:")
        for result in verification_results[:10]:  # Show only first 10
            print(f"  Control {result['control_id']}:")
            if result['missing']:
                print(f"    Missing policies: {result['missing']}")
            if result['extra']:
                print(f"    Extra policies: {result['extra']}")
            print(f"    Expected: {result['expected']}, Actual: {result['actual']}")
        if len(verification_results) > 10:
            print(f"  ... and {len(verification_results) - 10} more")
    else:
        print("All known Checkov mappings are correctly defined")

def suggest_improvements(controls: List[Dict[str, Any]]) -> None:
    """Suggest potential improvements to the enhanced benchmark"""
    print("\n===== Improvement Suggestions =====")
    
    # Identify controls that could benefit from additional resource types
    potential_improvements = []
    
    for control in controls:
        # Example: IAM controls mentioning "role" but not having aws_iam_role
        if re.match(r"^1\.", control["control_id"]) and "role" in control["title"].lower():
            if "aws_iam_role" not in control.get("resource_type", []):
                potential_improvements.append((control["control_id"], "Consider adding aws_iam_role"))
        
        # Example: Controls mentioning "encryption" but not having resource types for encryption
        if "encryption" in control.get("description", "").lower() and not any("encryption" in rt for rt in control.get("resource_type", [])):
            potential_improvements.append((control["control_id"], "Consider adding encryption-related resource type"))
        
        # S3 controls missing versioning attributes
        if re.match(r"^2\.2", control["control_id"]) and "versioning" in control["title"].lower():
            if not any("versioning" in attr for attr in control.get("relevant_attributes", [])):
                potential_improvements.append((control["control_id"], "Consider adding versioning-related attributes"))
    
    if potential_improvements:
        print(f"Found {len(potential_improvements)} potential improvements:")
        for control_id, suggestion in potential_improvements[:10]:  # Show only first 10
            print(f"  Control {control_id}: {suggestion}")
        if len(potential_improvements) > 10:
            print(f"  ... and {len(potential_improvements) - 10} more")
    else:
        print("No obvious improvements identified")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enhanced_file = os.path.join(script_dir, "cis_benchmark_enhanced.json")
    
    if not os.path.exists(enhanced_file):
        print(f"Error: Enhanced CIS benchmark file not found at {enhanced_file}")
        return
    
    print(f"Validating enhanced CIS benchmark at {enhanced_file}")
    controls = load_enhanced_benchmark(enhanced_file)
    print(f"Loaded {len(controls)} controls from the enhanced benchmark")
    
    # Run validations
    validate_resource_types(controls)
    validate_attributes(controls)
    validate_checkov_policies(controls)
    check_consistency(controls)
    
    # Check for specific fixes
    print("\n===== Validation Summary =====")
    issues_fixed = True
    
    # Check if control 1.16 has correct Checkov policy
    control_1_16 = next((c for c in controls if c["control_id"] == "1.16"), None)
    if control_1_16 and "CKV_AWS_62" in control_1_16.get("checkov_policies", []):
        print("✓ Control 1.16 correctly maps to CKV_AWS_62")
    else:
        print("✗ Control 1.16 does not correctly map to CKV_AWS_62")
        issues_fixed = False
    
    # Check if control 5.2 has correct Checkov policy
    control_5_2 = next((c for c in controls if c["control_id"] == "5.2"), None)
    if control_5_2 and "CKV_AWS_24" in control_5_2.get("checkov_policies", []):
        print("✓ Control 5.2 correctly maps to CKV_AWS_24")
    else:
        print("✗ Control 5.2 does not correctly map to CKV_AWS_24")
        issues_fixed = False
    
    # Check if control 5.5 has correct Checkov policy
    control_5_5 = next((c for c in controls if c["control_id"] == "5.5"), None)
    if control_5_5 and "CKV2_AWS_12" in control_5_5.get("checkov_policies", []):
        print("✓ Control 5.5 correctly maps to CKV2_AWS_12")
    else:
        print("✗ Control 5.5 does not correctly map to CKV2_AWS_12")
        issues_fixed = False
    
    # Check if control 3.5 has encryption-related resource type
    control_3_5 = next((c for c in controls if c["control_id"] == "3.5"), None)
    if control_3_5 and any("encryption" in rt or "kms" in rt for rt in control_3_5.get("resource_type", [])):
        print("✓ Control 3.5 includes encryption-related resource types")
    else:
        print("✗ Control 3.5 does not include encryption-related resource types")
        issues_fixed = False
    
    # Overall validation result
    if issues_fixed:
        print("\nAll previously identified issues have been fixed! ✓")
    else:
        print("\nSome issues still need to be addressed. ✗")
    
    verify_known_checkov_mappings(controls)
    suggest_improvements(controls)
    
    print("\n===== Validation Complete =====")
    print("The enhanced benchmark appears to be formatted correctly.")
    print("For complete verification, sample several controls manually and verify their resource types and attributes against AWS documentation.")

if __name__ == "__main__":
    main() 