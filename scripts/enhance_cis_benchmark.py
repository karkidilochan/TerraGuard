#!/usr/bin/env python3

import json
import re
import os
from typing import Dict, List, Any, Optional

# Define mappings between CIS controls and AWS resource types
# This is based on analysis of the control content
RESOURCE_TYPE_MAPPINGS = {
    # IAM-related controls (1.x)
    r"^1\.": {
        "resource_type": ["aws_iam_account_password_policy", "aws_iam_user", "aws_iam_role", "aws_iam_policy"],
        "patterns": {
            r"password policy": ["aws_iam_account_password_policy"],
            r"'root' user|root.+account": ["aws_iam_account_alias"],
            r"IAM user|IAM role|IAM policy": ["aws_iam_user", "aws_iam_role", "aws_iam_policy"],
            r"MFA": ["aws_iam_user"],
            r"access key": ["aws_iam_access_key"],
            r"Support": ["aws_iam_role"],
            r"SSL/TLS certificate": ["aws_iam_server_certificate"],
            r"IAM instance role": ["aws_iam_role", "aws_iam_instance_profile"],
            r"IAM Access Analyzer": ["aws_accessanalyzer_analyzer"]
        }
    },
    # S3-related controls (2.x)
    r"^2\.": {
        "resource_type": ["aws_s3_bucket"],
        "patterns": {
            r"bucket public access": ["aws_s3_bucket_public_access_block"],
            r"encryption": ["aws_s3_bucket_server_side_encryption_configuration"],
            r"versioning": ["aws_s3_bucket_versioning"],
            r"logging": ["aws_s3_bucket_logging"],
            r"event notifications": ["aws_s3_bucket_notification"],
            r"replication": ["aws_s3_bucket_replication_configuration"]
        }
    },
    # Logging-related controls (3.x and 4.x)
    r"^3\.|^4\.": {
        "resource_type": ["aws_cloudtrail", "aws_cloudwatch_log_group", "aws_config_configuration_recorder"],
        "patterns": {
            r"CloudTrail": ["aws_cloudtrail"],
            r"CloudWatch": ["aws_cloudwatch_log_group", "aws_cloudwatch_metric_alarm"],
            r"VPC flow log": ["aws_flow_log"],
            r"Guard ?Duty": ["aws_guardduty_detector"],
            r"Config": ["aws_config_configuration_recorder"],
            r"Security Hub": ["aws_securityhub_account"],
            r"encryption|encrypted": ["aws_kms_key", "aws_cloudtrail", "aws_s3_bucket_server_side_encryption_configuration"]
        }
    },
    # Networking-related controls (5.x)
    r"^5\.": {
        "resource_type": ["aws_vpc", "aws_security_group", "aws_network_acl"],
        "patterns": {
            r"security group": ["aws_security_group"],
            r"Network ACL": ["aws_network_acl"],
            r"VPC": ["aws_vpc"],
            r"routing tables": ["aws_route_table"],
            r"peering": ["aws_vpc_peering_connection"],
            r"subnet": ["aws_subnet"],
            r"ingress|egress": ["aws_security_group", "aws_network_acl"]
        }
    }
}

# Define attribute mappings for specific controls
RELEVANT_ATTRIBUTES = {
    "1.4": ["access_key_id", "status"],
    "1.5": ["mfa_enabled"],
    "1.8": ["minimum_password_length"],
    "1.9": ["password_reuse_prevention"],
    "1.10": ["mfa_enabled"],
    "1.13": ["status"],
    "1.14": ["create_date"],
    "1.16": ["policy", "policy_arn"],
    "1.20": ["analyzer_name", "type"],
    "2.1.1": ["block_public_acls", "block_public_policy", "ignore_public_acls", "restrict_public_buckets"],
    "2.1.2": ["bucket", "encrypted", "sse_algorithm"],
    "2.2": ["versioning", "enabled"],
    "3.5": ["kms_key_id", "encrypted"],
    "4.1": ["is_multi_region_trail", "include_global_service_events", "is_logging"],
    "4.2": ["kms_key_id"],
    "4.6": ["metric_name", "pattern", "threshold", "actions_enabled"],
    "5.1": ["filter", "log_destination"],
    "5.3": ["ingress", "from_port", "to_port", "cidr_blocks"],
    "5.4": ["ingress", "from_port", "to_port", "ipv6_cidr_blocks"]
}

# Map CIS controls to Checkov policies where possible
CHECKOV_POLICY_MAPPINGS = {
    "1.4": ["CKV_AWS_1"],
    "1.5": ["CKV_AWS_41"],
    "1.8": ["CKV_AWS_10"],
    "1.9": ["CKV_AWS_13"],
    "1.10": ["CKV_AWS_41"],
    "1.16": ["CKV_AWS_62", "CKV_AWS_63"],
    "1.20": ["CKV_AWS_85"],
    "2.1.1": ["CKV_AWS_53", "CKV_AWS_54", "CKV_AWS_55", "CKV_AWS_56"],
    "2.1.2": ["CKV_AWS_19"],
    "2.2": ["CKV_AWS_21"],
    "4.1": ["CKV_AWS_67", "CKV_AWS_153"],
    "4.2": ["CKV_AWS_35"],
    "5.2": ["CKV_AWS_24"],
    "5.3": ["CKV_AWS_24"],
    "5.4": ["CKV_AWS_25"],
    "5.5": ["CKV2_AWS_12"]
}

def identify_resource_types(control: Dict[str, Any]) -> List[str]:
    """Identify AWS resource types for a CIS control based on content analysis"""
    resource_types = set()
    
    # Check control ID against section patterns
    for section_pattern, mapping in RESOURCE_TYPE_MAPPINGS.items():
        if re.match(section_pattern, control["control_id"]):
            # Add default resource types for this section
            for rt in mapping["resource_type"]:
                resource_types.add(rt)
            
            # Check for specific patterns in the description and title
            text_to_analyze = f"{control['title']} {control.get('description', '')}"
            for pattern, resources in mapping["patterns"].items():
                if re.search(pattern, text_to_analyze, re.IGNORECASE):
                    for rt in resources:
                        resource_types.add(rt)
    
    return list(resource_types)

def get_relevant_attributes(control_id: str) -> List[str]:
    """Get relevant Terraform attributes for a specific CIS control"""
    # Remove any non-numeric suffix from control ID for mapping purposes
    base_id = re.sub(r'(\d+\.\d+).*', r'\1', control_id)
    
    # Return predefined attributes if they exist
    if control_id in RELEVANT_ATTRIBUTES:
        return RELEVANT_ATTRIBUTES[control_id]
    elif base_id in RELEVANT_ATTRIBUTES:
        return RELEVANT_ATTRIBUTES[base_id]
    
    # Return empty list if no specific attributes are defined
    return []

def get_checkov_policies(control_id: str) -> List[str]:
    """Get matching Checkov policies for a specific CIS control"""
    # Remove any non-numeric suffix from control ID for mapping purposes
    base_id = re.sub(r'(\d+\.\d+).*', r'\1', control_id)
    
    if control_id in CHECKOV_POLICY_MAPPINGS:
        return CHECKOV_POLICY_MAPPINGS[control_id]
    elif base_id in CHECKOV_POLICY_MAPPINGS:
        return CHECKOV_POLICY_MAPPINGS[base_id]
    
    return []

def enhance_cis_benchmark(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Enhance CIS benchmark JSON with resource_type and relevant_attributes fields
    
    Args:
        input_file: Path to the input CIS benchmark JSON file
        output_file: Path to the output enhanced JSON file (defaults to input_file if None)
    """
    if output_file is None:
        output_file = input_file
    
    try:
        with open(input_file, 'r') as f:
            benchmarks = json.load(f)
        
        enhanced_benchmarks = []
        
        for benchmark in benchmarks:
            # Add resource_type field
            benchmark["resource_type"] = identify_resource_types(benchmark)
            
            # Add relevant_attributes field
            benchmark["relevant_attributes"] = get_relevant_attributes(benchmark["control_id"])
            
            # Add checkov_policies field
            benchmark["checkov_policies"] = get_checkov_policies(benchmark["control_id"])
            
            enhanced_benchmarks.append(benchmark)
        
        with open(output_file, 'w') as f:
            json.dump(enhanced_benchmarks, f, indent=2)
        
        print(f"Enhanced CIS benchmark saved to {output_file}")
        print(f"Added metadata to {len(enhanced_benchmarks)} controls")
        
    except Exception as e:
        print(f"Error enhancing CIS benchmark: {str(e)}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "cis_benchmark.json")
    output_file = os.path.join(script_dir, "cis_benchmark_enhanced.json")
    
    enhance_cis_benchmark(input_file, output_file) 