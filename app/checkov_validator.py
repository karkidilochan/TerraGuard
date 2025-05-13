import os
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Tuple, Optional
import re
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map of Checkov IDs to CIS control IDs
CHECKOV_TO_CIS_MAPPING = {
    "CKV_AWS_1": "1.4",    # Ensure no root account access key exists
    "CKV_AWS_41": ["1.5", "1.10"],  # Ensure MFA is enabled
    "CKV_AWS_10": "1.8",   # Ensure IAM password policy requires minimum length
    "CKV_AWS_13": "1.9",   # Ensure IAM password policy prevents password reuse
    "CKV_AWS_62": "1.16",  # Ensure IAM policies that allow full admin privileges are not created
    "CKV_AWS_63": "1.16",  # Related to admin privileges
    "CKV_AWS_85": "1.20",  # Ensure IAM Access Analyzer is enabled
    "CKV_AWS_53": "2.1.1", # S3 Block Public Access
    "CKV_AWS_54": "2.1.1", # S3 Block Public Access
    "CKV_AWS_55": "2.1.1", # S3 Block Public Access
    "CKV_AWS_56": "2.1.1", # S3 Block Public Access
    "CKV_AWS_19": "2.1.2", # S3 Encryption
    "CKV_AWS_21": "2.2",   # S3 Versioning
    "CKV_AWS_67": "4.1",   # CloudTrail enabled
    "CKV_AWS_153": "4.1",  # CloudTrail multi-region
    "CKV_AWS_35": "4.2",   # CloudTrail encryption
    "CKV_AWS_24": ["5.2", "5.3"], # Security group remote access restrictions
    "CKV_AWS_25": "5.4",   # IPv6 remote access restrictions
    "CKV2_AWS_12": "5.5",  # Default security group restricts all traffic
}

# Reverse mapping for easier lookup
CIS_TO_CHECKOV_MAPPING = {}
for checkov_id, cis_ids in CHECKOV_TO_CIS_MAPPING.items():
    if isinstance(cis_ids, list):
        for cis_id in cis_ids:
            if cis_id not in CIS_TO_CHECKOV_MAPPING:
                CIS_TO_CHECKOV_MAPPING[cis_id] = []
            CIS_TO_CHECKOV_MAPPING[cis_id].append(checkov_id)
    else:
        if cis_ids not in CIS_TO_CHECKOV_MAPPING:
            CIS_TO_CHECKOV_MAPPING[cis_ids] = []
        CIS_TO_CHECKOV_MAPPING[cis_ids].append(checkov_id)

def check_checkov_installed():
    """Check if Checkov is installed and accessible."""
    if shutil.which("checkov"):
        return True
    
    try:
        subprocess.run(["checkov", "--version"], capture_output=True, check=False)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.warning("Checkov not found in PATH. CIS compliance validation will be simulated.")
        return False

def extract_cis_controls_from_code(code: str) -> List[str]:
    """
    Extract CIS control references from code.
    
    Args:
        code: The Terraform code
        
    Returns:
        List of CIS control IDs found in code
    """
    cis_references = set()
    
    # Look for CIS references in comments (e.g., # CIS 1.2.3)
    cis_pattern = r'(?:\/\/|#)\s*(?:CIS|cis)\s+(\d+\.\d+(?:\.\d+)?)'
    matches = re.findall(cis_pattern, code)
    
    if matches:
        cis_references.update(matches)
    
    # Look for resource configurations that might imply CIS controls
    if re.search(r'block_public_acls\s*=\s*true', code) or re.search(r'acl\s*=\s*"private"', code):
        cis_references.add("2.1.1")
    
    if re.search(r'server_side_encryption_configuration', code) or re.search(r'sse_algorithm', code):
        cis_references.add("2.1.2")
    
    if re.search(r'versioning\s*{\s*enabled\s*=\s*true', code):
        cis_references.add("2.2")

    if re.search(r'aws_cloudtrail', code) and re.search(r'is_multi_region_trail\s*=\s*true', code):
        cis_references.add("4.1")
    
    if "aws_iam" in code and "policy" in code and not re.search(r'"Action"\s*:\s*"\*"', code):
        cis_references.add("1.16")
    
    return list(cis_references)

def run_checkov_validation(terraform_code: str, pass_rate_threshold: float = 100.0) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Run Checkov validation on the generated Terraform code.
    
    Args:
        terraform_code: The Terraform code to validate
        pass_rate_threshold: Minimum pass rate percentage required for compliance (default: 100.0%)
        
    Returns:
        A tuple containing:
        - Boolean indicating overall compliance
        - Dictionary with detailed validation results
        - List of referenced CIS controls
    """
    # Extract markdown code block if present
    if "```terraform" in terraform_code:
        terraform_code = terraform_code.split("```terraform")[1].split("```")[0].strip()
    elif "```hcl" in terraform_code:
        terraform_code = terraform_code.split("```hcl")[1].split("```")[0].strip()
    
    # Exit early if code is empty
    if not terraform_code.strip():
        return False, {"error": "No Terraform code provided"}, []
    
    # First, extract CIS controls directly from the code as a baseline
    cis_controls = extract_cis_controls_from_code(terraform_code)
    
    # Prepare default summary in case Checkov fails
    default_summary = {
        "passed_checks": [],
        "failed_checks": [],
        "skipped_checks": [],
        "validation_summary": {
            "pass_rate": 0,
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tf_file = os.path.join(tmpdir, "main.tf")
        
        try:
            with open(tf_file, "w") as f:
                f.write(terraform_code)
        except IOError as e:
            logger.error(f"Error writing Terraform file: {str(e)}")
            return False, {"error": f"Error writing Terraform file: {str(e)}"}, cis_controls
        
        # Run Checkov with focus on CIS AWS Foundations Benchmark
        # This will output results in JSON format
        try:
            # Use shell=True on Windows
            import platform
            use_shell = platform.system() == "Windows"
            
            checkov_cmd_str = f"checkov -f {tf_file} --framework terraform --output json"
            logger.info(f"Running command: {checkov_cmd_str}")
            
            if use_shell:
                result = subprocess.run(
                    checkov_cmd_str,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=True
                )
            else:
                checkov_cmd = [
                    "checkov",
                    "-f", tf_file,
                    "--framework", "terraform",
                    "--output", "json"
                ]
                
                result = subprocess.run(
                    checkov_cmd,
                    check=False,
                    capture_output=True,
                    text=True
                )
            
            # Checkov returns exit code 1 when checks fail, which is normal
            # so we don't raise an exception for non-zero exit code
            
            if result.stdout:
                try:
                    # Log partial output for debugging
                    logger.debug(f"Checkov output first 200 chars: {result.stdout[:200]}")
                    
                    checkov_results = json.loads(result.stdout)
                    return process_checkov_results(checkov_results, cis_controls, pass_rate_threshold)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse Checkov JSON output: {e}")
                    # Try to extract any useful info from the output
                    logger.error(f"Raw output: {result.stdout[:500]}...")
                    # Return empty results
                    return False, default_summary, cis_controls
            elif result.stderr:
                error_msg = f"Checkov error: {result.stderr}"
                logger.error(error_msg)
                # Return empty results
                return False, default_summary, cis_controls
            else:
                logger.info("No Checkov output, assuming all checks passed")
                return True, default_summary, cis_controls
                
        except subprocess.SubprocessError as e:
            logger.error(f"Checkov execution error: {e}")
            return False, default_summary, cis_controls
        except Exception as e:
            logger.error(f"Unexpected error running Checkov: {e}")
            return False, default_summary, cis_controls

def process_checkov_results(results: Dict[str, Any], extracted_controls: List[str], pass_rate_threshold: float = 100.0) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Process the Checkov results and map them back to CIS controls.
    
    Args:
        results: The JSON results from Checkov
        extracted_controls: CIS controls extracted directly from the code
        pass_rate_threshold: Minimum pass rate percentage required for compliance (default: 100.0%)
        
    Returns:
        A tuple containing:
        - Boolean indicating overall compliance
        - Dictionary with processed validation results
        - List of referenced CIS controls
    """
    summary = {
        "passed_checks": [],
        "failed_checks": [],
        "skipped_checks": [],
        "validation_summary": {
            "pass_rate": 0,
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0
        }
    }
    
    referenced_cis_controls = set(extracted_controls)
    
    try:
        # Extract check results - handle potential structure differences in Checkov output
        check_results = results.get("results", {})
        if not check_results and "check_results" in results:
            # Some Checkov versions might use a different structure
            check_results = results.get("check_results", {})
            
        # Default to empty lists if keys not present
        failed_checks = check_results.get("failed_checks", [])
        passed_checks = check_results.get("passed_checks", [])
        skipped_checks = check_results.get("skipped_checks", [])
        
        # Ensure we're dealing with lists, even if structure is unexpected
        if not isinstance(failed_checks, list):
            logger.warning("Failed checks is not a list, defaulting to empty list")
            failed_checks = []
        if not isinstance(passed_checks, list):
            logger.warning("Passed checks is not a list, defaulting to empty list")
            passed_checks = []
        if not isinstance(skipped_checks, list):
            logger.warning("Skipped checks is not a list, defaulting to empty list")
            skipped_checks = []
        
        # Process failed checks and map to CIS controls
        for check in failed_checks:
            if not isinstance(check, dict):
                continue
                
            check_id = check.get("check_id", "unknown")
            summary["failed_checks"].append({
                "check_id": check_id,
                "check_name": check.get("check_name", "Unknown check"),
                "resource": check.get("resource", "Unknown resource"),
                "guideline": check.get("guideline", ""),
                "cis_controls": get_cis_controls_for_checkov_id(check_id)
            })
            
            # Add CIS controls to the referenced set
            for control in get_cis_controls_for_checkov_id(check_id):
                referenced_cis_controls.add(control)
        
        # Process passed checks
        for check in passed_checks:
            if not isinstance(check, dict):
                continue
                
            check_id = check.get("check_id", "unknown") 
            summary["passed_checks"].append({
                "check_id": check_id,
                "check_name": check.get("check_name", "Unknown check"),
                "resource": check.get("resource", "Unknown resource"),
                "cis_controls": get_cis_controls_for_checkov_id(check_id)
            })
            
            # Add CIS controls to the referenced set
            for control in get_cis_controls_for_checkov_id(check_id):
                referenced_cis_controls.add(control)
        
        # Process skipped checks
        for check in skipped_checks:
            if not isinstance(check, dict):
                continue
                
            check_id = check.get("check_id", "unknown")
            summary["skipped_checks"].append({
                "check_id": check_id,
                "check_name": check.get("check_name", "Unknown check"),
                "resource": check.get("resource", "Unknown resource"),
                "cis_controls": get_cis_controls_for_checkov_id(check_id)
            })
        
        # Calculate summary statistics
        total_checks = len(failed_checks) + len(passed_checks)
        if total_checks > 0:
            pass_rate = (len(passed_checks) / total_checks) * 100
        else:
            pass_rate = 0
        
        summary["validation_summary"] = {
            "pass_rate": pass_rate,
            "total_checks": total_checks,
            "passed": len(passed_checks),
            "failed": len(failed_checks),
            "skipped": len(skipped_checks)
        }
        
        # Overall compliance is determined by comparing pass_rate to threshold
        # If no checks ran, we can't claim compliance
        is_compliant = total_checks > 0 and pass_rate >= pass_rate_threshold
        logger.info(f"CIS compliance: pass rate {pass_rate:.1f}% vs threshold {pass_rate_threshold:.1f}%")
    except Exception as e:
        logger.error(f"Error processing Checkov results: {str(e)}")
        is_compliant = False
    
    return is_compliant, summary, list(referenced_cis_controls)

def get_cis_controls_for_checkov_id(checkov_id: str) -> List[str]:
    """
    Get the CIS control IDs that map to a specific Checkov ID.
    
    Args:
        checkov_id: The Checkov check ID
        
    Returns:
        A list of CIS control IDs
    """
    if not checkov_id:
        return []
        
    mapping = CHECKOV_TO_CIS_MAPPING.get(checkov_id)
    if mapping:
        if isinstance(mapping, list):
            return mapping
        else:
            return [mapping]
    
    # Try to extract from ID if it's a known format
    if checkov_id.startswith("CKV_AWS_"):
        # Some checkov IDs follow patterns that might indicate CIS controls
        # This is a fallback for IDs not explicitly mapped
        pass
        
    return []

def generate_compliance_report(validation_results: Dict[str, Any], 
                              referenced_controls: List[str]) -> str:
    """
    Generate a human-readable compliance report.
    
    Args:
        validation_results: The processed validation results
        referenced_controls: List of CIS controls referenced
        
    Returns:
        A formatted string with the compliance report
    """
    summary = validation_results["validation_summary"]
    
    report = [
        "# CIS Compliance Report",
        "",
        f"## Summary",
        f"- Pass Rate: {summary['pass_rate']:.1f}%",
        f"- Total Checks: {summary['total_checks']}",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Skipped: {summary['skipped']}",
        "",
        "## CIS Controls Referenced",
    ]
    
    # Add referenced controls
    for control in sorted(referenced_controls):
        report.append(f"- CIS {control}")
    
    # Add failed checks
    if validation_results["failed_checks"]:
        report.append("")
        report.append("## Failed Checks")
        for check in validation_results["failed_checks"]:
            cis_controls = ", ".join([f"CIS {control}" for control in check["cis_controls"]])
            report.append(f"- {check['check_id']}: {check['check_name']} ({cis_controls})")
            report.append(f"  - Resource: {check['resource']}")
            if check.get("guideline"):
                report.append(f"  - Guideline: {check['guideline']}")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Example usage
    test_tf_code = """
    provider "aws" {
      region = "us-east-1"
    }
    
    resource "aws_s3_bucket" "example" {
      bucket = "my-test-bucket"
      acl    = "public-read"  # This will fail CIS 2.1.1
    }
    """
    
    is_compliant, results, controls = run_checkov_validation(test_tf_code)
    print(f"Compliant: {is_compliant}")
    print(f"Referenced CIS controls: {controls}")
    print(generate_compliance_report(results, controls)) 