import re
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Common Terraform error patterns
TF_ERROR_PATTERNS = {
    "missing_required_argument": re.compile(r"The argument \"(\w+)\" is required"),
    "unsupported_argument": re.compile(r"An argument named \"(\w+)\" is not expected here"),
    "reference_error": re.compile(r"Reference to undeclared (resource|input variable|local value) \"([^\"]+)\""),
    "syntax_error": re.compile(r"Error: (Syntax error|Invalid expression|Parse error)"),
    "invalid_value": re.compile(r"Expected (\w+), got (\w+)")
}

# Checkov error categories mapped to human-readable guidance
CHECKOV_GUIDANCE = {
    "CKV_AWS_": "The resource violates AWS best practices. Consider adding required security configurations.",
    "CKV2_AWS_": "The resource configuration doesn't meet AWS security recommendations.",
    "CKV_GCP_": "The GCP resource configuration might have security weaknesses.",
    "CKV_AZURE_": "The Azure resource needs additional security hardening.",
    "CKV_K8S_": "The Kubernetes configuration needs security improvements.",
}

def extract_tf_validation_errors(validation_output: str) -> List[Dict[str, str]]:
    """
    Parse Terraform validation errors into structured format
    
    Args:
        validation_output: Raw Terraform validation error output
        
    Returns:
        List of structured error dictionaries with type, resource, and message
    """
    structured_errors = []
    
    # Process the validation output line by line
    for line in validation_output.split('\n'):
        line = line.strip()
        
        # Look for the "on main.tf line X" pattern to identify the resource context
        resource_match = re.search(r'on main\.tf line (\d+), in resource \"([^\"]+)\"', line)
        resource_context = resource_match.group(2) if resource_match else "unknown"
        
        # Check each error pattern
        for error_type, pattern in TF_ERROR_PATTERNS.items():
            match = pattern.search(line)
            if match:
                error_info = {
                    "type": error_type,
                    "resource": resource_context,
                    "message": line
                }
                
                # Add specific details based on error type
                if error_type == "missing_required_argument":
                    error_info["argument"] = match.group(1)
                    error_info["human_message"] = f"Add the required '{match.group(1)}' argument to the {resource_context} resource"
                    
                elif error_type == "unsupported_argument":
                    error_info["argument"] = match.group(1)
                    error_info["human_message"] = f"Remove the unsupported '{match.group(1)}' argument from the {resource_context} resource"
                
                structured_errors.append(error_info)
                break
    
    return structured_errors

def extract_checkov_issues(checkov_results: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Extract actionable guidance from Checkov scan results
    
    Args:
        checkov_results: Parsed Checkov JSON results
        (now can be found under validation_results['checkov_output'])
        
    Returns:
        List of structured compliance issue dictionaries
    """
    issues = []
    
    if not checkov_results:
        return issues
    
    # Try all possible locations where failed_checks might be found
    failed_checks = []
    
    # Option 1: Direct top-level failed_checks
    if "failed_checks" in checkov_results:
        failed_checks = checkov_results.get("failed_checks", [])
    
    # Option 2: In results.failed_checks
    elif "results" in checkov_results and isinstance(checkov_results["results"], dict):
        failed_checks = checkov_results["results"].get("failed_checks", [])
    
    # Option 3: In checkov_output.results.failed_checks
    elif "checkov_output" in checkov_results and isinstance(checkov_results["checkov_output"], dict):
        results_section = checkov_results["checkov_output"].get("results", {})
        if isinstance(results_section, dict):
            failed_checks = results_section.get("failed_checks", [])
    
    # Log the number of failed checks found for debugging
    logger.info(f"Found {len(failed_checks)} failed Checkov checks")
    
    # Process the failed checks to create actionable issues
    for check in failed_checks:
        if not isinstance(check, dict):
            continue
            
        check_id = check.get("check_id", "")
        check_name = check.get("check_name", "")
        resource = check.get("resource", "")
        
        # Find the guidance category based on the check ID prefix
        guidance = "This resource has a security or compliance issue."
        for prefix, guide_text in CHECKOV_GUIDANCE.items():
            if check_id.startswith(prefix):
                guidance = guide_text
                break
        
        issue = {
            "check_id": check_id,
            "resource": resource,
            "description": check_name,
            "guidance": guidance,
            "human_message": f"Fix {check_id}: {check_name} in {resource}"
        }
        
        issues.append(issue)
    
    return issues

def summarize_validation_errors(validation_results: Dict[str, Any]) -> str:
    """
    Create a concise, actionable summary of validation errors for the LLM
    
    Args:
        validation_results: The complete validation results dictionary
        
    Returns:
        A formatted error summary string for prompt insertion
    """
    summary_parts = []
    
    # Handle syntax validation errors
    if not validation_results.get("syntax_valid", True):
        syntax_errors = extract_tf_validation_errors(validation_results.get("terraform_output", ""))
        
        if syntax_errors:
            summary_parts.append("## Terraform Syntax Issues")
            for i, error in enumerate(syntax_errors, 1):
                if "human_message" in error:
                    summary_parts.append(f"{i}. {error['human_message']}")
                else:
                    summary_parts.append(f"{i}. Fix issue in {error['resource']}: {error['message']}")
    
    # Handle compliance validation errors
    if not validation_results.get("cis_compliant", True):
        checkov_issues = extract_checkov_issues(validation_results.get("checkov_results", {}))
        
        if checkov_issues:
            summary_parts.append("## Compliance Issues")
            for i, issue in enumerate(checkov_issues, 1):
                summary_parts.append(f"{i}. {issue['human_message']}")
    
    # If there are no structured errors but we know validation failed
    if not summary_parts and validation_results.get("errors"):
        summary_parts.append("## Validation Errors")
        for i, error in enumerate(validation_results.get("errors", []), 1):
            summary_parts.append(f"{i}. {error}")
    
    # Return the formatted summary
    if summary_parts:
        return "\n\n".join(summary_parts)
    else:
        return "No specific validation errors were identified. Please review the code for any issues." 