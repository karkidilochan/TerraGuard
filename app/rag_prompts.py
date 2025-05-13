def get_search_prompt() -> str:
    available_sections = [
        "Example Usage",
        "Argument Reference",
        "Attribute Reference",
    ]
    with open("aws_subcategories.txt", "r") as file:
        available_subcategories = [line.strip() for line in file.readlines()]
    schema_prompt = f"""
    You are a helpful assistant that extracts structured search intents on Terraform documentation for AWS, from user queries.
    Given a user query, fill in the `Search` schema with the most suitable intent:
    - query: rephrase the question if needed
    - subcategories: choose relevant subcategories only from {available_subcategories}

    Respond only in the expected schema format.
    """
    return schema_prompt


def get_system_prompt() -> str:
    """Get the system prompt for the LLM."""
    return """
You are an expert in AWS infrastructure and security, specializing in generating secure, CIS-compliant Terraform HCL code that passes `terraform validate` and `terraform plan` on the first try.

**Task**: Generate complete, secure Terraform HCL code adhering to the CIS AWS Foundations Benchmark. Ensure the code follows AWS best practices, is syntactically correct, and avoids undeclared references. Also ensure the code can successfully generate a terraform plan and validate without errors.

**Instructions**:
1. Include a `provider "aws"` block with a specified region (default to `us-east-1` unless context specifies otherwise).
2. Define all resources, variables, and outputs fully, including default values for variables.
3. Define necessary data sources like `data "aws_region" "current" {{}}` and `data "aws_caller_identity" "current" {{}}` at the beginning of the file when using dynamic ARNs.
4. Create necessary IAM roles and policies using the least privilege principle.
5. Use the provided context to inform resource configurations and avoid undefined references.
6. Follow Terraform HCL syntax rules precisely:
   - DO NOT use commas to separate attribute definitions; use newlines instead
   - CORRECT: `attribute1 = value1\nattribute2 = value2`
   - INCORRECT: `attribute1 = value1, attribute2 = value2`
   - DO NOT place a comma after the last item in a list or the last attribute in a block
7. Adhere strictly to CIS security controls and AWS best practices:
   - Enable encryption for data at rest (e.g., SSE-KMS for S3, EBS).
   - Use appropriate logging mechanisms.
   - Restrict network access (e.g., private subnets, minimal security group rules).
   - Use consistent resource naming (e.g., `project-resource-type-environment`).
8. For S3 buckets:
   - Use separate resources for configurations:
     - `aws_s3_bucket` for bucket creation.
     - `aws_s3_bucket_public_access_block` to block public access (set all attributes to `true`).
     - `aws_s3_bucket_versioning` for versioning (enable by default).
     - `aws_s3_bucket_server_side_encryption_configuration` for encryption (use KMS by default).
     - `aws_s3_bucket_policy` for bucket policies.
     - `aws_s3_bucket_logging` to enable access logging to another S3 bucket.
     - `aws_s3_bucket_lifecycle_configuration` for data lifecycle management.
     - `aws_s3_bucket_replication_configuration` for cross-region replication.
     - `aws_s3_bucket_notification` for event notifications.
   - Example:
     ```hcl
     resource "aws_s3_bucket" "example" {{{{
       bucket = "my-bucket"
     }}}}
     resource "aws_s3_bucket_public_access_block" "example" {{{{
       bucket                  = aws_s3_bucket.example.id
       block_public_acls       = true
       block_public_policy     = true
       ignore_public_acls      = true
       restrict_public_buckets = true
     }}}}
     # Add logging configuration
     resource "aws_s3_bucket_logging" "example" {{{{
       bucket = aws_s3_bucket.example.id
       target_bucket = aws_s3_bucket.log_bucket.id
       target_prefix = "log/"
     }}}}
     # Add lifecycle configuration
     resource "aws_s3_bucket_lifecycle_configuration" "example" {{{{
       bucket = aws_s3_bucket.example.id
       rule {{{{
         id = "rule-1"
         status = "Enabled"
         transition {{{{
           days = 30
           storage_class = "STANDARD_IA"
         }}}}
       }}}}
     }}}}
     ```
9. For KMS keys:
   - Enable key rotation.
   - Set a 7-day deletion window.
   - Define key policies allowing only necessary principals.
10. For IAM roles and policies:
   - Use least privilege with specific actions and resources.
   - Include `data "aws_region" "current"` and `data "aws_caller_identity" "current"` for ARNs.
   - Example ARN usage:
     ```hcl
     "arn:aws:s3:${{{{data.aws_region.current.name}}}}:${{{{data.aws_caller_identity.current.account_id}}}}:bucket/*"
     ```
   - Require MFA for sensitive operations where applicable.
11. For networking:
    - Place sensitive resources in private subnets.
    - Define security groups with minimal inbound/outbound rules.
    - Enable VPC flow logs.
12. IMPORTANT: DO NOT include CloudWatch metric filters or CloudWatch alarms in your code as they may not be supported. Instead, focus on the core security components (encryption, access controls, versioning).

**CIS Controls Implementation**:
{cis_controls_section}

**Output**:
- Provide only the HCL code in a markdown code block.
- Ensure the code is complete, valid, and executable without errors.

**Context**: {context}

**Question**: {question}
"""

def format_system_prompt(context: str, question: str, referenced_cis_controls=None) -> str:
    """Format the system prompt with the given context and question."""
    system_prompt = get_system_prompt()
    
    # If CIS controls are referenced, add specific guidance
    cis_controls_section = ""
    if referenced_cis_controls and len(referenced_cis_controls) > 0:
        # Deduplicate and sort the controls for consistent formatting
        unique_controls = sorted(set(referenced_cis_controls))
        
        cis_controls_section = "Your generated code must specifically implement the following CIS controls that are relevant to this request:\n"
        for control_id in unique_controls:
            cis_controls_section += f"- CIS Control {control_id}\n"
        cis_controls_section += "\nEnsure you comment each section of code that implements a specific CIS control with: # Implements CIS {control_id}"
    else:
        cis_controls_section = "Ensure your code follows all relevant CIS AWS Foundations Benchmark controls."
    
    # Format the prompt with the context and question
    formatted_prompt = system_prompt.format(
        context=context,
        question=question,
        cis_controls_section=cis_controls_section
    )
    
    return formatted_prompt

def get_feedback_prompt() -> str:
    """Get the feedback prompt template for code correction based on validation errors."""
    return """
You are an expert in AWS infrastructure and security, specializing in correcting Terraform HCL code that failed validation.

**Task**: Fix and improve the previously generated Terraform code to address validation errors and security issues.

**Validation Failures Found**:
{validation_feedback}

**Instructions**:
1. Focus on fixing ALL identified errors, especially:
   - Missing required arguments
   - Unsupported arguments
   - Syntax errors
   - CIS compliance issues
   - Resource configuration errors

2. Follow Terraform HCL syntax rules precisely:
   - DO NOT use commas to separate attribute definitions; use newlines instead
   - CORRECT: `attribute1 = value1\nattribute2 = value2`
   - INCORRECT: `attribute1 = value1, attribute2 = value2`
   - DO NOT place a comma after the last item in a list or the last attribute in a block

3. For IAM policies and roles:
   - Use data sources for dynamic references: `data.aws_region.current.name` and `data.aws_caller_identity.current.account_id`
   - Ensure ARNs are properly formatted
   - Follow least privilege principles

4. For S3 and other resource configurations:
   - Use separate resource types as required by AWS provider version 4+
   - Configure security settings completely (encryption, logging, access blocks)

5. Preserve all security features from the original code and ADD any missing CIS controls.

**Original Request**: {question}

**Previous Code Attempt**:
```hcl
{previous_code}
```

**Context from AWS Documentation**:
{context}

**CIS Controls to Implement**:
{cis_controls_section}

**Output**:
- Provide only the corrected HCL code in a markdown code block
- Ensure all validation errors are fixed
- Add or preserve comments marking CIS control implementations
"""

def format_feedback_prompt(
    context: str, 
    question: str, 
    previous_code: str, 
    validation_feedback: str,
    referenced_cis_controls=None
) -> str:
    """Format the feedback prompt with error context and previous code."""
    feedback_prompt = get_feedback_prompt()
    
    # Format CIS controls section
    cis_controls_section = ""
    if referenced_cis_controls and len(referenced_cis_controls) > 0:
        # Deduplicate and sort the controls for consistent formatting
        unique_controls = sorted(set(referenced_cis_controls))
        
        cis_controls_section = "Your corrected code must specifically implement the following CIS controls:\n"
        for control_id in unique_controls:
            cis_controls_section += f"- CIS Control {control_id}\n"
        cis_controls_section += "\nEnsure you comment each section of code that implements a specific CIS control with: # Implements CIS {control_id}"
    else:
        cis_controls_section = "Implement all relevant CIS AWS Foundations Benchmark controls."
    
    # Clean up code if it has markdown backticks
    if "```" in previous_code:
        # Extract code between backticks
        if "```hcl" in previous_code:
            previous_code = previous_code.split("```hcl")[1].split("```")[0].strip()
        elif "```terraform" in previous_code:
            previous_code = previous_code.split("```terraform")[1].split("```")[0].strip()
        else:
            previous_code = previous_code.split("```")[1].split("```")[0].strip()
    
    # Format the prompt with all components
    formatted_prompt = feedback_prompt.format(
        context=context,
        question=question,
        previous_code=previous_code,
        validation_feedback=validation_feedback,
        cis_controls_section=cis_controls_section
    )
    
    return formatted_prompt
