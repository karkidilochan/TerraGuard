import os
import sys
import json
import subprocess
import tempfile
import datetime
from typing import Dict, Any
import logging
import shutil
from app.checkov_validator import run_checkov_validation, generate_compliance_report

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_terraform_installed():
    """Check if Terraform is installed and accessible."""
    try:
        # Try different ways to check for terraform
        for cmd in [
            ["terraform", "--version"],
            ["C:\\Program Files\\Terraform\\terraform.exe", "--version"],
            ["C:\\terraform\\terraform.exe", "--version"],
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, check=False)
                if result.returncode == 0:
                    logger.info(f"Terraform found: {cmd[0]}")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        # Try with shell=True as a last resort
        result = subprocess.run(
            "terraform --version", capture_output=True, check=False, shell=True
        )
        if result.returncode == 0:
            logger.info("Terraform found using shell command")
            return True

        # If we get here, we couldn't find Terraform
        logger.warning(
            "Terraform not found in PATH. Skipping Terraform validation steps."
        )
        return False
    except Exception as e:
        logger.warning(f"Error checking for Terraform: {str(e)}")
        return False


def check_checkov_installed():
    """Check if Checkov is installed and accessible."""
    try:
        # Try different ways to check for Checkov
        for cmd in [
            ["checkov", "--version"],
            ["python", "-m", "checkov", "--version"],
            [sys.executable, "-m", "checkov", "--version"],
        ]:
            try:
                result = subprocess.run(cmd, capture_output=True, check=False)
                if result.returncode == 0:
                    logger.info(f"Checkov found: {' '.join(cmd)}")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                continue

        # Try with shell=True as a last resort
        result = subprocess.run(
            "checkov --version", capture_output=True, check=False, shell=True
        )
        if result.returncode == 0:
            logger.info("Checkov found using shell command")
            return True

        # Try python module directly
        try:
            import checkov

            logger.info("Checkov found as Python module")
            return True
        except ImportError:
            pass

        # If we get here, we couldn't find Checkov
        logger.warning(
            "Checkov not found in PATH. Skipping CIS compliance validation steps."
        )
        return False
    except Exception as e:
        logger.warning(f"Error checking for Checkov: {str(e)}")
        return False


def validate_terraform_code(
    code: str, output_dir: str = None, result_file_prefix: str = None
) -> Dict[str, Any]:
    """
    Validate Terraform code for syntax and CIS compliance.

    Args:
        code: The Terraform code to validate
        output_dir: The directory to store Checkov results (default: checkov_output folder in project root)
        result_file_prefix: Prefix for the Checkov result filename (default: "checkov_results")

    Returns:
        Dictionary with validation results
    """
    result = {
        "syntax_valid": False,
        "cis_compliant": False,
        "terraform_output": "",
        "checkov_output": None,
        "referenced_cis_controls": [],
        "errors": [],
    }

    # Strip markdown block if present
    if "```terraform" in code:
        code = code.split("```terraform")[1].split("```")[0].strip()
    elif "```hcl" in code:
        code = code.split("```hcl")[1].split("```")[0].strip()

    # Exit early if code is empty
    if not code.strip():
        result["errors"].append("No Terraform code provided")
        return result

    # Check if terraform is installed
    terraform_installed = check_terraform_installed()
    checkov_installed = check_checkov_installed()

    if not terraform_installed:
        result["errors"].append("Terraform not installed. Skipping syntax validation.")
        return result

    if not checkov_installed:
        result["errors"].append(
            "Checkov not installed. Skipping CIS compliance validation."
        )

    # Create output directory for Checkov results if it doesn't exist
    if output_dir is None:
        # Use a directory in the current working directory instead of user's home
        output_dir = os.path.join(os.getcwd(), "checkov_output")

    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(output_dir, ".test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Using output directory for Checkov results: {output_dir}")
    except (OSError, IOError) as e:
        logger.error(f"Error accessing output directory {output_dir}: {str(e)}")
        result["errors"].append(f"Error accessing output directory: {str(e)}")
        # Try an alternative location
        try:
            # Use a directory in the user's home directory as fallback
            home_dir = os.path.expanduser("~")
            output_dir = os.path.join(home_dir, "checkov_output")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Using alternative output directory: {output_dir}")
        except (OSError, IOError) as e:
            # Last resort - use system temp directory
            try:
                output_dir = os.path.join(tempfile.gettempdir(), "checkov_output")
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Using temp directory for output: {output_dir}")
            except (OSError, IOError) as e:
                logger.error(f"Error accessing all output directories: {str(e)}")
                result["errors"].append(
                    f"Error accessing all output directories: {str(e)}"
                )
                return result

    # Set default result file prefix if not provided
    if result_file_prefix is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file_prefix = f"checkov_results_{timestamp}"

    with tempfile.TemporaryDirectory() as tmpdir:
        tf_file = os.path.join(tmpdir, "main.tf")

        try:
            with open(tf_file, "w") as f:
                f.write(code)
        except IOError as e:
            result["errors"].append(f"Error writing Terraform file: {str(e)}")
            return result

        # Step 1: Check Terraform formatting and auto-format
        try:
            # Try formatting the code first
            fmt_result = subprocess.run(
                "terraform fmt main.tf",
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
                shell=True,
            )

            if fmt_result.returncode == 0:
                logger.info("Successfully formatted Terraform code")
            else:
                # If auto-format fails, check the formatting
                check_fmt_result = subprocess.run(
                    "terraform fmt -check main.tf",
                    cwd=tmpdir,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=True,
                )
                if check_fmt_result.returncode != 0:
                    result["errors"].append(
                        f"Formatting error: {check_fmt_result.stderr or 'Code is not properly formatted'}"
                    )
                else:
                    logger.info("Terraform code formatting checked successfully")
        except Exception as e:
            result["errors"].append(f"Formatting check error: {str(e)}")

        # Step 2: Initialize Terraform (required for validate)
        try:
            init_result = subprocess.run(
                "terraform init -backend=false",
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
                shell=True,
            )
            if init_result.returncode != 0:
                result["errors"].append(
                    f"Initialization error: {init_result.stderr or 'Failed to initialize Terraform'}"
                )
            else:
                logger.info("Terraform initialized successfully")
        except Exception as e:
            result["errors"].append(f"Initialization error: {str(e)}")

        # Step 3: Validate Terraform configuration
        try:
            validate_result = subprocess.run(
                "terraform validate",
                cwd=tmpdir,
                check=False,
                capture_output=True,
                text=True,
                shell=True,
            )

            result["terraform_output"] = (
                validate_result.stdout or validate_result.stderr
            )

            if validate_result.returncode == 0:
                result["syntax_valid"] = True
                logger.info("Terraform code validated successfully")
            else:
                result["errors"].append(
                    f"Validation error: {validate_result.stderr or 'Failed to validate Terraform configuration'}"
                )
        except Exception as e:
            result["errors"].append(f"Validation error: {str(e)}")

        # Step 4: Run Checkov for CIS compliance validation if installed
        if checkov_installed and result["syntax_valid"]:
            try:
                # Generate a filename for the output based on the prefix
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = os.path.join(
                    output_dir, f"{result_file_prefix}_{timestamp}.json"
                )

                # Run Checkov directly with output to stdout, then save results ourselves
                # This avoids issues with Checkov creating directories
                checkov_cmd_str = (
                    f"checkov -f {tf_file} --framework terraform --output json"
                )
                logger.info(f"Running Checkov command: {checkov_cmd_str}")

                checkov_result = subprocess.run(
                    checkov_cmd_str,
                    check=False,
                    capture_output=True,
                    text=True,
                    shell=True,
                    cwd=tmpdir,
                )

                # Process stdout regardless of exit code
                if checkov_result.stdout:
                    try:
                        checkov_data = json.loads(checkov_result.stdout)

                        # Save the results ourselves
                        with open(output_file, "w") as f:
                            json.dump(checkov_data, f, indent=2)

                        # Process the results using our CIS mapper
                        is_compliant, checkov_results, referenced_controls = (
                            run_checkov_validation(code)
                        )

                        result["cis_compliant"] = is_compliant
                        result["checkov_results"] = checkov_results
                        result["checkov_output"] = (
                            checkov_data  # Store the full JSON output
                        )
                        result["referenced_cis_controls"] = referenced_controls

                        # Generate compliance report
                        report = generate_compliance_report(
                            checkov_results, referenced_controls
                        )
                        result["compliance_report"] = report

                        # Save a link to the output file in the results
                        result["checkov_output_file"] = output_file

                        logger.info(
                            f"Checkov validation complete. Results saved to {output_file}"
                        )
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing Checkov results: {str(e)}")
                        logger.error(
                            f"Raw stdout (first 500 chars): {checkov_result.stdout[:500]}"
                        )
                        result["errors"].append(
                            f"Error parsing Checkov results: {str(e)}"
                        )
                else:
                    error_msg = f"Checkov error: {checkov_result.stderr}"
                    logger.error(error_msg)
                    result["errors"].append(error_msg)

                # If we had issues, try the direct approach as a fallback
                if not result.get("checkov_output"):
                    # Try running the internal function without subprocess
                    try:
                        is_compliant, checkov_results, referenced_controls = (
                            run_checkov_validation(code)
                        )
                        result["cis_compliant"] = is_compliant
                        result["checkov_results"] = checkov_results
                        result["referenced_cis_controls"] = referenced_controls

                        # Generate compliance report
                        report = generate_compliance_report(
                            checkov_results, referenced_controls
                        )
                        result["compliance_report"] = report
                        logger.info("Used fallback internal Checkov validation method")
                    except Exception as e:
                        logger.error(f"Error in fallback Checkov validation: {str(e)}")

            except Exception as e:
                logger.error(f"CIS compliance check error: {str(e)}")
                result["errors"].append(f"CIS compliance check error: {str(e)}")

    return result


if __name__ == "__main__":
    # Test the validator with a fully CIS-compliant S3 bucket example
    test_code = """
# Fully CIS-compliant S3 bucket example
# Following CIS controls 2.1.1, 2.1.2, 2.2, etc.

provider "aws" {
  region = "us-east-1"
}

# -----------------------------
# Main bucket configuration
# -----------------------------
resource "aws_s3_bucket" "example" {
  bucket = "my-secure-bucket-example"
  
  tags = {
    Name        = "My Secure Bucket"
    Environment = "Production"
    Compliance  = "CIS"
  }
}

# Use separate resource for versioning as per AWS provider best practices
resource "aws_s3_bucket_versioning" "example" {
  bucket = aws_s3_bucket.example.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption with KMS
resource "aws_s3_bucket_server_side_encryption_configuration" "example" {
  bucket = aws_s3_bucket.example.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_key.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "example" {
  bucket                  = aws_s3_bucket.example.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable access logging
resource "aws_s3_bucket_logging" "example" {
  bucket = aws_s3_bucket.example.id

  target_bucket = aws_s3_bucket.log_bucket.id
  target_prefix = "log/"
}

# Add lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "example" {
  bucket = aws_s3_bucket.example.id

  rule {
    id     = "archive-rule"
    status = "Enabled"
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}

# Add event notification
resource "aws_s3_bucket_notification" "example" {
  bucket = aws_s3_bucket.example.id

  topic {
    topic_arn     = aws_sns_topic.example.arn
    events        = ["s3:ObjectCreated:*", "s3:ObjectRemoved:*"]
    filter_prefix = "logs/"
  }
}

# Cross-region replication role
resource "aws_iam_role" "replication" {
  name = "s3-bucket-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

# Cross-region replication policy
resource "aws_iam_policy" "replication" {
  name = "s3-bucket-replication-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Effect   = "Allow"
        Resource = [aws_s3_bucket.example.arn]
      },
      {
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging"
        ]
        Effect   = "Allow"
        Resource = ["${aws_s3_bucket.example.arn}/*"]
      },
      {
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete",
          "s3:ReplicateTags"
        ]
        Effect   = "Allow"
        Resource = ["${aws_s3_bucket.replication.arn}/*"]
      }
    ]
  })
}

# Attach replication policy to role
resource "aws_iam_role_policy_attachment" "replication" {
  role       = aws_iam_role.replication.name
  policy_arn = aws_iam_policy.replication.arn
}

# Replication bucket in another region
resource "aws_s3_bucket" "replication" {
  provider = aws.replica
  bucket   = "my-secure-bucket-replication"
  
  tags = {
    Name        = "Replication Bucket"
    Environment = "Production"
    Compliance  = "CIS"
  }
}

# Replication configuration
resource "aws_s3_bucket_replication_configuration" "example" {
  # Must have bucket versioning enabled first
  depends_on = [aws_s3_bucket_versioning.example]

  bucket = aws_s3_bucket.example.id
  role   = aws_iam_role.replication.arn

  rule {
    id       = "replicate-all"
    status   = "Enabled"
    priority = 1

    destination {
      bucket        = aws_s3_bucket.replication.arn
      storage_class = "STANDARD"
    }
  }
}

# -----------------------------
# Log bucket configuration
# -----------------------------
resource "aws_s3_bucket" "log_bucket" {
  bucket = "my-secure-bucket-logs"
  
  tags = {
    Name        = "Log Bucket"
    Environment = "Production"
    Compliance  = "CIS"
  }
}

# Versioning for log bucket
resource "aws_s3_bucket_versioning" "log_bucket" {
  bucket = aws_s3_bucket.log_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption for log bucket
resource "aws_s3_bucket_server_side_encryption_configuration" "log_bucket" {
  bucket = aws_s3_bucket.log_bucket.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3_key.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# Block public access for log bucket
resource "aws_s3_bucket_public_access_block" "log_bucket" {
  bucket                  = aws_s3_bucket.log_bucket.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Add lifecycle configuration for log bucket
resource "aws_s3_bucket_lifecycle_configuration" "log_bucket" {
  bucket = aws_s3_bucket.log_bucket.id

  rule {
    id     = "log-archive-rule"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    expiration {
      days = 365
    }
  }
}

# Add event notification for log bucket
resource "aws_s3_bucket_notification" "log_bucket" {
  bucket = aws_s3_bucket.log_bucket.id

  topic {
    topic_arn     = aws_sns_topic.example.arn
    events        = ["s3:ObjectCreated:*"]
    filter_prefix = "log/"
  }
}

# -----------------------------
# Additional resources
# -----------------------------
# SNS topic for bucket notifications
resource "aws_sns_topic" "example" {
  name = "s3-event-notification-topic"
  
  tags = {
    Name        = "S3 Event Notifications"
    Environment = "Production"
    Compliance  = "CIS"
  }
}

# KMS key for S3 encryption
resource "aws_kms_key" "s3_key" {
  description             = "KMS key for S3 encryption"
  deletion_window_in_days = 10
  enable_key_rotation     = true
  
  # Define KMS key policy
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::ACCOUNT-ID:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow S3 to use the key"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
  
  tags = {
    Name        = "S3 Encryption Key"
    Environment = "Production"
    Compliance  = "CIS"
  }
}

# KMS key alias
resource "aws_kms_alias" "s3_key_alias" {
  name          = "alias/s3-encryption-key"
  target_key_id = aws_kms_key.s3_key.key_id
}

# Provider for replication region
provider "aws" {
  alias  = "replica"
  region = "us-west-2"
}
"""

    results = validate_terraform_code(test_code)
    print(f"Syntax valid: {results['syntax_valid']}")
    print(f"CIS compliant: {results['cis_compliant']}")
    print(f"Referenced CIS controls: {results['referenced_cis_controls']}")

    if results.get("compliance_report"):
        print("\nCompliance Report:")
        print(results["compliance_report"])

    if results.get("checkov_output_file"):
        print(f"\nCheckov results saved to: {results['checkov_output_file']}")

    if results["errors"]:
        print("\nErrors:")
        for error in results["errors"]:
            print(f"- {error}")
