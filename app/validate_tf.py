import os
import tempfile
import subprocess


def validate_terraform_code(code: str):

    # Strip markdown block if present
    code = code.split("```terraform")[1].split("```")[0].strip()

    # Exit early if code is empty
    if not code.strip():
        print("Error: No Terraform code provided")

    with tempfile.TemporaryDirectory() as tmpdir:
        tf_file = os.path.join(tmpdir, "main.tf")

        try:
            with open(tf_file, "w") as f:
                f.write(code)
        except IOError as e:
            print(f"Error writing Terraform file: {str(e)}")

        try:
            result = subprocess.run(
                ["terraform", "fmt", "-check", tf_file],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Terraform code formatting checked successfully", result)
        except subprocess.CalledProcessError as e:
            print(f"Formatting error: {e.stderr or str(e)}")

        # Step 2: Initialize Terraform (required for validate)
        try:
            result = subprocess.run(
                ["terraform", "init", "-backend=false"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Initialization error: {e.stderr or str(e)}")

        # Step 3: Validate the configuration
        try:
            result = subprocess.run(
                ["terraform", "validate"],
                cwd=tmpdir,
                check=True,
                capture_output=True,
                text=True,
            )
            print("Terraform code validated sucessfully", result)
        except subprocess.CalledProcessError as e:
            print(f"Validation error: {e.stderr or str(e)}")
