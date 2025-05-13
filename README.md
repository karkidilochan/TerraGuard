# TerraGuard: Secure Terraform RAG Pipeline with CIS Benchmark Compliance

TerraGuard is an intelligent Terraform code generation system that leverages Retrieval-Augmented Generation (RAG) to produce Infrastructure as Code (IaC) that automatically adheres to AWS CIS Foundations Benchmark security controls. This project combines LLM capabilities with security validation to ensure generated infrastructure code meets industry-standard security practices.

## Project Overview

TerraGuard addresses the challenge of writing secure infrastructure code by:
1. Using RAG to retrieve relevant Terraform and security documentation
2. Generating compliant Terraform code based on natural language requirements
3. Validating the code against CIS Benchmark security controls
4. Providing feedback loops to improve code generation quality
5. Offering both API and CLI interfaces for flexibility

## Features

- **CIS Benchmark Compliance**: Generates Terraform code that adheres to AWS CIS Foundations Benchmark security controls v1.4, covering over 100 security best practices.
- **RAG Pipeline Architecture**: 
  - Uses ChromaDB for efficient vector embeddings of Terraform documentation
  - Leverages Mistral LLM for high-quality code generation
  - Incorporates security documentation in the retrieval process
- **Automated Validation**: 
  - Validates generated code for Terraform syntax correctness
  - Checks security compliance using Checkov's policy-as-code engine
  - Provides detailed validation reports with specific issues
- **Feedback Loop System**: 
  - Implements an automated feedback mechanism to improve code quality
  - Tracks historical failures to prevent repeat issues
  - Uses reinforcement learning principles to enhance model outputs
- **Dual Interface**: 
  - FastAPI-based REST API for integration with other systems
  - Interactive CLI for direct developer usage

## Technical Architecture

TerraGuard consists of several key components:

1. **Vector Store**: ChromaDB instance containing embedded Terraform documentation
2. **LLM Interface**: Connection to Mistral AI for code generation
3. **CIS Benchmark Parser**: Extracts and enhances security controls from CIS documentation
4. **Validation Engine**: Integrates with Checkov for security validation
5. **Feedback System**: Tracks and learns from validation failures
6. **API/CLI Layer**: Provides user interfaces

## Prerequisites

Ensure you have the following installed:

- Python 3.9+
- [Poetry](https://python-poetry.org/docs/) for dependency management
- [ChromaDB](https://github.com/chroma-core/chroma) for vector database (installed via Poetry)
- [Checkov](https://www.checkov.io/) for security validation (installed via Poetry)
- A Mistral API key (if using the hosted service)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/terraguard.git
cd terraguard
```

### Step 2: Install Dependencies with Poetry

First, ensure Poetry is installed. Then, install the project dependencies:

```bash
poetry install
```

This will create a virtual environment and install all required packages.

### Step 3: Setup Environment Variables

Create a `.env` file in the root directory:

```
MISTRAL_API_KEY=<your-api-key>
```

### Step 4: Set Up the Vector Store

Ensure the `chroma_rag_db` directory exists and contains your indexed data. If not, you'll need to create it:

```bash
poetry run python -m app.vector_store
```

This script will:
- Download Terraform documentation
- Process and clean the documentation
- Create embeddings using the specified embedding model
- Store the embeddings in ChromaDB

### Step 5: Run the CIS Benchmark Setup Scripts

To prepare the CIS benchmark data for use in the pipeline:

1. Extract the benchmarks from the PDF (if not already done):
```bash
cd scripts
python cis_benchmark_extract.py
```
This extracts raw text from the CIS AWS Foundations Benchmark PDF.

2. Enhance the benchmarks with resource type and attribute metadata:
```bash
python enhance_cis_benchmark.py
```
This adds machine-readable metadata to make benchmarks actionable in code validation.

## Using TerraGuard

### As a Command Line Tool

Run TerraGuard in interactive CLI mode:

```bash
poetry run python main.py
```

You'll be prompted to enter your requirements in natural language. For example:
- "Create an S3 bucket with server-side encryption and versioning enabled"
- "Set up a VPC with public and private subnets that follows security best practices"
- "Configure an ECS cluster with least privilege IAM roles"

The system will:
1. Process your request
2. Retrieve relevant documentation
3. Generate Terraform code
4. Validate the code
5. Present the results

### As a Web API

To run the TerraGuard web API:

```bash
RUN_MODE=server poetry run python main.py
```

The API will be available at http://localhost:8000 with Swagger documentation at http://localhost:8000/docs

#### API Usage Examples

**Generate Terraform Code:**

```
POST /generate
{
  "query": "How do I set up an S3 bucket with versioning and encryption that complies with CIS benchmarks?"
}
```

Response:
```json
{
  "query": "How do I set up an S3 bucket with versioning and encryption that complies with CIS benchmarks?",
  "terraform_code": "resource \"aws_s3_bucket\" \"compliant_bucket\" {\n  bucket = \"my-compliant-bucket\"\n\n  # Other configuration...\n}\n\nresource \"aws_s3_bucket_versioning\" \"versioning\" {\n  bucket = aws_s3_bucket.compliant_bucket.id\n  versioning_configuration {\n    status = \"Enabled\"\n  }\n}\n\nresource \"aws_s3_bucket_server_side_encryption_configuration\" \"encryption\" {\n  bucket = aws_s3_bucket.compliant_bucket.id\n\n  rule {\n    apply_server_side_encryption_by_default {\n      sse_algorithm = \"AES256\"\n    }\n  }\n}",
  "referenced_cis_controls": ["2.1.1", "2.1.2"],
  "validation_summary": {
    "syntax_valid": true,
    "cis_compliant": true,
    "referenced_cis_controls": ["2.1.1", "2.1.2"],
    "error_count": 0
  },
  "compliance_report": "CIS Compliance Report: All checks passed..."
}
```

## Scripts Documentation

TerraGuard includes several utility scripts to help you test, evaluate, and run different aspects of the system:

### Core Scripts

- **main.py**: The main entry point for both CLI and API modes
- **app/vector_store.py**: Creates and manages the vector store for RAG
- **app/generator.py**: Handles LLM-based Terraform code generation
- **app/validator.py**: Validates generated code against security controls
- **app/feedback.py**: Implements the feedback loop system

### Testing Scripts

- **scripts/test_compliance_pipeline.py**: Tests the entire compliance validation pipeline
  ```bash
  python scripts/test_compliance_pipeline.py
  ```
  This verifies that validation works correctly by running several test cases.

- **scripts/test_feedback_loop.py**: Tests the feedback loop system
  ```bash
  # Basic testing using hardcoded scenarios
  python scripts/test_feedback_loop.py
  
  # Real failure testing (uses actual LLM failures)
  python scripts/test_feedback_loop.py --use-real-failures
  
  # Comprehensive testing (both hardcoded and real failures)
  python scripts/test_feedback_loop.py --use-both
  ```
  
- **scripts/analyse_feed_logs.py**: Analyzes generated logs from feedback loop tests
  ```bash
  python scripts/analyse_feed_logs.py
  ```
  Provides statistics and insights on feedback system performance.

### CIS Benchmark Scripts

- **scripts/cis_benchmark_extract.py**: Extracts CIS controls from PDF documentation
- **scripts/enhance_cis_benchmark.py**: Enhances CIS controls with additional metadata

## Customizing Prompts

TerraGuard allows you to customize the prompts used for code generation. This can be useful for adapting the system to specific requirements or improving generation quality.

### Modifying Prompt Templates

Prompt templates are located in `app/templates/`. To modify them:

1. Edit the appropriate template file:
   - `base_prompt.txt`: The main RAG prompt template
   - `correction_prompt.txt`: Used in the feedback loop

2. You can use the following variables in your templates:
   - `{query}`: The user's natural language query
   - `{context}`: Retrieved documentation
   - `{feedback}`: Previous validation feedback
   - `{cis_references}`: Relevant CIS controls

### Example Prompt Customization

To optimize for brevity in generated code:

```
Given the following user request:
{query}

And this Terraform documentation:
{context}

Generate concise, minimal Terraform code that implements the request while satisfying these security requirements:
{cis_references}

Focus on producing the most compact working code possible while maintaining security compliance.
```

## Troubleshooting

### Common Issues

- **Vector Store Connection Errors**:
  - Ensure ChromaDB is properly installed
  - Check that the `chroma_rag_db` directory exists and has permissions
  - Verify it contains data by running `python -m app.vector_store --verify`

- **API Key Issues**:
  - Verify `MISTRAL_API_KEY` is correctly set in your `.env` file
  - Test the key with `python scripts/test_mistral_connection.py`

- **Validation Failures**:
  - Ensure Checkov is properly installed (`poetry run checkov --version`)
  - Check the generated code for syntax errors
  - Review the specific CIS control that's failing

### Logging

To enable verbose logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
poetry run python main.py
```

## Contribution

Contributions to TerraGuard are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.