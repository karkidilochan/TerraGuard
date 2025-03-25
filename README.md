# TerraGuard: Secure Terraform RAG Pipeline with CIS Benchmark Compliance

TerraGuard is an intelligent Terraform code generation system with integrated CIS AWS Foundations Benchmark compliance validation.

## Features

- **CIS Benchmark Compliance**: Generate Terraform code that adheres to AWS CIS Foundations Benchmark security controls.
- **RAG Pipeline**: Leverages Retrieval-Augmented Generation to produce accurate Terraform code based on your requirements.
- **Automated Validation**: Validates generated code for both syntax and security compliance using Checkov.
- **Compliance Reporting**: Provides detailed reports on which CIS controls are met or violated.
- **API Interface**: Exposes functionality through a FastAPI-based REST API.

## Prerequisites

Ensure you have the following installed:

- Python 3.9+
- [Poetry](https://python-poetry.org/docs/)
- [Chromadb](https://github.com/chroma-core/chroma)
- [Checkov](https://www.checkov.io/)
- A Mistral API key (if using the hosted service)

## Step 1: Install Dependencies with Poetry

First, ensure Poetry is installed. Then, install the project dependencies:

```bash
poetry install
```

## Step 2: Setup Environment Variables

Create a `.env` file in the root directory:

```
MISTRAL_API_KEY=<your-api-key>
```

## Step 3: Set Up the Vector Store

Ensure the `chroma_rag_db` directory exists and contains your indexed data. If not, you'll need to create it.
Running the vector_store.py file as main will create the required vector store with the Terraform documentation for you.

```bash
poetry run python -m app.vector_store
```

## Step 4: Run the CIS Benchmark Setup Scripts

To prepare the CIS benchmark data for use in the pipeline:

1. Extract the benchmarks from the PDF (if not already done):
```bash
cd scripts
python cis_benchmark_extract.py
```

2. Enhance the benchmarks with resource type and attribute metadata:
```bash
python enhance_cis_benchmark.py
```

## Step 5: Test the Compliance Pipeline

Run a test script to verify the compliance validation works correctly:

```bash
python scripts/test_compliance_pipeline.py
```

## Step 6: Run the Web API

To run the TerraGuard web API:

```bash
RUN_MODE=server poetry run python main.py
```

The API will be available at http://localhost:8000 with Swagger documentation at http://localhost:8000/docs

## API Usage

### Generate Terraform Code

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
  "terraform_code": "...",
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

## Using TerraGuard as a CLI

To run the script with Poetry:

```bash
poetry run python main.py
```

## Troubleshooting

- Ensure `MISTRAL_API_KEY` is correctly set.
- Verify the vector store path is correct and contains data.
- Ensure all dependencies are installed without errors.
- Check that Checkov is properly installed for security validation.
