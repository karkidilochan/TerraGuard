# RAG Pipeline Setup and Usage Guide

This README will guide you through setting up and running a RAG (Retrieval-Augmented Generation) pipeline using Mistral as the LLM.

## Prerequisites

Ensure you have the following installed:

- Python 3.9+
- [Poetry](https://python-poetry.org/docs/)
- [Chromadb](https://github.com/chroma-core/chroma)
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

```
poetry run python -m app.vector_store
```

## Step 4: Run the RAG Pipeline

To run the script with Poetry:

```bash
poetry run python -m app.rag_pipeline
```

### Expected Output

```plaintext
Final Result:
Question: How do I set up an AWS AccessAnalyzer?
Query: setup AWS AccessAnalyzer
Context:
- AWS AccessAnalyzer is a service that helps you identify resources... (Metadata: source:aws_docs)
Answer: To set up AWS AccessAnalyzer, navigate to the IAM console, select Access Analyzer, and create an analyzer...
```

## Troubleshooting

- Ensure `MISTRAL_API_KEY` is correctly set.
- Verify the vector store path is correct and contains data.
- Ensure all dependencies are installed without errors.
