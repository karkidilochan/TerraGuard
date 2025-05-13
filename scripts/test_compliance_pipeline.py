#!/usr/bin/env python
import os
import sys
import json
from dotenv import load_dotenv
import logging
import datetime

# Add the parent directory to the path to import the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import VectorStore, CHROMA_DB_NAME, CHROMA_COLLECTION_NAME
from app.rag_pipeline import RAGPipeline, test_and_validate_rag_pipeline
from app.models import LLMClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample queries to test
TEST_QUERIES = [
    "How do I create an S3 bucket with versioning and encryption that complies with CIS benchmarks?",
    "Create IAM roles with least privilege access following CIS security controls",
    "Set up CloudTrail logging and monitoring following security best practices",
    "Configure secure VPC with private subnets and security groups according to CIS benchmarks",
    "Create an encrypted RDS instance with enhanced monitoring",
]


def pretty_print_result(result, query_idx):
    """Pretty print the results of a query."""
    print("\n" + "=" * 80)
    print(f"TEST QUERY #{query_idx+1}: {result['question']}")
    print("=" * 80)

    # Print answer
    print("\nGENERATED TERRAFORM CODE:\n")
    print(result["answer"])

    # Print validation results
    print("\nVALIDATION RESULTS:")
    print(f"Status: {result['validation_summary']['status']}")
    print(f"Syntax Valid: {result['validation_summary']['syntax_valid']}")
    print(f"CIS Compliant: {result['validation_summary']['cis_compliant']}")
    print(
        f"CIS Controls Referenced: {', '.join(result['validation_summary']['referenced_cis_controls'])}"
    )

    # Print issues if any
    if issues := result["validation_summary"].get("issues", []):
        print(f"\nVALIDATION ISSUES: ({len(issues)})")
        for issue in issues:
            print(f"- {issue}")

    # Print Checkov output file path if available
    if result["validation_summary"].get("checkov_output_file"):
        print(
            f"\nCHECKOV RESULTS SAVED TO: {result['validation_summary']['checkov_output_file']}"
        )

    # Print compliance report if available
    if result["validation_results"].get("compliance_report"):
        print("\nCOMPLIANCE REPORT:")
        print(result["validation_results"]["compliance_report"])

    # Print full errors if any and not already shown in issues
    if (
        "errors" in result["validation_results"]
        and result["validation_results"]["errors"]
    ):
        print("\nDETAILED ERRORS:")
        for error in result["validation_results"]["errors"]:
            print(f"- {error}")


def save_results_to_json(results, output_file):
    """Save the results to a JSON file."""
    # Convert to serializable format
    serializable_results = []

    for i, result in enumerate(results):
        # Create a clean dictionary without Document objects
        clean_result = {
            "query": result["question"],
            "terraform_code": result["answer"],
            "referenced_cis_controls": result.get("referenced_cis_controls", []),
            "validation_summary": {
                "status": result["validation_summary"]["status"],
                "syntax_valid": result["validation_summary"]["syntax_valid"],
                "cis_compliant": result["validation_summary"]["cis_compliant"],
                "referenced_cis_controls": result["validation_summary"][
                    "referenced_cis_controls"
                ],
                "issues": result["validation_summary"].get("issues", []),
            },
            "compliance_report": result["validation_results"].get("compliance_report"),
            "checkov_output_file": result["validation_results"].get(
                "checkov_output_file"
            ),
            "errors": result["validation_results"].get("errors", []),
        }
        serializable_results.append(clean_result)

    # Save to file
    with open(output_file, "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    # Load environment variables
    load_dotenv()

    # Initialize LLM client
    llm_client = LLMClient(
        provider="gemini",
        model="gemini-2.5-pro-preview-03-25",
        temperature=0.7,
    )

    # Initialize vector store
    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
    
    # Check if CIS benchmark data is already loaded
    # Note: This check is approximate as we can't directly query by type in the Chroma DB instance
    existing_docs = vector_store.get_db_instance().get()
    doc_count = len(existing_docs.get("documents", []))
    
    # If the vector store is empty, populate it with AWS resources and CIS benchmark data
    if vector_store.is_store_empty:
        logger.info("Vector store is empty. Populating with AWS resources and CIS benchmark data...")
        from app.scrape_aws_tf import chunk_aws_resources
        
        # Load AWS resources
        vector_store.store_documents(chunk_aws_resources("aws_resources.json"))
        vector_store.store_documents(chunk_aws_resources("aws_data_sources.json"))
        vector_store.store_documents(chunk_aws_resources("aws_ephemeral.json"))
        
        # Load CIS benchmark data
        vector_store.load_cis_benchmark_data()
    else:
        logger.info(f"Vector store already contains {doc_count} documents")
        
        # Check if we need to add CIS benchmark data
        # Try a test query for CIS controls to check if they're in the database
        test_results = vector_store.get_db_instance().similarity_search(
            query="CIS benchmark S3 bucket compliance controls",
            k=5,
            filter={"type": "cis_control"},
        )
        
        if not test_results:
            logger.info("No CIS benchmark data found. Adding it to the vector store...")
            vector_store.load_cis_benchmark_data()
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        vector_store=vector_store,
        llm_client=llm_client,
    )
    
    # Run a test query
    query = TEST_QUERIES[0]  # Use the first query for testing
    
    logger.info(f"Testing query: {query}")
    
    test_and_validate_rag_pipeline(pipeline, query)


if __name__ == "__main__":
    main()
