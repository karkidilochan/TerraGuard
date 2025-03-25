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
from app.rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample queries to test
TEST_QUERIES = [
    "How do I create an S3 bucket with versioning and encryption that complies with CIS benchmarks?",
    "Create IAM roles with least privilege access following CIS security controls",
    "Set up CloudTrail logging and monitoring following security best practices",
    "Configure secure VPC with private subnets and security groups according to CIS benchmarks",
    "Create an encrypted RDS instance with enhanced monitoring"
]

def pretty_print_result(result, query_idx):
    """Pretty print the results of a query."""
    print("\n" + "="*80)
    print(f"TEST QUERY #{query_idx+1}: {result['question']}")
    print("="*80)
    
    # Print answer
    print("\nGENERATED TERRAFORM CODE:\n")
    print(result['answer'])
    
    # Print validation results
    print("\nVALIDATION RESULTS:")
    print(f"Status: {result['validation_summary']['status']}")
    print(f"Syntax Valid: {result['validation_summary']['syntax_valid']}")
    print(f"CIS Compliant: {result['validation_summary']['cis_compliant']}")
    print(f"CIS Controls Referenced: {', '.join(result['validation_summary']['referenced_cis_controls'])}")
    
    # Print issues if any
    if issues := result['validation_summary'].get('issues', []):
        print(f"\nVALIDATION ISSUES: ({len(issues)})")
        for issue in issues:
            print(f"- {issue}")
    
    # Print Checkov output file path if available
    if result["validation_summary"].get("checkov_output_file"):
        print(f"\nCHECKOV RESULTS SAVED TO: {result['validation_summary']['checkov_output_file']}")
    
    # Print compliance report if available
    if result["validation_results"].get("compliance_report"):
        print("\nCOMPLIANCE REPORT:")
        print(result["validation_results"]["compliance_report"])
    
    # Print full errors if any and not already shown in issues
    if "errors" in result["validation_results"] and result["validation_results"]["errors"]:
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
                "referenced_cis_controls": result["validation_summary"]["referenced_cis_controls"],
                "issues": result["validation_summary"].get("issues", [])
            },
            "compliance_report": result["validation_results"].get("compliance_report"),
            "checkov_output_file": result["validation_results"].get("checkov_output_file"),
            "errors": result["validation_results"].get("errors", [])
        }
        serializable_results.append(clean_result)
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if MISTRAL_API_KEY is set
    if not os.getenv("MISTRAL_API_KEY"):
        raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

    # Check if Terraform and Checkov are installed
    import subprocess
    try:
        subprocess.run(["terraform", "--version"], check=True, capture_output=True)
        logger.info("Terraform is installed and accessible")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Terraform not found in PATH. Please make sure Terraform is installed and accessible.")
        return
    
    try:
        subprocess.run("checkov --version", check=True, capture_output=True, shell=True)
        logger.info("Checkov is installed and accessible")
    except (subprocess.SubprocessError, FileNotFoundError):
        logger.error("Checkov not found in PATH. Please make sure Checkov is installed and accessible.")
        return


    logger.info("Initializing vector store...")
    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
    
    if not os.path.exists(CHROMA_DB_NAME) or not os.listdir(CHROMA_DB_NAME):
        logger.info("Vector store not found. Creating and populating vector store...")
        vector_store.initialize_store()
    else:
        logger.info("Vector store found. Using existing store...")

    logger.info("Initializing RAG pipeline...")
    rag_pipeline = RAGPipeline(vector_store)
    
    # Ensure checkov_output directory exists
    os.makedirs("checkov_output", exist_ok=True)
    
    # Run the tests
    results = []
    
    logger.info(f"Starting tests with {len(TEST_QUERIES)} queries...")
    
    for i, query in enumerate(TEST_QUERIES):
        logger.info(f"Running query #{i+1}: {query}")
        
        try:
            # Run the RAG pipeline
            result = rag_pipeline.run(query)
            results.append(result)
            
            # Pretty print the result
            pretty_print_result(result, i)
            
        except Exception as e:
            logger.error(f"Error processing query {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Save results to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results_to_json(results, f"compliance_test_results_{timestamp}.json")
    
    logger.info("All tests completed!")

if __name__ == "__main__":
    main() 