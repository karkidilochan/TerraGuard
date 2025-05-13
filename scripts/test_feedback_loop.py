#!/usr/bin/env python
"""
Test script to demonstrate the feedback loop in TerraGuard

This script runs the RAG pipeline with intentionally problematic queries
to observe how the validation-feedback-regeneration loop improves code quality.
"""

import os
import sys
import json
import logging
import glob
import argparse
from typing import List, Dict, Any

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.vector_store import VectorStore, CHROMA_DB_NAME, CHROMA_COLLECTION_NAME
from app.rag_pipeline import RAGPipeline
from app.models import LLMClient
from app.validate_tf import validate_terraform_code
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feedback_loop_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_real_failures(failures_dir="real_failures", limit=5) -> List[Dict[str, Any]]:
    """Load real failures collected during normal operation"""
    failures = []
    
    # Check if the directory exists
    if not os.path.exists(failures_dir):
        logger.warning(f"Real failures directory '{failures_dir}' does not exist")
        return failures
    
    # Find all failure JSON files
    failure_files = glob.glob(os.path.join(failures_dir, "failure_*.json"))
    
    # Sort by timestamp (newest first)
    failure_files.sort(reverse=True)
    
    # Load the failures
    for file_path in failure_files[:limit]:
        try:
            with open(file_path, "r") as f:
                failure = json.load(f)
                
                # Check if it has the required fields
                if "query" in failure:
                    # Convert to our scenario format
                    scenario = {
                        "name": failure.get("name", f"Real failure: {os.path.basename(file_path)}"),
                        "query": failure["query"],
                        "source": "real_failure"  # Add source tag
                    }
                    failures.append(scenario)
        except Exception as e:
            logger.error(f"Error loading failure file {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(failures)} real failures")
    return failures

def analyze_improvement(final_state):
    """Analyze how the code improved through iterations"""
    retry_count = final_state.get("retry_count", 0)
    
    logger.info(f"Completed after {retry_count} retries")
    
    # Check validation results
    validation_results = final_state.get("validation_results", {})
    syntax_valid = validation_results.get("syntax_valid", False)
    cis_compliant = validation_results.get("cis_compliant", False)
    
    logger.info(f"Final validation: Syntax valid: {syntax_valid}, CIS compliant: {cis_compliant}")
    
    # Report on error count changes
    error_count = len(validation_results.get("errors", []))
    logger.info(f"Final error count: {error_count}")
    
    # Read the feedback logs to track progression
    feedback_logs_dir = os.path.join(os.getcwd(), "feedback_logs")
    log_files = [f for f in os.listdir(feedback_logs_dir) 
                if f.startswith("test_feedback_loop") and f.endswith(".json")]
    
    if log_files:
        logger.info("Error progression summary:")
        
        # Sort log files by iteration number - safely extracting iteration from JSON content
        # rather than relying on filename format which might change
        log_data_with_iteration = []
        
        for log_file in log_files:
            try:
                with open(os.path.join(feedback_logs_dir, log_file), "r") as f:
                    log_data = json.load(f)
                    # Extract iteration directly from file content
                    iteration = log_data.get("iteration", 0)
                    log_data_with_iteration.append((log_file, log_data, iteration))
            except Exception as e:
                logger.error(f"Error processing log file {log_file}: {str(e)}")
        
        # Sort by iteration
        log_data_with_iteration.sort(key=lambda x: x[2])  # Sort by iteration number (3rd element in tuple)
        
        # Process the sorted files
        for log_file, log_data, iteration in log_data_with_iteration:
            feedback = log_data.get("validation_feedback", "")
            
            # Count errors in feedback (approximate)
            error_lines = [line for line in feedback.split("\n") if line.strip()]
            error_count = len(error_lines)
            
            logger.info(f"Iteration {iteration}: ~{error_count} issues")

def run_test_scenario(pipeline, query, scenario_name, is_real_failure=False):
    """Run a test scenario with a specific query"""
    # Update run_type based on scenario source
    original_run_type = pipeline.run_type
    if is_real_failure:
        pipeline.run_type = "real_failure_test"
    else:
        pipeline.run_type = "test_feedback_loop"
    
    logger.info(f"=== Running scenario: {scenario_name} ===")
    logger.info(f"Query: {query}")
    logger.info(f"Using run_type: {pipeline.run_type}")
    
    # Run the full pipeline with the feedback loop
    result = pipeline.run(query)
    
    logger.info(f"Feedback loop completed for '{scenario_name}'")
    analyze_improvement(result)
    logger.info("=" * 50)
    
    # Restore original run_type
    pipeline.run_type = original_run_type
    
    return result

def main():
    """Main test function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the TerraGuard feedback loop")
    parser.add_argument("--use-real-failures", action="store_true", 
                        help="Use real failures collected during normal operation")
    parser.add_argument("--use-both", action="store_true",
                        help="Use both predefined scenarios and real failures")
    parser.add_argument("--max-failures", type=int, default=3,
                        help="Maximum number of real failures to test")
    args = parser.parse_args()
    
    load_dotenv()
    
    # Initialize components
    llm_client = LLMClient(
        provider="gemini",
        model="gemini-2.5-pro-preview-03-25",
        temperature=0.7,
    )
    
    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
    
    # Create a pipeline specifically for this test
    pipeline = RAGPipeline(
        vector_store,
        llm_client=llm_client,
        run_type="test_feedback_loop",
        max_retries=3
    )
    
    # Predefined test scenarios designed to trigger validation issues
    predefined_scenarios = [
        {
            "name": "S3 Bucket Missing Security Controls",
            "query": "Create an S3 bucket called terraform-data-bucket",
            "source": "predefined"
        },
        {
            "name": "IAM Role Without Principle of Least Privilege",
            "query": "Create an IAM role with admin access to S3",
            "source": "predefined"
        },
        {
            "name": "EC2 Instance Without Security Groups",
            "query": "Create an EC2 instance running Amazon Linux 2",
            "source": "predefined"
        }
    ]
    
    # Determine which scenarios to use
    test_scenarios = []
    
    if args.use_real_failures or args.use_both:
        # Load real failures
        real_failures = load_real_failures(limit=args.max_failures)
        test_scenarios.extend(real_failures)
        
        if not real_failures:
            logger.warning("No real failures found. Using predefined scenarios instead.")
            test_scenarios = predefined_scenarios
    
    if not args.use_real_failures or args.use_both or not test_scenarios:
        # Include predefined scenarios
        test_scenarios.extend(predefined_scenarios)
    
    logger.info(f"Running {len(test_scenarios)} test scenarios")
    
    # Run each scenario
    for scenario in test_scenarios:
        # Determine if this is a real failure
        is_real_failure = scenario.get("source") == "real_failure"
        run_test_scenario(pipeline, scenario["query"], scenario["name"], is_real_failure)
    
    logger.info("All test scenarios completed.")

if __name__ == "__main__":
    main() 