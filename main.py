import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from app.vector_store import VectorStore, CHROMA_DB_NAME, CHROMA_COLLECTION_NAME
from app.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import uvicorn

app = FastAPI(title="TerraGuard API", description="API for generating secure AWS Terraform configurations")

def check_vector_store_exists():
    """Check if the vector store directory exists and is populated"""
    chroma_dir = Path(CHROMA_DB_NAME)
    return chroma_dir.exists() and any(chroma_dir.iterdir())

class QueryRequest(BaseModel):
    """Request model for the generate endpoint"""
    query: str

class ValidationSummary(BaseModel):
    """Summary of validation results"""
    syntax_valid: bool
    cis_compliant: bool
    referenced_cis_controls: List[str]
    error_count: int

class QueryResponse(BaseModel):
    """Response model for the generate endpoint"""
    query: str
    terraform_code: str
    referenced_cis_controls: List[str]
    validation_summary: ValidationSummary
    compliance_report: Optional[str] = None

# Initialize the pipeline on server startup
@app.on_event("startup")
def startup_event():
    # Load environment variables
    load_dotenv()
    
    # Check if MISTRAL_API_KEY is set
    if not os.getenv("MISTRAL_API_KEY"):
        raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

    # Initialize VectorStore with correct parameters
    global vector_store, rag_pipeline
    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
    
    if not check_vector_store_exists():
        vector_store.initialize_store()

    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(vector_store)

@app.get("/")
def read_root():
    """Root endpoint to verify the API is running"""
    return {"status": "ok", "message": "TerraGuard API is running"}

@app.post("/generate", response_model=QueryResponse)
def generate_terraform_code(request: QueryRequest):
    """Generate Terraform code based on the query with CIS compliance validation"""
    try:
        # Run the RAG pipeline
        result = rag_pipeline.run(request.query)
        
        # Prepare and return the response
        return QueryResponse(
            query=request.query,
            terraform_code=result["answer"],
            referenced_cis_controls=result.get("referenced_cis_controls", []),
            validation_summary=ValidationSummary(
                syntax_valid=result["validation_summary"]["syntax_valid"],
                cis_compliant=result["validation_summary"]["cis_compliant"],
                referenced_cis_controls=result["validation_summary"]["referenced_cis_controls"],
                error_count=result["validation_summary"]["error_count"]
            ),
            compliance_report=result["validation_results"].get("compliance_report")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    # Load environment variables
    load_dotenv()
    
    # Check if MISTRAL_API_KEY is set
    if not os.getenv("MISTRAL_API_KEY"):
        raise EnvironmentError("MISTRAL_API_KEY not found in environment variables")

    print("Checking vector store...")
    # Initialize VectorStore with correct parameters
    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)
    
    if not check_vector_store_exists():
        print("Vector store not found. Creating and populating vector store...")
        vector_store.initialize_store()
    else:
        print("Vector store found. Loading existing store...")

    print("\nInitializing RAG pipeline...")
    rag = RAGPipeline(vector_store)
    
    try:
        # Example query
        query = "How do I set up an AWS S3 bucket with versioning and encryption that complies with CIS benchmarks?"
        print(f"\nProcessing query: {query}")
        
        # Use the run method which properly handles the state
        result = rag.run(query)
        
        # Print results in structured way
        print("\nFinal Result:")
        print(f"Question: {result['question']}")
        print(f"Query: {result['search']}")
        print("Context:")
        for doc in result["context"]:
            print(f"- {doc.page_content[:100]}... (Metadata: {doc.metadata})")
        print(f"Answer: {result['answer']}")
        
        # Print validation results
        print("\nValidation Results:")
        print(f"Syntax Valid: {result['validation_summary']['syntax_valid']}")
        print(f"CIS Compliant: {result['validation_summary']['cis_compliant']}")
        print(f"Referenced CIS Controls: {result['validation_summary']['referenced_cis_controls']}")
        
        if result["validation_results"].get("compliance_report"):
            print("\nCompliance Report:")
            print(result["validation_results"]["compliance_report"])
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    # Check if we should run as a script or as a web server
    if os.getenv("RUN_MODE", "script").lower() == "server":
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        main()
