import os
from typing import List
import json
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.scrape_aws_tf import chunk_aws_resources


CHROMA_DB_NAME = "./chroma/rag_db"
# CHROMA_DB_NAME = "./chroma_rag_db"
CHROMA_COLLECTION_NAME = "tf_aws_resources"


class VectorStore:

    def __init__(self, persist_directory, collection_name, embedding_model="mpnet"):
        self.persistent_directory = persist_directory
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )

        self.check_if_empty()

    def store_documents(
        self,
        documents: List[Document],
        batch_size=150,
    ):
        for i in tqdm(
            range(0, len(documents), batch_size), desc="Storing documents..."
        ):
            batch_docs = documents[i : i + batch_size]
            self.vector_db.add_documents(documents=batch_docs)

    def get_db_instance(self):
        return self.vector_db

    def check_if_empty(self):
        is_empty = False
        if not os.path.exists(self.persistent_directory):
            print(
                f"Vector store directory '{self.persistent_directory}' does not exist. Creating ... /n Run the store_documents() function to populate it."
            )
            is_empty = True
        else:
            # Check if the collection is empty
            existing_docs = self.get_db_instance().get()
            doc_count = len(existing_docs.get("documents", []))
            if doc_count == 0:
                print(
                    f"Vector store '{self.persistent_directory}' exists but is empty. Run the store_documents() function to populate it."
                )
                is_empty = True
            else:
                print(
                    f"Vector store contains {doc_count} documents. Skipping population."
                )
        self.is_store_empty = is_empty

    def load_cis_benchmark_data(self, cis_benchmark_file: str = "scripts/cis_benchmark_enhanced.json") -> bool:
        """
        Load CIS benchmark data from enhanced JSON file and add it to the vector store.
        
        Args:
            cis_benchmark_file: Path to the enhanced CIS benchmark JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if absolute path was provided, if not prepend current working directory
            if not os.path.isabs(cis_benchmark_file):
                # Try both the current directory and one level up
                potential_paths = [
                    os.path.join(os.getcwd(), cis_benchmark_file),
                    os.path.join(os.getcwd(), '..', cis_benchmark_file),
                    os.path.join(os.path.dirname(os.getcwd()), cis_benchmark_file)
                ]
                
                for path in potential_paths:
                    if os.path.exists(path):
                        cis_benchmark_file = path
                        break
            
            if not os.path.exists(cis_benchmark_file):
                print(f"Error: CIS benchmark file not found at {cis_benchmark_file}")
                print("Checked locations:")
                for path in potential_paths:
                    print(f"- {path}")
                return False
            
            print(f"Loading CIS benchmark data from {cis_benchmark_file}...")
                
            with open(cis_benchmark_file, 'r') as f:
                cis_controls = json.load(f)
                
            print(f"Loaded {len(cis_controls)} CIS controls.")
            
            # Convert CIS controls to Documents
            documents = []
            for control in cis_controls:
                control_id = control.get('control_id', 'unknown')
                
                # Create a rich text representation of the control
                content = f"""
                # CIS Control {control_id}: {control['title']}
                
                ## Description
                {control.get('description', '')}
                
                ## Rationale
                {control.get('rationale', '')}
                
                ## Implementation Requirements
                {control.get('remediation', '')}
                
                ## Audit Procedure
                {control.get('audit_procedure', '')}
                
                ## AWS Resources
                {', '.join(control.get('resource_type', []))}
                
                ## Related Checkov Policies
                {', '.join(control.get('checkov_policies', []))}
                """
                
                # Create metadata for filtering
                metadata = {
                    "type": "cis_control",
                    "control_id": control_id,
                    "title": control['title'],
                    "automation_status": control.get('automation_status', ''),
                    "profile": control.get('profile', ''),
                    "resource_types": ", ".join(control.get('resource_type', [])),
                    "checkov_policies": ", ".join(control.get('checkov_policies', [])),
                    "section": "CIS Controls"
                }
                
                # Create the document
                doc = Document(
                    page_content=content.strip(),
                    metadata=metadata
                )
                
                documents.append(doc)
                
            # Keep chunks larger to preserve context - don't split up controls
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,  # Larger chunks to keep controls together
                chunk_overlap=100,
                separators=["\n\n", "\n", " ", ""],
                keep_separator=False,
            )
            
            chunked_docs = text_splitter.split_documents(documents)
            print(f"Split into {len(chunked_docs)} chunks.")
            
            # Store the documents
            self.store_documents(chunked_docs)
            print("CIS benchmark data successfully added to vector store.")
            
            return True
            
        except Exception as e:
            print(f"Error loading CIS benchmark data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":

    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)

    # TODO: move this to automatically populate db after done experimenting with vector store
    if vector_store.is_store_empty:
        vector_store.store_documents(chunk_aws_resources("aws_resources.json"))
        vector_store.store_documents(chunk_aws_resources("aws_data_sources.json"))
        vector_store.store_documents(chunk_aws_resources("aws_ephemeral.json"))
        # Add CIS benchmark data to the vector store
        vector_store.load_cis_benchmark_data()
    
    # Test query using CIS benchmark terms
    retrieved_docs = vector_store.get_db_instance().similarity_search(
        query="How do I configure an S3 bucket that complies with CIS benchmarks for encryption and public access?",
        k=5,
        filter=None  # Remove the filter to allow CIS controls to be retrieved
    )
    
    print("\nTest query results:")
    for i, doc in enumerate(retrieved_docs):
        print(f"\nResult {i+1}:")
        print(f"Type: {doc.metadata.get('type', 'Unknown')}")
        if doc.metadata.get('type') == 'cis_control':
            print(f"Control: {doc.metadata.get('control_id')} - {doc.metadata.get('title')}")
        else:
            print(f"Resource: {doc.metadata.get('resource_name', 'Unknown')}")
        print(f"First 200 chars: {doc.page_content[:200]}...")
