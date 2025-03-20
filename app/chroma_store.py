import json
import os
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import (
    HuggingFaceBgeEmbeddings,
    HuggingFaceEmbeddings,
    SentenceTransformerEmbeddings,
)

# Choose one of these local embedding models


def get_bge_embeddings():
    """Get BGE embeddings (best quality but slower)."""
    model_name = (
        "BAAI/bge-large-en-v1.5"  # Options: bge-small-en, bge-base-en, bge-large-en
    )

    return HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use 'cuda' if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )


def get_e5_embeddings():
    """Get E5 embeddings (fast and high quality)."""
    model_name = "intfloat/e5-small-v2"  # Options: e5-small, e5-base, e5-large

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # Use 'cuda' if you have a GPU
    )


def get_mpnet_embeddings():
    """Get MPNet embeddings (balanced speed/quality)."""
    model_name = "sentence-transformers/all-mpnet-base-v2"

    return SentenceTransformerEmbeddings(model_name=model_name)


def get_minilm_embeddings():
    """Get MiniLM embeddings (fastest)."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return SentenceTransformerEmbeddings(model_name=model_name)


# Load and process the JSON data
def load_data_from_json(file_path):
    """Load data from a JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def process_json_data(data):
    """Extract and process text from the JSON data."""
    documents = []

    for item in tqdm(data, desc="Processing documents"):
        # Extract the important content from each item
        title = item.get("title", "")
        description = item.get("description", "")
        resource_name = item.get("resource_name", "")

        # Extract full content or construct from sections
        content = item.get("full_content", "")
        if not content and "sections" in item:
            sections = item["sections"]
            content = "\n\n".join([sections.get(key, "") for key in sections])

        # Create metadata for better retrieval
        metadata = {
            "title": title,
            "resource_name": resource_name,
            "description": description,
        }

        # Add subcategory and other metadata if available
        if "metadata" in item:
            for key, value in item["metadata"].items():
                metadata[f"metadata_{key}"] = value

        # Create a document object
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    return documents


# Split documents into chunks for better retrieval
def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=False,
    )

    return text_splitter.split_documents(documents)


# Create and persist the vector store with batching
def create_vector_store_batched(documents, embeddings, batch_size=100):
    """Create and persist a vector store with batching."""
    # Initialize a vector store
    vector_store = None

    # Process in batches
    for i in tqdm(range(0, len(documents), batch_size), desc="Processing batches"):
        batch = documents[i : i + batch_size]

        # Create or update the vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory="./terraform_docs_vectorstore",
            )
        else:
            # Add documents to existing store
            vector_store.add_documents(documents=batch)

        # Persist after each batch
        vector_store.persist()

    return vector_store


# Main function to build the vector store
def build_vector_store(json_file_path, embedding_type="minilm", batch_size=100):
    """Build a vector store from a JSON file.

    Args:
        json_file_path: Path to the JSON file
        embedding_type: Type of embedding to use ("bge", "e5", "mpnet", or "minilm")
        batch_size: Number of documents to process in each batch
    """
    print("Loading data from JSON...")
    data = load_data_from_json(json_file_path)

    print("Processing JSON data...")
    documents = process_json_data(data)

    print(f"Splitting {len(documents)} documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print(f"Creating {embedding_type} embeddings...")
    if embedding_type == "bge":
        embeddings = get_bge_embeddings()
    elif embedding_type == "e5":
        embeddings = get_e5_embeddings()
    elif embedding_type == "mpnet":
        embeddings = get_mpnet_embeddings()
    else:  # default to minilm
        embeddings = get_minilm_embeddings()

    print("Building vector store in batches...")
    vector_store = create_vector_store_batched(chunks, embeddings, batch_size)

    print("Vector store built successfully!")
    return vector_store


# Example usage in a RAG pipeline
def query_rag_model(query, vector_store, top_k=5):
    """Query the RAG model."""
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=top_k)
    return docs


# Example execution
if __name__ == "__main__":
    # Path to your JSON file containing all 2086 items
    json_file_path = "aws_resources.json"

    try:
        # Choose embedding type: "bge", "e5", "mpnet", or "minilm"
        embedding_type = "bge"  # Fastest option

        # Build the vector store
        print(
            f"Starting vector store creation process with {embedding_type} embeddings..."
        )
        vector_store = build_vector_store(
            json_file_path, embedding_type=embedding_type, batch_size=200
        )

        # Example query
        query = "How do I set up an AWS AccessAnalyzer?"
        print(f"\nRunning example query: {query}")
        results = query_rag_model(query, vector_store)

        print(f"Found {len(results)} relevant documents")

        # Display the first result
        if results:
            print("\nTop result:")
            print(f"Title: {results[0].metadata.get('title', 'No title')}")
            print(
                f"Description: {results[0].metadata.get('description', 'No description')}"
            )
            print("Content excerpt: " + results[0].page_content[:200] + "...")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
