import json
import os
import time
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.schema import Document


# 2. Load and process the JSON data
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


# 3. Split documents into chunks for better retrieval
def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        keep_separator=False,
    )

    return text_splitter.split_documents(documents)


# 4. Create embeddings using Mistral AI
def create_embeddings():
    """Create embeddings using Mistral AI."""
    return MistralAIEmbeddings(model="mistral-embed")  # or the appropriate model name


# 5. Create and persist the vector store
def create_vector_store(documents, embeddings, batch_size):
    """Create and persist a vector store."""
    # Create a new vector store
    vector_store = None

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

        # Sleep to prevent rate limiting
        time.sleep(1)

    return vector_store


# 6. Load the vector store for usage
def load_vector_store(embeddings):
    """Load an existing vector store."""
    return Chroma(
        persist_directory="./terraform_docs_vectorstore", embedding_function=embeddings
    )


# 7. Main function to build the vector store
def build_vector_store(json_file_path, batch_size):
    """Build a vector store from a JSON file."""
    print("Loading data from JSON...")
    data = load_data_from_json(json_file_path)

    print("Processing JSON data...")
    documents = process_json_data(data)

    print(f"Splitting {len(documents)} documents...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = create_embeddings()

    print("Building vector store...")
    vector_store = create_vector_store(chunks, embeddings, batch_size)

    print("Vector store built successfully!")
    return vector_store


# 8. Example usage in a RAG pipeline
def query_rag_model(query, vector_store, top_k=5):
    """Query the RAG model."""
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=top_k)

    # Here you would typically pass these documents to your LLM
    # For example with Mistral AI:
    # from langchain_mistralai import ChatMistralAI
    # llm = ChatMistralAI(model="mistral-medium")
    # result = llm.invoke(query + "\n\nContext: " + "\n\n".join([doc.page_content for doc in docs]))

    return docs


# Example execution
if __name__ == "__main__":
    # Path to your JSON file containing all 2086 items
    json_file_path = "aws_resources.json"

    # Build the vector store
    vector_store = build_vector_store(json_file_path, batch_size=20)

    # Example query
    query = "How do I set up an AWS AccessAnalyzer?"
    results = query_rag_model(query, vector_store)

    print(f"\nExample query: {query}")
    print(f"Found {len(results)} relevant documents")

    # Display the first result
    if results:
        print("\nTop result:")
        print(f"Title: {results[0].metadata.get('title', 'No title')}")
        print(
            f"Description: {results[0].metadata.get('description', 'No description')}"
        )
        print("Content excerpt: " + results[0].page_content[:200] + "...")
