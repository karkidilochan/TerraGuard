import os
from typing import List
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from app.scrape_aws_tf import chunk_aws_resources


CHROMA_DB_NAME = "./chroma_rag_db"
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
        chunk_size=1000,
        chunk_overlap=50,
        batch_size=150,
    ):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"],
            keep_separator=False,
        )
        split_documents = text_splitter.split_documents(documents)
        for i in tqdm(
            range(0, len(split_documents), batch_size), desc="Storing documents..."
        ):
            batch_docs = split_documents[i : i + batch_size]
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


if __name__ == "__main__":

    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)

    # TODO: move this to automatically populate db after done experimenting with vector store
    if vector_store.is_store_empty:
        vector_store.store_documents(chunk_aws_resources())
    retrieved_docs = vector_store.get_db_instance().similarity_search(
        "How do I setup AWS Access Analyzer?"
    )
    print(retrieved_docs)
