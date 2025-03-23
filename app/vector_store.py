import time
from typing import List
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from scrape_aws_tf import chunk_aws_resources


class VectorStore:

    def __init__(self, embedding_model, persist_directory, collection_name):
        self.persistent_directory = persist_directory
        self.embedding_model = embedding_model
        self.vector_db = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )

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
            range(0, len(split_documents), batch_size), desc="Storing documents"
        ):
            batch_docs = split_documents[i : i + batch_size]
            self.vector_db.add_documents(documents=batch_docs)

    def get_db_instance(self):
        return self.vector_db


if __name__ == "__main__":
    # TODO: compare performance of codebert and sentencetransformers

    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        # model_name="sentence-transformers/all-MiniLM-L6-v2"
        model_name="microsoft/codebert-base"
        # model_name="sentence-transformers/all-mpnet-base-v2"
    )

    vector_store = VectorStore(embeddings, "./chroma_rag_db", "tf_aws_resources")
    vector_store.store_documents(chunk_aws_resources())
    retrieved_docs = vector_store.get_db_instance().similarity_search(
        "How do I setup AWS Access Analyzer?"
    )
    print(retrieved_docs)

    # vector_store.get_db_instance().similarity_search(
    #     state["query"]["query"],
    #     filter=lambda doc: doc.metadata.get("section") == state["query"]["section"],
    # )

    # vector_store.store_documents(chunk_aws_resources())
    # retrieved_docs = vector_store.get_db_instance().similarity_search(
    #     "How do I setup AWS Access Analyzer?"
    # )
    # print(retrieved_docs)

    # import bs4
    # from langchain_community.document_loaders import WebBaseLoader

    # bs4_strainer = bs4.SoupStrainer(
    #     class_=("post-title", "post-header", "post-content")
    # )

    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs={"parse_only": bs4_strainer},
    # )

    # docs = loader.load()
    # from langchain_text_splitters import RecursiveCharacterTextSplitter

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     add_start_index=True,
    # )

    # all_splits = text_splitter.split_documents(docs)

    # vector_store = VectorStore(embedding_model, "./dummy_db", "resources")

    # document_ids = vector_store.store_documents(all_splits)
    # question = "what is self reflection?"

    # retrieved_docs = vector_store.get_db_instance().similarity_search(question)
    # print(retrieved_docs)
