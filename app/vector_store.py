import faiss

from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from embeddings import embedding_model

from langchain.vectorstores import Chroma

vector_store = FAISS(
    embedding_function=embeddings,
    index_path="app/indexes/rag",
    docstore=InMemoryDocstore(),
)
