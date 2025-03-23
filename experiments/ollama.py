from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import Language
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os

from app.scrape_aws_tf import chunk_aws_resources


all_chunks = chunk_aws_resources()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
split_docs = splitter.split_documents(all_chunks)


embeddings = OllamaEmbeddings(model="llama2:7b")
doc_result = embeddings.embed_documents([all_chunks])

db = FAISS.from_documents(split_docs, embeddings)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

code_llm = Ollama(model="codellama")

prompt_RAG = """
    You are a proficient python developer. Respond with the syntactically correct code for the question below. Make sure you follow these rules:
    1. Use context to understand the APIs and how to use them.
    2. Ensure all the requirements in the question are met.
    3. Ensure the output code syntax is correct.
    4. All required dependencies should be imported above the code.
    Question:
    {question}
    Context:
    {context}
    Helpful Response:
    """
prompt_RAG_template = PromptTemplate(
    template=prompt_RAG, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_llm(
    llm=code_llm,
    prompt=prompt_RAG_template,
    retriever=retriever,
    return_source_documents=True,
)

user_question = input("Enter Query to generate Code: ")
# user_question = "Create a python function that will load the data from vectord using similarity score threshold retriever."
results = qa_chain({"query": user_question})

print(results)
