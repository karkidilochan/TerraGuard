from langchain import hub

from app.schemas import State
from app.vector_store import vector_store


prompt = hub.pull("rlm/rag-prompt")


def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(
        state["search"]["query"],
        filter=lambda doc: doc.metadata.get("section") == state["query"]["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    answer = llm.invoke(
        prompt.invoke({"question": state["question"], "context": docs_content})
    )
    return {"answer": answer.content}


def analyze_query(state: State):
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}
