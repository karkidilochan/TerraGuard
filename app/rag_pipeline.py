from typing_extensions import TypedDict, List, Annotated, Literal
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from embeddings import embedding_model


# search
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run"]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section of the document to search in",
    ]


# state
class State(TypedDict):
    question: str
    context: List[Document]
    query: Search
    answer: str


persist_directory = "./terraform_docs_vectorstore"

vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory,
)


# steps
def retrieve(state: State):

    retrieved_docs = vector_store.similarity_search(
        state["query"]["query"],
        filter=lambda doc: doc.metadata.get("section") == state["query"]["section"],
    )
    print(retrieved_docs)
    return {"context": retrieved_docs}


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     answer = llm.invoke(
#         prompt.invoke({"question": state["question"], "context": docs_content})
#     )
#     return {"answer": answer.content}


def analyze_query(state: State):
    from langchain.chat_models import init_chat_model

    llm = init_chat_model("mistral-large-latest", model_provider="mistralai")
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


# control flow
# from langgraph.graph import START, StateGraph

# graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
# graph_builder.add_edge(START, "analyze_query")
# graph = graph_builder.compile()

if __name__ == "__main__":
    question = "How do I set up an AWS AccessAnalyzer?"

    retrieved_docs = vector_store.similarity_search(question)
    print(retrieved_docs)

    # answer = llm.invoke(prompt.invoke({"question": question, "context": docs_content}))
