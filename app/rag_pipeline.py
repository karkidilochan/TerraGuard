from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from app.validate_tf import validate_terraform_code
from app.prompt_template import get_system_prompt
from app.vector_store import CHROMA_COLLECTION_NAME, CHROMA_DB_NAME, VectorStore
from app.schemas import Search, State
from dotenv import load_dotenv


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        llm_model: str = "mistral-large-latest",
        llm_provider: str = "mistralai",
    ):
        # Vector store
        self.vector_store = vector_store

        # LLM setup
        self.llm = init_chat_model(llm_model, model_provider=llm_provider)
        self.structured_llm = self.llm.with_structured_output(Search)

        # Build and compile the LangGraph
        self.graph = self._build_graph()

    def _build_graph(self):
        graph_builder = StateGraph(State)
        graph_builder.add_sequence([self.analyze_query, self.retrieve, self.generate])

        graph_builder.add_edge(START, "analyze_query")
        graph_builder.add_edge("generate", END)

        return graph_builder.compile()

    def analyze_query(self, state: State) -> State:
        """Parse the question into a structured Search object."""
        query = self.structured_llm.invoke(state["question"])
        return {"search": query}

    def retrieve(self, state: State) -> State:
        """Retrieve documents based on the query and section."""
        print(state["search"])
        retrieved_docs = self.vector_store.get_db_instance().similarity_search(
            query=state["search"]["query"],
            # filter={
            #     "resource_name": {"$in": state["search"]["resources"]},
            # },
        )
        return {"context": retrieved_docs}

    def generate(self, state: State) -> State:
        """Generate an answer using the retrieved context."""
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = get_system_prompt().format(
            question=state["question"], context=docs_content
        )
        answer = self.llm.invoke(prompt)
        return {"answer": answer.content}

    def run(self, question: str) -> State:
        """Run the RAG pipeline for a given question."""
        initial_state = {
            "question": question,
        }
        result = self.graph.invoke(initial_state)
        return result


# Example usage
if __name__ == "__main__":

    load_dotenv()

    vector_store = VectorStore(CHROMA_DB_NAME, CHROMA_COLLECTION_NAME)

    pipeline = RAGPipeline(
        vector_store,
        llm_model="mistral-large-latest",
        llm_provider="mistralai",
    )

    question = "How do I set up an AWS AccessAnalyzer?"

    # Run the pipeline
    result = pipeline.run(question)

    # Print results
    print("\nFinal Result:")
    print(f"Question: {result['question']}")
    print(f"Query: {result['search']}")
    print("Context:")
    for doc in result["context"]:
        print(f"- {doc.page_content} (Metadata: {doc.metadata})")
    print(f"Answer: {result['answer']}")

    # validate_terraform_code(result["answer"])
