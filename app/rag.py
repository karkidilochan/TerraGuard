from langchain import hub
import os
from typing_extensions import TypedDict, List, Annotated, Literal
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from schemas import State


prompt = hub.pull("rlm/rag-prompt")


class RAGPipeline:
    def __init__(self, persist_directory="./terraform_docs_vectorstore"):
        """Initialize the RAG pipeline with ChromaDB and embeddings."""
        # Set up embedding model (replace with your preferred model)
        self.embedding_function = MistralAIEmbeddings(model="mistral-embed")

        # Initialize ChromaDB with persistent storage
        self.vectorstore = Chroma(
            embedding_function=self.embedding_function,
            persist_directory=persist_directory,
        )

        self.llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

        # Define prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["question", "context"],
            template="""Given the following question and context, provide a concise and accurate answer:

            Question: {question}

            Context: {context}

            Answer:""",
        )

    def add_documents(self, documents: List[Document]):
        """Add documents to the vectorstore."""
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()

    def retrieve(self, state: State) -> State:
        """Retrieve relevant documents based on the search query."""
        # Perform similarity search
        context = self.vectorstore.similarity_search(
            state["search"]["query"],
            k=5,  # Number of documents to retrieve
            filter=(
                {"section": state["search"]["section"]}
                if state["search"]["section"]
                else None
            ),
        )

        # Update state with retrieved context
        state["context"] = context
        return state

    def generate(self, state: State) -> State:
        """Generate answer using the LLM based on question and context."""
        # Format context
        context_str = "\n".join([doc.page_content for doc in state["context"]])

        # Create prompt
        prompt = self.prompt_template.format(
            question=state["question"], context=context_str
        )

        # Generate answer
        answer = self.llm.predict(prompt)
        state["answer"] = answer.strip()
        return state

    def process_query(self, question: str, section: str = None) -> State:
        """Process a complete query through the RAG pipeline."""
        # Initialize state
        state: State = {
            "question": question,
            "context": [],
            "search": {"query": question, "section": section or ""},
            "answer": "",
        }

        # Run pipeline
        state = self.retrieve(state)
        state = self.generate(state)

        return state


# Example usage
def main():
    # Initialize pipeline
    pipeline = RAGPipeline(persist_directory="./terraform_docs_vectorstore")

    # Example documents (you'd normally load these from your data source)
    # sample_docs = [
    #     Document(
    #         page_content="The capital of France is Paris.",
    #         metadata={"section": "geography"},
    #     ),
    #     Document(
    #         page_content="France is known for its wine production.",
    #         metadata={"section": "culture"},
    #     ),
    # ]

    # # Add documents to vectorstore
    # pipeline.add_documents(sample_docs)

    # Process a query
    result = pipeline.process_query(
        question="What is the capital of France?", section="geography"
    )

    # Print results
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print("Context:")
    for doc in result["context"]:
        print(f"- {doc.page_content}")


if __name__ == "__main__":
    main()


# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(
#         state["search"]["query"],
#         filter=lambda doc: doc.metadata.get("section") == state["query"]["section"],
#     )
#     return {"context": retrieved_docs}


# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     answer = llm.invoke(
#         prompt.invoke({"question": state["question"], "context": docs_content})
#     )
#     return {"answer": answer.content}


# def analyze_query(state: State):
#     structured_llm = llm.with_structured_output(Search)
#     query = structured_llm.invoke(state["question"])
#     return {"query": query}
