from typing_extensions import TypedDict, List, Annotated, Literal
from langchain_core.documents import Document


# search
class Search(TypedDict):
    query: str
    section: str


# state
class State(TypedDict):
    question: str
    context: List[Document]
    search: Search
    answer: str
