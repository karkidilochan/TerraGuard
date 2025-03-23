from typing_extensions import TypedDict, List, Annotated, Literal
from langchain_core.documents import Document


# search
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run"]
    # resources: Annotated[List[str], ..., "List of AWS resource names to filter by"]


# state
class State(TypedDict):
    question: str
    context: List[Document]
    search: Search
    answer: str
