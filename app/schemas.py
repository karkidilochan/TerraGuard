from typing_extensions import TypedDict, List, Annotated, Literal, Dict, Union, Optional
from langchain_core.documents import Document


# search
class Search(TypedDict):
    query: Annotated[str, ..., "Search query to run"]
    resource_names: Annotated[
        List[str],
        ...,
        "List of AWS resource names in Terraform to filter by (e.g., ['aws_s3_bucket', 'aws_iam_role']",
    ]


# Validation results schema
class ValidationSummary(TypedDict):
    syntax_valid: bool
    cis_compliant: bool
    referenced_cis_controls: List[str]
    error_count: int


# state
class State(TypedDict):
    question: str
    context: List[Document]
    search: Search
    answer: str
    referenced_cis_controls: Optional[List[str]]
    # validation_results: Optional[Dict[str, Union[bool, str, List[str], Dict]]]
    # validation_summary: Optional[ValidationSummary]
