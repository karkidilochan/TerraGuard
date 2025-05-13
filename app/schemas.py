from typing_extensions import TypedDict, List, Annotated, Literal, Dict, Union, Optional
from langchain_core.documents import Document


# search
class Search(TypedDict):
    query: Annotated[str, ..., "User's natural language query"]
    subcategories: Annotated[
        List[str],
        ...,
        "AWS subcategories like: ['IAM Access Analyzer', 'Account Management', ...]",
    ]


# Validation results schema
class ValidationSummary(TypedDict):
    syntax_valid: bool
    cis_compliant: bool
    pass_rate: Optional[float]  # Added to track overall pass rate percentage
    total_checks: Optional[int]  # Total number of checks run
    passed: Optional[int]  # Number of passed checks
    failed: Optional[int]  # Number of failed checks
    skipped: Optional[int]  # Number of skipped checks
    referenced_cis_controls: List[str]  # CIS controls found in the code
    issues: List[str]  # List of validation issues
    checkov_output_file: Optional[str]  # Path to the Checkov output file


# state
class State(TypedDict):
    question: str
    context: List[Document]
    search: Search
    answer: str
    referenced_cis_controls: Optional[List[str]]
    # Added fields for feedback loop
    retry_count: Optional[int]
    validation_feedback: Optional[str]
    previous_attempt_code: Optional[str]
    max_retries: Optional[int]
    validation_results: Optional[Dict]
