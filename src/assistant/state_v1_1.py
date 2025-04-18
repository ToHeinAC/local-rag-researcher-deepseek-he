import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict

class ResearcherState(TypedDict):
    query: str
    research_queries: list[str]
    search_summaries: Annotated[list, operator.add]
    current_position: int
    final_answer: str
    detected_language: str  # Added field to store detected language
    additional_context: Optional[str]  # Added field to store additional context from document retrieval
