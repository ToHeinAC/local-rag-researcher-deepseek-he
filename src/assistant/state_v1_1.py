import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_community.schema import Document

class ResearcherState(TypedDict):
    user_instructions: str  # Changed from query to user_instructions
    research_queries: list[str]
    retrieved_documents: list
    search_summaries: Annotated[list, operator.add]
    current_position: int
    final_answer: str
    detected_language: str  # Added field to store detected language
    additional_context: Optional[str]  # Added field to store additional context from document retrieval
    all_query_documents: Optional[Dict[str, List[Document]]]  # Added field to store all query documents
