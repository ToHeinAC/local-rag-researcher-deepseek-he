import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
# Updated import path for Document to fix ModuleNotFoundError
from langchain_core.documents import Document

class ResearcherState(TypedDict):
    user_query: str
    current_position: int
    detected_language: str  # Added field to store detected language
    research_queries: list[str]
    retrieved_documents: dict[str, list[Document]]  
    search_summaries: dict[str, list[Document]]
    final_answer: str
    additional_context: Optional[list[Document]]  # Added field to store additional context from document retrieval

