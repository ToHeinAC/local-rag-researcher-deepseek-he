import operator
from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
# Updated import path for Document to fix ModuleNotFoundError
from langchain_core.documents import Document

class ResearcherState(TypedDict):
    user_instructions: str  # Changed from query to user_instructions
    research_queries: list[str]
    retrieved_documents: list[str]  
    search_summaries: list[str]
    current_position: int
    final_answer: str
    detected_language: str  # Added field to store detected language
    additional_context: Optional[str]  # Added field to store additional context from document retrieval
    all_query_documents: Dict[str, List[Document]]  # Dictionary mapping queries to their retrieved documents

