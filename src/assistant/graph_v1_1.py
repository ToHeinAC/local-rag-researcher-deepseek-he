import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.assistant.state_v1_1 import ResearcherState
from src.assistant.prompts import (
    # Language detection prompts
    LANGUAGE_DETECTOR_SYSTEM_PROMPT, LANGUAGE_DETECTOR_HUMAN_PROMPT,
    # Research query generation prompts
    RESEARCH_QUERY_WRITER_SYSTEM_PROMPT, RESEARCH_QUERY_WRITER_HUMAN_PROMPT,
    # Document summarization prompts
    SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT,
    # Report writing prompts
    REPORT_WRITER_SYSTEM_PROMPT, REPORT_WRITER_HUMAN_PROMPT,
)
from src.assistant.utils import format_documents_with_metadata, invoke_ollama, parse_output, tavily_search, DetectedLanguage, Queries
from src.assistant.rag_helpers import source_summarizer_ollama
import re
import time


# Initialize the researcher graph
researcher_graph = StateGraph(ResearcherState)

# Detect language of user query
def detect_language(state: ResearcherState, config: RunnableConfig):
    print("--- Detecting language of user query ---")
    query = state["user_instructions"]  # Get the query from user_instructions
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # First check if a language is already set in the config (from GUI)
    user_selected_language = config["configurable"].get("selected_language", None)
    
    if user_selected_language:
        print(f"Using user-selected language: {user_selected_language}")
        return {"detected_language": user_selected_language}
    
    # If no language is set in config, detect it from the query
    print("No language selected by user, detecting from query...")
    
    # Format the system prompt
    system_prompt = LANGUAGE_DETECTOR_SYSTEM_PROMPT
    
    # Format the human prompt
    human_prompt = LANGUAGE_DETECTOR_HUMAN_PROMPT.format(
        query=query
    )
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=llm_model,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=DetectedLanguage
    )
    
    detected_language = result.language
    print(f"Detected language: {detected_language}")
    
    return {"detected_language": detected_language}

# Display embedding model information
def display_embedding_model_info(state: ResearcherState):
    """Display information about which embedding model is being used."""
    from src.assistant.configuration import get_config_instance
    config = get_config_instance()
    embedding_model = config.embedding_model
    print(f"\n=== Using embedding model: {embedding_model} ===\n")
    return {}

def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    print("--- Generating research queries ---")
    query = state["user_instructions"]  # Get the query from user_instructions
    detected_language = state["detected_language"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    llm_model = config["configurable"].get("summarization_llm", "deepseek-r1:latest")
    
    # Get additional context if available
    additional_context = state.get("additional_context", "")
    
    # Format the system prompt
    system_prompt = RESEARCH_QUERY_WRITER_SYSTEM_PROMPT.format(
        max_queries=max_queries,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M"),
        language=detected_language
    )
    
    # Format the human prompt
    human_prompt = RESEARCH_QUERY_WRITER_HUMAN_PROMPT.format(
        query=query,
        language=detected_language,
        additional_context=f"Consider this additional context when generating queries: {additional_context}" if additional_context else ""
    )
    
    # Using local llm model with Ollama
    result = invoke_ollama(
        model=llm_model,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=Queries
    )
    
    # Add the original human query to the list of research queries
    all_queries = result.queries
    
    return {"research_queries": all_queries}


def retrieve_rag_documents(state: ResearcherState, config: RunnableConfig):
    """Retrieve documents from the RAG database."""
    print("--- Retrieving documents ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    
    query = state["user_instructions"]  # Get the query from user_instructions
    detected_language = state.get("detected_language", "English")
    
    # Get the number of results to retrieve from config
    k_results = config["configurable"].get("k_results", 3)  # Default to 3 if not specified
    
    # Display embedding model information for this retrieval operation
    # Use the global configuration instance instead of creating a new one
    from src.assistant.configuration import get_config_instance
    config_obj = get_config_instance()
    embedding_model = config_obj.embedding_model
    
    # Get the detected language from the global configuration if available
    # This ensures consistency with the language detected in the main workflow
    if hasattr(config_obj, 'detected_language'):
        # Use the language from global config, which was set in initiate_query_research
        detected_language = config_obj.detected_language
        print(f"  [Using language from global config: {detected_language}]")
    else:
        # Fallback to the language from state if not in global config
        print(f"  [Using language from state: {detected_language}]")
    
    print(f"  [Using embedding model for retrieval: {embedding_model}]")
    print(f"  [Retrieving {k_results} results per query]")
    
    # Use the new search_documents function from vector_db.py with user-specified k and language
    documents = search_documents(query, k=k_results, language=detected_language)
    
    # Add detailed debugging
    print(f"  [DEBUG] Retrieved {len(documents)} documents")
    if documents and len(documents) > 0:
        print(f"  [DEBUG] First document type: {type(documents[0])}")
        if hasattr(documents[0], "page_content"):
            print(f"  [DEBUG] First doc content (sample): {documents[0].page_content[:100]}...")
        if hasattr(documents[0], "metadata"):
            print(f"  [DEBUG] First doc metadata: {documents[0].metadata}")
    else:
        print("  [WARNING] No documents were retrieved. Check vector database.")
    
    # Make sure documents is at least an empty list if None
    if documents is None:
        documents = []
    
    # Return documents directly in state dictionary
    return {
        "retrieved_documents": documents
    }


def summarize_query_research(state: ResearcherState, config: RunnableConfig):
    print("--- Summarizing query research ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    
    query = state["user_instructions"]  # Get the query from user_instructions
    # Use the summarization LLM model instead of the general purpose LLM model
    summarization_llm = config["configurable"].get("summarization_llm", "llama3.2")
    detected_language = state.get("detected_language", config["configurable"].get("detected_language", "English"))
    print(f"  [Using language: {detected_language}]")
    print(f"  [Using summarization LLM: {summarization_llm}]")
    
    # Properly retrieve documents from state with appropriate fallbacks
    # First try direct access from state
    information = state.get("retrieved_documents", None)
    
    # If not found, try to get from the retrieve_rag_documents node output
    if information is None:
        # Get from node output if it exists
        retrieve_output = state.get("retrieve_rag_documents", {})
        if isinstance(retrieve_output, dict):
            information = retrieve_output.get("retrieved_documents", [])
    
    # Ensure information is always a list
    if information is None:
        information = []
        
    print(f"  [DEBUG] Retrieved documents found: {len(information) > 0}")
    print(f"  [DEBUG] Number of documents: {len(information)}")
    print(f"  [DEBUG] Full state keys: {list(state.keys())}")
    
    # Dump first document for debugging if available
    if information and len(information) > 0:
        print("  [DEBUG] First document available:" )
        doc = information[0]
        print(f"  Type: {type(doc)}")
        if hasattr(doc, "page_content"):
            print(f"  Snippet: {doc.page_content[:100]}...")
    
    # Initialize a default summary in case no documents are found
    summary = ""
    
    if not information:
        print("  [WARNING] No documents were retrieved from the previous step!")
        # Create a fallback summary when no documents are found
        summary = f"Keine relevanten Dokumente wurden in der Datenbank gefunden für die Anfrage: '{query}'. " 
        if detected_language.lower() != 'german':
            summary = f"No relevant documents were found in the database for the query: '{query}'. "
    else:
        # Format documents with metadata but simplified without emphasis on citations
        # Preserve original content as much as possible
        context_documents = format_documents_with_metadata(information, preserve_original=True)
        
        # Format the system prompt for source_summarizer_ollama
        system_prompt = SUMMARIZER_SYSTEM_PROMPT.format(
            language=detected_language
        )
        
        # Use source_summarizer_ollama from rag_helpers.py
        summary_result = source_summarizer_ollama(
            query=query,
            context_documents=context_documents,
            language=detected_language,
            system_message=system_prompt,
            llm_model=summarization_llm
        )
        
        # Extract the content from the result
        summary = summary_result["content"]

    # Always return a list with at least one summary to maintain workflow consistency
    return {
        "search_summaries": [summary],
    }


def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    print("--- Generating final answer ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    # Use the report writing LLM model instead of the general purpose LLM model
    report_llm = config["configurable"].get("report_llm", "deepseek-r1:latest")
    
    # Get detected language
    detected_language = state.get("detected_language", "English")
    
    # Format the system prompt
    system_prompt = REPORT_WRITER_SYSTEM_PROMPT.format(
        language=detected_language,
        date=datetime.datetime.now().strftime("%Y-%m-%d")
    )
    
    # Determine report structure based on the query
    report_structure = """
    1. Introduction
    2. Main Findings
    3. Detailed Analysis
    4. Conclusion
    """
    
    # Get the search summaries from the state
    search_summaries = state.get("search_summaries", None)
    print(f"  [DEBUG] Search summaries found: {search_summaries is not None}")
    
    # If not found directly, look for it in nested objects
    if not search_summaries:
        # Look for search_summaries in the summarize_query_research node output
        summarize_output = state.get("summarize_query_research", {})
        if isinstance(summarize_output, dict) and "search_summaries" in summarize_output:
            search_summaries = summarize_output["search_summaries"]
            print(f"  [DEBUG] Found search_summaries in summarize_query_research output")
    
    # If still not found, use empty string as fallback
    if not search_summaries:
        information = ""
        print("  [WARNING] No search summaries found in state. Using empty information.")
    else:
        information = search_summaries[0] if isinstance(search_summaries, list) and len(search_summaries) > 0 else ""
        print(f"  [DEBUG] Using information from search_summaries (length: {len(information)})")
        if information:
            print(f"  [DEBUG] Information preview: {information[:100]}...")
    
    # Format the human prompt
    human_prompt = REPORT_WRITER_HUMAN_PROMPT.format(
        instruction=state["user_instructions"],
        report_structure=report_structure,
        information=information,
        language=detected_language
    )
    
    # Call the LLM with the improved prompts
    final_answer = invoke_ollama(
        model=report_llm,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    # Remove thinking part if present
    final_answer = parse_output(final_answer)["response"]
    
    try:
        final_answer = final_answer['final_answer']
    except:
        final_answer = final_answer
    


# Define main researcher nodes
researcher_graph.add_node(display_embedding_model_info)
researcher_graph.add_node(detect_language)
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(retrieve_rag_documents)
researcher_graph.add_node(summarize_query_research)
researcher_graph.add_node(generate_final_answer)

# Define transitions for the main graph
researcher_graph.add_edge(START, "display_embedding_model_info")
researcher_graph.add_edge("display_embedding_model_info", "detect_language")
researcher_graph.add_edge("detect_language", "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "retrieve_rag_documents")
researcher_graph.add_edge("retrieve_rag_documents", "summarize_query_research")
researcher_graph.add_edge("summarize_query_research", "generate_final_answer")
researcher_graph.add_edge("generate_final_answer", END)

# Compile the researcher graph
researcher = researcher_graph.compile()

# Make sure researcher_graph is exported
__all__ = ["researcher", "researcher_graph"]