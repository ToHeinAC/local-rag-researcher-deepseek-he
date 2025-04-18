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
from src.assistant.utils import format_documents_with_metadata, invoke_ollama, parse_output, tavily_search, DetectedLanguage
from src.assistant.rag_helpers import source_summarizer_ollama
import re
import time


# Detect language of user query
def detect_language(state: ResearcherState, config: RunnableConfig):
    print("--- Detecting language of user query ---")
    query = state["query"]
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
        query=user_instructions
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
    query = state["query"]
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
    query = state["query"]
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
    
    return {"retrieved_documents": documents}


def summarize_query_research(state: QuerySearchState, config: RunnableConfig):
    print("--- Summarizing query research ---")
    query = state["query"]
    # Use the summarization LLM model instead of the general purpose LLM model
    summarization_llm = config["configurable"].get("summarization_llm", "llama3.2")
    detected_language = state.get("detected_language", config["configurable"].get("detected_language", "English"))
    print(f"  [Using language: {detected_language}]")
    print(f"  [Using summarization LLM: {summarization_llm}]")

    information = state["retrieved_documents"]

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

    return {
        "search_summaries": [summary],
    }


def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    print("--- Generating final answer ---")
    user_instructions = state["user_instructions"]
    
    # Get all types of summaries from state for debugging
    search_summaries = state.get("search_summaries", [])
    improved_summaries = state.get("improved_summaries", [])
    filtered_summaries = state.get("filtered_summaries", [])
    ranked_summaries = state.get("ranked_summaries", [])
    
    # Comprehensive debug information to help diagnose issues
    print(f"  [DEBUG] State of summaries at final answer generation:")
    print(f"  [DEBUG] - Search summaries: {len(search_summaries)}")
    print(f"  [DEBUG] - Improved summaries: {len(improved_summaries)}")
    print(f"  [DEBUG] - Filtered summaries: {len(filtered_summaries)}")
    print(f"  [DEBUG] - Ranked summaries: {len(ranked_summaries)}")
    
    # Original debug info
    print(f"  [Number of ranked summaries available: {len(ranked_summaries)}]")
    
    # Use the report writing LLM model instead of the general purpose LLM model
    report_llm = config["configurable"].get("report_llm", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "en")
    print(f"  [Using language: {detected_language}]")
    print(f"  [Using report writing LLM: {report_llm}]")
    
    # Determine report structure based on the query
    report_structure = """
    1. Introduction
    2. Main Findings
    3. Detailed Analysis
    4. Conclusion
    """
        
    # Format the system prompt
    system_prompt = REPORT_WRITER_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Format the human prompt
    human_prompt = REPORT_WRITER_HUMAN_PROMPT.format(
        language=detected_language,
        instruction=user_instructions,
        report_structure=report_structure,
        information=combined_information
    )
    
    # Call the LLM with the improved prompts
    final_answer = invoke_ollama(
        model=report_llm,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    # Remove thinking part if present
    final_answer = parse_output(final_answer)["response"]
    
    return {"final_answer": final_answer}

# Create subghraph for searching each query
query_search_subgraph = StateGraph(QuerySearchState, input=QuerySearchStateInput, output=QuerySearchStateOutput)

# Define subgraph nodes for searching the query
query_search_subgraph.add_node(retrieve_rag_documents)
query_search_subgraph.add_node(evaluate_retrieved_documents)
query_search_subgraph.add_node(web_research)
query_search_subgraph.add_node(summarize_query_research)
query_search_subgraph.add_node(quality_check_summary)
query_search_subgraph.add_node(improve_summary)

# Set entry point and define transitions for the subgraph
query_search_subgraph.add_edge(START, "retrieve_rag_documents")
query_search_subgraph.add_edge("retrieve_rag_documents", "evaluate_retrieved_documents")
query_search_subgraph.add_conditional_edges("evaluate_retrieved_documents", route_research)
query_search_subgraph.add_edge("web_research", "summarize_query_research")
query_search_subgraph.add_conditional_edges("summarize_query_research", route_after_summarization, ["quality_check_summary", END])
query_search_subgraph.add_conditional_edges("quality_check_summary", route_quality_check, ["improve_summary", END])
query_search_subgraph.add_edge("improve_summary", "quality_check_summary")

# Define a collector function to explicitly collect and merge search summaries from the subgraph
def collect_search_summaries(state: ResearcherState):
    print("--- Collecting search summaries ---")
    # Get the current search summaries (if any)
    current_summaries = state.get("search_summaries", [])
    print(f"  [DEBUG] Current number of collected summaries: {len(current_summaries)}")
    
    # If there are no summaries, create a placeholder
    if not current_summaries:
        print("  [WARNING] No search summaries found, creating placeholder")
        user_instructions = state["user_instructions"]
        placeholder = f"No relevant information was found for the query: {user_instructions}"
        # Initialize all summary states with the placeholder to maintain information flow
        return {
            "search_summaries": [placeholder],
            "improved_summaries": [placeholder],  # Initialize improved_summaries
            "filtered_summaries": [placeholder],  # Initialize filtered_summaries
            "ranked_summaries": [placeholder]     # Initialize ranked_summaries
        }
    
    # Print a sample of the summaries for debugging
    for i, summary in enumerate(current_summaries):
        if summary:
            print(f"  [DEBUG] Summary {i+1} first 100 chars: {summary[:100]}...")
        else:
            print(f"  [DEBUG] Summary {i+1} is empty or None")
    
    # Create a dictionary with properly formatted keys for the UI
    # The UI looks for keys that start with 'search_and_summarize_query'
    result = {}
    for i, summary in enumerate(current_summaries):
        if summary:
            # Format the key to match what the UI expects
            key = f"search_and_summarize_query_{i+1}"
            result[key] = summary
    
    # Set all summary states to maintain information flow
    # This ensures that search_summaries is available for the next steps
    result["search_summaries"] = current_summaries
    
    # Initialize improved_summaries with search_summaries to ensure data flow
    # This will be overwritten by improve_summary if that node runs
    result["improved_summaries"] = current_summaries.copy()
    
    return result

# Create main research agent graph
researcher_graph = StateGraph(ResearcherState, input=ResearcherStateInput, output=ResearcherStateOutput, config_schema=Configuration)

# Define main researcher nodes
researcher_graph.add_node(display_embedding_model_info)
researcher_graph.add_node(detect_language)
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(search_queries)
researcher_graph.add_node("search_and_summarize_query", query_search_subgraph.compile())
researcher_graph.add_node(collect_search_summaries)  # Add the collector node
researcher_graph.add_node(filter_search_summaries)
researcher_graph.add_node(rank_search_summaries)
researcher_graph.add_node(generate_final_answer)

# Define transitions for the main graph
researcher_graph.add_edge(START, "display_embedding_model_info")
researcher_graph.add_edge("display_embedding_model_info", "detect_language")
researcher_graph.add_edge("detect_language", "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "search_queries")
researcher_graph.add_conditional_edges("search_queries", initiate_query_research, ["search_and_summarize_query"])
researcher_graph.add_conditional_edges("search_and_summarize_query", check_more_queries, ["search_queries", "collect_search_summaries"])  # Route to collector node
researcher_graph.add_edge("collect_search_summaries", "filter_search_summaries")  # Then to filter
researcher_graph.add_edge("filter_search_summaries", "rank_search_summaries")
researcher_graph.add_edge("rank_search_summaries", "generate_final_answer")
researcher_graph.add_edge("generate_final_answer", END)

# Compile the researcher graph
researcher = researcher_graph.compile()

# Make sure researcher_graph is exported
__all__ = ["researcher", "researcher_graph"]