import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db_v1_1 import get_or_create_vector_db, search_documents, get_embedding_model_path
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
from src.assistant.rag_helpers_v1_1 import source_summarizer_ollama
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
    """Retrieve documents from the RAG database for all research queries."""
    print("--- Retrieving documents for all research queries ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    
    # Get research queries
    research_queries = state.get("research_queries", [])
    
    if not research_queries:
        # If no research queries were generated, use the original user query
        print("  [WARNING] No research queries found, using original user query")
        research_queries = [state["user_instructions"]]
    
    print(f"  [INFO] Processing {len(research_queries)} research queries")
    
    # Get detected language
    detected_language = state.get("detected_language", "English")
    
    # Get the number of results to retrieve from config
    k_results = config["configurable"].get("k_results", 3)  # Default to 3 if not specified
    
    # Display embedding model information
    from src.assistant.configuration import get_config_instance
    config_obj = get_config_instance()
    embedding_model = config_obj.embedding_model
    
    # Get the detected language from the global configuration if available
    if hasattr(config_obj, 'detected_language'):
        detected_language = config_obj.detected_language
        print(f"  [Using language from global config: {detected_language}]")
    else:
        print(f"  [Using language from state: {detected_language}]")
    
    print(f"  [Using embedding model for retrieval: {embedding_model}]")
    print(f"  [Retrieving {k_results} results per query]")
    
    # Dictionary to store documents for each query
    all_query_documents = {}
    
    # Process each research query
    for i, query in enumerate(research_queries):
        print(f"  [INFO] Processing query {i+1}/{len(research_queries)}: '{query}'")
        
        # Use the search_documents function with user-specified k and language
        documents = search_documents(query, k=k_results, language=detected_language)
        
        # Ensure documents is at least an empty list if None
        if documents is None:
            documents = []
            
        # Add detailed debugging
        print(f"  [DEBUG] Retrieved {len(documents)} documents for query: '{query}'")
        if documents and len(documents) > 0:
            if hasattr(documents[0], "page_content"):
                print(f"  [DEBUG] First doc content (sample): {documents[0].page_content[:100]}...")
        else:
            print(f"  [WARNING] No documents were retrieved for query: '{query}'")
        
        # Store documents for this query
        all_query_documents[query] = documents
    
    print(f"  [INFO] Completed retrieval for all {len(research_queries)} queries")
    
    # Return all retrieved documents organized by query
    # Make sure to keep a full trace of all retrieved documents for debugging
    print(f"  [DEBUG] Returning all_query_documents with {len(all_query_documents)} queries")
    for query, docs in all_query_documents.items():
        print(f"  [DEBUG] Query '{query[:50]}...' has {len(docs)} documents")
    
    # In LangGraph, the returned dictionary represents the specific state updates
    # We need to make sure this is properly merged with the existing state
    return {
        "all_query_documents": all_query_documents,
        "research_queries": research_queries  # Include for reference
    }


def summarize_query_research(state: ResearcherState, config: RunnableConfig):
    print("--- Summarizing all query research ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    
    # Get all query documents from the previous step
    all_query_documents = state.get("all_query_documents", {})
    research_queries = state.get("research_queries", [])
    
    # Debug the all_query_documents structure if it exists
    if "all_query_documents" in state:
        print(f"  [DEBUG] Found all_query_documents in state with {len(state['all_query_documents'])} queries")
        for query, docs in state['all_query_documents'].items():
            print(f"  [DEBUG] Query '{query[:50]}...' has {len(docs)} documents")
    else:
        print("  [DEBUG] all_query_documents not found in state!")
    
    if not all_query_documents:
        print("  [WARNING] No query documents found in state")
        # If we have research queries but no documents, create empty summaries
        if research_queries:
            print(f"  [INFO] Creating empty summaries for {len(research_queries)} queries")
        else:
            print("  [WARNING] No research queries found either")
            research_queries = [state["user_instructions"]]
            print(f"  [INFO] Using original query: '{research_queries[0]}'")
    
    # Get language and LLM model for summarization
    summarization_llm = config["configurable"].get("summarization_llm", "llama3.2")
    detected_language = state.get("detected_language", config["configurable"].get("detected_language", "English"))
    print(f"  [Using language: {detected_language}]")
    print(f"  [Using summarization LLM: {summarization_llm}]")
    
    # Initialize list to store all summaries
    all_summaries = []
    
    # Format the system prompt for summarization
    system_prompt = SUMMARIZER_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Process each query and its documents
    for i, query in enumerate(research_queries):
        print(f"  [INFO] Summarizing query {i+1}/{len(research_queries)}: '{query}'")
        
        # Get documents for this query
        documents = all_query_documents.get(query, [])
        print(f"  [DEBUG] Found {len(documents)} documents for this query")
        
        # Generate summary for this query
        if not documents:
            # Create a fallback summary for queries with no documents
            if detected_language.lower() == 'german':
                summary = f"Keine relevanten Dokumente wurden in der Datenbank gefunden für die Anfrage: '{query}'"
            else:
                summary = f"No relevant documents were found in the database for the query: '{query}'"
            print("  [WARNING] No documents found for this query, using fallback summary")
        else:
            # Format documents for summarization
            context_documents = format_documents_with_metadata(documents, preserve_original=True)
            
            # Use source_summarizer_ollama to create a summary
            try:
                summary_result = source_summarizer_ollama(
                    query=query,
                    context_documents=context_documents,
                    language=detected_language,
                    system_message=system_prompt,
                    llm_model=summarization_llm
                )
                
                # Extract the content from the result
                summary = summary_result["content"]
                print(f"  [DEBUG] Summary generated successfully")
            except Exception as e:
                print(f"  [ERROR] Failed to generate summary: {str(e)}")
                if detected_language.lower() == 'german':
                    summary = f"Fehler bei der Generierung der Zusammenfassung für Anfrage: '{query}'"
                else:
                    summary = f"Error generating summary for query: '{query}'"
        
        # Create a labeled summary object
        labeled_summary = {
            "query": query,
            "content": summary,
            "position": i
        }
        
        # Add to our list of summaries
        all_summaries.append(labeled_summary)
        print(f"  [INFO] Added summary for query '{query}'")
    
    print(f"  [INFO] Generated {len(all_summaries)} summaries for all queries")
    
    # Return all summaries, but also preserve the all_query_documents in the state
    # In LangGraph, we need to explicitly return all state keys we want to preserve
    result = {
        "search_summaries": all_summaries,
        "all_query_documents": all_query_documents,  # Always return this regardless of source
        "research_queries": research_queries  # Ensure research_queries is preserved
    }
    
    # Include additional context if it exists
    if "additional_context" in state:
        result["additional_context"] = state["additional_context"]
    
    return result


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
    
    # Get the report structure from config or use default
    report_structure = config["configurable"].get("report_structure", """
    1. Introduction
    2. Main Findings
    3. Detailed Analysis
    4. Conclusion
    """)
    
    # Get the search summaries from the state
    search_summaries = state.get("search_summaries", [])
    print(f"  [DEBUG] Search summaries found: {len(search_summaries) if search_summaries else 0}")
    
    # Get research queries for reference
    research_queries = state.get("research_queries", [])
    print(f"  [DEBUG] Research queries found: {len(research_queries) if research_queries else 0}")
    
    # Process and format the accumulated summaries
    if not search_summaries:
        information = "No search summaries were found. Please try again with a different query."
        print("  [WARNING] No search summaries found in state. Using empty information.")
    else:
        # Format the information by organizing all summaries
        formatted_info = []
        
        # Process labeled summaries (new format with query, content, position)
        if isinstance(search_summaries[0], dict) and "query" in search_summaries[0] and "content" in search_summaries[0]:
            # Sort summaries by position for consistent ordering
            sorted_summaries = sorted(search_summaries, key=lambda x: x.get("position", 0))
            
            for i, summary_obj in enumerate(sorted_summaries):
                query = summary_obj.get("query", f"Research Query {i+1}")
                content = summary_obj.get("content", "")
                
                # Add formatted content with query and summary
                formatted_info.append(f"## Research Query {i+1}: {query}\n\n{content}\n")
            
            # Join all formatted summaries
            information = "\n\n".join(formatted_info)
        
        # Fallback for older format summaries (plain strings)
        elif all(isinstance(s, str) for s in search_summaries):
            for i, summary in enumerate(search_summaries):
                # Try to pair with research query if available
                query = research_queries[i] if i < len(research_queries) else f"Research Query {i+1}"
                formatted_info.append(f"## Research Query {i+1}: {query}\n\n{summary}\n")
            
            information = "\n\n".join(formatted_info)
        
        # Handle any other format as stringified content
        else:
            information = str(search_summaries)
        
        print(f"  [DEBUG] Using information from {len(search_summaries)} summaries (length: {len(information)})")
        if information:
            print(f"  [DEBUG] Information preview (first 100 chars): {information[:100]}...")
    
    # Format the human prompt
    human_prompt = REPORT_WRITER_HUMAN_PROMPT.format(
        instruction=state["user_instructions"],
        report_structure=report_structure,
        information=information,
        language=detected_language
    )
    
    # Call the LLM with the prompt to generate the final answer
    final_answer = invoke_ollama(
        model=report_llm,
        system_prompt=system_prompt,
        user_prompt=human_prompt
    )
    
    # Remove thinking part if present
    parsed_output = parse_output(final_answer)
    report_content = parsed_output.get("response", final_answer)
    
    # For LangGraph, we need to return an update to state as a dictionary
    # but ensure the content itself is just the report text
    print(f"  [INFO] Returning final answer (length: {len(report_content) if isinstance(report_content, str) else 'unknown'})")
    
    # Make sure the report content is not None
    if report_content is None:
        # Fallback to using search summaries if available
        if search_summaries:
            report_content = "# Research Report\n\n"
            if isinstance(search_summaries, list):
                for i, summary in enumerate(search_summaries):
                    report_content += f"## Research Finding {i+1}\n\n{summary}\n\n"
            else:
                report_content += f"## Research Findings\n\n{search_summaries}\n\n"
            report_content += "\n## Conclusion\n\nThe above findings represent the key information found during research."
        else:
            report_content = "No research findings could be generated. Please try again with a different query."
    
    # Since we're now getting plain markdown text directly from the LLM (based on updated prompt),
    # we just need basic cleanup of markdown code blocks if present
    if isinstance(report_content, str):
        # Clean up common prefixes that might still appear
        prefixes_to_remove = ["```markdown\n", "```md\n", "```\n"]
        for prefix in prefixes_to_remove:
            if report_content.startswith(prefix):
                report_content = report_content[len(prefix):]
        
        # Clean up common suffixes
        suffixes_to_remove = ["\n```"]
        for suffix in suffixes_to_remove:
            if report_content.endswith(suffix):
                report_content = report_content[:-len(suffix)]
        
        # Ensure text is properly stripped of whitespace
        report_content = report_content.strip()
        
        print(f"  [INFO] Ensured report content is clean markdown for display")
    
    # Set the final_answer in the state and return it with the node name
    # This ensures both the state is updated and the UI can access it
    return {
        "final_answer": report_content,  # Update the state's final_answer field
        "generate_final_answer": report_content  # Return for the UI display
    }


# Define main researcher nodes
researcher_graph.add_node(display_embedding_model_info)
researcher_graph.add_node(detect_language)
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(retrieve_rag_documents)
researcher_graph.add_node(summarize_query_research)
researcher_graph.add_node(generate_final_answer)

# Add a router function to decide whether to process more queries or go to the final answer
def query_router(state: ResearcherState):
    print("--- Routing based on research queries ---")
    
    # Get the research queries and current position
    research_queries = state.get("research_queries", [])
    current_position = state.get("current_position", 0)
    
    print(f"  [DEBUG] Current position: {current_position}, Total queries: {len(research_queries)}")
    
    # If we have more queries to process, continue with the next one
    if current_position < len(research_queries):
        # We have more queries to process
        print(f"  [INFO] Processing next query: '{research_queries[current_position]}'")
    else:
        # We've processed all queries, move to final answer
        print("  [INFO] All queries processed, generating final answer")
    
    # Return empty dict for state updates, conditional edges will handle the routing
    return {}

# Set the current position after each query is processed
def update_position(state: ResearcherState):
    print("--- Updating position for next query ---")
    
    # Get the current position and increment it
    current_position = state.get("current_position", 0) + 1
    
    # Return the updated state with the new position
    print(f"  [INFO] Updated position to {current_position}")
    return {"current_position": current_position}

# Define transitions for the simplified graph - linear flow without router
researcher_graph.add_edge(START, "display_embedding_model_info")
researcher_graph.add_edge("display_embedding_model_info", "detect_language")
researcher_graph.add_edge("detect_language", "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "retrieve_rag_documents")
researcher_graph.add_edge("retrieve_rag_documents", "summarize_query_research")
researcher_graph.add_edge("summarize_query_research", "generate_final_answer")
researcher_graph.add_edge("generate_final_answer", END)

# Since we're using the StateGraph with ResearcherState, state is preserved by default
# But we'll explicitly ensure the required keys are preserved in the researcher state class

# No need for routing nodes or conditional edges since each node now processes all queries at once

# Compile the researcher graph
researcher = researcher_graph.compile()

# Make sure researcher_graph is exported
__all__ = ["researcher", "researcher_graph"]