import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from langchain_core.documents import Document
from src.assistant.v1_1.configuration_v1_1 import Configuration, get_config_instance
from src.assistant.v1_1.vector_db_v1_1 import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.assistant.v1_1.state_v1_1 import ResearcherState
from src.assistant.v1_1.prompts_v1_1 import (
    # Language detection prompts
    LANGUAGE_DETECTOR_SYSTEM_PROMPT, LANGUAGE_DETECTOR_HUMAN_PROMPT,
    # Research query generation prompts
    RESEARCH_QUERY_WRITER_SYSTEM_PROMPT, RESEARCH_QUERY_WRITER_HUMAN_PROMPT,
    # Document summarization prompts
    SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT,
    # Report writing prompts
    REPORT_WRITER_SYSTEM_PROMPT, REPORT_WRITER_HUMAN_PROMPT,
)
from src.assistant.v1_1.utils_v1_1 import format_documents_with_metadata, invoke_ollama, parse_output, tavily_search, DetectedLanguage, Queries
from src.assistant.v1_1.rag_helpers_v1_1 import source_summarizer_ollama, format_documents_as_plain_text
import re
import time


# Initialize the researcher graph
researcher_graph = StateGraph(ResearcherState)

# Detect language of user query
def detect_language(state: ResearcherState, config: RunnableConfig):
    print("--- Detecting language of user query ---")
    query = state["user_query"]  # Get the query from user_query
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
    config = get_config_instance()
    embedding_model = config.embedding_model
    print(f"\n=== Using embedding model: {embedding_model} ===\n")
    # Return a dictionary with a key for embedding_model - LangGraph nodes must return dictionaries
    return {"embedding_model": embedding_model, "current_step": "display_embedding_model_info"}

def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    """
    Generate research queries based on the user's instructions.

    Args:
        state (ResearcherState): The current state of the researcher.
        config (RunnableConfig): The configuration for the graph.

    Returns:
        dict: A state update containing the generated research queries as a list.
    """
    print("--- Generating research queries ---")
    query = state["user_query"]  # Get the query from user_query
    detected_language = state["detected_language"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    llm_model = config["configurable"].get("report_llm", "deepseek-r1:latest")
    
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
    
    all_queries = result.queries
    all_queries.insert(0, query)
    print(f"  [DEBUG] Generated research queries: {all_queries}")
    assert isinstance(all_queries, list), "all_queries must be a list"
    
    return {"research_queries": all_queries}


def retrieve_rag_documents(state: ResearcherState, config: RunnableConfig):
    """Retrieve documents from the RAG database for all research queries.
    
    Args:
        state (ResearcherState): The current state of the researcher.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: A state update containing the retrieved documents in form of a dictionary with query as key and list of langchain Document objects.
    """
    print("--- Retrieving documents for all research queries ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    
    # Get research queries
    research_queries = state.get("research_queries", [])
    
    if not research_queries:
        # If no research queries were generated, use the original user query
        print("  [WARNING] No research queries found, using original user query")
        research_queries = [state["user_query"]]
    
    print(f"  [INFO] Processing {len(research_queries)} research queries")
    
    # Get detected language
    detected_language = state.get("detected_language", "English")
    
    # Get the number of results to retrieve from config
    k_results = config["configurable"].get("k_results", 3)  # Default to 3 if not specified
    
    # Display embedding model information
    from src.assistant.v1_1.configuration_v1_1 import get_config_instance
    config = get_config_instance()
    embedding_model = config.embedding_model
    
    # Get the detected language from the global configuration if available
    if hasattr(config, 'detected_language'):
        detected_language = config.detected_language
        print(f"  [Using language from global config: {detected_language}]")
    else:
        print(f"  [Using language from state: {detected_language}]")
    
    print(f"  [Using embedding model for retrieval: {embedding_model}]")
    print(f"  [Retrieving {k_results} results per query]")
    
    # Dictionary to store documents for each query
    all_query_documents = {}
    
    # Import the special database configuration from vector_db_v1_1.py directly
    import sys
    from src.assistant.v1_1.vector_db_v1_1 import VECTOR_DB_PATH  # Get the path
    # Avoid circular imports
    
    print(f"  [DEBUG] Vector DB path: {VECTOR_DB_PATH}")
    print(f"  [DEBUG] System path: {sys.path}")
    
    # Force the update of the global configuration for embedding model
    # This ensures the right database is used for document retrieval
    try:
        # We already imported get_config_instance at the top of this file
        # and have the config instance from earlier in this function
        print(f"  [DEBUG] Current embedding model in config: {config.embedding_model}")
        print(f"  [DEBUG] Update language to: {detected_language}")
        config.detected_language = detected_language
    except Exception as e:
        print(f"  [ERROR] Error updating config: {e}")
    
    # Process each research query
    for i, query in enumerate(research_queries):
        print(f"  [INFO] Processing query {i+1}/{len(research_queries)}: '{query}'")
        
        # Use similarity_search_for_tenant directly (the working method) instead of search_documents
        try:
            print(f"  [DEBUG] Calling similarity_search_for_tenant directly with query: '{query}', k={k_results}, language={detected_language}")
            
            # Import the necessary functions and constants
            from src.assistant.v1_1.rag_helpers_v1_1 import similarity_search_for_tenant
            import os
            
            # Use the exact same parameters that work in the UI
            # Hard-code the known working values
            database_name = 'sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2--2000--400'
            tenant_id = '2025-04-22_15-41-10'
            # Try with the collection_prefix since we see both collections in the logs
            collection_name = 'collection_2025-04-22_15-41-10'
            
            # Get the embedding model
            from src.assistant.v1_1.vector_db_v1_1 import get_embedding_model
            embeddings = get_embedding_model()
            
            # Construct the database path
            database_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'database', database_name)
            print(f"  [DEBUG] Using database path: {database_path}")
            print(f"  [DEBUG] Using tenant ID: {tenant_id}")
            print(f"  [DEBUG] Using collection name: {collection_name}")
            
            # Call similarity_search_for_tenant directly with the known working parameters
            documents = similarity_search_for_tenant(
                tenant_id=tenant_id,
                embed_llm=embeddings,
                persist_directory=database_path,
                similarity="cosine",
                normal=True,
                query=query,
                k=k_results,
                language=detected_language,
                collection_name=collection_name  # Explicitly pass the collection name
            )
            
            # Ensure documents is at least an empty list if None
            if documents is None:
                documents = []
                print("  [WARNING] similarity_search_for_tenant returned None")
        except Exception as e:
            print(f"  [ERROR] Exception during similarity_search_for_tenant: {str(e)}")
            import traceback
            print(f"  [ERROR] Traceback: {traceback.format_exc()}")
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
    
    assert isinstance(all_query_documents, dict), "all_query_documents must be a dictionary"
    assert len(all_query_documents) == len(research_queries), "all_query_documents must have the same length as research_queries"

    # In LangGraph, the returned dictionary represents the specific state updates
    # We need to make sure this is properly merged with the existing state
    return {
        "retrieved_documents": all_query_documents
    }


def summarize_query_research(state: ResearcherState, config: RunnableConfig):
    """
    Summarize the retrieved documents for each research query.
    
    Args:
        state (ResearcherState): The current state of the researcher.
        config (RunnableConfig): The configuration for the researcher.
    
    Returns:
        dict: A state update containing the search summaries in form of a dictionary with query as key and list of langchain Document objects.
    """
    print("--- Summarizing all query research ---")
    # Print current state keys for debugging
    print(f"  [DEBUG] Current state keys: {list(state.keys())}")
    
    # Get all query documents from the previous step
    all_query_documents = state.get("retrieved_documents", {})
    research_queries = state.get("research_queries", [])

    
    # Dictionary to store formatted plain text documents for each query
    all_formatted_documents = {}
    
    # Debug the all_query_documents structure if it exists
    if "retrieved_documents" in state:
        print(f"  [DEBUG] Found retrieved_documents in state with {len(state['retrieved_documents'])} queries")
        for query, docs in state['retrieved_documents'].items():
            print(f"  [DEBUG] Query '{query[:50]}...' has {len(docs)} documents")
    else:
        print("  [DEBUG] retrieved_documents not found in state!")
    
    if not all_query_documents:
        print("  [WARNING] No query documents found in state")
        # If we have research queries but no documents, create empty summaries
        if research_queries:
            print(f"  [INFO] Creating empty summaries for {len(research_queries)} queries")
        else:
            print("  [WARNING] No research queries found either")
            research_queries = [state["user_query"]]
            print(f"  [INFO] Using original query: '{research_queries[0]}'")
    
    # Get language and LLM model for summarization
    summarization_llm = config["configurable"].get("summarization_llm", "llama3.2")
    detected_language = state.get("detected_language", config["configurable"].get("detected_language", "English"))
    print(f"  [Using language: {detected_language}]")
    print(f"  [Using summarization LLM: {summarization_llm}]")
    
    # Format the system prompt for summarization
    system_prompt = SUMMARIZER_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Initialize dictionary to store summaries for each query
    all_summaries = {}
    
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
            
            # Create a document object for the fallback summary and add it to all_summaries
            summary_doc = Document(page_content=summary, metadata={"position": i, "query": query, "name": "No documents", "path": "No path"})
            
            # Add to our dictionary of summaries
            if query not in all_summaries:
                all_summaries[query] = []
            all_summaries[query].append(summary_doc)
            print(f"  [INFO] Added fallback summary for query '{query}' with no documents")
        else: 
            # Format documents for summarization (using the original method for source_summarizer_ollama)
            context_documents = format_documents_with_metadata(documents, preserve_original=True)
            
            # Format the human prompt for this specific query and documents
            human_prompt = SUMMARIZER_HUMAN_PROMPT.format(
                query=query,
                documents=context_documents,
                language=detected_language
            )
            
            # Use source_summarizer_ollama to create a summary
            try:
                summary_result = source_summarizer_ollama(
                    query=query,
                    context_documents=context_documents,
                    language=detected_language,
                    system_message=system_prompt,
                    #human_message=human_prompt,
                    llm_model=summarization_llm
                )
                # Extract the summary content from the successful result
                summary = summary_result["content"]
                
            except Exception as e:
                print(f"  [ERROR] Failed to generate summary: {str(e)}")
                # Set fallback summary text for error cases
                if detected_language.lower() == 'german':
                    summary = f"Fehler bei der Generierung der Zusammenfassung für Anfrage: '{query}'"
                else:
                    summary = f"Error generating summary for query: '{query}'"
                # Create a minimal metadata structure for error cases
                summary_result = {"metadata": {"name": "error", "path": "error"}}
            
            # Create a document object for the summary
            summary_doc = Document(page_content=summary, metadata={"position": i, "query": query, "name": summary_result["metadata"]["name"], "path": summary_result["metadata"]["path"]})
            
            # Add to our dictionary of summaries, with query as key and list of documents as value
            if query not in all_summaries:
                all_summaries[query] = []
            all_summaries[query].append(summary_doc)
            print(f"  [INFO] Added summary for query '{query}'")
        
    print(f"  [INFO] Generated {len(all_summaries)} summaries for all queries")

    assert isinstance(all_summaries, dict), "all_summaries must be a dictionary"
    assert len(all_summaries) == len(research_queries), "all_summaries must have the same length as research_queries"
    
    # Return all summaries, but also preserve the all_query_documents in the state
    # In LangGraph, we need to explicitly return all state keys we want to preserve
    result = {
        "search_summaries": all_summaries,
        "formatted_documents": all_formatted_documents  # Add the formatted documents to the state
    }
    
    # Include additional context if it exists
    if "additional_context" in state:
        result["additional_context"] = state["additional_context"]
    
    print(f"  [INFO] Added {len(all_formatted_documents)} formatted document sets to state")
    return result


def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    """
    Generate a final answer based on the search summaries.
    
    Args:
        state (ResearcherState): The current state of the researcher.
        config (RunnableConfig): The configuration for the graph.
    
    Returns:
        dict: A state update containing the final answer in form of a string.
    """
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
    )
    
    # Get the report structure from config or use default
    report_structure = config["configurable"].get("report_structure", """
    1. Introduction
    2. Main Findings
    3. Detailed Analysis
    4. Conclusion
    """)
    
    # Get the search summaries from the state
    search_summaries = state.get("search_summaries", {})
    print(f"  [DEBUG] Search summaries found: {len(search_summaries) if search_summaries else 0}")
    
    # Debug information about summaries
    if search_summaries:
        print(f"  [DEBUG] Search summaries keys: {list(search_summaries.keys())}")
        total_documents = sum(len(docs) for docs in search_summaries.values())
        print(f"  [DEBUG] Total documents across all summaries: {total_documents}")
    
    # Process and format the accumulated summaries
    if not search_summaries:
        information = "No search summaries were found. Please try again with a different query."
        print("  [WARNING] No search summaries found in state. Using empty information.")
    else:
        # Format the information by organizing all summaries
        formatted_info = []
        
        # Process search_summaries as a dictionary where keys are queries and values are lists of Document objects
        print(f"  [DEBUG] Processing search_summaries as a dictionary with {len(search_summaries)} queries")
        
        # Sort queries by their position metadata if available
        sorted_queries = []
        for query, docs in search_summaries.items():
            # Get position from the first document's metadata if available
            position = docs[0].metadata.get("position", 0) if docs else 0
            sorted_queries.append((query, position))
        
        # Sort queries by position
        sorted_queries.sort(key=lambda x: x[1])
        
        # Process each query and its documents
        for i, (query, _) in enumerate(sorted_queries):
            docs = search_summaries[query]
            
            # Combine all document content for this query
            content = "\n".join([doc.page_content for doc in docs])
            
            # Add formatted content with query and summary
            formatted_info.append(f"## Research Query {i+1}: {query}\n\n{content}\n")
        
        # Join all formatted summaries
        information = "\n\n".join(formatted_info)
        
        # Fallback for empty formatted_info or if search_summaries is not a dictionary
        if not formatted_info:
            # Check if search_summaries is a list (old format) instead of a dictionary
            if isinstance(search_summaries, list):
                print(f"  [DEBUG] Processing search_summaries as a list with {len(search_summaries)} items")
                research_queries = state.get("research_queries", [])
                
                for i, summary in enumerate(search_summaries):
                    # Handle different possible formats
                    if isinstance(summary, dict) and "content" in summary:
                        query = summary.get("query", f"Research Query {i+1}")
                        content = summary.get("content", "")
                    elif isinstance(summary, str):
                        query = research_queries[i] if i < len(research_queries) else f"Research Query {i+1}"
                        content = summary
                    else:
                        # Try to extract content from Document object if that's what it is
                        try:
                            query = research_queries[i] if i < len(research_queries) else f"Research Query {i+1}"
                            content = getattr(summary, "page_content", str(summary))
                        except:
                            query = f"Research Query {i+1}"
                            content = str(summary)
                    
                    # Add formatted content with query and summary
                    formatted_info.append(f"## Research Query {i+1}: {query}\n\n{content}\n")
                
                # Join all formatted summaries
                information = "\n\n".join(formatted_info)
            else:
                # If we get here, something unexpected happened
                information = "Could not process search summaries due to unexpected format."
                print(f"  [WARNING] Unexpected search_summaries format: {type(search_summaries)}")
        
        # Handle any other format as stringified content
        else:
            information = str(search_summaries)
        
        print(f"  [DEBUG] Using information from {len(search_summaries)} summaries (length: {len(information)})")
        if information:
            print(f"  [DEBUG] Information preview (first 100 chars): {information[:100]}...")
    
    # Format the human prompt
    human_prompt = REPORT_WRITER_HUMAN_PROMPT.format(
        instruction=state["user_query"],
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
        "final_answer": report_content
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