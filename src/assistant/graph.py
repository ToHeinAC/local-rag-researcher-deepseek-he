import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.assistant.state import ResearcherState, ResearcherStateInput, ResearcherStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput, SummaryRanking
from src.assistant.prompts import (
    # Language detection prompts
    LANGUAGE_DETECTOR_SYSTEM_PROMPT, LANGUAGE_DETECTOR_HUMAN_PROMPT,
    # Research query generation prompts
    RESEARCH_QUERY_WRITER_SYSTEM_PROMPT, RESEARCH_QUERY_WRITER_HUMAN_PROMPT,
    # Document relevance evaluation prompts
    RELEVANCE_EVALUATOR_SYSTEM_PROMPT, RELEVANCE_EVALUATOR_HUMAN_PROMPT,
    # Document summarization prompts
    SUMMARIZER_SYSTEM_PROMPT, SUMMARIZER_HUMAN_PROMPT,
    # Quality checking prompts
    QUALITY_CHECKER_SYSTEM_PROMPT, QUALITY_CHECKER_HUMAN_PROMPT,
    # Summary improvement prompts
    SUMMARY_IMPROVEMENT_SYSTEM_PROMPT, SUMMARY_IMPROVEMENT_HUMAN_PROMPT,
    # Report writing prompts
    REPORT_WRITER_SYSTEM_PROMPT, REPORT_WRITER_HUMAN_PROMPT,
    # Ranking prompts
    RANKING_SYSTEM_PROMPT
)
from src.assistant.utils import format_documents_with_metadata, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries, SummaryRankings, SummaryRelevance, QualityCheckResult, DetectedLanguage
from src.assistant.rag_helpers import source_summarizer_ollama
import re
import time

# Number of query to process in parallel for each batch
# Change depending on the performance of the system
BATCH_SIZE = 3

# Detect language of user query
def detect_language(state: ResearcherState, config: RunnableConfig):
    print("--- Detecting language of user query ---")
    user_instructions = state["user_instructions"]
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
    user_instructions = state["user_instructions"]
    detected_language = state["detected_language"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
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
        query=user_instructions,
        language=detected_language,
        additional_context=f"Consider this additional context when generating queries: {additional_context}" if additional_context else ""
    )
    
    # Using local Deepseek R1 model with Ollama
    result = invoke_ollama(
        model=llm_model,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=Queries
    )
    
    # Add the original human query to the list of research queries
    all_queries = [user_instructions] + result.queries
    
    return {"research_queries": all_queries}

def search_queries(state: ResearcherState):
    # Kick off the search for each query by calling initiate_query_research
    print("--- Searching queries ---")
    # Get the current processing position from state or initialize to 0
    current_position = state.get("current_position", 0)

    return {"current_position": current_position + BATCH_SIZE}


def check_more_queries(state: ResearcherState) -> Literal["search_queries", "collect_search_summaries"]:
    """Check if there are more queries to process"""
    current_position = state.get("current_position", 0)
    research_queries = state.get("research_queries", [])
    
    # Print debug information
    print(f"  [DEBUG] check_more_queries: current_position={current_position}, len(research_queries)={len(research_queries)}")
    
    # Check if there are more queries to process
    # We need to ensure current_position is less than the length of research_queries
    # to avoid processing empty batches
    if current_position < len(research_queries):
        return "search_queries"
    return "collect_search_summaries"

def initiate_query_research(state: ResearcherState, config: RunnableConfig):
    # Get the next batch of queries
    queries = state["research_queries"]
    current_position = state["current_position"]
    
    # Print debug information
    print(f"  [DEBUG] initiate_query_research: current_position={current_position}, len(queries)={len(queries)}")
    
    # Calculate the start and end indices for the current batch
    batch_start = max(0, current_position - BATCH_SIZE)
    batch_end = min(current_position, len(queries))
    
    # Get the current batch of queries
    current_batch = queries[batch_start:batch_end]
    
    # Print debug information about the batch
    print(f"  [DEBUG] Processing batch: {batch_start}:{batch_end}, batch size: {len(current_batch)}")
    
    # Get the detected language from state - this comes from the detect_language node
    # and should be preserved throughout the workflow
    detected_language = state.get("detected_language", "English")
    print(f"Using language for query research: {detected_language}")
    
    # Store the detected language in the global configuration to ensure consistency
    from src.assistant.configuration import get_config_instance
    config_obj = get_config_instance()
    if hasattr(config_obj, 'detected_language'):
        config_obj.detected_language = detected_language
    else:
        # Add detected_language attribute dynamically if it doesn't exist
        setattr(config_obj, 'detected_language', detected_language)
    print(f"Stored detected language '{detected_language}' in global configuration")
    
    # Get the quality check loops from the main config
    quality_check_loops = state.get("quality_check_loops", 1)
    
    # Get the LLM models from state if available
    llm_model = state.get("llm_model", "llama3.2")
    summarization_llm = state.get("summarization_llm", "llama3.2")
    report_llm = state.get("report_llm", "deepseek-r1:latest")
    
    # Get the number of results to retrieve from config
    k_results = config["configurable"].get("k_results", 3)  # Default to 3 if not specified

    # Return the batch of queries to process with detected language and quality_check_loops in config
    return [
        Send("search_and_summarize_query", {
            "query": s, 
            "detected_language": detected_language,  # Pass language directly to the state
            "configurable": {
                "detected_language": detected_language,
                "quality_check_loops": quality_check_loops,
                "llm_model": llm_model,
                "summarization_llm": summarization_llm,
                "report_llm": report_llm,
                "k_results": k_results  # Pass the k_results to the subgraph
            }
        })
        for s in current_batch
    ]

def retrieve_rag_documents(state: QuerySearchState, config: RunnableConfig):
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

def evaluate_retrieved_documents(state: QuerySearchState, config: RunnableConfig):
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]
    detected_language = state.get("detected_language", "English")
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # Format the system prompt
    system_prompt = RELEVANCE_EVALUATOR_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Format the documents for the user prompt
    formatted_documents = format_documents_with_metadata(retrieved_documents)
    
    # Format the human prompt
    human_prompt = RELEVANCE_EVALUATOR_HUMAN_PROMPT.format(
        query=query,
        documents=formatted_documents,
        language=detected_language
    )
    
    # Using local Deepseek R1 model with Ollama
    evaluation = invoke_ollama(
        model=llm_model,
        system_prompt=system_prompt,
        user_prompt=human_prompt,
        output_format=Evaluation
    )
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # evaluation = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=evaluation_prompt,
    #     user_prompt=f"Evaluate the relevance of the retrieved documents for this query: {query}",
    #     output_format=Evaluation
    # )

    return {"are_documents_relevant": evaluation.is_relevant}

def route_research(state: QuerySearchState, config: RunnableConfig) -> Literal["summarize_query_research", "web_research", "__end__"]:
    """ Route the research based on the documents relevance """

    if state["are_documents_relevant"]:
        return "summarize_query_research"
    elif config["configurable"].get("enable_web_search", False):
        return "web_research"
    else:
        print("Skipping query due to irrelevant documents and web search disabled.")
        return "__end__"

def web_research(state: QuerySearchState):
    print("--- Web research ---")
    query = state["query"]
    detected_language = state.get("detected_language", "English")
    print(f"  [Using language: {detected_language}]")
    
    output = tavily_search(query)
    search_results = output["results"]
    
    # Format web search results to include links
    formatted_results = []
    for result in search_results:
        title = result.get('title', 'Untitled')
        url = result.get('url', '')
        content = result.get('content', '')
        
        # Format as a document with source and link
        formatted_result = f"SOURCE: [{title}]({url})\n\nContent: {content}"
        formatted_results.append(formatted_result)

    return {"web_search_results": formatted_results}

def summarize_query_research(state: QuerySearchState, config: RunnableConfig):
    print("--- Summarizing query research ---")
    query = state["query"]
    # Use the summarization LLM model instead of the general purpose LLM model
    summarization_llm = config["configurable"].get("summarization_llm", "llama3.2")
    detected_language = state.get("detected_language", config["configurable"].get("detected_language", "English"))
    print(f"  [Using language: {detected_language}]")
    print(f"  [Using summarization LLM: {summarization_llm}]")

    information = None
    if state["are_documents_relevant"]:
        # If documents are relevant: Use RAG documents
        information = state["retrieved_documents"]
    else:
        # If documents are irrelevant: Use web search results,
        # if enabled, otherwise query will be skipped in the previous router node
        information = state["web_search_results"]

    # Format documents with metadata but simplified without emphasis on citations
    # Preserve original content as much as possible
    context_documents = format_documents_with_metadata(information, preserve_original=True) if state["are_documents_relevant"] else information
    
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
        "summary_improvement_iterations": 0,  # Initialize the iteration counter
    }

def route_after_summarization(state: QuerySearchState, config: RunnableConfig) -> Literal["quality_check_summary", "__end__"]:
    """Route based on whether quality checking is enabled."""
    enable_quality_checker = config["configurable"].get("enable_quality_checker", True)
    
    if enable_quality_checker:
        print("Quality checker is enabled, proceeding to quality check")
        return "quality_check_summary"
    else:
        print("Quality checker is disabled, skipping quality check")
        return "__end__"

def route_quality_check(state: QuerySearchState, config: RunnableConfig) -> Literal["improve_summary", "__end__"]:
    """Route based on quality check results."""
    # Get quality check results
    quality_results = state.get("quality_check_results", {})
    
    # Check if improvement is needed based on the quality metrics
    improvement_needed = quality_results.get("improvement_needed", False)
    quality_score = quality_results.get("quality_score", 0.7)
    is_sufficient = quality_results.get("is_accurate", True) and quality_results.get("is_complete", True)
    
    # Get current improvement iteration for this document
    iterations = state.get("summary_improvement_iterations", 0)
    
    # Get the maximum number of quality check loops from config
    quality_check_loops = config["configurable"].get("quality_check_loops", 1)
    
    # Log quality check information
    print(f"Document '{state['query']}': Quality check - Score: {quality_score}, Sufficient: {is_sufficient}, Improvement needed: {improvement_needed}")
    
    # Check if improvement is needed and possible
    if improvement_needed and iterations < quality_check_loops:
        print(f"Document '{state['query']}': Proceeding with summary improvement (iteration {iterations + 1}/{quality_check_loops})")
        return "improve_summary"
    else:
        # If no improvement is needed or possible, end the process
        if not improvement_needed:
            print(f"Document '{state['query']}': No improvement needed")
        else:
            print(f"Document '{state['query']}': Reached maximum improvement iterations ({quality_check_loops})")
        return "__end__"

def improve_summary(state: QuerySearchState, config: RunnableConfig):
    """Improve the summary based on quality check feedback."""
    print("--- Improving summary based on quality check ---")
    query = state["query"]
    
    # Get the initial summary - this is crucial to ensure we always have a summary to return
    assert state["search_summaries"] != []
    initial_summary = state["search_summaries"]
    
    # Get quality check results and other configuration
    quality_check_results = state["quality_check_results"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "English")
    print(f"  [Using language: {detected_language}]")
    
    # Get current improvement iteration and increment it
    iterations = state.get("summary_improvement_iterations", 0)
    iterations += 1
    
    # Get the maximum number of quality check loops from config
    quality_check_loops = config["configurable"].get("quality_check_loops", 1)
    
    print(f"Document '{query}': Improving summary (iteration {iterations}/{quality_check_loops})")
    
    # Get the information source used for the summary
    information = None
    if state["are_documents_relevant"]:
        information = state["retrieved_documents"]
    else:
        information = state["web_search_results"]
    
    # Format the information
    formatted_information = format_documents_with_metadata(information, preserve_original=True) if state["are_documents_relevant"] else information
    
    # Create a prompt for improving the summary
    improvement_suggestions = quality_check_results.get("improvement_suggestions", "")
    issues_found = quality_check_results.get("issues_found", [])
    missing_elements = quality_check_results.get("missing_elements", [])
    citation_issues = quality_check_results.get("citation_issues", [])
    
    # Format the issues as bullet points
    formatted_issues = ""
    if issues_found:
        formatted_issues += "\nIssues found:\n" + "\n".join([f"- {issue}" for issue in issues_found])
    if missing_elements:
        formatted_issues += "\nMissing elements:\n" + "\n".join([f"- {element}" for element in missing_elements])
    if citation_issues:
        formatted_issues += "\nCitation issues:\n" + "\n".join([f"- {issue}" for issue in citation_issues])
    
    # Format the system prompt
    system_prompt = SUMMARY_IMPROVEMENT_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Format the human prompt
    human_prompt = SUMMARY_IMPROVEMENT_HUMAN_PROMPT.format(
        query=query,
        summary=initial_summary,
        feedback=f"{improvement_suggestions}\n{formatted_issues}",
        documents=formatted_information,
        language=detected_language
    )
    
    try:
        # Using local model with Ollama
        improved_summary_result = invoke_ollama(
            model=report_writer_llm,
            system_prompt=system_prompt,
            user_prompt=human_prompt
        )
        
        # Remove thinking part if present
        improved_summary = parse_output(improved_summary_result)["response"]
        
        # Validate the improved summary - if it's empty or too short, use the initial summary
        if not improved_summary or len(improved_summary) < 50:
            print(f"Document '{query}': Improved summary is too short or empty, using initial summary")
            improved_summary = initial_summary
        else:
            print(f"Document '{query}': Summary successfully improved (iteration {iterations}/{quality_check_loops})")
    
    except Exception as e:
        print(f"Document '{query}': Error improving summary: {str(e)}, using initial summary")
        improved_summary = initial_summary
    
    # Return the improved summary and the updated iteration count
    # Always ensure we have a non-empty search_summaries list
    return {
        "improved_summaries": [improved_summary],
        "summary_improvement_iterations": iterations
    }

def filter_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Filter out irrelevant search summaries with a simpler approach"""
    print("--- Filtering search summaries ---")
    user_instructions = state["user_instructions"]
    
    # Get improved summaries, with fallback to search_summaries if improved_summaries is empty
    improved_summaries = state.get("improved_summaries", [])
    if not improved_summaries:
        print("  [DEBUG] No improved summaries found, falling back to search_summaries")
        improved_summaries = state.get("search_summaries", [])
        
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "English")
    print(f"  [Using language: {detected_language}]")
    
    # Debug logging to check if summaries are being received
    print(f"  [DEBUG] Number of improved summaries received: {len(improved_summaries)}")
    if improved_summaries:
        print(f"  [DEBUG] First few characters of first summary: {improved_summaries[0][:100] if improved_summaries[0] else 'Empty summary'}...")
    
    # If there are no summaries, return empty filtered summaries
    if not improved_summaries:
        return {"filtered_summaries": []}
    
    # Keep track of filtered summaries
    filtered_summaries = []
    
    # Process each summary
    for summary in improved_summaries:
        # Skip empty summaries
        if not summary or len(summary.strip()) < 50:
            continue
            
        # Check if the summary contains phrases indicating no relevant information
        irrelevance_indicators = [
            "no relevant information",
            "not relevant",
            "irrelevant",
            "no information",
            "does not provide",
            "doesn't provide",
            "not related",
            "no direct information",
            "keine relevanten Informationen",
            "keine relevanten",
            "kein Bezug"
        ]
        
        # Check if the summary contains any irrelevance indicators
        is_irrelevant = False
        for indicator in irrelevance_indicators:
            if indicator.lower() in summary.lower():
                is_irrelevant = True
                break
                
        # If the summary is not irrelevant, add it to filtered summaries
        if not is_irrelevant:
            filtered_summaries.append(summary)
    
    # Get the number of original research queries to determine minimum summaries to keep
    research_queries = state.get("research_queries", [])
    min_summaries_to_keep = max(1, len(research_queries))  # At least 1, or the number of queries
    print(f"  [DEBUG] Minimum summaries to keep: {min_summaries_to_keep}")
    
    # If we have fewer filtered summaries than the minimum required, add back some of the filtered out ones
    if len(filtered_summaries) < min_summaries_to_keep and len(improved_summaries) > len(filtered_summaries):
        print(f"  [DEBUG] Not enough relevant summaries, adding back some filtered ones")
        
        # Create a list of summaries that were filtered out
        filtered_out_summaries = []
        for summary in improved_summaries:
            if summary and len(summary.strip()) >= 50 and summary not in filtered_summaries:
                filtered_out_summaries.append(summary)
        
        # Add back filtered out summaries until we reach the minimum
        for summary in filtered_out_summaries:
            if len(filtered_summaries) >= min_summaries_to_keep:
                break
            filtered_summaries.append(summary)
            print(f"  [DEBUG] Added back a filtered summary, now have {len(filtered_summaries)}")
    
    # If we still have no summaries (e.g., all were empty), include a placeholder
    if not filtered_summaries:
        placeholder = f"No relevant information was found for the query: {user_instructions}"
        filtered_summaries = [placeholder]
        print(f"  [DEBUG] Using placeholder summary")
    
    print(f"Filtered {len(improved_summaries)} summaries to {len(filtered_summaries)} relevant ones")
    
    return {"filtered_summaries": filtered_summaries}

def rank_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Simplified ranking of search summaries by relevance"""
    print("--- Ranking search summaries ---")
    user_instructions = state["user_instructions"]
    filtered_summaries = state.get("filtered_summaries", [])
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "English")
    print(f"  [Using language: {detected_language}]")
    
    # Debug logging to check if filtered summaries are being received
    print(f"  [DEBUG] Number of filtered summaries received: {len(filtered_summaries)}")
    if filtered_summaries:
        print(f"  [DEBUG] First few characters of first filtered summary: {filtered_summaries[0][:100] if filtered_summaries[0] else 'Empty summary'}...")
    
    # If there are no filtered summaries, return empty rankings
    if not filtered_summaries:
        print("  [DEBUG] No filtered summaries to rank")
        return {
            "ranked_summaries": [],
            "relevance_scores": []
        }
        
    # If there's only a placeholder summary, use it but don't try to rank it
    if len(filtered_summaries) == 1 and "No relevant information" in filtered_summaries[0]:
        print("  [DEBUG] Only have placeholder summary, using as is")
        return {
            "ranked_summaries": filtered_summaries,
            "relevance_scores": [1.0]
        }
    
    # If there's only one summary, return it as is
    if len(filtered_summaries) == 1:
        return {
            "ranked_summaries": filtered_summaries,
            "relevance_scores": [1.0]
        }
    
    # For multiple summaries, use the LLM to rank them
    # Create a formatted string of all summaries to pass to the prompt
    formatted_summaries = ""
    for i, summary in enumerate(filtered_summaries):
        formatted_summaries += f"\n\nSummary {i+1}:\n{summary}"
        
    ranking_prompt = RANKING_SYSTEM_PROMPT.format(
        language=detected_language,
        user_instructions=user_instructions,
        summaries=formatted_summaries
    )
    # Add instructions for the JSON output format
    user_prompt = "Provide your ranking as a JSON object with the following structure:"
    user_prompt += "\n{\"rankings\": [{\"summary_index\": 0, \"relevance_score\": 9, \"justification\": \"reason\"}, ...]}"
    user_prompt += "\nwhere 'summary_index' is the 0-based index of the summary."
    
    # Using local model with Ollama
    try:
        rankings = invoke_ollama(
            model=llm_model,
            system_prompt=ranking_prompt,
            user_prompt=user_prompt,
            output_format=SummaryRankings
        )
        
        # Sort the rankings by relevance score (descending)
        sorted_rankings = sorted(rankings.rankings, key=lambda x: x.relevance_score, reverse=True)
        
        # Extract the ranked summaries and scores
        ranked_summaries = [filtered_summaries[r.summary_index] for r in sorted_rankings]
        relevance_scores = [r.relevance_score for r in sorted_rankings]
        
    except Exception as e:
        print(f"Error ranking summaries: {str(e)}")
        # If there's an error, use the original order
        ranked_summaries = filtered_summaries
        relevance_scores = [1.0] * len(filtered_summaries)
    
    return {
        "ranked_summaries": ranked_summaries,
        "relevance_scores": relevance_scores
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
    
    # Combine the information from the ranked summaries - this is the primary source
    if ranked_summaries and len(ranked_summaries) > 0:
        combined_information = "\n\n".join([summary for summary in ranked_summaries])
        print(f"  [Using ranked summaries: {len(ranked_summaries)} summaries, {len(combined_information)} characters]")
    else:
        # If no ranked summaries, try to get filtered summaries as first fallback
        filtered_summaries = state.get("filtered_summaries", [])
        if filtered_summaries and len(filtered_summaries) > 0:
            print(f"  [No ranked summaries found, using {len(filtered_summaries)} filtered summaries as fallback]")
            combined_information = "\n\n".join([summary for summary in filtered_summaries])
        else:
            # If no filtered summaries, try to get improved summaries as second fallback
            improved_summaries = state.get("improved_summaries", [])
            if improved_summaries and len(improved_summaries) > 0:
                print(f"  [No ranked or filtered summaries found, using {len(improved_summaries)} improved summaries as fallback]")
                combined_information = "\n\n".join([summary for summary in improved_summaries])
            else:
                # If no improved summaries, try to get raw search summaries as last resort
                search_summaries = state.get("search_summaries", [])
                if search_summaries and len(search_summaries) > 0:
                    print(f"  [No ranked, filtered, or improved summaries found, using {len(search_summaries)} raw search summaries as last resort]")
                    combined_information = "\n\n".join([summary for summary in search_summaries])
                else:
                    print("  [WARNING: No summaries found at all!]")
                    combined_information = f"No relevant information was found for the query: {user_instructions}"
    
    # Print a sample of the combined information to help with debugging
    if len(combined_information) > 200:
        print(f"  [Sample of combined information: {combined_information[:200]}...]")
    else:
        print(f"  [Full combined information: {combined_information}]")
        
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

def quality_check_summary(state: QuerySearchState, config: RunnableConfig):
    """Quality check to ensure the summary contains sufficient information."""
    print("--- Quality checking summary ---")
    query = state["query"]
    
    # Ensure we have a summary to check
    if not state.get("search_summaries") or not state["search_summaries"][0]:
        print(f"Document '{query}': No summary to check, skipping quality check")
        # Return default quality check results indicating no improvement needed
        return {
            "quality_check_results": {
                "quality_score": 7.0,  # Acceptable score
                "is_accurate": True,
                "is_complete": True,
                "issues_found": [],
                "missing_elements": [],
                "citation_issues": [],
                "improvement_needed": False,
                "improvement_suggestions": "No summary to check."
            }
        }
    
    # Get the current summary to check
    current_summary = state["search_summaries"][0]
    
    # Use the summarization LLM model for quality checking
    report_llm = config["configurable"].get("report_llm", "llama3.2")
    detected_language = state.get("detected_language", config["configurable"].get("detected_language", "English"))
    quality_check_loops = config["configurable"].get("quality_check_loops", 1)
    print(f"  [Using summarization LLM for quality check: {report_llm}]")
    
    # Get the source documents
    information = None
    if state["are_documents_relevant"]:
        information = state["retrieved_documents"]
    else:
        information = state["web_search_results"]
    
    # Format documents with metadata
    formatted_information = format_documents_with_metadata(information) if state["are_documents_relevant"] else information
    
    # Get current iteration for this document
    iterations = state.get("summary_improvement_iterations", 0)
    print(f"Document '{query}': Quality check iteration {iterations + 1}/{quality_check_loops}")
    
    # Format the system prompt with the detected language
    system_prompt = QUALITY_CHECKER_SYSTEM_PROMPT.format(
        language=detected_language
    )
    
    # Format the human prompt with the detected language
    human_prompt = QUALITY_CHECKER_HUMAN_PROMPT.format(
        summary=current_summary,
        documents=formatted_information,
        language=detected_language
    )
    
    try:
        # Using the configured summarization LLM model with Ollama
        quality_check = invoke_ollama(
            model=summarization_llm,
            system_prompt=system_prompt,
            user_prompt=human_prompt,
            output_format=QualityCheckResult
        )
        
        # Extract quality metrics
        quality_score = quality_check.quality_score
        is_sufficient = quality_check.is_accurate and quality_check.is_complete
        improvement_needed = quality_check.improvement_needed
        
        # Log quality results
        print(f"Document '{query}': Summary quality score: {quality_score}")
        print(f"Document '{query}': Is summary sufficient: {is_sufficient}")
        print(f"Document '{query}': Improvement needed: {improvement_needed}")
        
        # If the quality is very high (score > 8), we can override improvement_needed to false
        # This ensures we don't waste time improving already good summaries
        if quality_score > 8 and improvement_needed:
            print(f"Document '{query}': Quality score is high ({quality_score}), overriding improvement_needed to false")
            quality_check.improvement_needed = False
        
        return {"quality_check_results": quality_check.dict()}
        
    except Exception as e:
        print(f"Error during quality check for document '{query}': {str(e)}")
        # Default quality check results in case of error
        default_results = {
            "quality_score": 7.0,  # Acceptable score
            "is_accurate": True,
            "is_complete": True,
            "issues_found": [],
            "missing_elements": [],
            "citation_issues": [],
            "improvement_needed": False,  # No improvement needed by default
            "improvement_suggestions": "Error during quality check, using default results."
        }
        print(f"Document '{query}': Using default quality check results")
        return {"quality_check_results": default_results}

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
        return {"search_summaries": [placeholder]}
    
    # Print a sample of the summaries for debugging
    for i, summary in enumerate(current_summaries):
        if summary:
            print(f"  [DEBUG] Summary {i+1} first 100 chars: {summary[:100]}...")
        else:
            print(f"  [DEBUG] Summary {i+1} is empty or None")
    
    # Explicitly return the summaries to ensure they're passed to the next step
    # The operator.add annotation in ResearcherState will handle merging
    return {"search_summaries": current_summaries}

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