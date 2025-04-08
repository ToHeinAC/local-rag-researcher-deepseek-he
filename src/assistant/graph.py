import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.assistant.state import ResearcherState, ResearcherStateInput, ResearcherStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput, SummaryRanking
from src.assistant.prompts import RESEARCH_QUERY_WRITER_PROMPT, RELEVANCE_EVALUATOR_PROMPT, SUMMARIZER_PROMPT, REPORT_WRITER_PROMPT, QUALITY_CHECKER_PROMPT, LANGUAGE_DETECTOR_PROMPT
from src.assistant.utils import format_documents_with_metadata, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries, SummaryRankings, SummaryRelevance, QualityCheckResult, DetectedLanguage
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
    
    language_detector_prompt = LANGUAGE_DETECTOR_PROMPT.format(
        query=user_instructions
    )
    
    # Using local model with Ollama
    result = invoke_ollama(
        model=llm_model,
        system_prompt=language_detector_prompt,
        user_prompt=f"Detect the language of this query: {user_instructions}",
        output_format=DetectedLanguage
    )
    
    detected_language = result.language
    print(f"Detected language: {detected_language}")
    
    return {"detected_language": detected_language}

# Display embedding model information
def display_embedding_model_info(state: ResearcherState):
    """Display information about which embedding model is being used."""
    config = Configuration()
    embedding_model = config.embedding_model
    print(f"\n=== Using embedding model: {embedding_model} ===\n")
    return {}

def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    print("--- Generating research queries ---")
    user_instructions = state["user_instructions"]
    detected_language = state["detected_language"]
    max_queries = config["configurable"].get("max_search_queries", 3)
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    query_writer_prompt = RESEARCH_QUERY_WRITER_PROMPT.format(
        max_queries=max_queries,
        date=datetime.datetime.now().strftime("%Y/%m/%d %H:%M")
    )
    
    # Using local Deepseek R1 model with Ollama
    result = invoke_ollama(
        model=llm_model,
        system_prompt=query_writer_prompt,
        user_prompt=f"Generate research queries for this user instruction in {detected_language} language: {user_instructions}",
        output_format=Queries
    )
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # result = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=query_writer_prompt,
    #     user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
    #     output_format=Queries
    # )

    # Add the original human query to the list of research queries
    all_queries = [user_instructions] + result.queries
    
    return {"research_queries": all_queries}

def search_queries(state: ResearcherState):
    # Kick off the search for each query by calling initiate_query_research
    print("--- Searching queries ---")
    # Get the current processing position from state or initialize to 0
    current_position = state.get("current_position", 0)

    return {"current_position": current_position + BATCH_SIZE}


def check_more_queries(state: ResearcherState) -> Literal["search_queries", "filter_search_summaries"]:
    """Check if there are more queries to process"""
    current_position = state.get("current_position", 0)
    if current_position < len(state["research_queries"]):
        return "search_queries"
    return "filter_search_summaries"

def initiate_query_research(state: ResearcherState):
    # Get the next batch of queries
    queries = state["research_queries"]
    current_position = state["current_position"]
    batch_end = min(current_position, len(queries))
    current_batch = queries[current_position - BATCH_SIZE:batch_end]
    detected_language = state.get("detected_language", "en")
    
    # Get the quality check loops from the main config
    quality_check_loops = state.get("quality_check_loops", 1)

    # Return the batch of queries to process with detected language and quality_check_loops in config
    return [
        Send("search_and_summarize_query", {"query": s, "configurable": {
            "detected_language": detected_language,
            "quality_check_loops": quality_check_loops
        }})
        for s in current_batch
    ]

def retrieve_rag_documents(state: QuerySearchState):
    """Retrieve documents from the RAG database."""
    print("--- Retrieving documents ---")
    query = state["query"]
    
    # Display embedding model information for this retrieval operation
    config = Configuration()
    embedding_model = config.embedding_model
    print(f"  [Using embedding model for retrieval: {embedding_model}]")
    
    # Use the new search_documents function from vector_db.py
    documents = search_documents(query, k=3)
    
    return {"retrieved_documents": documents}

def evaluate_retrieved_documents(state: QuerySearchState, config: RunnableConfig):
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = config["configurable"].get("detected_language", "en")
    
    evaluation_prompt = RELEVANCE_EVALUATOR_PROMPT.format(
        query=query,
        documents=format_documents_with_metadata(retrieved_documents)
    )
    
    # Using local Deepseek R1 model with Ollama
    evaluation = invoke_ollama(
        model=llm_model,
        system_prompt=evaluation_prompt,
        user_prompt=f"Evaluate the relevance of the retrieved documents for this query in {detected_language} language: {query}",
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
    output = tavily_search(state["query"])
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
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = config["configurable"].get("detected_language", "en")

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
    formatted_information = format_documents_with_metadata(information, preserve_original=True) if state["are_documents_relevant"] else information
    
    summary_prompt = SUMMARIZER_PROMPT.format(
        query=query,
        documents=formatted_information
    )
    
    # Using local model with Ollama
    summary = invoke_ollama(
        model=llm_model,
        system_prompt=summary_prompt,
        user_prompt=f"Extract and include relevant information from the documents that answers this query in {detected_language} language, preserving original wording: {query}"
    )
    # Remove thinking part if present
    summary = parse_output(summary)["response"]

    return {
        "search_summaries": [summary],
        "summary_improvement_iterations": 0,  # Initialize the iteration counter
        "original_summary": summary  # Store the original summary for reference
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
    quality_results = state.get("quality_check_results", {})
    
    # Check if improvement is needed based on the simplified quality metrics
    improvement_needed = quality_results.get("improvement_needed", False)
    quality_score = quality_results.get("quality_score", 0.7)
    is_sufficient = quality_results.get("is_accurate", True) and quality_results.get("is_complete", True)
    
    # Get current improvement iteration for this specific document
    iterations = state.get("summary_improvement_iterations", 0)
    
    # Get the maximum number of quality check loops from config
    quality_check_loops = config["configurable"].get("quality_check_loops", 1)
    
    # Only improve if we haven't reached the maximum number of iterations for this document
    if iterations < quality_check_loops:
        print(f"Document '{state['query']}': Proceeding with summary improvement. Quality score: {quality_score}, Sufficient: {is_sufficient}")
        print(f"Document '{state['query']}': Improvement iteration {iterations + 1}/{quality_check_loops}")
        return "improve_summary"
    else:
        print(f"Document '{state['query']}': Reached maximum number of improvement iterations ({quality_check_loops}). Quality score: {quality_score}, Sufficient: {is_sufficient}")
        return "__end__"

def improve_summary(state: QuerySearchState, config: RunnableConfig):
    """Improve the summary based on quality check feedback."""
    print("--- Improving summary based on quality check ---")
    query = state["query"]
    original_summary = state.get("original_summary", state["search_summaries"][0])
    current_summary = state["search_summaries"][0]
    quality_check_results = state["quality_check_results"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = config["configurable"].get("detected_language", "en")
    
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
    
    improvement_prompt = f"""
    You are tasked with improving a summary based on quality check feedback. The summary should accurately reflect the information in the source documents and address the query.
    
    Query: {query}
    
    Current Summary:
    {current_summary}
    
    Quality Check Feedback:
    {improvement_suggestions}
    {formatted_issues}
    
    Source Documents:
    {formatted_information}
    
    Please provide an improved summary that addresses the feedback and better answers the query. Ensure that all information is accurate and properly cited.
    """
    
    # Using local model with Ollama
    improved_summary = invoke_ollama(
        model=llm_model,
        system_prompt="You are an expert summarizer. Improve the summary based on the feedback provided.",
        user_prompt=f"Improve this summary in {detected_language} language based on the feedback: {improvement_prompt}"
    )
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # improved_summary = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt="You are an expert summarizer. Improve the summary based on the feedback provided.",
    #     user_prompt=improvement_prompt
    # )
    
    # Remove thinking part if present
    improved_summary = parse_output(improved_summary)["response"]
    
    print(f"Document '{query}': Summary improved (iteration {iterations}/{quality_check_loops})")
    
    # Return the improved summary and the updated iteration count
    return {
        "search_summaries": [improved_summary],
        "summary_improvement_iterations": iterations
    }

def filter_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Filter out irrelevant search summaries with a simpler approach"""
    print("--- Filtering search summaries ---")
    user_instructions = state["user_instructions"]
    search_summaries = state.get("search_summaries", [])
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "en")
    
    # If there are no summaries, return empty filtered summaries
    if not search_summaries:
        return {"filtered_summaries": []}
    
    # Keep track of filtered summaries
    filtered_summaries = []
    
    # Process each summary
    for summary in search_summaries:
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
            "no direct information"
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
    
    # If we have filtered out all summaries, include a placeholder
    if not filtered_summaries:
        placeholder = f"No relevant information was found for the query: {user_instructions}"
        filtered_summaries = [placeholder]
    
    print(f"Filtered {len(search_summaries)} summaries to {len(filtered_summaries)} relevant ones")
    
    return {"filtered_summaries": filtered_summaries}

def rank_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Simplified ranking of search summaries by relevance"""
    print("--- Ranking search summaries ---")
    user_instructions = state["user_instructions"]
    filtered_summaries = state.get("filtered_summaries", [])
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "en")
    
    # If there are no filtered summaries, return empty rankings
    if not filtered_summaries or (len(filtered_summaries) == 1 and "No relevant information" in filtered_summaries[0]):
        return {
            "ranked_summaries": [],
            "relevance_scores": []
        }
    
    # If there's only one summary, return it as is
    if len(filtered_summaries) == 1:
        return {
            "ranked_summaries": filtered_summaries,
            "relevance_scores": [1.0]
        }
    
    # For multiple summaries, use the LLM to rank them
    ranking_prompt = f"""
    Rank the following information summaries based on their relevance to the user's query. 
    Assign a score from 0.0 to 1.0 for each summary, where 1.0 is most relevant.
    
    User Query: {user_instructions}
    
    Summaries to rank:
    """
    
    # Add each summary to the prompt
    for i, summary in enumerate(filtered_summaries):
        ranking_prompt += f"\n\nSummary {i+1}:\n{summary}"
    
    ranking_prompt += "\n\nProvide your ranking as a JSON object with the following structure:"
    ranking_prompt += "\n{\"rankings\": [{\"summary_index\": 0, \"relevance_score\": 0.9, \"justification\": \"reason\"}, ...]}"
    ranking_prompt += "\nwhere 'summary_index' is the 0-based index of the summary."
    
    # Using local model with Ollama
    try:
        rankings = invoke_ollama(
            model=llm_model,
            system_prompt=ranking_prompt,
            user_prompt=f"Rank these summaries in {detected_language} language based on their relevance to: {user_instructions}",
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
    ranked_summaries = state.get("ranked_summaries", [])
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = state.get("detected_language", "en")
    
    # Determine report structure based on the query
    report_structure = """
    1. Introduction
    2. Main Findings
    3. Detailed Analysis
    4. Conclusion
    """
    
    # Combine the information from the ranked summaries
    if ranked_summaries:
        combined_information = "\n\n".join([summary for summary in ranked_summaries])
    else:
        combined_information = "No relevant information found."
    
    # Generate the final answer
    report_prompt = REPORT_WRITER_PROMPT.format(
        instruction=user_instructions,
        report_structure=report_structure,
        information=combined_information
    )
    
    final_answer = invoke_ollama(
        model=llm_model,
        system_prompt=report_prompt,
        user_prompt=f"Create a comprehensive report in {detected_language} language for: {user_instructions}"
    )
    
    # Remove thinking part if present
    final_answer = parse_output(final_answer)["response"]
    
    return {"final_answer": final_answer}

def quality_check_summary(state: QuerySearchState, config: RunnableConfig):
    """Simple quality check to ensure the summary contains sufficient information."""
    print("--- Quality checking summary ---")
    query = state["query"]
    current_summary = state["search_summaries"][0]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    detected_language = config["configurable"].get("detected_language", "en")
    quality_check_loops = config["configurable"].get("quality_check_loops", 1)  # Get configured loop count
    
    # Get the source documents
    information = None
    if state["are_documents_relevant"]:
        information = state["retrieved_documents"]
    else:
        information = state["web_search_results"]
    
    # Format documents with metadata
    formatted_information = format_documents_with_metadata(information) if state["are_documents_relevant"] else information
    
    quality_prompt = QUALITY_CHECKER_PROMPT.format(
        summary=current_summary,
        documents=formatted_information
    )
    
    try:
        # Initialize quality check results
        quality_check = None
        
        # Get current iteration for this document
        iterations = state.get("summary_improvement_iterations", 0)
        
        # Perform quality check for this document
        print(f"Document '{query}': Quality check iteration {iterations + 1}/{quality_check_loops}")
        
        # Using local model with Ollama
        quality_check = invoke_ollama(
            model=llm_model,
            system_prompt=quality_prompt,
            user_prompt=f"Evaluate the quality of this summary in {detected_language} language for the query: {query}",
            output_format=QualityCheckResult
        )
        
        # Log quality results
        quality_score = quality_check.quality_score
        is_sufficient = quality_check.is_accurate and quality_check.is_complete  # Determine if sufficient based on accuracy and completeness
        improvement_needed = quality_check.improvement_needed
        
        print(f"Document '{query}': Summary quality score: {quality_score}")
        print(f"Document '{query}': Is summary sufficient: {is_sufficient}")
        print(f"Document '{query}': Improvement needed: {improvement_needed}")
        
        return {"quality_check_results": quality_check.dict()}
        
    except Exception as e:
        print(f"Error during quality check for document '{query}': {str(e)}")
        # Default quality check results in case of error
        default_results = {
            "quality_score": 0.7,  # Medium quality score
            "is_accurate": True,  # Assume accurate
            "is_complete": True,  # Assume complete
            "issues_found": [],
            "missing_elements": [],
            "citation_issues": [],
            "improvement_needed": False,
            "improvement_suggestions": "No specific suggestions due to error in quality check."
        }
        print(f"Document '{query}': Proceeding with summary improvement using quality feedback. Quality score: {default_results['quality_score']}, Sufficient: {default_results['is_accurate'] and default_results['is_complete']}")
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

# Create main research agent graph
researcher_graph = StateGraph(ResearcherState, input=ResearcherStateInput, output=ResearcherStateOutput, config_schema=Configuration)

# Define main researcher nodes
researcher_graph.add_node(display_embedding_model_info)
researcher_graph.add_node(detect_language)
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(search_queries)
researcher_graph.add_node("search_and_summarize_query", query_search_subgraph.compile())
researcher_graph.add_node(filter_search_summaries)
researcher_graph.add_node(rank_search_summaries)
researcher_graph.add_node(generate_final_answer)

# Define transitions for the main graph
researcher_graph.add_edge(START, "display_embedding_model_info")
researcher_graph.add_edge("display_embedding_model_info", "detect_language")
researcher_graph.add_edge("detect_language", "generate_research_queries")
researcher_graph.add_edge("generate_research_queries", "search_queries")
researcher_graph.add_conditional_edges("search_queries", initiate_query_research, ["search_and_summarize_query"])
researcher_graph.add_conditional_edges("search_and_summarize_query", check_more_queries, ["search_queries", "filter_search_summaries"])
researcher_graph.add_edge("filter_search_summaries", "rank_search_summaries")
researcher_graph.add_edge("rank_search_summaries", "generate_final_answer")
researcher_graph.add_edge("generate_final_answer", END)

# Compile the researcher graph
researcher = researcher_graph.compile()

# Make sure researcher_graph is exported
__all__ = ["researcher", "researcher_graph"]