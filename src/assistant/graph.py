import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db, search_documents
from src.assistant.state import ResearcherState, ResearcherStateInput, ResearcherStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput, SummaryRanking
from src.assistant.prompts import RESEARCH_QUERY_WRITER_PROMPT, RELEVANCE_EVALUATOR_PROMPT, SUMMARIZER_PROMPT, REPORT_WRITER_PROMPT, QUALITY_CHECKER_PROMPT
from src.assistant.utils import format_documents_with_metadata, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries, SummaryRankings, SummaryRelevance, QualityCheckResult
import re
import time

# Number of query to process in parallel for each batch
# Change depending on the performance of the system
BATCH_SIZE = 3

def generate_research_queries(state: ResearcherState, config: RunnableConfig):
    print("--- Generating research queries ---")
    user_instructions = state["user_instructions"]
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
        user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
        output_format=Queries
    )
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # result = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=query_writer_prompt,
    #     user_prompt=f"Generate research queries for this user instruction: {user_instructions}",
    #     output_format=Queries
    # )

    return {"research_queries": result.queries}

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

    # Return the batch of queries to process
    return [
        Send("search_and_summarize_query", {"query": s})
        for s in current_batch
    ]

def retrieve_rag_documents(state: QuerySearchState):
    """Retrieve documents from the RAG database."""
    print("--- Retrieving documents ---")
    query = state["query"]
    
    # Use the new search_documents function from vector_db.py
    documents = search_documents(query, k=3)
    
    return {"retrieved_documents": documents}

def evaluate_retrieved_documents(state: QuerySearchState, config: RunnableConfig):
    query = state["query"]
    retrieved_documents = state["retrieved_documents"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    evaluation_prompt = RELEVANCE_EVALUATOR_PROMPT.format(
        query=query,
        documents=format_documents_with_metadata(retrieved_documents)
    )
    
    # Using local Deepseek R1 model with Ollama
    evaluation = invoke_ollama(
        model=llm_model,
        system_prompt=evaluation_prompt,
        user_prompt=f"Evaluate the relevance of the retrieved documents for this query: {query}",
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
        user_prompt=f"Extract and include relevant information from the documents that answers this query, preserving original wording: {query}"
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

def quality_check_summary(state: QuerySearchState, config: RunnableConfig):
    """Simple quality check to ensure the summary contains sufficient information."""
    print("--- Quality checking summary ---")
    query = state["query"]
    current_summary = state["search_summaries"][0]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
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
        
        # Perform quality checks in a loop based on configured count
        for i in range(quality_check_loops):
            print(f"Quality check iteration {i+1}/{quality_check_loops}")
            
            # Using local model with Ollama
            quality_check = invoke_ollama(
                model=llm_model,
                system_prompt=quality_prompt,
                user_prompt=f"Evaluate if this summary contains sufficient information to answer the query: {query}",
                output_format=QualityCheckResult
            )
            
            # Log quality results
            quality_score = quality_check.quality_score
            is_sufficient = quality_check.is_sufficient
            improvement_needed = quality_check.improvement_needed
            
            print(f"Summary quality score: {quality_score}")
            print(f"Is summary sufficient: {is_sufficient}")
            print(f"Improvement needed: {improvement_needed}")
            
            # If the quality is good enough, break the loop
            if is_sufficient and not improvement_needed:
                print(f"Quality check passed on iteration {i+1}, breaking loop")
                break
            
            # If this is not the last iteration and improvement is needed, wait briefly
            if i < quality_check_loops - 1 and improvement_needed:
                print("Waiting before next quality check iteration...")
                time.sleep(1)  # Brief pause between iterations
        
        return {"quality_check_results": quality_check.dict()}
        
    except Exception as e:
        print(f"Error during quality check: {str(e)}")
        # Default quality check results in case of error
        default_results = {
            "quality_score": 0.7,  # Medium quality score
            "is_sufficient": True,  # Assume sufficient to continue
            "improvement_needed": False,  # Don't force improvement to avoid loops
            "improvement_suggestions": "No specific suggestions due to error in quality check."
        }
        return {"quality_check_results": default_results}

def route_quality_check(state: QuerySearchState) -> Literal["improve_summary", "__end__"]:
    """Route based on quality check results."""
    quality_results = state.get("quality_check_results", {})
    
    # Check if improvement is needed based on the simplified quality metrics
    improvement_needed = quality_results.get("improvement_needed", False)
    quality_score = quality_results.get("quality_score", 0.7)
    is_sufficient = quality_results.get("is_sufficient", True)
    
    # Get current improvement iteration
    iterations = state.get("summary_improvement_iterations", 0)
    
    # Only improve the summary once, regardless of quality metrics
    if iterations == 0:
        print(f"Proceeding with summary improvement using quality feedback. Quality score: {quality_score}, Sufficient: {is_sufficient}")
        return "improve_summary"
    else:
        print(f"Summary has already been improved once, proceeding with current version. Quality score: {quality_score}, Sufficient: {is_sufficient}")
        return "__end__"

def improve_summary(state: QuerySearchState, config: RunnableConfig):
    """Improve the summary based on quality check feedback."""
    print("--- Improving summary with quality feedback ---")
    query = state["query"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    current_summary = state["search_summaries"][0]
    quality_results = state["quality_check_results"]
    
    # Get the source documents
    information = None
    if state["are_documents_relevant"]:
        information = state["retrieved_documents"]
    else:
        information = state["web_search_results"]
    
    # Format documents with metadata but simplified
    formatted_information = format_documents_with_metadata(information) if state["are_documents_relevant"] else information
    
    # Extract quality feedback
    quality_score = quality_results.get("quality_score", 0.7)
    is_sufficient = quality_results.get("is_sufficient", False)
    improvement_suggestions = quality_results.get("improvement_suggestions", "")
    
    # Create an improved prompt that explicitly incorporates quality feedback
    improvement_prompt = f"""Improve this summary by addressing the specific quality feedback provided.

Original Query: {query}

Current Summary:
{current_summary}

Source Documents:
{formatted_information}

Quality Feedback:
- Quality Score: {quality_score}
- Is sufficient: {'Yes' if is_sufficient else 'No'}
- Improvement Suggestions: {improvement_suggestions}

Your task is to create ONE improved version of the summary that:
1. Addresses the specific improvement suggestions
2. Extracts the most relevant information from the source documents
3. Directly answers the original query
4. Preserves original wording from the source documents where appropriate
"""
    
    try:
        # Generate improved summary incorporating quality feedback
        improved_summary = invoke_ollama(
            model=llm_model,
            system_prompt=improvement_prompt,
            user_prompt=f"Create ONE improved version of this summary based on the quality feedback provided."
        )
        # Remove thinking part if present
        improved_summary = parse_output(improved_summary)["response"]
            
    except Exception as e:
        print(f"Error generating improved summary: {str(e)}")
        # If there's an error, keep the original summary
        improved_summary = current_summary
    
    # Set iteration counter to 1 to indicate one improvement has been made
    return {
        "search_summaries": [improved_summary],
        "summary_improvement_iterations": 1
    }

def filter_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Filter out irrelevant search summaries with a simpler approach"""
    print("--- Filtering search summaries ---")
    user_instructions = state["user_instructions"]
    search_summaries = state["search_summaries"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # Skip empty or minimal summaries
    valid_summaries = []
    for summary in search_summaries:
        if summary and len(summary.strip()) > 50:
            valid_summaries.append(summary)
    
    # If all summaries were excluded, keep at least one with the most content
    if not valid_summaries and search_summaries:
        valid_summaries = [max(search_summaries, key=lambda s: len(s.strip()) if s else 0)]
    
    # Simple evaluation of relevance with minimal overhead
    filtered_summaries = []
    for summary in valid_summaries:
        evaluation_prompt = f"""Determine if this summary is relevant to the original query.
        
Original Query: {user_instructions}

Summary:
{summary}

Your output must only be a valid JSON object with keys "is_relevant" (boolean) and "confidence" (float 0-1).
"""
        
        result = invoke_ollama(
            model=llm_model,
            system_prompt=evaluation_prompt,
            user_prompt="Is this summary relevant to the original query?",
            output_format=SummaryRelevance
        )
        
        if result.is_relevant:
            filtered_summaries.append(summary)
    
    # If all summaries were filtered out, keep the most relevant one
    if not filtered_summaries and valid_summaries:
        filtered_summaries = [valid_summaries[0]]
    
    return {"filtered_summaries": filtered_summaries}

def rank_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Simplified ranking of search summaries by relevance"""
    print("--- Ranking search summaries ---")
    user_instructions = state["user_instructions"]
    filtered_summaries = state["filtered_summaries"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # If we have only one summary, no need to rank
    if len(filtered_summaries) <= 1:
        return {
            "ranked_summaries": filtered_summaries,
            "relevance_scores": [1.0] if filtered_summaries else []
        }
    
    ranked_results = []
    
    # For each filtered summary, calculate a simple relevance score
    for i, summary in enumerate(filtered_summaries):
        ranking_prompt = f"""Rate how relevant this information is to answering the query.

Query: {user_instructions}

Information:
{summary}

Your output must be a JSON object with keys "relevance_score" (float 0-1) and "justification" (brief explanation).
"""
        
        try:
            ranking = invoke_ollama(
                model=llm_model,
                system_prompt=ranking_prompt,
                user_prompt="Rate the relevance of this information",
                output_format=SummaryRanking
            )
            
            ranked_results.append({
                "summary_index": i,
                "relevance_score": ranking.relevance_score,
                "justification": ranking.justification
            })
            
        except Exception as e:
            print(f"Error ranking summary {i}: {str(e)}")
            # Assign a default middle score
            ranked_results.append({
                "summary_index": i,
                "relevance_score": 0.5,
                "justification": "Ranking failed - assigned default score"
            })
    
    # Sort by relevance score, highest first
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Prepare the ranked summaries and scores
    ranked_summaries = [filtered_summaries[result["summary_index"]] for result in ranked_results]
    relevance_scores = [result["relevance_score"] for result in ranked_results]
    
    return {
        "ranked_summaries": ranked_summaries,
        "relevance_scores": relevance_scores
    }

def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    """Generate the final answer using relevant information from ranked summaries"""
    print("--- Generating final answer ---")
    user_instructions = state["user_instructions"]
    ranked_summaries = state.get("ranked_summaries", [])
    relevance_scores = state.get("relevance_scores", [])
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # Get report structure configuration
    report_structure_file = config["configurable"].get("report_structure_file", "./report_structures/standard report.md")
    try:
        with open(report_structure_file, "r") as f:
            report_structure = f.read()
    except Exception as e:
        print(f"Error reading report structure file: {str(e)}")
        report_structure = "# Report\n\n## Summary\n\n## Key Findings\n\n## Conclusion"
    
    # Combine all relevant information from ranked summaries
    combined_information = ""
    
    if ranked_summaries:
        # Include all ranked summaries, sorted by relevance
        for i, (summary, score) in enumerate(zip(ranked_summaries, relevance_scores)):
            # Only include if it has a reasonable amount of content
            if summary and len(summary.strip()) > 50:
                combined_information += f"\n\n--- Information {i+1} (Relevance Score: {score}) ---\n\n{summary}"
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
        user_prompt=f"Create a comprehensive report for: {user_instructions}"
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

# Create main research agent graph
researcher_graph = StateGraph(ResearcherState, input=ResearcherStateInput, output=ResearcherStateOutput, config_schema=Configuration)

# Define main researcher nodes
researcher_graph.add_node(generate_research_queries)
researcher_graph.add_node(search_queries)
researcher_graph.add_node("search_and_summarize_query", query_search_subgraph.compile())
researcher_graph.add_node(filter_search_summaries)
researcher_graph.add_node(rank_search_summaries)
researcher_graph.add_node(generate_final_answer)

# Define transitions for the main graph
researcher_graph.add_edge(START, "generate_research_queries")
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