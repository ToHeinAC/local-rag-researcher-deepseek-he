import datetime
from typing_extensions import Literal
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langchain_core.runnables.config import RunnableConfig
from src.assistant.configuration import Configuration
from src.assistant.vector_db import get_or_create_vector_db, search_documents
from src.assistant.state import ResearcherState, ResearcherStateInput, ResearcherStateOutput, QuerySearchState, QuerySearchStateInput, QuerySearchStateOutput, SummaryRanking
from src.assistant.prompts import RESEARCH_QUERY_WRITER_PROMPT, RELEVANCE_EVALUATOR_PROMPT, SUMMARIZER_PROMPT, REPORT_WRITER_PROMPT
from src.assistant.utils import format_documents_with_metadata, invoke_llm, invoke_ollama, parse_output, tavily_search, Evaluation, Queries, SummaryRankings, SummaryRelevance
import re

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
    pass

def initiate_query_research(state: ResearcherState):
    # Kick off the search for each query in parallel using Send method and calling the "search_and_summarize_query" subgraph
    return [
        Send("search_and_summarize_query", {"query": s})
        for s in state["research_queries"]
    ]

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

    # Format documents with metadata to include sources and document links
    formatted_information = format_documents_with_metadata(information) if state["are_documents_relevant"] else information
    
    summary_prompt = SUMMARIZER_PROMPT.format(
        query=query,
        documents=formatted_information
    )
    
    # Using local Deepseek R1 model with Ollama
    summary = invoke_ollama(
        model=llm_model,
        system_prompt=summary_prompt,
        user_prompt=f"Generate a research summary for this query: {query}"
    )
    # Remove thinking part (reasoning between <think> tags)
    summary = parse_output(summary)["response"]
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # summary = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=summary_prompt,
    #     user_prompt=f"Generate a research summary for this query: {query}"
    # )

    return {"search_summaries": [summary]}

def filter_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Filter out irrelevant search summaries"""
    print("--- Filtering search summaries ---")
    user_instructions = state["user_instructions"]
    search_summaries = state["search_summaries"]
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # For each summary, determine if it's relevant to the original query
    filtered_summaries = []
    filtering_details = []
    
    for i, summary in enumerate(search_summaries):
        evaluation_prompt = f"""
        Determine if this summary is relevant to the original query.
        
        Original Query: {user_instructions}
        
        Summary {i+1}:
        {summary}
        
        Evaluate the relevance of this summary to the original query.
        Your output must only be a valid JSON object with keys "is_relevant" (boolean), "confidence" (float 0-1), and "justification" (string explaining your decision).
        """
        
        result = invoke_ollama(
            model=llm_model,
            system_prompt=evaluation_prompt,
            user_prompt="Is this summary relevant to the original query?",
            output_format=SummaryRelevance
        )
        
        # Store details about this evaluation
        filtering_details.append({
            "summary_index": i+1,
            "is_relevant": result.is_relevant,
            "confidence": result.confidence,
            "justification": getattr(result, 'justification', 'No justification provided'),
            "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary
        })
        
        if result.is_relevant:
            filtered_summaries.append(summary)
    
    # If all summaries were filtered out, keep at least one (the most relevant one)
    if not filtered_summaries and search_summaries:
        # Find the summary with highest confidence
        if filtering_details:
            most_confident_idx = max(range(len(filtering_details)), 
                                   key=lambda i: filtering_details[i]["confidence"] if not filtering_details[i]["is_relevant"] else 0)
            filtered_summaries = [search_summaries[most_confident_idx]]
            # Mark this summary as included despite being irrelevant
            filtering_details[most_confident_idx]["included_anyway"] = True
        else:
            filtered_summaries = [search_summaries[0]]
        
    print(f"Filtered {len(search_summaries)} summaries down to {len(filtered_summaries)}")
    
    return {
        "filtered_summaries": filtered_summaries,
        "filtering_details": filtering_details
    }

def rank_search_summaries(state: ResearcherState, config: RunnableConfig):
    """Rank search summaries by relevance to the original query"""
    print("--- Ranking search summaries ---")
    user_instructions = state["user_instructions"]
    
    # Use filtered summaries if available, otherwise use all summaries
    summaries = state.get("filtered_summaries", state["search_summaries"])
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # If there's only one summary, no need to rank
    if len(summaries) <= 1:
        return {
            "ranked_summaries": summaries,
            "relevance_scores": [10] if summaries else []
        }
    
    # Create a prompt to rank summaries
    ranking_prompt = """
    You are a research assistant tasked with ranking information summaries by relevance to the original query.
    
    Original Query: {query}
    
    For each summary, assign a relevance score from 1-10 and provide a brief justification.
    Your output must be a valid JSON array of objects with 'summary_index', 'relevance_score', and 'justification' keys.
    
    Summaries:
    {summaries}
    """
    
    formatted_summaries = "\n\n---\n\n".join([f"Summary {i+1}:\n{summary}" for i, summary in enumerate(summaries)])
    
    result = invoke_ollama(
        model=llm_model,
        system_prompt=ranking_prompt.format(query=user_instructions, summaries=formatted_summaries),
        user_prompt="Rank these research summaries by relevance to the original query",
        output_format=SummaryRankings
    )
    
    # Sort summaries by relevance score
    ranked_summaries = sorted(result.rankings, key=lambda x: x.relevance_score, reverse=True)
    
    # Return the ranked summaries and their scores
    return {
        "ranked_summaries": [summaries[r.summary_index-1] for r in ranked_summaries],
        "relevance_scores": [r.relevance_score for r in ranked_summaries]
    }

def generate_final_answer(state: ResearcherState, config: RunnableConfig):
    print("--- Generating final answer ---")
    user_instructions = state["user_instructions"]
    report_structure = config["configurable"].get("report_structure", "")
    llm_model = config["configurable"].get("llm_model", "deepseek-r1:latest")
    
    # Get the ranked summaries (top 5 most relevant)
    ranked_summaries = state.get("ranked_summaries", state.get("filtered_summaries", state["search_summaries"]))
    top_summaries = ranked_summaries[:5] if len(ranked_summaries) > 5 else ranked_summaries
    
    # Extract document links from the summaries to ensure we cite original documents
    document_links = []
    for summary in top_summaries:
        # Extract all markdown links from the summary using regex
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', summary)
        for link_text, link_url in links:
            document_links.append((link_text, link_url))
    
    # Format the summaries with document links
    formatted_summaries = []
    for i, summary in enumerate(top_summaries):
        relevance_score = state.get('relevance_scores', [0] * len(top_summaries))[i]
        formatted_summary = f"Summary {i+1}" + (f" (Relevance: {relevance_score:.2f})" if relevance_score else "") + f":\n{summary}"
        formatted_summaries.append(formatted_summary)
    
    # Join the formatted summaries
    all_summaries = "\n\n---\n\n".join(formatted_summaries)
    
    # Add document links information for the report writer
    document_links_info = "\n\nOriginal Document Links:\n"
    if document_links:
        for i, (doc_name, doc_link) in enumerate(document_links):
            document_links_info += f"- [{doc_name}]({doc_link})\n"
    else:
        document_links_info += "No document links found in the summaries."
    
    all_summaries += document_links_info
    
    # Generate the final answer using the report writer prompt
    report_prompt = REPORT_WRITER_PROMPT.format(
        instruction=user_instructions,
        report_structure=report_structure,
        information=all_summaries
    )
    
    # Using local Deepseek R1 model with Ollama
    final_answer = invoke_ollama(
        model=llm_model,
        system_prompt=report_prompt,
        user_prompt=f"Generate a comprehensive report that answers this query: {user_instructions}"
    )
    # Remove thinking part (reasoning between <think> tags)
    final_answer = parse_output(final_answer)["response"]
    
    # Using external LLM providers with OpenRouter: GPT-4o, Claude, Deepseek R1,... 
    # final_answer = invoke_llm(
    #     model='gpt-4o-mini',
    #     system_prompt=report_prompt,
    #     user_prompt=f"Generate a comprehensive report that answers this query: {user_instructions}"
    # )
    
    return {"final_answer": final_answer}

# Create subghraph for searching each query
query_search_subgraph = StateGraph(QuerySearchState, input=QuerySearchStateInput, output=QuerySearchStateOutput)

# Define subgraph nodes for searching the query
query_search_subgraph.add_node(retrieve_rag_documents)
query_search_subgraph.add_node(evaluate_retrieved_documents)
query_search_subgraph.add_node(web_research)
query_search_subgraph.add_node(summarize_query_research)

# Set entry point and define transitions for the subgraph
query_search_subgraph.add_edge(START, "retrieve_rag_documents")
query_search_subgraph.add_edge("retrieve_rag_documents", "evaluate_retrieved_documents")
query_search_subgraph.add_conditional_edges("evaluate_retrieved_documents", route_research)
query_search_subgraph.add_edge("web_research", "summarize_query_research")
query_search_subgraph.add_edge("summarize_query_research", END)

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