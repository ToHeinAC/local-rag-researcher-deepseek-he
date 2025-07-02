import json
import asyncio
import os
import sys
import importlib.util
from datetime import datetime
from types import ModuleType

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_core.tools import tool

# Create a dummy tavily module to prevent import errors
class DummyTavilyClient:
    """Dummy implementation of TavilyClient to prevent import errors"""
    def __init__(self, *args, **kwargs):
        pass
        
    def search(self, *args, **kwargs):
        print("DummyTavilyClient.search called")
        return []

# Create dummy tavily module
dummy_tavily = ModuleType('tavily')
dummy_tavily.TavilyClient = DummyTavilyClient

# Add dummy module to sys.modules if tavily is not installed
if 'tavily' not in sys.modules:
    print("Adding dummy tavily module to sys.modules")
    sys.modules['tavily'] = dummy_tavily

# Custom implementation of evaluators since openevals is not available
from langchain.prompts import PromptTemplate

# Define prompts for evaluators
RAG_RETRIEVAL_RELEVANCE_PROMPT = """
You are an objective judge evaluating the relevance of search results to a user's query.

<query>
{query}
</query>

<search_results>
{search_results}
</search_results>

Based on the search results provided, are they relevant to the query?
Return ONLY 'true' if the results are relevant to the query, or 'false' if they are not relevant.
"""

RAG_HELPFULNESS_PROMPT = """
You are an objective judge evaluating the helpfulness of an answer to a user's question.

<question>
{question}
</question>

<answer>
{answer}
</answer>

Based on the answer provided, is it helpful to the user's question?
"""

# Use the absolute path to local-rag-researcher-deepseek-he directory
local_rag_path = '/home/he/ai/dev/langgraph/local-rag-researcher-deepseek-he'

# Import local RAG functionality
sys.path.insert(0, local_rag_path)

# Import necessary modules from src.assistant
from src.assistant.rag_helpers import (
    load_embed,
    get_tenant_vectorstore,
    get_tenant_collection_name,
    calculate_chunk_ids,
    extract_text_from_pdf,
    transform_documents,
    source_summarizer_ollama,
    similarity_search_for_tenant
)
from src.assistant.utils import clear_cuda_memory

# Default model
model = init_chat_model("ollama:llama3.2", temperature=0.1)

current_date = datetime.now().strftime("%A, %B %d, %Y")

MAX_SEARCH_RETRIES = 5

# Default database configuration
DEFAULT_DATABASE = "Qwen--Qwen3-Embedding-0.6B--3000--600"
DEFAULT_TENANT = "default"


class GraphState(MessagesState):
    original_question: str
    attempted_search_queries: list[str]
    database_path: str = None
    tenant_id: str = None


# Create prompt templates for the evaluators
relevance_prompt_template = PromptTemplate(
    template=RAG_RETRIEVAL_RELEVANCE_PROMPT + f"\n\nThe current date is {current_date}.",
    input_variables=["query", "search_results"]
)

helpfulness_prompt_template = PromptTemplate(
    template=RAG_HELPFULNESS_PROMPT + f'\nReturn "true" if the answer is helpful, and "false" otherwise.\n\nThe current date is {current_date}.',
    input_variables=["question", "answer"]
)

# Implement async evaluator functions
async def relevance_evaluator(inputs):
    """
    Evaluate the relevance of search results to a query.
    
    Args:
        inputs: A dictionary with keys 'query' and 'search_results'
        
    Returns:
        'true' if the results are relevant, 'false' otherwise
    """
    try:
        query = inputs["query"]
        search_results = inputs["search_results"]
        
        # Format the prompt with the inputs
        formatted_prompt = relevance_prompt_template.format(
            query=query,
            search_results=search_results
        )
        
        # Use the model to evaluate relevance
        result = await model.ainvoke(formatted_prompt)
        result_text = result.content.strip().lower()
        
        # Extract just 'true' or 'false' from the response
        if "true" in result_text:
            return "true"
        else:
            return "false"
    except Exception as e:
        print(f"Error in relevance evaluation: {e}")
        # Default to true in case of error
        return "true"

async def helpfulness_evaluator(inputs):
    """
    Evaluate the helpfulness of an answer to a question.
    
    Args:
        inputs: A dictionary with keys 'question' and 'answer'
        
    Returns:
        'true' if the answer is helpful, 'false' otherwise
    """
    try:
        question = inputs["question"]
        answer = inputs["answer"]
        
        # Format the prompt with the inputs
        formatted_prompt = helpfulness_prompt_template.format(
            question=question,
            answer=answer
        )
        
        # Use the model to evaluate helpfulness
        result = await model.ainvoke(formatted_prompt)
        result_text = result.content.strip().lower()
        
        # Extract just 'true' or 'false' from the response
        if "true" in result_text:
            return "true"
        else:
            return "false"
    except Exception as e:
        print(f"Error in helpfulness evaluation: {e}")
        # Default to true in case of error
        return "true"


SYSTEM_PROMPT = """
Use the provided local database retrieval tool to find information relevant to the user's question.
"""


# Function to detect language of text
def detect_language(text):
    """Detect the language of the input text.
    
    This is a simple implementation that checks for common German characters.
    For a production system, consider using a proper language detection library like langdetect.
    """
    # Check for common German characters/words
    german_chars = ['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü']
    german_words = ['der', 'die', 'das', 'und', 'ist', 'von', 'für', 'mit']
    
    # Convert to lowercase for word matching
    text_lower = text.lower()
    
    # Check for German characters
    for char in german_chars:
        if char in text:
            return "German"
    
    # Check for common German words
    for word in german_words:
        if f" {word} " in f" {text_lower} ":
            return "German"
    
    # Default to English if no German indicators found
    return "English"

# Create a local database retrieval tool
@tool
async def local_retrieval_tool(query: str, database_path: str = None, tenant_id: str = None, language: str = None, k: int = 3):
    """Search the local database for information relevant to the query.
    
    Args:
        query: The search query
        database_path: Optional path to the database directory
        tenant_id: Optional tenant ID for the database
        language: Optional language of the query (English or German)
        k: Number of results to return (default: 3)
    """
    import logging
    from src.assistant.rag_helpers import similarity_search_for_tenant
    
    # Import embedding models with fallbacks
    try:
        from src.assistant.embeddings import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Use default database path and tenant if not provided
    if database_path is None:
        database_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "database",
            DEFAULT_DATABASE
        )
    
    if tenant_id is None:
        tenant_id = DEFAULT_TENANT
    
    # Detect language if not provided
    if language is None:
        language = detect_language(query)
        logger.info(f"Detected language: {language} for query: {query}")
    
    # Get embedding model
    try:
        # Extract embedding model name from database path
        db_name = os.path.basename(database_path)
        parts = db_name.split('--')
        if len(parts) >= 2:
            model_name = parts[0].replace('--', '/') + '/' + parts[1]
        else:
            model_name = "jinaai/jina-embeddings-v2-base-de"  # Default model
            
        # Initialize embedding model
        embed_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )
        
        # Perform similarity search
        results = similarity_search_for_tenant(
            tenant_id=tenant_id,
            embed_llm=embed_model,
            persist_directory=database_path,
            similarity="cosine",
            normal=True,
            query=query,
            k=k,
            language=language
        )
        
        return {
            "results": results, 
            "detected_language": language, 
            "database_path": database_path, 
            "tenant_id": tenant_id
        }
    except Exception as e:
        logger.error(f"Error in local retrieval: {str(e)}")
        return {
            "error": str(e),
            "detected_language": language,
            "database_path": database_path,
            "tenant_id": tenant_id
        }


model_with_tools = model.bind_tools([local_retrieval_tool])


async def relevance_filter(state: GraphState):
    """Filter out irrelevant search results."""
    query = state["original_question"]
    search_results = state["messages"][-1].content
    
    # Instead of using ainvoke, directly call the relevance_evaluator function
    try:
        # Check if relevance_evaluator is callable directly
        if callable(relevance_evaluator):
            is_relevant = await relevance_evaluator({"query": query, "search_results": search_results})
        else:
            # Fallback: assume all results are relevant if we can't evaluate
            print("Warning: relevance_evaluator is not callable, assuming results are relevant")
            is_relevant = "true"
    except Exception as e:
        print(f"Error in relevance evaluation: {e}")
        # Fallback: assume all results are relevant if evaluation fails
        is_relevant = "true"
    
    if is_relevant == "true":
        return {"messages": state["messages"]}
    return {}


async def should_continue(state: GraphState):
    if len(state["attempted_search_queries"]) > MAX_SEARCH_RETRIES:
        return END
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "local_retrieval"
    return "reflect"


async def call_model(state: GraphState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    response = await model_with_tools.ainvoke(messages)
    if response.tool_calls and response.tool_calls[0]["name"] == local_retrieval_tool.name:
        search_query = response.tool_calls[0]["args"]["query"]
        
        # Extract database_path and tenant_id from tool_calls if provided
        tool_args = response.tool_calls[0]["args"]
        database_path = tool_args.get("database_path", state.get("database_path"))
        tenant_id = tool_args.get("tenant_id", state.get("tenant_id"))
        
        return {
            "messages": [response],
            "attempted_search_queries": state["attempted_search_queries"] + [search_query],
            "database_path": database_path,
            "tenant_id": tenant_id,
        }
    return {"messages": [response]}


async def local_retrieval(state: GraphState):
    last_message = state["messages"][-1]
    
    # Extract the query and optional database parameters
    tool_args = last_message.tool_calls[0]["args"]
    query = tool_args["query"]
    database_path = tool_args.get("database_path", state.get("database_path"))
    tenant_id = tool_args.get("tenant_id", state.get("tenant_id"))
    language = tool_args.get("language", None)
    k = tool_args.get("k", 3)  # Default to 3 results if not specified
    
    try:
        # Call the local retrieval tool with the appropriate parameters
        search_results = await local_retrieval_tool.ainvoke(input={
            "query": query,
            "database_path": database_path,
            "tenant_id": tenant_id,
            "language": language,
            "k": k
        })
        
        # Format the response as a proper message with role and content
        if isinstance(search_results, dict) and "error" in search_results:
            # Format error message properly
            error_message = {
                "role": "assistant",
                "content": f"Error during retrieval: {search_results['error']}\nNo documents found. Please try a different query or database configuration."
            }
            return {"messages": [error_message]}
        elif isinstance(search_results, dict) and "results" in search_results:
            # Format successful response with results
            results = search_results["results"]
            
            # Format the results as a string if they're not already
            if not isinstance(results, str):
                from src.assistant.rag_helpers import format_documents_as_plain_text
                results_text = format_documents_as_plain_text(results)
            else:
                results_text = results
                
            return {"messages": [{
                "role": "assistant",
                "content": results_text
            }]}
        else:
            # Format successful response with no specific results
            return {"messages": [{
                "role": "assistant",
                "content": "No results found. Please try a different query or database configuration."
            }]}
    except Exception as e:
        # Handle any exceptions and format as proper message
        error_message = {
            "role": "assistant",
            "content": f"Error during retrieval: {str(e)}\nNo documents found. Please try a different query or database configuration."
        }
        return {"messages": [error_message]}



async def reflect(state: GraphState):
    """Reflect on the answer and decide whether to retry."""
    question = state["original_question"]
    answer = state["messages"][-1].content

    # Evaluate the helpfulness of the answer
    try:
        # Check if helpfulness_evaluator is callable directly
        if callable(helpfulness_evaluator):
            is_helpful = await helpfulness_evaluator({"question": question, "answer": answer})
        else:
            # Fallback: assume answer is helpful if we can't evaluate
            print("Warning: helpfulness_evaluator is not callable, assuming answer is helpful")
            is_helpful = "true"
    except Exception as e:
        print(f"Error in helpfulness evaluation: {e}")
        # Fallback: assume answer is helpful if evaluation fails
        is_helpful = "true"

    if is_helpful == "true":
        # If the answer is helpful, we're done
        return {"messages": state["messages"]}

    # If we've already tried the maximum number of times, we're done
    if len(state["attempted_search_queries"]) >= MAX_SEARCH_RETRIES:
        # Add a reflection message
        reflection_message = {
            "role": "assistant",
            "content": "I apologize, but I'm having trouble finding relevant information to answer your question accurately. Let me provide the best answer I can based on my general knowledge.",
        }
        return {"messages": state["messages"] + [reflection_message]}

    # Otherwise, try again with a different approach
    reflection_message = {
        "role": "assistant",
        "content": f"""
I originally asked you the following question:

<original_question>
{state["original_question"]}
</original_question>

Your answer was not helpful for the following reason:

<reason>
The answer was not relevant to the question.
</reason>

Please check the conversation history carefully and try again. You may choose to fetch more information if you think the answer
to the original question is not somewhere in the conversation, but carefully consider if the answer is already in the conversation.

You have already attempted to answer the original question using the following search queries,
so if you choose to search again, you must rephrase your search query to be different from the ones below to avoid fetching redundant information:

<attempted_search_queries>
{state['attempted_search_queries']}
</attempted_search_queries>

As a reminder, check the previous conversation history and fetched context carefully before searching again!
""",
        }
    
    return {"messages": state["messages"] + [reflection_message]}


async def retry_or_end(state: GraphState):
    if state["messages"][-1].type == "human":
        return "agent"
    return END


async def store_database_config(state: GraphState):
    """Store the database configuration in the state."""
    # Get the database path and tenant ID from the parameters or use defaults
    database_path = state.get("database_path")
    tenant_id = state.get("tenant_id")
    
    # If not provided, use the defaults
    if database_path is None:
        database_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "local-rag-researcher-deepseek-he",
            "database",
            DEFAULT_DATABASE
        )
    
    if tenant_id is None:
        tenant_id = DEFAULT_TENANT
    
    return {
        "database_path": database_path,
        "tenant_id": tenant_id
    }


workflow = StateGraph(GraphState, input=MessagesState, output=MessagesState)

workflow.add_node(
    "store_original_question",
    lambda state: {
        "original_question": state["messages"][-1].content,
        "attempted_search_queries": [],
    },
)
workflow.add_node("store_database_config", store_database_config)
workflow.add_node("agent", call_model)
workflow.add_node("local_retrieval", local_retrieval)
workflow.add_node("relevance_filter", relevance_filter)
workflow.add_node("reflect", reflect)

workflow.add_edge(START, "store_original_question")
workflow.add_edge("store_original_question", "store_database_config")
workflow.add_edge("store_database_config", "agent")
workflow.add_conditional_edges("agent", should_continue, ["local_retrieval", "reflect", END])
workflow.add_edge("local_retrieval", "relevance_filter")
workflow.add_edge("relevance_filter", "agent")
workflow.add_conditional_edges(
    "reflect",
    retry_or_end,
    ["agent", END],
)

agent = workflow.compile()


# Function to initialize the agent with custom database configuration
def init_agent(database_path=None, tenant_id=None):
    """
    Initialize the corrective RAG agent with custom database configuration.
    
    Args:
        database_path (str, optional): Path to the database directory.
        tenant_id (str, optional): Tenant ID for the database.
        
    Returns:
        The initialized agent.
    """
    # The agent is already compiled, but we can set the initial state
    # when we invoke it later
    return agent


# Example usage
async def run_agent(question, database_path=None, tenant_id=None):
    """
    Run the agent with a question and optional database configuration.
    
    Args:
        question (str): The question to ask the agent.
        database_path (str, optional): Path to the database directory.
        tenant_id (str, optional): Tenant ID for the database.
        
    Returns:
        The agent's response.
    """
    # Create initial state with database configuration
    initial_state = {
        "messages": [{"role": "user", "content": question}],
        "database_path": database_path,
        "tenant_id": tenant_id,
    }
    
    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    # Return the last message from the agent
    return result["messages"][-1].content
