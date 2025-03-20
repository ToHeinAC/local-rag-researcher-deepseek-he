import os
import re
import shutil
from ollama import chat
from tavily import TavilyClient
from pydantic import BaseModel
from langchain_community.document_loaders import CSVLoader, TextLoader, PDFPlumberLoader
import torch

class Evaluation(BaseModel):
    is_relevant: bool

class Queries(BaseModel):
    queries: list[str]

def parse_output(text):
    # First try to extract thinking part if it exists
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    
    if think_match:
        think = think_match.group(1).strip()
        output = re.search(r'</think>\s*(.*?)$', text, re.DOTALL).group(1).strip()
    else:
        think = None
        output = text.strip()
    
    # Check if the output is in JSON format with key-value pairs
    try:
        import json
        
        # Check if the text looks like JSON
        if (output.startswith('{') and output.endswith('}')) or (output.startswith('[') and output.endswith(']')):
            # Try to parse as JSON
            json_obj = json.loads(output)
            
            # If it's a dict with a 'final_answer' or similar key, extract just the value
            if isinstance(json_obj, dict):
                # Look for common keys that might contain the main content
                for key in ['final_answer', 'answer', 'response', 'content', 'result', 'output']:
                    if key in json_obj:
                        output = json_obj[key]
                        break
                # If no specific key was found but there's only one value, use that
                if len(json_obj) == 1:
                    output = list(json_obj.values())[0]
    except (json.JSONDecodeError, ValueError, AttributeError):
        # If it's not valid JSON or any other error occurs, keep the original output
        pass
    
    return {
        "reasoning": think,
        "response": output
    }

def format_documents_with_metadata(documents):
    """
    Convert a list of Documents into a formatted string including metadata.

    Args:
        documents: List of Document objects

    Returns:
        String containing document content and metadata
    """
    formatted_docs = []
    for doc in documents:
        source = doc.metadata.get('source', 'Unknown source')
        formatted_doc = f"Source: {source}\nContent: {doc.page_content}"
        formatted_docs.append(formatted_doc)

    return "\n\n---\n\n".join(formatted_docs)

def get_configured_llm_model(default_model='deepseek-r1:latest'):
    """
    Get the configured LLM model name from environment variable or use the default.
    
    Args:
        default_model (str): Default model to use if not configured
        
    Returns:
        str: The model name to use
    """
    return os.environ.get('LLM_MODEL', default_model)

def invoke_ollama(model, system_prompt, user_prompt, output_format=None):
    # Use the configured model if none is specified
    if model is None:
        model = get_configured_llm_model()
        
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = chat(
        messages=messages,
        model=model,
        format=output_format.model_json_schema() if output_format else None
    )

    if output_format:
        return output_format.model_validate_json(response.message.content)
    else:
        return response.message.content
    
def invoke_llm(
    model,  # Specify the model name from OpenRouter
    system_prompt,
    user_prompt,
    output_format=None,
    temperature=0
):
        
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model, 
        temperature=temperature,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base= "https://openrouter.ai/api/v1",
    )
    
    # If Response format is provided use structured output
    if output_format:
        llm = llm.with_structured_output(output_format)
    
    # Invoke LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm.invoke(messages)
    
    if output_format:
        return response
    return response.content # str response

def tavily_search(query, include_raw_content=True, max_results=3):
    """ Search the web using the Tavily API.

    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return

    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""

    tavily_client = TavilyClient()
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content
    )

def get_report_structures(reports_folder="report_structures"):
    """
    Loads report structures from .md or .txt files in the specified folder.
    Each file should be named as 'report_name.md' or 'report_name.txt' and contain the report structure.
    Returns a dictionary of report structures.
    """
    report_structures = {}

    # Create the folder if it doesn't exist
    os.makedirs(reports_folder, exist_ok=True)

    try:
        # List all .md and .txt files in the folder
        for filename in os.listdir(reports_folder):
            if filename.endswith(('.md', '.txt')):
                report_name = os.path.splitext(filename)[0]  # Remove extension
                file_path = os.path.join(reports_folder, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        report_structures[report_name] = {
                            "content": content
                        }
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")

    except Exception as e:
        print(f"Error accessing reports folder: {str(e)}")

    return report_structures

def process_uploaded_files(uploaded_files):
    # Create files directory if it doesn't exist
    files_folder = "files"
    os.makedirs(files_folder, exist_ok=True)
    
    try:
        for uploaded_file in uploaded_files:
            # Save file to the files folder
            file_path = os.path.join(files_folder, uploaded_file.name)
            
            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
        # Process all files in the folder using the new embedding approach
        from src.assistant.rag_helpers import load_embed
        from src.assistant.vector_db import get_embedding_model, VECTOR_DB_PATH, DEFAULT_TENANT_ID
        
        # Get the embedding model
        embeddings = get_embedding_model()
        
        # Load and embed the documents
        load_embed(
            folder=files_folder,
            vdbdir=VECTOR_DB_PATH,
            embed_llm=embeddings,
            similarity="cosine",
            c_size=2000,
            c_overlap=400,
            normal=True,
            clean=True,
            tenant_id=DEFAULT_TENANT_ID
        )
        
        return True
    except Exception as e:
        print(f"Error processing files: {e}")
        return False

def clear_cuda_memory():
    """
    Clear CUDA memory cache to free up GPU resources between queries.
    Only has an effect if CUDA is available.
    """
    if torch.cuda.is_available():
        # Empty the cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("CUDA memory cache cleared")
    return