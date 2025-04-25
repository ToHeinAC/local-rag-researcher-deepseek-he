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
    
class SummaryRanking(BaseModel):
    summary_index: int
    relevance_score: float
    justification: str
    
class SummaryRankings(BaseModel):
    rankings: list[SummaryRanking]

class SummaryRelevance(BaseModel):
    is_relevant: bool
    confidence: float
    justification: str = "No justification provided"

class QualityCheckResult(BaseModel):
    quality_score: float
    is_accurate: bool
    is_complete: bool
    issues_found: list[str]
    missing_elements: list[str]
    citation_issues: list[str] = []
    improvement_needed: bool
    improvement_suggestions: str

class DetectedLanguage(BaseModel):
    language: str

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

def format_documents_with_metadata(documents, preserve_original=False):
    formatted_docs = []
    for doc in documents:
        # Get the source filename from metadata
        source = doc.metadata.get('source', 'Unknown source')
        
        # Ensure we have an absolute path to the document
        doc_path = ''
        if 'path' in doc.metadata and os.path.isfile(doc.metadata['path']):
            doc_path = doc.metadata['path']
        elif 'source' in doc.metadata:
            # Try to construct an absolute path to the file in the files directory
            potential_path = os.path.abspath(os.path.join(os.getcwd(), 'files', source))
            if os.path.isfile(potential_path):
                doc_path = potential_path
            else:
                # If file doesn't exist in the current directory structure, still use the path format
                # This ensures consistent citation format even for documents that might be processed later
                doc_path = os.path.abspath(os.path.join(os.getcwd(), 'files', source))
        
        # Extract just the filename for display
        filename = os.path.basename(source) if source != 'Unknown source' else 'Unknown source'
        
        # Format with markdown link using the required format: [local_document_filename](local_document_full_path)
        # Ensure the path includes the /files folder as specified in the updated requirements
        if doc_path:
            # Make sure the path contains the /files directory for consistency
            if '/files/' not in doc_path and '\\files\\' not in doc_path:
                files_dir = os.path.join(os.getcwd(), 'files')
                doc_path = os.path.join(files_dir, filename)
            source_link = f"[{filename}]({doc_path})"
        else:
            # If no path is available, still create a standard format with a placeholder path
            files_dir = os.path.join(os.getcwd(), 'files')
            doc_path = os.path.join(files_dir, filename)
            source_link = f"[{filename}]({doc_path})"
        
        # When preserve_original is True, include the full original content without any modifications
        if preserve_original:
            formatted_doc = f"SOURCE: {source_link}\n\nContent: {doc.page_content}"
        else:
            formatted_doc = f"SOURCE: {source_link}\n\nContent: {doc.page_content}"
            
        formatted_docs.append(formatted_doc)
    return "\n\n---\n\n".join(formatted_docs)

def get_configured_llm_model(default_model='deepseek-r1:latest'):
    """
    Get the configured general purpose LLM model name from environment variable or use the default.
    
    Args:
        default_model (str): Default model to use if not configured
        
    Returns:
        str: The model name to use
    """
    return os.environ.get('LLM_MODEL', default_model)

def get_configured_report_llm_model(default_model='deepseek-r1:latest'):
    """
    Get the configured report writing LLM model name from environment variable or use the default.
    
    Args:
        default_model (str): Default model to use if not configured
        
    Returns:
        str: The model name to use for report writing
    """
    return os.environ.get('REPORT_LLM', default_model)

def get_configured_summarization_llm_model(default_model='llama3.2'):
    """
    Get the configured summarization LLM model name from environment variable or use the default.
    
    Args:
        default_model (str): Default model to use if not configured
        
    Returns:
        str: The model name to use for summarization
    """
    return os.environ.get('SUMMARIZATION_LLM', default_model)

def invoke_ollama(model, system_prompt, user_prompt, output_format=None):
    # Use the configured model if none is specified
    if model is None:
        model = get_configured_llm_model()
    
    # Print the actual model being used for debugging
    print(f"  [DEBUG] Actually using model in invoke_ollama: {model}")
        
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
        from src.assistant.vector_db import get_embedding_model, get_vector_db_path, DEFAULT_TENANT_ID
        
        # Get the embedding model
        embeddings = get_embedding_model()
        
        # Get the vector DB path with embedding model name
        vector_db_path = get_vector_db_path()
        
        # Load and embed the documents
        load_embed(
            folder=files_folder,
            vdbdir=vector_db_path,
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