import os, re
from datetime import datetime
from langchain_core.runnables import chain
from langchain.schema import Document
# Use updated import path to avoid deprecation warning
try:
    from langchain_chroma import Chroma
except ImportError:
    # Fallback to original import if package is not installed
    from langchain_community.vectorstores import Chroma
# Use updated import path to avoid deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to original import if package is not installed
    from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any
import nltk
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

# Try to download nltk data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Define constants - must match the value in vector_db.py
VECTOR_DB_PATH = "database"


def get_tenant_collection_name(tenant_id):
    """Get the collection name for a tenant."""
    return f"collection_{tenant_id}"


def get_tenant_vectorstore(tenant_id, embed_llm, persist_directory, similarity, normal=True):
    """Get the vector store for a tenant."""
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    
    # Create directory if it doesn't exist
    os.makedirs(tenant_vdb_dir, exist_ok=True)
    
    # Get collection name for tenant
    collection_name = get_tenant_collection_name(tenant_id)
    
    return Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm,
        collection_metadata={"hnsw:space": similarity, "normalize_embeddings": normal}
    )


def similarity_search_for_tenant(tenant_id, embed_llm, persist_directory, similarity, normal, query, k=2, language="English"):
    """Perform similarity search for a tenant."""
    # Import clear_cuda_memory here to avoid circular imports
    from src.assistant.utils import clear_cuda_memory
    
    # Clear CUDA memory before search
    clear_cuda_memory()
    
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    
    # Check if directory exists
    if not os.path.exists(tenant_vdb_dir):
        raise Exception(f"Vector database directory for tenant {tenant_id} does not exist at {tenant_vdb_dir}")
    
    # Get collection name for tenant
    collection_name = get_tenant_collection_name(tenant_id)
    
    # Initialize vectorstore for search
    vectorstore = Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm,
        collection_metadata={"hnsw:space": similarity, "normalize_embeddings": normal}
    )
    
    try:
        # Print language being used for retrieval
        print(f"Using language for retrieval: {language}")
        
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=k)
        
        # Add language metadata to each document for downstream processing
        for doc in results:
            if "metadata" in doc.__dict__:
                doc.metadata["language"] = language
        
        # Clean up
        vectorstore._client = None
        del vectorstore
        
        # Clear CUDA memory after search
        clear_cuda_memory()
        
        return results
    except Exception as e:
        # Clean up in case of error
        if 'vectorstore' in locals():
            vectorstore._client = None
            del vectorstore
        
        # Clear CUDA memory in case of error
        clear_cuda_memory()
        
        # Re-raise the exception
        raise e


def transform_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    Transforms a list of Document objects into a specific dictionary format for the simplified workflow.
    
    Args:
        documents (list): List of Document objects with metadata and page_content
        
    Returns:
        list: List of dictionaries with content and metadata in the required format
    """
    transformed_docs = []
    
    for doc in documents:
        transformed_doc = {
            "content": doc.page_content,
            "metadata": {}
        }
        
        # Copy metadata if available
        if hasattr(doc, "metadata") and doc.metadata:
            for key, value in doc.metadata.items():
                transformed_doc["metadata"][key] = value
        
        transformed_docs.append(transformed_doc)
    
    return transformed_docs


def format_documents_as_plain_text(documents):
    """
    Format LangChain Documents into a plain text representation with ID, source, and content information.
    
    Args:
        documents (list): List of LangChain Document objects to format
        
    Returns:
        str: A formatted string with document information in plain text format
    """
    if not documents:
        return "No documents found."
    
    formatted_docs = []
    for i, doc in enumerate(documents):
        # Extract document ID, source, and content
        doc_id = doc.metadata.get('id', f'Unknown-ID-{i}')
        doc_source = doc.metadata.get('source', 'Unknown source')
        doc_content = doc.page_content
        
        # Format the document information
        formatted_doc = f"Document{i+1}:\nID is: {doc_id},\nSOURCE is: {doc_source},\nCONTENT is: {doc_content}\n"
        formatted_docs.append(formatted_doc)
    
    return "\n".join(formatted_docs)


def source_summarizer_ollama(query: str, context_documents: List, language: str, system_message: str, human_message: str, llm_model: str = "deepseek-r1:latest") -> Dict[str, str]:
    """
    Summarizes source documents using Ollama with the simplified workflow.
    
    Args:
        query (str): User query to guide the summarization
        context_documents (list): List of documents to summarize (can be dicts or Document objects)
        language (str): Language to use for the summary
        system_message (str): System prompt for the model
        llm_model (str): Ollama model name to use
        
    Returns:
        dict: Dictionary containing the summary content
    """
    # Import here to avoid circular imports
    from src.assistant.v1_1.utils_v1_1 import invoke_ollama, parse_output
    
    # Format the documents into a readable string
    formatted_docs = ""
    for i, doc in enumerate(context_documents):
        # Handle different document types (dict, Document object, or string)
        if isinstance(doc, dict):
            # If it's a dictionary, use get method
            content = doc.get("content", "No content available")
            metadata = doc.get("metadata", {})
        elif hasattr(doc, "page_content") and hasattr(doc, "metadata"):
            # If it's a Document object, access attributes directly
            content = doc.page_content
            metadata = doc.metadata
        elif isinstance(doc, str):
            # If it's a string, use it directly as content
            content = doc
            metadata = {}
        else:
            # For any other type, try to convert to string
            print(f"  [WARNING] Unknown document type: {type(doc)}, attempting to convert to string")
            try:
                content = str(doc)
                metadata = {}
            except:
                content = "Error: Could not process document"
                metadata = {}
        
        # Create document header with metadata
        source = metadata.get("source", "Unknown Source")
        path = metadata.get("path", "")
        
        # Handle path validation without character-by-character errors
        if path:
            try:
                # Check if path is a string
                if not isinstance(path, str):
                    print(f"  [WARNING] Path is not a string: {type(path)}")
                    path = ""
                # Check for special characters or non-printable characters
                elif any(ord(c) < 32 or ord(c) > 126 for c in path) or '\\x' in repr(path):
                    print(f"  [WARNING] Path contains special or non-printable characters, sanitizing")
                    # Sanitize by removing problematic characters
                    path = re.sub(r'[^\w\-\.\s\/:]+', '', path)
                # Validate general path format
                elif not re.match(r'^[\w\-\.\s\/:]+$', path.strip()):
                    print(f"  [WARNING] Invalid path format, sanitizing: {path[:20]}...")
                    path = re.sub(r'[^\w\-\.\s\/:]+', '', path)
                else:
                    # Valid path
                    if len(path) > 100:
                        print(f"  [DEBUG] Valid path extracted (truncated): {path[:50]}...")
                    else:
                        print(f"  [DEBUG] Valid path extracted: {path}")
            except Exception as e:
                print(f"  [WARNING] Error processing path: {type(e).__name__}")
                path = ""  # Set to empty to avoid further issues
        else:
            # No path in metadata
            path = ""
        
        # Format this document with its source
        formatted_docs += f"\n---\nDOCUMENT {i+1}: {source}\n---\n{content}\n\n"
    
    # Create the human prompt
    human_prompt = human_message
    
    # Call Ollama to generate the summary
    try:
        response = invoke_ollama(model=llm_model, system_prompt=system_message, user_prompt=human_prompt)
        
        # Parse the response to extract just the content if using the parse_output function
        parsed_output = parse_output(response).get("response", response)
        
        return {"content": parsed_output}
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        # Return a fallback summary in case of error
        return {"content": f"Error generating summary: {str(e)}. Please try again with a different model or query."}
