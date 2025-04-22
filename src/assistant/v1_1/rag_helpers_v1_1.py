import os, re
from datetime import datetime
from typing import List
from langchain_core.runnables import chain
from langchain_core.documents import Document
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
from src.assistant.v1_1.prompts_v1_1 import (
    # Document summarization prompts
    SUMMARIZER_HUMAN_PROMPT, SUMMARIZER_SYSTEM_PROMPT
)

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


def source_summarizer_ollama(query, context_documents, language, system_message, llm_model="deepseek-r1"):
    # Make sure language is explicitly passed through the entire pipeline
    print(f"Generating summary using language: {language}")
    # Override system_message to ensure language is set properly
    from src.assistant.v1_1.prompts_v1_1 import SUMMARIZER_SYSTEM_PROMPT
    system_message = SUMMARIZER_SYSTEM_PROMPT.format(language=language)
    # Check if context_documents is already a formatted string
    if isinstance(context_documents, str):
        formatted_context = context_documents
    else:
        # Handle the case where context_documents is a list of dictionary objects
        try:
            formatted_context = "\n".join(
                f"Content: {doc['content']}\nSource: {doc['metadata']['name']}\nPath: {doc['metadata']['path']}"
                for doc in context_documents
            )
        except (TypeError, KeyError):
            # Fallback: try to use the documents as they are
            formatted_context = str(context_documents)
    #formatted_context = "\n".join(
    #    f"{str(doc)}"
    #    for doc in context_documents
    #)
    prompt = SUMMARIZER_HUMAN_PROMPT.format(query=query, documents=formatted_context, language=language)
    
    # Initialize ChatOllama with the specified model and temperature
    llm = Ollama(model=llm_model, temperature=0.1, repeat_penalty=1.2) 
    # For RAG systems like your summarizer, consider:
    #    Using lower temperatures (0.1-0.3) for factual accuracy
    #   Combining with repeat_penalty=1.1-1.3 to avoid redundant content
    #   Monitoring token usage with num_ctx for long documents
    
    # Format messages for LangChain
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    # Get response from the model
    response = llm.invoke(messages)
    
    # Extract content from response
    response_content = response
    
    # Clean markdown formatting if present
    try:
        final_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    except:
        final_content = response_content.strip()

    # Extract metadata from all documents with added checks for structure
    document_names = []
    for doc in context_documents:
        if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
            # Try to get name from metadata, with fallbacks to source or id if name doesn't exist
            if 'name' in doc['metadata']:
                document_names.append(doc['metadata']['name'])
            elif 'source' in doc['metadata']:
                document_names.append(doc['metadata']['source'])
            elif 'id' in doc['metadata']:
                # Extract filename from id if it contains a path
                doc_id = doc['metadata']['id']
                if ':' in doc_id:
                    doc_id = doc_id.split(':', 1)[0]  # Get the part before the first colon
                document_names.append(doc_id)
            else:
                # Use a default name if no identifiers are available
                document_names.append(f"Document-{len(document_names)+1}")
    
    document_paths = []
    for doc in context_documents:
        if isinstance(doc, dict) and 'metadata' in doc and isinstance(doc['metadata'], dict):
            # Try to get path from metadata, with fallback to source if path doesn't exist
            if 'path' in doc['metadata']:
                document_paths.append(doc['metadata']['path'])
            elif 'source' in doc['metadata']:
                document_paths.append(doc['metadata']['source'])
            else:
                # Use a default path if no path information is available
                document_paths.append("Unknown path")

    return {
        "content": final_content,
        "metadata": {
            "name": document_names,
            "path": document_paths
        }
    }   
