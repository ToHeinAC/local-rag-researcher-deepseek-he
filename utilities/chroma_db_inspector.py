#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import argparse
import re

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Import necessary libraries
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Import specific functions from the project
from src.assistant.v1_1.rag_helpers_v1_1 import get_tenant_collection_name, similarity_search_for_tenant
from src.assistant.v1_1.vector_db_v1_1 import get_embedding_model_path

def clean_model_name(model_name):
    """Create a clean directory name from embedding model"""
    return model_name.replace('/', '--').replace('\\', '--')

def extract_embedding_model(db_dir_name):
    # Convert from format like 'sentence-transformers--all-mpnet-base-v2--2000--400'
    # or 'sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2'
    # to 'sentence-transformers/all-mpnet-base-v2' or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    parts = db_dir_name.split('--')
    if len(parts) >= 2:
        # The first two parts are the embedding model name
        model_name = parts[0].replace('--', '/') + '/' + parts[1]
        return model_name
    return None

def get_embedding_model(model_name):
    """Get the embedding model"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def inspect_chroma_db(db_path, tenant_id, preview_length=25):
    """Inspect a Chroma database and display document information using similarity_search_for_tenant"""
    print(f"\nInspecting Chroma DB at: {db_path}")
    print(f"Tenant ID: {tenant_id}\n")
    
    # Extract embedding model name from the path
    db_dir_name = os.path.basename(db_path)
    embedding_model_pattern = r'(.*?)--\d+--\d+'
    match = re.search(embedding_model_pattern, db_dir_name)
    
    if match:
        embedding_model = match.group(1).replace('--', '/')
        print(f"Embedding model: {embedding_model}\n")
    else:
        embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"Could not extract embedding model from path. Using default: {embedding_model}\n")
    
    # Get the embedding model
    embed_model = get_embedding_model(embedding_model)
    
    # First, list all collections in the database
    try:
        from chromadb import PersistentClient
        client = PersistentClient(path=db_path)
        collections = client.list_collections()
        print(f"Available collections in the database:\n")
        for i, collection_name in enumerate(collections):
            print(f"{i+1}. {collection_name}")
        print()
    except Exception as e:
        print(f"Error listing collections: {str(e)}\n")
        collections = []
    
    # Get the correct collection name for the tenant
    collection_name = get_tenant_collection_name(tenant_id)
    print(f"Collection name for tenant: {collection_name}\n")
    
    # Try to get all documents using a dummy query
    try:
        # Use a dummy query that should match everything
        dummy_query = "*"
        
        # Set a high k value to get all documents
        k_value = 100  # Adjust as needed
        
        # Use similarity_search_for_tenant to get documents
        print(f"Retrieving documents using similarity_search_for_tenant...\n")
        results = similarity_search_for_tenant(
            tenant_id=tenant_id,
            embed_llm=embed_model,
            persist_directory=db_path,
            similarity="cosine",  # Default similarity metric
            normal=True,  # Default normalization setting
            query=dummy_query,
            k=k_value,
            language="English"  # Default language
        )
        
        if not results:
            print("No documents found in the database.\n")
            return
        
        print(f"Found {len(results)} documents in the database.\n")
        print("-" * 80)
        print(f"{'Document ID':<40} | {'Content Preview':<{preview_length+5}}")
        print("-" * 80)
        
        for i, doc in enumerate(results):
            # Get document content and ID
            doc_content = doc.page_content
            doc_id = doc.metadata.get('source', f"Document {i+1}")
            
            # Get the first n characters of the document content
            content_preview = doc_content[:preview_length] + "..." if len(doc_content) > preview_length else doc_content
            print(f"{doc_id:<40} | {content_preview}")
        
        print("-" * 80)
        
        # Print metadata sample
        if results and hasattr(results[0], 'metadata') and results[0].metadata:
            print("\nMetadata sample (first document):")
            print(results[0].metadata)
    
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}\n")
        
        # Fallback to direct Chroma access if similarity_search_for_tenant fails
        print("Falling back to direct Chroma access...\n")
        try:
            # Try to access the collection directly
            db = Chroma(
                persist_directory=db_path,
                embedding_function=embed_model,
                collection_name=collection_name
            )
            
            # Get all documents
            all_docs = db.get()
            
            if not all_docs['documents']:
                print("No documents found in the database.\n")
                return
            
            print(f"Found {len(all_docs['documents'])} documents in the database.\n")
            print("-" * 80)
            print(f"{'Document ID':<40} | {'Content Preview':<{preview_length+5}}")
            print("-" * 80)
            
            for i, (doc_id, doc_content) in enumerate(zip(all_docs['ids'], all_docs['documents'])):
                # Get the first n characters of the document content
                content_preview = doc_content[:preview_length] + "..." if len(doc_content) > preview_length else doc_content
                print(f"{doc_id:<40} | {content_preview}")
            
            print("-" * 80)
            
            # Print metadata if available
            if all_docs['metadatas'] and all_docs['metadatas'][0]:
                print("\nMetadata sample (first document):")
                print(all_docs['metadatas'][0])
                
        except Exception as e2:
            print(f"Error in fallback method: {str(e2)}\n")

def list_tenant_dirs(db_path):
    """List all tenant directories in the database path"""
    # Get the absolute path to the database
    if not os.path.isabs(db_path):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_full_path = os.path.join(project_root, db_path)
    else:
        db_full_path = db_path
        
    # Check if the path exists
    if not os.path.exists(db_full_path):
        print(f"Database path does not exist: {db_full_path}")
        return []
        
    # List all directories in the database path
    tenant_dirs = []
    try:
        for item in os.listdir(db_full_path):
            item_path = os.path.join(db_full_path, item)
            if os.path.isdir(item_path):
                tenant_dirs.append(item)
    except Exception as e:
        print(f"Error listing tenant directories: {str(e)}")
        
    return tenant_dirs

def main():
    # Import the DEFAULT_TENANT_ID from app_v1_1.py
    from src.assistant.v1_1.app_v1_1 import DEFAULT_TENANT_ID
    
    parser = argparse.ArgumentParser(description='Inspect a Chroma database')
    parser.add_argument('--db_path', type=str, 
                        default="database/sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2--2000--400",
                        help='Path to the Chroma database')
    parser.add_argument('--tenant_id', type=str, default=DEFAULT_TENANT_ID,
                        help='Tenant ID')
    parser.add_argument('--preview_length', type=int, default=25,
                        help='Length of content preview')
    parser.add_argument('--list_tenants', action='store_true',
                        help='List all tenant directories in the database')
    
    args = parser.parse_args()
    
    # Get the absolute path to the database
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    db_full_path = os.path.join(project_root, args.db_path)
    
    # If --list_tenants flag is provided, list all tenant directories and exit
    if args.list_tenants:
        print(f"Listing tenant directories in {db_full_path}:")
        tenant_dirs = list_tenant_dirs(db_full_path)
        if tenant_dirs:
            for i, tenant_dir in enumerate(tenant_dirs):
                print(f"{i+1}. {tenant_dir}")
        else:
            print("No tenant directories found.")
        return
    
    # Inspect the database
    inspect_chroma_db(db_full_path, args.tenant_id, args.preview_length)

if __name__ == "__main__":
    import re  # Import re here for the regex pattern matching
    main()
