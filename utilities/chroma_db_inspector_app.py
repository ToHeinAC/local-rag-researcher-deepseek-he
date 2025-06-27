#!/usr/bin/env python3

import os
import sys
import re
import streamlit as st
from pathlib import Path

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Chroma DB Inspector",
    page_icon="🔍",
    layout="wide"
)

# Import necessary libraries
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient

# Import specific functions from the project
from src.assistant.v1_1.rag_helpers_v1_1 import get_tenant_collection_name, similarity_search_for_tenant
from src.assistant.v1_1.vector_db_v1_1 import get_embedding_model_path

# Define constants - avoid importing from app_v1_1 which has its own st.set_page_config()
DEFAULT_TENANT_ID = 'default_test'

# Title and description
st.title("Chroma Database Inspector 🔍")
st.markdown("This app allows you to inspect the contents of your Chroma vector database.")

# Define functions
def clean_model_name(model_name):
    """Create a clean directory name from embedding model"""
    return model_name.replace('/', '--').replace('\\', '--')

def get_embedding_model(model_name):
    """Get the embedding model"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def list_embedding_models(database_path):
    """List all embedding model directories in the database path"""
    embedding_models = []
    if os.path.exists(database_path):
        for item in os.listdir(database_path):
            item_path = os.path.join(database_path, item)
            if os.path.isdir(item_path) and '--' in item:
                embedding_models.append(item)
    return embedding_models

def list_tenant_dirs(db_path):
    """List all tenant directories in the database path that contain valid Chroma collections"""
    tenant_dirs = []
    if os.path.exists(db_path):
        for item in os.listdir(db_path):
            item_path = os.path.join(db_path, item)
            if os.path.isdir(item_path):
                # Check if this directory contains a Chroma collection
                # A valid tenant directory should contain either:
                # 1. A chroma.sqlite3 file (Chroma DB file)
                # 2. A 'chroma' subdirectory
                # 3. A collection directory with Chroma files
                
                # Check for chroma.sqlite3
                if os.path.exists(os.path.join(item_path, 'chroma.sqlite3')):
                    tenant_dirs.append(item)
                    continue
                
                # Check for 'chroma' subdirectory
                if os.path.exists(os.path.join(item_path, 'chroma')) and os.path.isdir(os.path.join(item_path, 'chroma')):
                    tenant_dirs.append(item)
                    continue
                
                # Check if there are any subdirectories that might be collection directories
                # (containing typical Chroma files like data_level0.bin, header.bin, etc.)
                has_collection = False
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    if os.path.isdir(subitem_path):
                        # Check for typical Chroma collection files
                        if (os.path.exists(os.path.join(subitem_path, 'data_level0.bin')) or
                            os.path.exists(os.path.join(subitem_path, 'header.bin')) or
                            os.path.exists(os.path.join(subitem_path, 'index_metadata.pickle'))):
                            has_collection = True
                            break
                
                if has_collection:
                    tenant_dirs.append(item)
    
    return tenant_dirs

def extract_embedding_model(db_dir_name):
    """Extract embedding model name from database directory name"""
    embedding_model_pattern = r'(.*?)--\d+--\d+'
    match = re.search(embedding_model_pattern, db_dir_name)
    
    if match:
        return match.group(1).replace('--', '/')
    return None

def get_collection_doc_count(client, collection_name):
    """Get the number of documents in a collection"""
    try:
        collection = client.get_collection(collection_name)
        return collection.count()
    except Exception as e:
        return 0

def get_collections_with_doc_counts(db_path):
    """Get all collections with their document counts"""
    collections_with_counts = []
    collection_name_prefix = "collection_"  # Common prefix in ChromaDB
    try:
        # Create a Chroma client and use the API
        client = PersistentClient(path=db_path)
        
        # Look for the chroma subdirectory, which is often where collections are stored
        chroma_dir = os.path.join(db_path, 'chroma')
        check_path = chroma_dir if os.path.exists(chroma_dir) else db_path
        st.write(f"Checking for collections in: {check_path}")
        
        # Track collections we've already processed to avoid duplicates
        processed_collections = set()
        
        # Use Chroma API to list collections - this is the most reliable method
        try:
            api_collections = client.list_collections()
            st.write(f"API found {len(api_collections)} collections: {api_collections}")
            
            # Process each collection from the API
            for collection_name in api_collections:
                # Skip if we've already processed this collection (normalized name)
                normalized_name = collection_name
                if normalized_name in processed_collections:
                    continue
                
                # Mark as processed
                processed_collections.add(normalized_name)
                
                try:
                    # Get the collection
                    collection = client.get_collection(collection_name)
                    doc_count = collection.count()
                    
                    # Add to our results
                    collections_with_counts.append({
                        "name": collection_name,
                        "count": doc_count
                    })
                    st.write(f"Collection '{collection_name}' has {doc_count} documents")
                except Exception as collection_error:
                    st.error(f"Error accessing collection '{collection_name}': {str(collection_error)}")
        except Exception as api_error:
            st.error(f"Error listing collections via API: {str(api_error)}")
            
            # Fallback: try to find collections from directory structure
            try:
                for item in os.listdir(check_path):
                    item_path = os.path.join(check_path, item)
                    if os.path.isdir(item_path) and not item.startswith('.'):
                        # Try with and without prefix
                        collection_names_to_try = [item]
                        if item.startswith(collection_name_prefix):
                            # Also try without prefix
                            collection_names_to_try.append(item[len(collection_name_prefix):])
                        else:
                            # Also try with prefix
                            collection_names_to_try.append(f"{collection_name_prefix}{item}")
                        
                        # Try each possible name
                        for name_to_try in collection_names_to_try:
                            # Skip if we've already processed this collection
                            if name_to_try in processed_collections:
                                continue
                                
                            try:
                                collection = client.get_collection(name_to_try)
                                # If we get here, the collection exists
                                doc_count = collection.count()
                                
                                # Mark as processed
                                processed_collections.add(name_to_try)
                                
                                # Add to our results
                                collections_with_counts.append({
                                    "name": name_to_try,
                                    "count": doc_count
                                })
                                st.write(f"Collection '{name_to_try}' has {doc_count} documents")
                                
                                # We found a working name, no need to try others
                                break
                            except:
                                # This name didn't work, try the next one
                                pass
            except Exception as dir_error:
                st.error(f"Error listing directories: {str(dir_error)}")
        
        # Sort collections by count (descending) then by name
        collections_with_counts = sorted(collections_with_counts, key=lambda x: (-x["count"], x["name"]))
        
        return collections_with_counts
    except Exception as e:
        st.error(f"Error getting collections with counts: {str(e)}")
        return []

def inspect_chroma_db(db_path, tenant_id, collection_name=None, preview_length=50, k_value=100):
    """Inspect a Chroma database and display document information using similarity_search_for_tenant"""
    results = {}
    
    # Check if we're at the root database level or already in a specific path
    # If db_path contains 'chroma', we're likely already at the correct level
    is_chroma_dir = 'chroma' in db_path.split(os.path.sep)
    
    # Get path components to determine where we are
    path_components = db_path.split(os.path.sep)
    st.write(f"Database path components: {path_components}")
    
    # Extract embedding model name from the path if not in chroma dir
    if not is_chroma_dir:
        db_dir_name = os.path.basename(db_path)
        embedding_model = extract_embedding_model(db_dir_name)
        
        if not embedding_model:
            embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            st.info(f"Could not extract embedding model from path. Using default: {embedding_model}")
        else:
            st.info(f"Embedding model: **{embedding_model}**")
    else:
        # If we're already in a chroma directory, use default embedding model
        embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        st.info(f"Using default embedding model as we're in a chroma directory: {embedding_model}")
    
    # Get the embedding model
    embed_model = get_embedding_model(embedding_model)
    
    # If no collection name is provided, use the tenant collection name
    if not collection_name:
        collection_name = get_tenant_collection_name(tenant_id)
    
    st.write(f"Using collection: **{collection_name}**")
    
    # Try to get all documents using a dummy query
    try:
        # Use a dummy query that should match everything
        dummy_query = "*"
        
        # If we're using a custom collection (not the tenant collection)
        if collection_name != get_tenant_collection_name(tenant_id):
            # Use direct Chroma access for custom collections
            with st.spinner("Accessing Chroma database directly..."):
                db = Chroma(
                    persist_directory=db_path,
                    embedding_function=embed_model,
                    collection_name=collection_name
                )
                
                # Get all documents
                all_docs = db.get()
                
                if not all_docs['documents']:
                    st.warning("No documents found in the collection.")
                    return None
                
                st.success(f"Found {len(all_docs['documents'])} documents in the collection.")
                
                # Convert to a format similar to similarity_search_for_tenant results
                from langchain_core.documents import Document
                results = []
                for i, (doc_id, doc_content, metadata) in enumerate(zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas'])):
                    doc = Document(
                        page_content=doc_content,
                        metadata={**metadata, 'id': doc_id}
                    )
                    results.append(doc)
                
                return results
        else:
            # Use similarity_search_for_tenant for tenant collections
            with st.spinner("Retrieving documents using similarity_search_for_tenant..."):
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
                st.warning("No documents found in the database.")
                return None
            
            st.success(f"Found {len(results)} documents in the database.")
            return results
    
    except Exception as e:
        st.error(f"Error retrieving documents: {str(e)}")
        
        # Fallback to direct Chroma access if similarity_search_for_tenant fails
        st.info("Falling back to direct Chroma access...")
        try:
            # Try to access the collection directly
            with st.spinner("Accessing Chroma database directly..."):
                db = Chroma(
                    persist_directory=db_path,
                    embedding_function=embed_model,
                    collection_name=collection_name
                )
                
                # Get all documents
                all_docs = db.get()
                
                if not all_docs['documents']:
                    st.warning("No documents found in the database.")
                    return None
                
                st.success(f"Found {len(all_docs['documents'])} documents in the database.")
                
                # Convert to a format similar to similarity_search_for_tenant results
                from langchain_core.documents import Document
                results = []
                for i, (doc_id, doc_content, metadata) in enumerate(zip(all_docs['ids'], all_docs['documents'], all_docs['metadatas'])):
                    doc = Document(
                        page_content=doc_content,
                        metadata={**metadata, 'id': doc_id}
                    )
                    results.append(doc)
                
                return results
                
        except Exception as e2:
            st.error(f"Error in fallback method: {str(e2)}")
            return None

# Main app code
# Sidebar for configuration
st.sidebar.header("Configuration")

# Get project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
default_db_path = os.path.join(project_root, "database")

# Database path selection
db_path = st.sidebar.text_input("Database Path", value=default_db_path)

# List embedding models (databases)
embedding_models = list_embedding_models(db_path)
selected_model = None
full_db_path = db_path

if embedding_models:
    selected_model = st.sidebar.selectbox(
        "Select Database", 
        embedding_models,
        index=0
    )
    
    # Construct the full path to the database
    # For Chroma DB, the path structure is typically database/embedding_model/tenant_id/chroma
    full_db_path = os.path.join(db_path, selected_model)
    
    # Check if there's a 'chroma' directory in the embedding model directory
    # This might be the actual db path in some setups
    chroma_path = os.path.join(full_db_path, 'chroma')
    if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
        st.info(f"Found direct 'chroma' directory at embedding model level: {chroma_path}")
else:
    st.sidebar.warning("No databases found in the specified path.")

# List tenant directories only if a database is selected
selected_tenant = None
tenant_dirs = []

if selected_model:
    # Try to find tenant directories
    tenant_dirs = list_tenant_dirs(full_db_path)
    
    # Check if we need to look directly at the 'chroma' directory
    if not tenant_dirs:
        chroma_path = os.path.join(full_db_path, 'chroma')
        if os.path.exists(chroma_path) and os.path.isdir(chroma_path):
            st.info(f"Looking for collections directly in the 'chroma' directory: {chroma_path}")
            # In this case, we don't have separate tenant directories,
            # so we'll create a virtual 'current' tenant for the UI
            tenant_dirs = ['current']
            # Update the full path to point to the chroma directory
            full_db_path = chroma_path
    
    if tenant_dirs:
        # Sort tenant directories to put default and default_test first
        sorted_tenants = sorted(tenant_dirs, key=lambda x: (0 if x == DEFAULT_TENANT_ID else (1 if x == "default" else 2), x))
        
        selected_tenant = st.sidebar.selectbox(
            "Select Tenant ID", 
            sorted_tenants,
            index=0
        )
    else:
        st.sidebar.warning("No tenant directories found in the selected database.")
        selected_tenant = DEFAULT_TENANT_ID

# Initialize session state variables for tracking UI state if they don't exist
if 'preview_clicked' not in st.session_state:
    st.session_state.preview_clicked = False
if 'collections_with_counts' not in st.session_state:
    st.session_state.collections_with_counts = []

# Preview button - only enabled when database and tenant are selected
preview_button_disabled = not (selected_model and selected_tenant)

if preview_button_disabled:
    st.sidebar.warning("Please select a database and tenant ID to preview available collections.")

# Function to handle preview button click
def on_preview_click():
    # Reset the preview state
    st.session_state.preview_clicked = True
    
    # Store current database and tenant info to help with tracking changes
    st.session_state.current_db_path = full_db_path
    st.session_state.current_tenant = selected_tenant
    
    # Determine actual path to search for collections
    search_path = full_db_path
    
    # If we selected a tenant and it's not a 'virtual' tenant
    if selected_tenant != 'current':
        # Check if we need to append the tenant ID to the path
        tenant_path = os.path.join(full_db_path, selected_tenant)
        if os.path.exists(tenant_path) and os.path.isdir(tenant_path):
            search_path = tenant_path
            st.info(f"Searching for collections in tenant directory: {search_path}")
    
    # Get collections with document counts for the selected path
    with st.spinner("Loading collections... This may take a moment for large databases"):
        st.session_state.collections_with_counts = get_collections_with_doc_counts(search_path)
    
    # Reset any previously selected collection
    if 'selected_collection' in st.session_state:
        del st.session_state.selected_collection
    
    # Also reset collection display
    if 'collection_display' in st.session_state:
        del st.session_state.collection_display

# Preview Collections button
st.sidebar.button("Preview Collections", disabled=preview_button_disabled, on_click=on_preview_click)

# Initialize session state for collection if it doesn't exist
if 'selected_collection' not in st.session_state:
    st.session_state.selected_collection = None

# Only show collection selection if preview was clicked and we have both database and tenant
default_collection_name = get_tenant_collection_name(selected_tenant) if selected_tenant else None

# Check if database or tenant has changed since last preview
db_tenant_changed = False
if hasattr(st.session_state, 'current_db_path') and hasattr(st.session_state, 'current_tenant'):
    if st.session_state.current_db_path != full_db_path or st.session_state.current_tenant != selected_tenant:
        db_tenant_changed = True
        # If changed, we need to click preview again
        st.warning("Database or tenant changed. Please click 'Preview Collections' again to update.")
        st.session_state.preview_clicked = False

if st.session_state.preview_clicked and selected_model and selected_tenant:
    # Get collections with documents
    doc_collections = st.session_state.collections_with_counts
    
    # Check if we actually have collections loaded
    if not doc_collections:
        st.warning("No collections found or loaded. Try clicking 'Preview Collections' again.")
    
    # Only show collections during the selection phase but not after a collection is selected or being inspected
    if (not 'selected_collection' in st.session_state or not st.session_state.selected_collection) and not st.session_state.inspection_active:
        # Show all available collections in the main panel
        with st.expander("Available collections", expanded=True):
            st.write(f"Found {len(doc_collections)} collections in the database.")
            for idx, coll_info in enumerate(doc_collections):
                st.write(f"{idx+1}. {coll_info['name']} ({coll_info['count']} docs)")
    
    # Format collections for selection - include document counts
    collection_options = []
    default_index = 0
    collection_name_prefix = "collection_"
    
    for i, collection_info in enumerate(doc_collections):
        name = collection_info["name"]
        count = collection_info["count"]
        display_name = f"{name} ({count} docs)"
        collection_options.append({"display": display_name, "name": name})
        
        # Check if this might be the default tenant collection
        if name.endswith(selected_tenant) or (
            name.startswith(collection_name_prefix) and 
            name[len(collection_name_prefix):].endswith(selected_tenant)
        ):
            default_index = i
    
    # Add collection selection if collections exist
    if collection_options:
        st.sidebar.subheader("Available Collections")
        
        # Use multiselect instead of selectbox to allow selecting multiple collections
        # But for now, keep it limited to one selection until we implement multi-collection viewing
        if 'collection_display' not in st.session_state:
            st.session_state.collection_display = collection_options[default_index if default_index < len(collection_options) else 0]["display"]
        
        # Function to update selected collection when dropdown changes
        def on_collection_change():
            # Currently only supporting a single selection
            # Find the selected collection name from the display name
            if st.session_state.collection_display:
                for collection_info in collection_options:
                    if collection_info["display"] == st.session_state.collection_display:
                        st.session_state.selected_collection = collection_info["name"]
                        break
            else:
                st.session_state.selected_collection = None
        
        # Display the collection selection dropdown - using selectbox for now
        st.sidebar.selectbox(
            "Select Collection", 
            [c["display"] for c in collection_options],
            index=[c["display"] for c in collection_options].index(st.session_state.collection_display) 
                  if st.session_state.collection_display in [c["display"] for c in collection_options] 
                  else (default_index if default_index < len(collection_options) else 0),
            key="collection_display",
            on_change=on_collection_change
        )
        
        # Make sure selected_collection is set based on the current selection
        on_collection_change()
        
        # Get the selected collection for use in the rest of the app
        selected_collection = st.session_state.selected_collection
        
        # Calculate total documents across all collections and in the selected collection
        total_docs = sum(c["count"] for c in doc_collections)
        selected_docs = next((c["count"] for c in doc_collections if c["name"] == selected_collection), 0)
        
        # Show document counts
        st.sidebar.info(f"Total documents across all collections: {total_docs}")
        st.sidebar.success(f"Selected collection has {selected_docs} documents")
        
        # Update the total documents display in the main UI
        st.write(f"Total documents across all collections: {total_docs}")
        st.write(f"Selected collection '{selected_collection}' has {selected_docs} documents")
        
        # Only show these settings after collection is selected
        st.sidebar.subheader("Document Settings")
        
        # Number of documents to retrieve
        k_value = st.sidebar.slider("Max Documents to Retrieve", min_value=10, max_value=500, value=100)
        
        # Document preview length
        preview_length = st.sidebar.slider("Content Preview Length", min_value=25, max_value=200, value=50)
    else:
        st.sidebar.warning("No collections found in the database.")
        st.session_state.selected_collection = default_collection_name
        selected_collection = default_collection_name
else:
    # If preview hasn't been clicked, set selected_collection to None for the rest of the app
    selected_collection = None

# Search functionality
st.sidebar.header("Search")
search_query = st.sidebar.text_input("Search Query (Optional)")

# Define default values for k_value and preview_length in case they're not set in the UI flow
if 'k_value' not in locals():
    k_value = 100  # Default value
    
if 'preview_length' not in locals():
    preview_length = 50  # Default value

# Button to inspect database
inspect_button_disabled = not (selected_model and selected_tenant and selected_collection)

if inspect_button_disabled:
    st.sidebar.warning("Please select a database, tenant ID, and collection to inspect.")

# Store the inspection state in session state
if 'inspection_active' not in st.session_state:
    st.session_state.inspection_active = False

if st.sidebar.button("Inspect Database", disabled=inspect_button_disabled):
    # Set inspection active to hide the collections list
    st.session_state.inspection_active = True
    
    if os.path.exists(full_db_path):
        st.write(f"## Inspecting Database")
        st.write(f"**Path:** {full_db_path}")
        st.write(f"**Tenant ID:** {selected_tenant}")
        st.write(f"**Collection:** {selected_collection}")
        st.write(f"**Max Documents:** {k_value}")
        st.write(f"**Preview Length:** {preview_length}")
        
        # Inspect the database with the selected collection
        results = inspect_chroma_db(full_db_path, selected_tenant, selected_collection, preview_length, k_value)
        
        if results:
            # Display documents in a table
            st.write("## Documents")
            
            # Filter documents if search query is provided
            if search_query:
                filtered_results = [doc for doc in results if search_query.lower() in doc.page_content.lower()]
                st.write(f"Found {len(filtered_results)} documents matching search query: '{search_query}'")
                display_results = filtered_results
            else:
                display_results = results
            
            # Create a dataframe for display
            import pandas as pd
            
            data = []
            for i, doc in enumerate(display_results):
                doc_id = doc.metadata.get('source', doc.metadata.get('id', f"Document {i+1}"))
                content_preview = doc.page_content[:preview_length] + "..." if len(doc.page_content) > preview_length else doc.page_content
                data.append({
                    "#": i+1,
                    "Document ID": doc_id,
                    "Content Preview": content_preview
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Document viewer
                st.write("## Document Viewer")
                
                # Store the current document index in session state
                if 'doc_index' not in st.session_state:
                    st.session_state.doc_index = 0
                
                # Function to update the document index when the number input changes
                def update_doc_index():
                    # The number_input is 1-indexed, but we need 0-indexed
                    st.session_state.doc_index = st.session_state.doc_number - 1
                
                # Use a number input with key and on_change to ensure updates
                doc_number = st.number_input(
                    "Select document number to view", 
                    min_value=1, 
                    max_value=len(display_results), 
                    value=st.session_state.doc_index + 1,  # Convert back to 1-indexed
                    key="doc_number",
                    on_change=update_doc_index
                )
                
                # Get the current document index from session state
                doc_index = st.session_state.doc_index
                
                if 0 <= doc_index < len(display_results):
                    selected_doc = display_results[doc_index]
                    
                    # Display document details in two columns
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("### Document Content")
                        st.text_area(
                            "Content", 
                            selected_doc.page_content, 
                            height=300,
                            key=f"content_{doc_index}"  # Use a unique key including the index
                        )
                    
                    with col2:
                        st.write("### Metadata")
                        for key, value in selected_doc.metadata.items():
                            st.write(f"**{key}:** {value}")
            else:
                st.warning("No documents match the search query.")
    else:
        st.error(f"Database path does not exist: {full_db_path}")

# Add a footer
st.markdown("---")
st.markdown("*Chroma Database Inspector for RAG Researcher*")
