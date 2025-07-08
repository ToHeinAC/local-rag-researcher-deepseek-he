import os
import sys
import time
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path
import re
from datetime import datetime
import torch

# Set PyTorch CUDA memory allocation configuration to mitigate fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from src.assistant.prompts import SUMMARIZER_SYSTEM_PROMPT

###Run command: streamlit run ./utilities/vdb_emb-retr_handler-app.py --server.port 8501  --server.headless False

# Set page configuration
st.set_page_config(
    page_title="Vector Database Handler",
    page_icon="🧠",
    layout="wide"
)

# CUDA memory clearing will be done via GUI button

# Set up session state variables
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "gpt-4o"
    
# Add CUDA memory status to session state
if 'cuda_memory_cleared' not in st.session_state:
    st.session_state.cuda_memory_cleared = False
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "jinaai/jina-embeddings-v2-base-en"
if 'selected_database' not in st.session_state:
    st.session_state.selected_database = ""
if 'tenant_id' not in st.session_state:
    st.session_state.tenant_id = "default"
if 'vdb_dir' not in st.session_state:
    st.session_state.vdb_dir = ""
if 'gpu_device' not in st.session_state:
    st.session_state.gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
if 'use_gpu' not in st.session_state:
    st.session_state.use_gpu = torch.cuda.is_available()
if 'use_fp16' not in st.session_state:
    st.session_state.use_fp16 = st.session_state.use_gpu
if 'force_cpu_large_docs' not in st.session_state:
    st.session_state.force_cpu_large_docs = False
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 2000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 400

# Title and description
st.title("Vector Database Handler App")
st.markdown("Use this app to manage your vector databases and search for similar documents.")

# (Memory clearing button moved to sidebar)

# Function to get current GPU memory info
def get_gpu_memory_info():
    if torch.cuda.is_available():
        free_memory = torch.cuda.mem_get_info()[0] / (1024**3)  # Convert to GB
        total_memory = torch.cuda.mem_get_info()[1] / (1024**3)  # Convert to GB
        used_memory = total_memory - free_memory  # Calculate used memory
        percent_used = (used_memory/total_memory)*100  # Calculate percentage used
        return used_memory, total_memory, percent_used
    return 0, 0, 0

# Store GPU memory info in session state if not already there
if 'gpu_memory_info' not in st.session_state:
    used_memory, total_memory, percent_used = get_gpu_memory_info()
    st.session_state.gpu_memory_info = {
        'used': used_memory,
        'total': total_memory,
        'percent': percent_used
    }

# Display GPU status
if torch.cuda.is_available():
    st.sidebar.success(f"🚀 GPU Acceleration: Enabled ({torch.cuda.get_device_name(0)})")
    
    # CUDA memory clearing button - prominent placement in sidebar with warning
    st.sidebar.warning("⚠️ **IMPORTANT: Clear CUDA memory before processing large documents to avoid out-of-memory errors!**")
    if st.sidebar.button("🧹 CLEAR CUDA MEMORY", type="primary", use_container_width=True):
        torch.cuda.empty_cache()
        st.session_state.cuda_memory_cleared = True
        
        # Update GPU memory info in session state after clearing
        used_memory, total_memory, percent_used = get_gpu_memory_info()
        st.session_state.gpu_memory_info = {
            'used': used_memory,
            'total': total_memory,
            'percent': percent_used
        }
        
        st.sidebar.success("✅ CUDA memory cleared successfully!")
    
    # Display memory stats using the session state values
    used_memory = st.session_state.gpu_memory_info['used']
    total_memory = st.session_state.gpu_memory_info['total']
    percent_used = st.session_state.gpu_memory_info['percent']
    st.sidebar.info(f"GPU memory: {used_memory:.2f}GB / {total_memory:.2f}GB used ({percent_used:.1f}%)")
    
    # GPU settings
    st.sidebar.subheader("GPU Settings")
    use_gpu = st.sidebar.checkbox("Use GPU for embeddings", value=st.session_state.use_gpu)
    if use_gpu != st.session_state.use_gpu:
        st.session_state.use_gpu = use_gpu
        st.session_state.gpu_device = "cuda" if use_gpu else "cpu"
        st.experimental_rerun()
    
    if st.session_state.use_gpu:
        use_fp16 = st.sidebar.checkbox("Use FP16 precision (faster)", value=st.session_state.use_fp16)
else:
    st.sidebar.warning("⚠️ GPU Acceleration: Not available (using CPU)")
    st.session_state.gpu_device = "cpu"
    st.session_state.use_gpu = False
    st.session_state.use_fp16 = False



# Define paths
DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "insert_data")
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_INSERTED_PATH = os.path.join(os.path.dirname(__file__), "db_inserted")

# Ensure the db_inserted directory exists
os.makedirs(DB_INSERTED_PATH, exist_ok=True)

# Function to create a clean directory name from embedding model
def clean_model_name(model_name):
    return model_name.replace('/', '--').replace('\\', '--')

# Function to get embedding model
def get_embedding_model(model_name):
    # Check if we already have this model in session state
    model_key = f"embedding_model_instance_{model_name}_{st.session_state.gpu_device}"
    if model_key in st.session_state:
        st.info(f"Using cached embedding model: {model_name}")
        return st.session_state[model_key]
    
    # Determine device to use (GPU or CPU)
    device = st.session_state.gpu_device
    
    # Log device being used
    if device == "cpu":
        st.info("Using CPU for embeddings (GPU not available)")
    else:
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "No GPU detected"
        st.success(f"Using {device} for embeddings ({gpu_info})")
    
    # Default to a known working model if the specified model has issues
    fallback_model = "jinaai/jina-embeddings-v2-base-en"
    
    try:
        # Set up model kwargs based on device and precision settings
        model_kwargs = {
            'device': device
        }
        
        # Create the HuggingFaceEmbeddings instance with the updated API
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Handle float16 precision separately after model creation if needed
        if st.session_state.use_fp16 and device != "cpu" and hasattr(embeddings, 'client'):
            embeddings.client = embeddings.client.half()
        
        # Cache the model in session state
        st.session_state[model_key] = embeddings
        return embeddings
    except Exception as e:
        st.warning(f"Error loading model {model_name}: {str(e)}\nFalling back to {fallback_model}")
        
        try:
            # Fallback approach if the primary method fails - use a known working model
            import sentence_transformers
            
            # Create the model directly without empty_init parameter
            model = sentence_transformers.SentenceTransformer(
                fallback_model,
                device=device
            )
            
            # Create a custom embedding function that uses the model
            class SentenceTransformerEmbeddings(HuggingFaceEmbeddings):
                def __init__(self, model):
                    self.client = model
                    
                def embed_documents(self, texts):
                    return self.client.encode(texts, convert_to_numpy=True).tolist()
                    
                def embed_query(self, text):
                    return self.client.encode(text, convert_to_numpy=True).tolist()
            
            # Create and cache the fallback embedding model
            fallback_embeddings = SentenceTransformerEmbeddings(model)
            st.session_state[f"embedding_model_instance_{fallback_model}_{device}"] = fallback_embeddings
            return fallback_embeddings
        except Exception as e:
            st.error(f"Failed to load fallback model: {str(e)}")
            raise

# Function to get file save date from filesystem
def get_file_save_date(file_path):
    """Get file save date from filesystem in YYMMDD format
    
    Args:
        file_path (str): The file path
        
    Returns:
        str: The file save date in YYMMDD format
    """
    try:
        # Get file creation time
        ctime = os.path.getctime(file_path)
        # Convert to datetime
        dt = datetime.fromtimestamp(ctime)
        # Format as YYMMDD
        return dt.strftime('%y%m%d')
    except Exception as e:
        print(f"Error getting file save date: {str(e)}")
        # Return current date as fallback
        return datetime.now().strftime('%y%m%d')

# Function to get all documents in a vector database
def get_all_documents_in_vectordb(tenant_id, embed_llm, persist_directory):
    """Retrieve all document filenames from a vector database for a specific tenant
    
    Args:
        tenant_id (str): The tenant ID
        embed_llm: The embedding model
        persist_directory (str): The directory where the vector database is stored
        
    Returns:
        list: A list of unique document filenames in the vector database
    """
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    
    # Check if directory exists
    if not os.path.exists(tenant_vdb_dir):
        return []
    
    # Get collection name for tenant
    collection_name = get_tenant_collection_name(tenant_id)
    
    # Initialize vectorstore
    vectorstore = Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm
    )
    
    try:
        # Get all documents
        result = vectorstore.get()
        
        # Extract unique source filenames from metadata
        documents = []
        if 'metadatas' in result and result['metadatas']:
            for metadata in result['metadatas']:
                if metadata and 'source' in metadata:
                    documents.append(metadata['source'])
        
        # Clean up
        vectorstore._client = None
        del vectorstore
        
        # Return unique document filenames
        return sorted(list(set(documents)))
    except Exception as e:
        # Clean up in case of error
        if 'vectorstore' in locals():
            vectorstore._client = None
            del vectorstore
        
        # Re-raise the exception
        raise e

# Function to extract embedding model name from database directory
def extract_embedding_from_db_name(db_name):
    """Extract the embedding model name from a database directory name
    
    Args:
        db_name (str): The database directory name
        
    Returns:
        str: The embedding model name or None if not found
    """
    parts = db_name.split('--')
    if len(parts) >= 2:
        # Reconstruct the model name by replacing -- with /
        model_parts = parts[:-2] if len(parts) > 2 else [parts[0]]
        model_name = '/'.join([p.replace('--', '/') for p in model_parts])
        return model_name
    return None

# Function to list available databases
def list_available_databases():
    """List all available databases in the database folder
    
    Returns:
        list: A list of database directory names
    """
    if not os.path.exists(DATABASE_PATH):
        return []
    
    return [d for d in os.listdir(DATABASE_PATH) if os.path.isdir(os.path.join(DATABASE_PATH, d))]

# Function to generate metadata from filename
def generate_metadata_from_file(filename):
    """Generate metadata from filename including current date in YYMMDD format
    
    Args:
        filename (str): The filename
        
    Returns:
        str: The metadata string (filename--YYMMDD)
    """
    # Get current date in YYMMDD format
    current_date = datetime.now().strftime('%y%m%d')
    
    # Create metadata string
    return f"{filename}--{current_date}"

# Function to check if a file is already in the database
def is_file_in_database(filename, date_str, tenant_id, embed_llm, persist_directory):
    """
    Check if a file with the same metadata is already in the database
    
    Args:
        filename (str): The filename
        date_str (str): The file save date string in YYMMDD format
        tenant_id (str): The tenant ID
        embed_llm: The embedding model
        persist_directory (str): The directory where the vector database is stored
        
    Returns:
        bool: True if the file is already in the database, False otherwise
    """
    try:
        # Extract original filename if it contains a date
        original_filename = filename
        if "--" in filename:
            parts = filename.split("--")
            if len(parts) >= 2:
                original_filename = parts[0]
        
        # Get all documents in the database
        all_docs = get_all_documents_in_vectordb(tenant_id, embed_llm, persist_directory)
        
        # Generate metadata string to check against source field
        metadata = f"{original_filename}--{date_str}"
        
        # Check if the file is already in the database by source field
        if metadata in all_docs:
            return True
            
        # Also check the vectorstore metadata for original_filename and file_save_date
        try:
            tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
            collection_name = get_tenant_collection_name(tenant_id)
            vectorstore = Chroma(
                persist_directory=tenant_vdb_dir,
                collection_name=collection_name,
                embedding_function=embed_llm
            )
            
            # Get all metadata
            result = vectorstore.get()
            if 'metadatas' in result and result['metadatas']:
                for metadata in result['metadatas']:
                    if metadata and 'original_filename' in metadata and 'file_save_date' in metadata:
                        # Check if both original filename and file save date match
                        if metadata['original_filename'] == original_filename and metadata['file_save_date'] == date_str:
                            return True
            
            # Clean up
            vectorstore._client = None
            del vectorstore
        except Exception as e:
            print(f"Error checking vectorstore metadata: {str(e)}")
        
        return False
    except Exception as e:
        print(f"Error checking if file is in database: {str(e)}")
        return False

# Function to check if a file is already in the db_inserted folder
def is_file_in_inserted_folder(filename, date_str):
    """Check if a file with the same name is already in the db_inserted folder
    
    Args:
        filename (str): The filename
        date_str (str): The file save date string in YYMMDD format
        
    Returns:
        bool: True if the file is already in the db_inserted folder, False otherwise
    """
    # Check if the db_inserted folder exists
    if not os.path.exists(DB_INSERTED_PATH):
        os.makedirs(DB_INSERTED_PATH, exist_ok=True)
        return False
    
    # Extract original filename if it contains a date
    original_filename = filename
    if "--" in filename:
        parts = filename.split("--")
        if len(parts) >= 2:
            original_filename = parts[0]
    
    # Get all files in the db_inserted folder
    inserted_files = os.listdir(DB_INSERTED_PATH)
    
    # Check if the original filename is already in the db_inserted folder
    # We check for the original filename since files are stored with their original names
    return original_filename in inserted_files

# Function to copy file to db_inserted folder with original filename
def copy_file_to_inserted(source_path, filename, date_str):
    """Copy a file to the db_inserted folder with its original filename
    
    Args:
        source_path (str): The source file path
        filename (str): The filename (can be original or with date)
        date_str (str): The insertion date string in YYMMDD format (not used in filename)
        
    Returns:
        str: The path to the copied file
    """
    # Create the db_inserted folder if it doesn't exist
    os.makedirs(DB_INSERTED_PATH, exist_ok=True)
    
    # Extract original filename if it contains a date
    original_filename = filename
    if "--" in filename:
        parts = filename.split("--")
        if len(parts) >= 2:
            original_filename = parts[0]
    
    # Copy the file to the db_inserted folder with its original filename
    target_path = os.path.join(DB_INSERTED_PATH, original_filename)
    shutil.copy2(source_path, target_path)
    
    st.info(f"Copied file to {target_path} using original filename {original_filename}")
    
    return target_path

# Create tabs for the different steps
tab1, tab2, tab3, tab4 = st.tabs(["Step 1: Select/Create Database", "Step 2: Insert Documents", "Step 3: View Database", "Step 4: Manage Deletions"])

# Step 1: Select or Create Database
with tab1:
    st.header("Select or Create Vector Database")
    
    # Get list of available databases
    available_dbs = list_available_databases()
    
    # Option to select existing database or create new one
    db_option = st.radio(
        "Choose an option",
        ["Select existing database", "Create new database"],
        index=0 if available_dbs else 1
    )
    
    if db_option == "Select existing database":
        if available_dbs:
            # Display available databases with their embedding models
            db_options_with_models = {}
            for db in available_dbs:
                embedding_model = extract_embedding_from_db_name(db)
                if embedding_model:
                    db_options_with_models[db] = f"{db} (Embedding: {embedding_model})"
                else:
                    db_options_with_models[db] = db
            
            selected_db = st.selectbox(
                "Select a database",
                options=list(db_options_with_models.keys()),
                format_func=lambda x: db_options_with_models[x],
                index=0
            )
            
            if st.button("Use Selected Database"):
                # Extract embedding model from database name
                embedding_model = extract_embedding_from_db_name(selected_db)
                if embedding_model:
                    st.session_state.embedding_model = embedding_model
                    st.session_state.selected_database = selected_db
                    st.session_state.vdb_dir = os.path.join(DATABASE_PATH, selected_db)
                    st.success(f"Selected database: {selected_db} with embedding model: {embedding_model}")
                else:
                    st.error(f"Could not extract embedding model from database name: {selected_db}")
        else:
            st.warning("No existing databases found. Please create a new one.")
    
    else:  # Create new database
        st.subheader("Create New Database")
        
        # Select embedding model
        embedding_options = [
            "Qwen/Qwen3-Embedding-0.6B",
            "sentence-transformers/all-mpnet-base-v2",
            "jinaai/jina-embeddings-v2-base-de",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
        
        selected_embedding = st.selectbox(
            "Choose an embedding model",
            options=embedding_options,
            index=embedding_options.index(st.session_state.embedding_model) if st.session_state.embedding_model in embedding_options else 0
        )
        
        # Inputs for chunk size and overlap
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=10000,
                value=st.session_state.chunk_size,
                step=100
            )
        
        with col2:
            chunk_overlap = st.number_input(
                "Chunk overlap",
                min_value=0,
                max_value=5000,
                value=st.session_state.chunk_overlap,
                step=50
            )
        
        # Preview the VDB directory name
        clean_embed_name = clean_model_name(selected_embedding)
        vdb_dir_name = f"{clean_embed_name}--{chunk_size}--{chunk_overlap}"
        vdb_full_path = os.path.join(DATABASE_PATH, vdb_dir_name)
        
        st.info(f"Vector database will be stored at: **{vdb_full_path}**")
        
        if st.button("Create Database"):
            st.session_state.embedding_model = selected_embedding
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.selected_database = vdb_dir_name
            st.session_state.vdb_dir = vdb_full_path
            
            # Create directory if it doesn't exist
            os.makedirs(vdb_full_path, exist_ok=True)
            
            st.success(f"Database created: {vdb_dir_name} with embedding model: {selected_embedding}")

# Step 2: Insert Documents
with tab2:
    st.header("Insert Documents to Vector Database")
    
    if not st.session_state.vdb_dir:
        st.warning("Please select or create a database in Step 1 first.")
    else:
        st.success(f"Using database: **{st.session_state.selected_database}**")
        st.info(f"Embedding model: **{st.session_state.embedding_model}**")
        st.info(f"Tenant ID: **{st.session_state.tenant_id}**")
        
        # Get files in insert_data folder
        insert_files = []
        if os.path.exists(DEFAULT_DATA_FOLDER):
            insert_files = [f for f in os.listdir(DEFAULT_DATA_FOLDER) if os.path.isfile(os.path.join(DEFAULT_DATA_FOLDER, f))]
        
        if not insert_files:
            st.warning(f"No files found in {DEFAULT_DATA_FOLDER}. Please add files to this folder.")
        else:
            st.success(f"Found {len(insert_files)} files in {DEFAULT_DATA_FOLDER}")
            
            # Display files with their metadata
            st.subheader("Files to Process")
            file_data = []
            current_date = datetime.now().strftime('%y%m%d')
            
            for filename in insert_files:
                file_path = os.path.join(DEFAULT_DATA_FOLDER, filename)
                
                # Get file save date from filesystem
                file_save_date = get_file_save_date(file_path)
                st.info(f"File: {filename} - Filesystem save date: {file_save_date}")
                
                # Extract original filename if it contains a date
                original_filename = filename
                if "--" in filename:
                    parts = filename.split("--")
                    if len(parts) >= 2:
                        original_filename = parts[0]
                
                # Use the filesystem file save date for metadata
                metadata = f"{original_filename}--{file_save_date}"
                
                # Check if file is already in database or inserted folder
                embed_model = get_embedding_model(st.session_state.embedding_model)
                is_in_db = is_file_in_database(filename, file_save_date, st.session_state.tenant_id, embed_model, st.session_state.vdb_dir)
                is_in_inserted = is_file_in_inserted_folder(filename, file_save_date)
                
                file_data.append({
                    "filename": filename,
                    "metadata": metadata,
                    "is_in_db": is_in_db,
                    "is_in_inserted": is_in_inserted
                })
            
            # Display files in a table
            file_df = pd.DataFrame(file_data)
            st.dataframe(file_df)
            
            # Sidebar
            with st.sidebar:
                # Create a spinner to show when loading the model
                with st.spinner("Loading model..."):
                    # Option to force CPU for very large documents
                    st.session_state.force_cpu_large_docs = st.checkbox(
                        "Force CPU for large documents", 
                        value=st.session_state.force_cpu_large_docs, 
                        help="Use CPU for processing very large documents to avoid CUDA OOM errors"
                    )
                    
            # Clear CUDA memory before starting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                st.info("Initial CUDA memory cleared")
                
            # Button to process files
            # Add a session state variable to track embedding completion
            if 'embedding_completed' not in st.session_state:
                st.session_state.embedding_completed = False
                
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    # Reset completion status at the start
                    st.session_state.embedding_completed = False
                    
                    # Get embedding model
                    embed_model = get_embedding_model(st.session_state.embedding_model)
                    
                    # Process each file
                    for file_info in file_data:
                        filename = file_info["filename"]
                        is_in_db = file_info["is_in_db"]
                        is_in_inserted = file_info["is_in_inserted"]
                        
                        if not is_in_db and not is_in_inserted:
                            # Create an expander for this document's processing logs
                            with st.expander(f"Processing: {filename}", expanded=True):
                                try:
                                    # Clear CUDA memory before processing each document
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        
                                        # Update GPU memory info in session state
                                        used_memory, total_memory, percent_used = get_gpu_memory_info()
                                        st.session_state.gpu_memory_info = {
                                            'used': used_memory,
                                            'total': total_memory,
                                            'percent': percent_used
                                        }
                                        
                                        st.info(f"CUDA memory cleared before processing {filename}")
                                    
                                    # Create embeddings
                                    file_path = os.path.join(DEFAULT_DATA_FOLDER, filename)
                                    
                                    # Get file save date from filesystem
                                    original_filename = filename
                                
                                    # Extract original filename if it contains a date
                                    if "--" in filename:
                                        parts = filename.split("--")
                                        if len(parts) >= 2:
                                            original_filename = parts[0]
                                
                                    # Get file save date from filesystem
                                    file_save_date = get_file_save_date(file_path)
                                    st.info(f"Using file save date from filesystem: {file_save_date} for {filename}")
                                
                                    # Get current date for insertion date
                                    insertion_date = datetime.now().strftime('%y%m%d')
                                    st.info(f"Using insertion date: {insertion_date} (current date)")
                                    
                                    # Display both dates for clarity
                                    st.info(f"File metadata: Original filename: {original_filename}, File save date: {file_save_date}, Insertion date: {insertion_date}")
                                    
                                    # Create a temporary folder with just this file
                                    temp_folder = os.path.join(DEFAULT_DATA_FOLDER, "temp_embed")
                                    os.makedirs(temp_folder, exist_ok=True)
                                    temp_file_path = os.path.join(temp_folder, filename)
                                    shutil.copy2(file_path, temp_file_path)
                                
                                    # Process the file directly instead of using a nested function
                                    try:
                                        # Get the file extension
                                        file_ext = os.path.splitext(file_path)[1].lower()
                                    
                                        # Process the file based on its type
                                        if file_ext == ".pdf":
                                            # Check file size and switch to CPU if it's a large document and the option is enabled
                                            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                                            original_device = st.session_state.gpu_device
                                        
                                            # If file is large (> 10MB) and force CPU option is enabled, switch to CPU
                                            if file_size_mb > 10 and st.session_state.force_cpu_large_docs:
                                                st.warning(f"Large document detected ({file_size_mb:.2f}MB). Switching to CPU processing.")
                                                st.session_state.gpu_device = "cpu"
                                                # Get a CPU-based embedding model for this document
                                                cpu_embed_model = get_embedding_model(st.session_state.embedding_model)
                                                # Use the CPU model for this operation
                                                embed_model = cpu_embed_model
                                            
                                            try:
                                                try:
                                                    # Clear CUDA memory before extracting text from PDF
                                                    if torch.cuda.is_available():
                                                        torch.cuda.empty_cache()
                                                    
                                                    # Extract text from PDF
                                                    text = extract_text_from_pdf(file_path)
                                                    
                                                    # Clear CUDA memory before creating vector database
                                                    if torch.cuda.is_available():
                                                        torch.cuda.empty_cache()
                                                except Exception as pdf_error:
                                                    st.error(f"Error extracting text from PDF: {str(pdf_error)}")
                                                    continue
                                                    
                                                # Create a document
                                                document = Document(
                                                    page_content=text,
                                                    metadata={
                                                        "source": f"{original_filename}--{file_save_date}",
                                                        "path": file_path,
                                                        "original_filename": original_filename,
                                                        "file_save_date": file_save_date,
                                                        "insertion_date": insertion_date
                                                    }
                                                )
                                                
                                                # Log metadata for debugging
                                                st.info(f"Document metadata: {document.metadata}")
                                                
                                                # Create text splitter
                                                text_splitter = RecursiveCharacterTextSplitter(
                                                    chunk_size=st.session_state.chunk_size,
                                                    chunk_overlap=st.session_state.chunk_overlap,
                                                    separators=["\n\n", "\n", ".", " ", ""]
                                                )
                                                
                                                # Split document into chunks
                                                doc_chunks = text_splitter.split_documents([document])
                                                
                                                # Add metadata to chunks
                                                chunks = []
                                                for chunk in doc_chunks:
                                                    chunk.metadata['source'] = document.metadata['source']
                                                    chunk.metadata['page'] = document.metadata.get('page', 0)
                                            except RuntimeError as cuda_error:
                                                if "CUDA out of memory" in str(cuda_error):
                                                    st.warning(f"CUDA out of memory when processing {filename}. Falling back to CPU...")
                                                    
                                                    # Save current device setting
                                                    original_device = st.session_state.gpu_device
                                                    
                                                    # Temporarily switch to CPU
                                                    st.session_state.gpu_device = "cpu"
                                                    
                                                    # Get a CPU-based embedding model
                                                    cpu_embed_model = get_embedding_model(st.session_state.embedding_model)
                                                    
                                                    # Clear CUDA memory again before retry with CPU
                                                    if torch.cuda.is_available():
                                                        torch.cuda.empty_cache()
                                                        
                                                    # Extract text from PDF
                                                    text = extract_text_from_pdf(file_path)
                                                    
                                                    # Create a document
                                                    document = Document(
                                                        page_content=text,
                                                        metadata={
                                                            "source": f"{original_filename}--{file_save_date}",
                                                            "path": file_path,
                                                            "original_filename": original_filename,
                                                            "file_save_date": file_save_date,
                                                            "insertion_date": insertion_date
                                                        }
                                                    )
                                                    
                                                    # Create text splitter
                                                    text_splitter = RecursiveCharacterTextSplitter(
                                                        chunk_size=st.session_state.chunk_size,
                                                        chunk_overlap=st.session_state.chunk_overlap,
                                                        separators=["\n\n", "\n", ".", " ", ""]
                                                    )
                                                    
                                                    # Split document into chunks
                                                    doc_chunks = text_splitter.split_documents([document])
                                                    
                                                    # Add metadata to chunks
                                                    chunks = []
                                                    for chunk in doc_chunks:
                                                        chunk.metadata['source'] = document.metadata['source']
                                                        chunk.metadata['page'] = document.metadata.get('page', 0)
                                                    
                                                    # Restore original device setting for future operations
                                                    st.session_state.gpu_device = original_device
                                                    
                                                    # Use the CPU model for this operation
                                                    embed_model = cpu_embed_model
                                                else:
                                                    # Re-raise if it's not a CUDA memory error
                                                    raise
                                        
                                        # Restore original device setting after processing if we switched to CPU for large file
                                        if file_size_mb > 10 and st.session_state.force_cpu_large_docs:
                                            st.session_state.gpu_device = original_device
                                                
                                        # Add additional metadata to chunks and collect them
                                        for chunk in doc_chunks:
                                            chunk.metadata['path'] = document.metadata.get('path', '')
                                            chunk.metadata['original_filename'] = document.metadata.get('original_filename', '')
                                            chunk.metadata['file_save_date'] = document.metadata.get('file_save_date', '')
                                            chunk.metadata['insertion_date'] = document.metadata.get('insertion_date', '')
                                            chunks.append(chunk)
                                        
                                        # Log number of chunks created
                                        st.info(f"Created {len(chunks)} chunks with file save date: {file_save_date} and insertion date: {insertion_date}")
                                        
                                        # Calculate human-readable chunk IDs
                                        chunks = calculate_chunk_ids(chunks)
                                        
                                        # Get tenant vectorstore
                                        tenant_vdb_dir = os.path.join(st.session_state.vdb_dir, st.session_state.tenant_id)
                                        collection_name = get_tenant_collection_name(st.session_state.tenant_id)
                                        vectorstore = Chroma(
                                            persist_directory=tenant_vdb_dir,
                                            collection_name=collection_name,
                                            embedding_function=embed_model
                                        )
                                        
                                        # Clear CUDA memory immediately before embedding allocation
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                            st.info("CUDA memory cleared before embedding allocation")
                                            
                                        # Process chunks in smaller batches to reduce memory usage
                                        batch_size = 10  # Adjust this based on your document size and GPU memory
                                        for i in range(0, len(chunks), batch_size):
                                            # Clear CUDA memory before each batch
                                            if torch.cuda.is_available():
                                                torch.cuda.empty_cache()
                                                
                                            # Get the current batch
                                            batch_chunks = chunks[i:i+batch_size]
                                            st.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} with {len(batch_chunks)} chunks")
                                            
                                            # Add documents to vectorstore
                                            vectorstore.add_documents(batch_chunks)
                                            
                                            # Give a small delay for memory cleanup between batches
                                            import time
                                            time.sleep(0.5)
                                        
                                        # Copy file to db_inserted folder using original filename
                                        inserted_path = copy_file_to_inserted(file_path, original_filename, insertion_date)
                                        st.success(f"File copied to inserted folder: {inserted_path}")
                                    except Exception as e:
                                        st.error(f"Error processing {filename}: {str(e)}")
                                except Exception as e:
                                    st.error(f"Error processing {filename}: {str(e)}")
                        else:
                            st.info(f"Skipped {filename} (already processed)")
                    
                    # Mark embedding as completed and show success message
                    st.session_state.embedding_completed = True
                    st.success("✅ All file embeddings completed successfully!")
                    
                    # Update GPU memory info after all processing is done
                    used_memory, total_memory, percent_used = get_gpu_memory_info()
                    st.session_state.gpu_memory_info = {
                        'used': used_memory,
                        'total': total_memory,
                        'percent': percent_used
                    }
                    

# Step 3: View Database
with tab3:
    st.header("View Database Contents")
    
    if not st.session_state.vdb_dir:
        st.warning("Please select or create a database in Step 1 first.")
    else:
        st.success(f"Using database: **{st.session_state.selected_database}**")
        st.info(f"Embedding model: **{st.session_state.embedding_model}**")
        st.info(f"Tenant ID: **{st.session_state.tenant_id}**")
        
        try:
            # Get embedding model
            embed_model = get_embedding_model(st.session_state.embedding_model)
            
            # Get all documents in the database
            all_docs = get_all_documents_in_vectordb(st.session_state.tenant_id, embed_model, st.session_state.vdb_dir)
            
            if all_docs:
                st.subheader("Documents in Database")
                
                # Parse metadata from vector database
                doc_data = []
                
                # Get tenant vectorstore once for all documents
                tenant_vdb_dir = os.path.join(st.session_state.vdb_dir, st.session_state.tenant_id)
                collection_name = get_tenant_collection_name(st.session_state.tenant_id)
                vectorstore = Chroma(
                    persist_directory=tenant_vdb_dir,
                    collection_name=collection_name,
                    embedding_function=embed_model
                )
                
                # Get all metadata at once
                result = vectorstore.get()
                
                for doc in all_docs:
                    # Initialize default values
                    file_save_date = "N/A"
                    insertion_date = "N/A"
                    original_filename = doc
                    all_metadata = {}
                    
                    # Extract the metadata from the vector database
                    try:
                        if 'metadatas' in result and result['metadatas']:
                            for i, metadata in enumerate(result['metadatas']):
                                if metadata and 'source' in metadata and metadata['source'] == doc:
                                    # Store all metadata for display in expander
                                    all_metadata = metadata.copy()
                                    
                                    # Get original_filename from metadata if available
                                    if 'original_filename' in metadata and metadata['original_filename']:
                                        original_filename = metadata['original_filename']
                                    
                                    # Get file_save_date from metadata if available
                                    if 'file_save_date' in metadata and metadata['file_save_date']:
                                        file_save_date_raw = metadata['file_save_date']
                                        # Format the date as YY-MM-DD if it's in YYMMDD format
                                        if len(file_save_date_raw) == 6 and file_save_date_raw.isdigit():
                                            file_save_date = f"{file_save_date_raw[:2]}-{file_save_date_raw[2:4]}-{file_save_date_raw[4:]}"
                                        else:
                                            file_save_date = file_save_date_raw
                                    
                                    # Get insertion_date from metadata if available
                                    if 'insertion_date' in metadata and metadata['insertion_date']:
                                        insertion_date_raw = metadata['insertion_date']
                                        # Format the date as YY-MM-DD if it's in YYMMDD format
                                        if len(insertion_date_raw) == 6 and insertion_date_raw.isdigit():
                                            insertion_date = f"{insertion_date_raw[:2]}-{insertion_date_raw[2:4]}-{insertion_date_raw[4:]}"
                                        else:
                                            insertion_date = insertion_date_raw
                                    
                                    break
                    except Exception as e:
                        # If there's an error, just use the default values
                        print(f"Error getting metadata: {str(e)}")
                        # Try to extract information from filename if it follows the pattern
                        if "--" in doc:
                            parts = doc.split("--")
                            if len(parts) >= 2:
                                original_filename = parts[0]
                                file_save_date_raw = parts[1]
                                if len(file_save_date_raw) == 6 and file_save_date_raw.isdigit():
                                    file_save_date = f"{file_save_date_raw[:2]}-{file_save_date_raw[2:4]}-{file_save_date_raw[4:]}"
                                else:
                                    file_save_date = file_save_date_raw
                    
                    # Add to document data list
                    doc_data.append({
                        "Document": original_filename,
                        "File Save Date": file_save_date,
                        "Insertion Date": insertion_date,
                        "Source": doc,
                        "Metadata": all_metadata
                    })
                
                # Clean up vectorstore
                vectorstore._client = None
                del vectorstore
                
                # Create a dataframe for display
                import pandas as pd
                display_df = pd.DataFrame([
                    {
                        "Document": item["Document"],
                        "File Save Date": item["File Save Date"],
                        "Insertion Date": item["Insertion Date"]
                    } for item in doc_data
                ])
                
                # Display the dataframe
                st.dataframe(display_df)
                
                # Show expandable metadata for each document
                st.subheader("Document Metadata Details")
                for i, doc_item in enumerate(doc_data):
                    with st.expander(f"📄 {doc_item['Document']}"):
                        st.markdown(f"**Source:** {doc_item['Source']}")
                        st.markdown(f"**File Save Date:** {doc_item['File Save Date']}")
                        st.markdown(f"**Insertion Date:** {doc_item['Insertion Date']}")
                        
                        # Display all metadata in a formatted way
                        st.markdown("### All Metadata")
                        if doc_item['Metadata']:
                            for key, value in doc_item['Metadata'].items():
                                st.markdown(f"**{key}:** {value}")
                        else:
                            st.info("No additional metadata available")
                
                # Show total count
                st.success(f"Total documents in database: {len(all_docs)}")
                
                # Option to perform a test query
                st.subheader("Test Retrieval")
                query = st.text_input("Enter a query to test retrieval")
                k_results = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=3)
                
                if st.button("Perform Query") and query:
                    with st.spinner("Retrieving documents..."):
                        try:
                            # Perform similarity search
                            results = similarity_search_for_tenant(
                                tenant_id=st.session_state.tenant_id,
                                embed_llm=embed_model,
                                persist_directory=st.session_state.vdb_dir,
                                similarity="cosine",
                                normal=True,
                                query=query,
                                k=k_results,
                                language="English"  # Default language
                            )
                            
                            # Display retrieved documents
                            st.subheader("Retrieved Documents")
                            
                            for i, doc in enumerate(results):
                                with st.expander(f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                                    st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                    st.write(f"**Chunk Nr:** {doc.metadata.get('chunk_id', 'Unknown')}")
                                    st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                                    st.write(f"**Content:**\n{doc.page_content}")
                        except Exception as e:
                            st.error(f"Error during retrieval: {str(e)}")
            else:
                st.warning("No documents found in the database. Please insert documents in Step 2.")
        except Exception as e:
            st.error(f"Error retrieving database contents: {str(e)}")

# Function to delete documents from vector database
def delete_documents_from_vectordb(tenant_id, embed_llm, persist_directory, document_source):
    """
    Delete all chunks associated with a specific document source from the vector database
    
    Args:
        tenant_id (str): The tenant ID
        embed_llm: The embedding model
        persist_directory (str): The directory where the vector database is stored
        document_source (str): The source identifier of the document to delete (format: filename--date)
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        # Get tenant-specific directory
        tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
        
        # Check if directory exists
        if not os.path.exists(tenant_vdb_dir):
            return False
        
        # Get collection name for tenant
        collection_name = get_tenant_collection_name(tenant_id)
        
        # Initialize vectorstore
        vectorstore = Chroma(
            persist_directory=tenant_vdb_dir,
            collection_name=collection_name,
            embedding_function=embed_llm
        )
        
        try:
            # Get all documents
            result = vectorstore.get()
            
            # Find IDs of chunks with matching source
            ids_to_delete = []
            if 'ids' in result and 'metadatas' in result:
                for i, metadata in enumerate(result['metadatas']):
                    if metadata and 'source' in metadata and metadata['source'] == document_source:
                        ids_to_delete.append(result['ids'][i])
            
            # Delete chunks if any found
            if ids_to_delete:
                vectorstore.delete(ids=ids_to_delete)
                st.success(f"Deleted {len(ids_to_delete)} chunks for document: {document_source}")
            else:
                st.warning(f"No chunks found for document: {document_source}")
            
            # Clean up
            vectorstore._client = None
            del vectorstore
            
            return len(ids_to_delete) > 0
        except Exception as e:
            # Clean up in case of error
            if 'vectorstore' in locals():
                vectorstore._client = None
                del vectorstore
            
            # Re-raise the exception
            st.error(f"Error deleting chunks: {str(e)}")
            return False
    except Exception as e:
        st.error(f"Error accessing vector database: {str(e)}")
        return False

# Function to rename file in db_inserted folder with _removed suffix
def rename_file_with_removed_suffix(filename):
    """
    Rename a file in the db_inserted folder with _removed suffix
    
    Args:
        filename (str): The original filename without path
        
    Returns:
        bool: True if renaming was successful, False otherwise
    """
    try:
        # Check if the db_inserted folder exists
        if not os.path.exists(DB_INSERTED_PATH):
            st.error(f"DB inserted folder not found: {DB_INSERTED_PATH}")
            return False
        
        # Get the file path
        file_path = os.path.join(DB_INSERTED_PATH, filename)
        
        # Check if the file exists
        if not os.path.exists(file_path):
            st.error(f"File not found in db_inserted folder: {filename}")
            return False
        
        # Get file name and extension
        file_name, file_ext = os.path.splitext(filename)
        
        # Create new filename with _removed suffix
        new_filename = f"{file_name}_removed{file_ext}"
        new_file_path = os.path.join(DB_INSERTED_PATH, new_filename)
        
        # Rename the file
        os.rename(file_path, new_file_path)
        
        st.success(f"Renamed file to: {new_filename}")
        return True
    except Exception as e:
        st.error(f"Error renaming file: {str(e)}")
        return False

# Function to get all files with _removed suffix in db_inserted folder
def get_removed_files():
    """
    Get all files with _removed suffix in db_inserted folder
    
    Returns:
        list: A list of filenames with _removed suffix
    """
    try:
        # Check if the db_inserted folder exists
        if not os.path.exists(DB_INSERTED_PATH):
            return []
        
        # Get all files in the db_inserted folder
        all_files = os.listdir(DB_INSERTED_PATH)
        
        # Filter files with _removed suffix
        removed_files = []
        for filename in all_files:
            if "_removed" in filename:
                removed_files.append(filename)
        
        return removed_files
    except Exception as e:
        st.error(f"Error getting removed files: {str(e)}")
        return []

# Function to permanently delete files
def permanently_delete_files(filenames):
    """
    Permanently delete files from db_inserted folder
    
    Args:
        filenames (list): List of filenames to delete
        
    Returns:
        tuple: (success_count, error_count, errors)
    """
    success_count = 0
    error_count = 0
    errors = []
    
    for filename in filenames:
        try:
            file_path = os.path.join(DB_INSERTED_PATH, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                success_count += 1
            else:
                error_count += 1
                errors.append(f"File not found: {filename}")
        except Exception as e:
            error_count += 1
            errors.append(f"Error deleting {filename}: {str(e)}")
    
    return success_count, error_count, errors

# Step 4: Delete Documents
with tab4:
    st.header("Delete Documents from Vector Database")
    
    if not st.session_state.vdb_dir:
        st.warning("Please select or create a database in Step 1 first.")
    else:
        st.success(f"Using database: **{st.session_state.selected_database}**")
        st.info(f"Embedding model: **{st.session_state.embedding_model}**")
        st.info(f"Tenant ID: **{st.session_state.tenant_id}**")
        
        # Create tabs for delete operations
        delete_tab1, delete_tab2 = st.tabs(["Delete from Vector Database", "Manage Removed Files"])
        
        # Tab for deleting from vector database
        with delete_tab1:
            try:
                # Get embedding model
                embed_model = get_embedding_model(st.session_state.embedding_model)
                
                # Get all documents in the database
                all_docs = get_all_documents_in_vectordb(st.session_state.tenant_id, embed_model, st.session_state.vdb_dir)
                
                if all_docs:
                    st.subheader("Documents Available for Deletion")
                    
                    # Parse metadata from vector database
                    doc_data = []
                    
                    # Get tenant vectorstore once for all documents
                    tenant_vdb_dir = os.path.join(st.session_state.vdb_dir, st.session_state.tenant_id)
                    collection_name = get_tenant_collection_name(st.session_state.tenant_id)
                    vectorstore = Chroma(
                        persist_directory=tenant_vdb_dir,
                        collection_name=collection_name,
                        embedding_function=embed_model
                    )
                    
                    # Get all metadata at once
                    result = vectorstore.get()
                    
                    for doc in all_docs:
                        # Initialize default values
                        file_save_date = "N/A"
                        insertion_date = "N/A"
                        original_filename = doc
                        all_metadata = {}
                        
                        # Extract the metadata from the vector database
                        try:
                            if 'metadatas' in result and result['metadatas']:
                                for i, metadata in enumerate(result['metadatas']):
                                    if metadata and 'source' in metadata and metadata['source'] == doc:
                                        # Store all metadata for display in expander
                                        all_metadata = metadata.copy()
                                        
                                        # Get original_filename from metadata if available
                                        if 'original_filename' in metadata and metadata['original_filename']:
                                            original_filename = metadata['original_filename']
                                        
                                        # Get file_save_date from metadata if available
                                        if 'file_save_date' in metadata and metadata['file_save_date']:
                                            file_save_date_raw = metadata['file_save_date']
                                            # Format the date as YY-MM-DD if it's in YYMMDD format
                                            if len(file_save_date_raw) == 6 and file_save_date_raw.isdigit():
                                                file_save_date = f"{file_save_date_raw[:2]}-{file_save_date_raw[2:4]}-{file_save_date_raw[4:]}"
                                            else:
                                                file_save_date = file_save_date_raw
                                        
                                        # Get insertion_date from metadata if available
                                        if 'insertion_date' in metadata and metadata['insertion_date']:
                                            insertion_date_raw = metadata['insertion_date']
                                            # Format the date as YY-MM-DD if it's in YYMMDD format
                                            if len(insertion_date_raw) == 6 and insertion_date_raw.isdigit():
                                                insertion_date = f"{insertion_date_raw[:2]}-{insertion_date_raw[2:4]}-{insertion_date_raw[4:]}"
                                            else:
                                                insertion_date = insertion_date_raw
                                        
                                        break
                        except Exception as e:
                            # If there's an error, just use the default values
                            print(f"Error getting metadata: {str(e)}")
                            # Try to extract information from filename if it follows the pattern
                            if "--" in doc:
                                parts = doc.split("--")
                                if len(parts) >= 2:
                                    original_filename = parts[0]
                                    file_save_date_raw = parts[1]
                                    if len(file_save_date_raw) == 6 and file_save_date_raw.isdigit():
                                        file_save_date = f"{file_save_date_raw[:2]}-{file_save_date_raw[2:4]}-{file_save_date_raw[4:]}"
                                    else:
                                        file_save_date = file_save_date_raw
                        
                        # Add to document data list
                        doc_data.append({
                            "Document": original_filename,
                            "File Save Date": file_save_date,
                            "Insertion Date": insertion_date,
                            "Source": doc,
                            "Metadata": all_metadata
                        })
                    
                    # Clean up vectorstore
                    vectorstore._client = None
                    del vectorstore
                    
                    # Create a dataframe for display
                    import pandas as pd
                    display_df = pd.DataFrame([
                        {
                            "Document": item["Document"],
                            "File Save Date": item["File Save Date"],
                            "Insertion Date": item["Insertion Date"]
                        } for item in doc_data
                    ])
                    
                    # Display the dataframe
                    st.dataframe(display_df)
                    
                    # Select document to delete
                    st.subheader("Select Document to Delete")
                    
                    # Create a list of document options with their metadata
                    doc_options = [f"{item['Document']} (Save Date: {item['File Save Date']})" for item in doc_data]
                    selected_doc_index = st.selectbox("Select a document to delete", range(len(doc_options)), format_func=lambda i: doc_options[i])
                    
                    # Get the selected document data
                    selected_doc = doc_data[selected_doc_index]
                    
                    # Show warning and confirmation
                    st.warning(f"⚠️ You are about to delete all chunks for document: **{selected_doc['Document']}**")
                    st.info(f"Original filename: {selected_doc['Document']}")
                    st.info(f"Source identifier: {selected_doc['Source']}")
                    
                    # Confirmation checkbox
                    confirm_delete = st.checkbox("I confirm that I want to delete this document from the vector database")
                    
                    if st.button("Delete Document") and confirm_delete:
                        with st.spinner("Deleting document from vector database..."):
                            # Delete document from vector database
                            delete_success = delete_documents_from_vectordb(
                                tenant_id=st.session_state.tenant_id,
                                embed_llm=embed_model,
                                persist_directory=st.session_state.vdb_dir,
                                document_source=selected_doc['Source']
                            )
                            
                            if delete_success:
                                # Rename file in db_inserted folder
                                rename_success = rename_file_with_removed_suffix(selected_doc['Document'])
                                
                                if rename_success:
                                    st.success(f"✅ Document successfully deleted and file renamed with _removed suffix")
                                else:
                                    st.warning(f"⚠️ Document was deleted from vector database but file could not be renamed")
                                    
                                # Refresh the page to update the document list
                                st.info("Please refresh the page to update the document list")
                                if st.button("Refresh Page"):
                                    st.experimental_rerun()
                else:
                    st.warning("No documents found in the database. Please insert documents in Step 2.")
            except Exception as e:
                st.error(f"Error retrieving database contents: {str(e)}")
        
        # Tab for managing removed files
        with delete_tab2:
            st.subheader("Manage Removed Files")
            
            # Get all files with _removed suffix
            removed_files = get_removed_files()
            
            if removed_files:
                st.success(f"Found {len(removed_files)} files marked for removal")
                
                # Initialize session state for file selection if not exists
                if 'selected_files_for_deletion' not in st.session_state:
                    st.session_state.selected_files_for_deletion = {}
                
                # Create a container for the file list
                with st.container():
                    st.markdown("### Files Marked for Removal")
                    st.markdown("Select files to permanently delete:")
                    
                    # Create checkboxes for each file
                    for file in removed_files:
                        # Initialize if not exists
                        if file not in st.session_state.selected_files_for_deletion:
                            st.session_state.selected_files_for_deletion[file] = False
                        
                        # Create checkbox and update session state
                        selected = st.checkbox(f"❌ {file}", value=st.session_state.selected_files_for_deletion[file], key=f"select_{file}")
                        st.session_state.selected_files_for_deletion[file] = selected
                
                # Count selected files
                selected_count = sum(1 for file, selected in st.session_state.selected_files_for_deletion.items() if selected)
                
                # Show delete button if files are selected
                if selected_count > 0:
                    st.warning(f"⚠️ You have selected {selected_count} files for permanent deletion")
                    
                    # Add select all / deselect all buttons in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Select All"):
                            for file in removed_files:
                                st.session_state.selected_files_for_deletion[file] = True
                            st.experimental_rerun()
                    with col2:
                        if st.button("Deselect All"):
                            for file in removed_files:
                                st.session_state.selected_files_for_deletion[file] = False
                            st.experimental_rerun()
                    
                    # Confirmation checkbox
                    confirm_permanent_delete = st.checkbox("I understand that this action will permanently delete the selected files and cannot be undone")
                    
                    # Delete button
                    if st.button("⚠️ Permanently Delete Selected Files") and confirm_permanent_delete:
                        with st.spinner("Deleting files..."):
                            # Get list of files to delete
                            files_to_delete = [file for file, selected in st.session_state.selected_files_for_deletion.items() if selected]
                            
                            # Delete files
                            success_count, error_count, errors = permanently_delete_files(files_to_delete)
                            
                            # Show results
                            if success_count > 0:
                                st.success(f"✅ Successfully deleted {success_count} files")
                            
                            if error_count > 0:
                                st.error(f"Failed to delete {error_count} files")
                                for error in errors:
                                    st.error(error)
                            
                            # Reset selection state
                            st.session_state.selected_files_for_deletion = {}
                            
                            # Refresh button
                            if st.button("Refresh List"):
                                st.experimental_rerun()
                else:
                    st.info("Select files to delete using the checkboxes above")
            else:
                st.info("No files marked for removal found in the db_inserted folder")

# Add a footer
st.markdown("---")
st.markdown("*Vector Database Handler App for RAG Researcher*")
