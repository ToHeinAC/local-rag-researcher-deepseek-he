import os
import sys
import time
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path
import re
from datetime import datetime

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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.schema import Document
from src.assistant.prompts import SUMMARIZER_SYSTEM_PROMPT

# Set page config
st.set_page_config(
    page_title="Vector Database Handler",
    page_icon="🗃️",
    layout="wide"
)

# Title and description
st.title("Vector Database Handler App")
st.markdown("This app allows you to select or create vector databases and insert new documents with proper metadata tracking.")

# Define paths
DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "insert_data")
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
DB_INSERTED_PATH = os.path.join(os.path.dirname(__file__), "db_inserted")

# Ensure the db_inserted directory exists
os.makedirs(DB_INSERTED_PATH, exist_ok=True)

# Initialize session state variables
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
if 'selected_database' not in st.session_state:
    st.session_state.selected_database = ""
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 2000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 400
if 'tenant_id' not in st.session_state:
    st.session_state.tenant_id = "default"
if 'vdb_dir' not in st.session_state:
    st.session_state.vdb_dir = ""

# Function to create a clean directory name from embedding model
def clean_model_name(model_name):
    return model_name.replace('/', '--').replace('\\', '--')

# Function to get embedding model
def get_embedding_model(model_name):
    try:
        # Try with model_kwargs that work with newer versions
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except NotImplementedError:
        # Fallback for meta tensor error
        import torch
        import sentence_transformers
        
        # Create the model directly with empty_init=False to avoid meta tensor issues
        model = sentence_transformers.SentenceTransformer(
            model_name,
            device='cpu',
            empty_init=False
        )
        
        # Then create the embedding with the pre-loaded model
        return HuggingFaceEmbeddings(
            model_name=model_name,
            client=model,
            encode_kwargs={'normalize_embeddings': True}
        )

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
tab1, tab2, tab3 = st.tabs(["Step 1: Select/Create Database", "Step 2: Insert Documents", "Step 3: View Database"])

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
            
            # Button to process files
            if st.button("Process Files"):
                with st.spinner("Processing files..."):
                    # Get embedding model
                    embed_model = get_embedding_model(st.session_state.embedding_model)
                    
                    # Process each file
                    for file_info in file_data:
                        filename = file_info["filename"]
                        is_in_db = file_info["is_in_db"]
                        is_in_inserted = file_info["is_in_inserted"]
                        
                        if not is_in_db and not is_in_inserted:
                            try:
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
                                        
                                        # Add documents to vectorstore
                                        vectorstore.add_documents(chunks)
                                        
                                        # Copy file to db_inserted folder using original filename
                                        copy_file_to_inserted(file_path, original_filename, insertion_date)
                                        
                                        st.success(f"Successfully processed {filename} with {len(chunks)} chunks and file save date {file_save_date}")
                                    else:
                                        st.error(f"Unsupported file type: {file_ext}")
                                except Exception as e:
                                    st.error(f"Error processing {filename}: {str(e)}")
                                finally:
                                    # Clean up temp folder
                                    shutil.rmtree(temp_folder, ignore_errors=True)
                            except Exception as e:
                                st.error(f"Error processing {filename}: {str(e)}")
                        else:
                            st.info(f"Skipped {filename} (already processed)")
                    
                    st.success("All files processed successfully!")

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
                for doc in all_docs:
                    # Initialize default values
                    file_save_date = "N/A"
                    insertion_date = "N/A"
                    original_filename = doc
                    
                    # Extract the metadata from the vector database
                    try:
                        # Get the document metadata from the vector database
                        tenant_vdb_dir = os.path.join(st.session_state.vdb_dir, st.session_state.tenant_id)
                        collection_name = get_tenant_collection_name(st.session_state.tenant_id)
                        vectorstore = Chroma(
                            persist_directory=tenant_vdb_dir,
                            collection_name=collection_name,
                            embedding_function=embed_model
                        )
                        
                        # Get metadata for this document
                        result = vectorstore.get()
                        if 'metadatas' in result and result['metadatas']:
                            for i, metadata in enumerate(result['metadatas']):
                                if metadata and 'source' in metadata and metadata['source'] == doc:
                                    # Get original_filename from metadata if available
                                    if 'original_filename' in metadata and metadata['original_filename']:
                                        original_filename = metadata['original_filename']
                                        st.info(f"Found original filename in metadata: {original_filename}")
                                    
                                    # Get file_save_date from metadata if available
                                    if 'file_save_date' in metadata and metadata['file_save_date']:
                                        file_save_date_raw = metadata['file_save_date']
                                        st.info(f"Found file save date in metadata: {file_save_date_raw}")
                                        # Format the date as YY-MM-DD if it's in YYMMDD format
                                        if len(file_save_date_raw) == 6 and file_save_date_raw.isdigit():
                                            file_save_date = f"{file_save_date_raw[:2]}-{file_save_date_raw[2:4]}-{file_save_date_raw[4:]}"
                                        else:
                                            file_save_date = file_save_date_raw
                                    
                                    # Get insertion_date from metadata if available
                                    if 'insertion_date' in metadata and metadata['insertion_date']:
                                        insertion_date_raw = metadata['insertion_date']
                                        st.info(f"Found insertion date in metadata: {insertion_date_raw}")
                                        # Format the date as YY-MM-DD if it's in YYMMDD format
                                        if len(insertion_date_raw) == 6 and insertion_date_raw.isdigit():
                                            insertion_date = f"{insertion_date_raw[:2]}-{insertion_date_raw[2:4]}-{insertion_date_raw[4:]}"
                                        else:
                                            insertion_date = insertion_date_raw
                                    
                                    break
                        
                        # Clean up
                        vectorstore._client = None
                        del vectorstore
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
                    
                    doc_data.append({
                        "Original Filename": original_filename,
                        "File Save Date": file_save_date,
                        "Insertion Date": insertion_date,
                        "Full Document Name": doc
                    })
                
                # Display documents in a table
                doc_df = pd.DataFrame(doc_data)
                st.dataframe(doc_df)
                
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
                                    st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                                    st.write(f"**Content:**\n{doc.page_content}")
                        except Exception as e:
                            st.error(f"Error during retrieval: {str(e)}")
            else:
                st.warning("No documents found in the database. Please insert documents in Step 2.")
        except Exception as e:
            st.error(f"Error retrieving database contents: {str(e)}")

# Add a footer
st.markdown("---")
st.markdown("*Vector Database Handler App for RAG Researcher*")
