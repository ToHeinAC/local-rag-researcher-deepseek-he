import streamlit as st
import asyncio
import os
import sys
import torch
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Add project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Import the corrective RAG agent
from src.assistant.v1_2.corrective_rag_agent_local import (
    init_agent, 
    run_agent, 
    DEFAULT_DATABASE, 
    DEFAULT_TENANT
)

# Import utilities
from src.assistant.utils import clear_cuda_memory
from src.assistant.rag_helpers import similarity_search_for_tenant

# Import embedding models
try:
    from src.assistant.embeddings import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database")

# Set page config
st.set_page_config(
    page_title="Corrective RAG Agent",
    page_icon="🔍",
    layout="wide"
)

# Function to create a clean directory name from embedding model
def clean_model_name(model_name):
    return model_name.replace('/', '--').replace('\\', '--')

# Function to extract embedding model name from database directory
def extract_embedding_model(db_dir_name):
    parts = db_dir_name.split('--')
    if len(parts) >= 2:
        model_name = parts[0].replace('--', '/') + '/' + parts[1]
        return model_name
    return None

# Function to get embedding model
def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

# Function to list available databases
def list_available_databases():
    if not os.path.exists(DATABASE_PATH):
        return []
    
    return [d for d in os.listdir(DATABASE_PATH) if os.path.isdir(os.path.join(DATABASE_PATH, d))]

# Function to list available tenants for a database
def list_available_tenants(database_path):
    if not os.path.exists(database_path):
        return []
    
    return [d for d in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, d))]

# Function to run the agent asynchronously
async def run_agent_async(question, database_path, tenant_id):
    try:
        response = await run_agent(question, database_path, tenant_id)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

# Function to run the agent and handle asyncio
def run_agent_with_asyncio(question, database_path, tenant_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(run_agent_async(question, database_path, tenant_id))
        return response
    finally:
        loop.close()

# Initialize session state variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'selected_database' not in st.session_state:
    st.session_state.selected_database = ""
if 'tenant_id' not in st.session_state:
    st.session_state.tenant_id = DEFAULT_TENANT
if 'vdb_dir' not in st.session_state:
    st.session_state.vdb_dir = ""
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "jinaai/jina-embeddings-v2-base-de"
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "deepseek-r1:latest"
if 'k_results' not in st.session_state:
    st.session_state.k_results = 3

# Function to clear chat history
def clear_chat():
    st.session_state.chat_history = []

# Main function
def main():
    st.title("Corrective RAG Agent")
    st.markdown("This application uses a corrective RAG agent to answer questions based on your local document database.")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # CUDA memory clearing button
    if torch.cuda.is_available():
        st.sidebar.success(f"🚀 GPU Acceleration: Enabled ({torch.cuda.get_device_name(0)})")
        if st.sidebar.button("🧹 Clear CUDA Memory", type="primary"):
            clear_cuda_memory()
            st.sidebar.success("✅ CUDA memory cleared successfully!")
    else:
        st.sidebar.warning("⚠️ GPU Acceleration: Not available (using CPU)")
    
    # Database selection
    st.sidebar.subheader("Database Selection")
    
    # List available databases
    available_databases = list_available_databases()
    
    if not available_databases:
        st.sidebar.warning("No databases found. Please create a database first.")
    else:
        # Select database
        selected_database = st.sidebar.selectbox(
            "Select Database",
            available_databases,
            index=0 if available_databases else None
        )
        
        if selected_database:
            st.session_state.selected_database = selected_database
            st.session_state.vdb_dir = os.path.join(DATABASE_PATH, selected_database)
            
            # Extract embedding model from database name
            embedding_model = extract_embedding_model(selected_database)
            if embedding_model:
                st.session_state.embedding_model = embedding_model
            
            # List available tenants
            available_tenants = list_available_tenants(st.session_state.vdb_dir)
            
            if not available_tenants:
                st.sidebar.warning(f"No tenants found in database {selected_database}.")
                st.session_state.tenant_id = DEFAULT_TENANT
            else:
                # Select tenant
                tenant_id = st.sidebar.selectbox(
                    "Select Tenant",
                    available_tenants,
                    index=0 if DEFAULT_TENANT in available_tenants else 0
                )
                
                if tenant_id:
                    st.session_state.tenant_id = tenant_id
    
    # LLM model selection
    st.sidebar.subheader("LLM Model Selection")
    llm_models = ["deepseek-r1:latest", "deepseek-r1:7b", "llama3:8b", "llama3:70b", "mistral:7b", "mixtral:8x7b"]
    selected_llm = st.sidebar.selectbox(
        "Select LLM Model",
        llm_models,
        index=llm_models.index(st.session_state.llm_model) if st.session_state.llm_model in llm_models else 0
    )
    st.session_state.llm_model = selected_llm
    
    # Number of results to retrieve
    st.sidebar.subheader("Retrieval Settings")
    k_results = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=10, value=st.session_state.k_results)
    st.session_state.k_results = k_results
    
    # Display current configuration
    st.sidebar.subheader("Current Configuration")
    st.sidebar.info(f"Database: {st.session_state.selected_database}")
    st.sidebar.info(f"Tenant ID: {st.session_state.tenant_id}")
    st.sidebar.info(f"Embedding Model: {st.session_state.embedding_model}")
    st.sidebar.info(f"LLM Model: {st.session_state.llm_model}")
    
    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        clear_chat()
        st.experimental_rerun()
    
    # Main chat interface
    st.subheader("Chat")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # User input
    user_question = st.chat_input("Ask a question...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Display user message
        st.chat_message("user").write(user_question)
        
        # Check if database is selected
        if not st.session_state.vdb_dir:
            st.error("Please select a database first.")
            return
        
        # Get response from agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Get embedding model
                embed_model = get_embedding_model(st.session_state.embedding_model)
                
                # Run agent
                response = run_agent_with_asyncio(
                    user_question,
                    st.session_state.vdb_dir,
                    st.session_state.tenant_id
                )
                
                # Display response
                st.write(response)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
