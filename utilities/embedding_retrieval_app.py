import os
import sys
import time
import streamlit as st
from pathlib import Path
import re

# Add the parent directory to the path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.assistant.rag_helpers import load_embed, similarity_search_for_tenant, get_tenant_collection_name
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.assistant.rag_helpers import transform_documents, source_summarizer_ollama
from src.assistant.prompts import SUMMARIZER_SYSTEM_PROMPT

# Set page config
st.set_page_config(
    page_title="Embedding & Retrieval Testing",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("Embedding & Retrieval Testing App")
st.markdown("This app allows you to test different embeddings, create vector databases, and test retrieval with various LLMs.")

# Define paths
DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "insert_data")
DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")

# Create tabs for the different steps
tab1, tab2, tab3 = st.tabs(["Step 1: Choose Embedding Model", "Step 2: Create Embeddings", "Step 3: Test Retrieval"])

# Global variables to store selections
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "Qwen/Qwen3-Embedding-0.6B"
if 'chunk_size' not in st.session_state:
    st.session_state.chunk_size = 2000
if 'chunk_overlap' not in st.session_state:
    st.session_state.chunk_overlap = 400
if 'data_folder' not in st.session_state:
    st.session_state.data_folder = DEFAULT_DATA_FOLDER
if 'vdb_dir' not in st.session_state:
    st.session_state.vdb_dir = ""
if 'tenant_id' not in st.session_state:
    st.session_state.tenant_id = ""

# Function to create a clean directory name from embedding model
def clean_model_name(model_name):
    return model_name.replace('/', '--').replace('\\', '--')

# Function to get embedding model
def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

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
        raise Exception(f"Vector database directory for tenant {tenant_id} does not exist at {tenant_vdb_dir}")
    
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

# Summarizer system prompt
SUMMARIZER_SYSTEM_PROMPT_ORIG = """
You are an expert summarizer working within a RAG system. Your task is to create a deep, comprehensive and accurate representation of the provided original information while properly attributing all facts to their sources.

Guidelines:
- Create a clear and coherent summary using neutral and professional language
- Maintain factual accuracy without adding new information
- Maintain exact figures, data points, sections and paragraphs
- For each piece of information, you MUST CITE the original source, i.e. the original document, using the format [Document Name](document_path)
- Return only the plain text summary without any reliminary remarks and without markdown formatting

Useful hint: The citation information [Document Name](document_path) is stored in a dict and provided within the 'metadata'
{{
"content": doc.page_content,
"metadata": {
    "name": doc.metadata['id'],
    "path": doc.metadata['source']
}}
"""

SUMMARIZER_SYSTEM_PROMPT_BUP = """
You are an expert AI summarizer. Create a factual summary from provided documents with EXACT source citations. Follow these rules:

1. **Citation Format**: For citations, ALWAYS use the EXACT format [Source_filename] after each fact. 
You find the Source_filename in the provided metadata with the following structure:
\nContent: some content
\nSource_filename: the corresponding Source_filename
\nSource_path: the corresponding fullpath

2. **Content Rules**:
   - Maintain exact figures, data points, sections and paragraphs
   - No markdown, formulate only plain text and complete sentences
   - NO new information or opinions

**Example Input**:
\nContent: 'The 2025 budget for infrastructure is €4.2M.',
\nSource_filename: 'City_Budget.pdf'
\nSource_path: './some/path/to/City_Budget.pdf'
  
**Example Output**:
The 2025 fiscal plan allocates €4.2 million for infrastructure [City_Budget.pdf].

**Current Task**:
Create a deep, comprehensive and accurate representation of the provided original information:
"""
# Step 1: Choose Embedding Model
with tab1:
    st.header("Select Embedding Model")
    
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
    
    if st.button("Set Embedding Model", key="set_embedding"):
        st.session_state.embedding_model = selected_embedding
        st.success(f"Embedding model set to: {selected_embedding}")

# Step 2: Create Embeddings
with tab2:
    st.header("Create Embeddings")
    
    # Display current embedding model
    st.info(f"Current embedding model: **{st.session_state.embedding_model}**")
    
    # Input for data folder
    data_folder = st.text_input(
        "Data folder path",
        value=st.session_state.data_folder
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
    clean_embed_name = clean_model_name(st.session_state.embedding_model)
    vdb_dir_name = f"{clean_embed_name}--{chunk_size}--{chunk_overlap}"
    vdb_full_path = os.path.join(DATABASE_PATH, vdb_dir_name)
    
    st.info(f"Vector database will be stored at: **{vdb_full_path}**")
    
    # Button to create embeddings
    if st.button("Create Embeddings", key="create_embeddings"):
        with st.spinner("Creating embeddings..."):
            try:
                # Update session state
                st.session_state.data_folder = data_folder
                st.session_state.chunk_size = chunk_size
                st.session_state.chunk_overlap = chunk_overlap
                
                # Get embedding model
                embed_model = get_embedding_model(st.session_state.embedding_model)
                
                # Create embeddings
                dirname, tenant_id = load_embed(
                    folder=data_folder,
                    vdbdir=vdb_full_path,
                    embed_llm=embed_model,
                    c_size=chunk_size,
                    c_overlap=chunk_overlap,
                    #clean=False
                )
                
                # Store the results in session state
                st.session_state.vdb_dir = dirname
                st.session_state.tenant_id = tenant_id
                
                st.success(f"Embeddings created successfully!\nTenant ID: {tenant_id}\nDirectory: {dirname}")
            except Exception as e:
                st.error(f"Error creating embeddings: {str(e)}")

# Step 3: Test Retrieval
with tab3:
    st.header("Test Retrieval")
    
    # Display current embedding and VDB info
    st.info(f"Current embedding model: **{st.session_state.embedding_model}**")
    
    # Initialize language in session state if not present
    if 'summary_language' not in st.session_state:
        st.session_state.summary_language = "English"
    
    if st.session_state.vdb_dir:
        st.success(f"Vector database is ready at: **{st.session_state.vdb_dir}**")
    else:
        st.warning("Please complete Step 2 to create a vector database first.")
    
    # Input for query
    query = st.text_input("Enter your query", value="")
    
    # Select LLM model for summarization
    llm_options = [
        "deepseek-r1:1.5b",
        "deepseek-r1:latest",
        "llama3.1:8b-instruct-q4_0",
        "llama3.2",
        "gemma3:4b",
        "phi4-mini",
        "mistral:instruct",
        "mistrallite",
        "deepseek-r1:70b",
        "qwq",
        "gemma3:27b",
        "llama3.3",
        "llama4:latest",
        "mistral-small",
        "mistral-nemo"
    ]
    
    selected_llm = st.selectbox(
        "Choose an LLM for summarization",
        options=llm_options,
        index=0
    )
    
    # Language selection for summary
    language_options = ["English", "German", "French", "Spanish", "Italian"]
    selected_language = st.selectbox(
        "Choose language for summary",
        options=language_options,
        index=language_options.index(st.session_state.summary_language) if st.session_state.summary_language in language_options else 0
    )
    
    # Update session state with selected language
    if selected_language != st.session_state.summary_language:
        st.session_state.summary_language = selected_language
    
    # Number of results to retrieve
    k_results = st.slider("Number of results to retrieve", min_value=1, max_value=10, value=3)
    
    # Button to perform retrieval
    if st.button("Perform Retrieval", key="perform_retrieval") and query and st.session_state.vdb_dir:
        with st.spinner("Retrieving and summarizing..."):
            try:
                # Get embedding model
                embed_model = get_embedding_model(st.session_state.embedding_model)
                
                # Store the selected language in session state for the graph workflow
                # This needs to happen BEFORE the retrieval to ensure correct language detection
                st.session_state.selected_language = selected_language
                
                # Perform similarity search
                results = similarity_search_for_tenant(
                    tenant_id=st.session_state.tenant_id,
                    embed_llm=embed_model,
                    persist_directory=st.session_state.vdb_dir,
                    similarity="cosine",
                    normal=True,
                    query=query,
                    k=k_results,
                    language=selected_language  # Pass the selected language to the retrieval function
                )

                transformed_results = transform_documents(results)
                print("# of transformed documents:", len(transformed_results))
                
                # Display retrieved documents
                st.subheader("Retrieved Documents")
                
                # Display all documents in the vector database
                st.markdown("**All Documents in Vector Database:**")
                try:
                    # Get all documents in the vector database
                    embed_model = get_embedding_model(st.session_state.embedding_model)
                    all_documents = get_all_documents_in_vectordb(
                        tenant_id=st.session_state.tenant_id,
                        embed_llm=embed_model,
                        persist_directory=st.session_state.vdb_dir
                    )
                    
                    if all_documents:
                        # Display as a formatted list
                        st.write(", ".join([os.path.basename(doc) for doc in all_documents]))
                    else:
                        st.write("No documents found in the vector database.")
                except Exception as e:
                    st.error(f"Error retrieving documents: {str(e)}")
                
                # Display the documents returned from the search query
                #st.markdown("**Documents Retrieved from Query:**")
                #doc_filenames = [os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in results]
                #st.write(", ".join(doc_filenames))
                
                for i, doc in enumerate(results):
                    with st.expander(f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                        st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                        st.write(f"**Content:**\n{doc.page_content}")
                
                # Display the formatted context
                st.subheader("Formatted Context")
                # Create the formatted context in the same way as in source_summarizer_ollama
                formatted_context = transformed_results
                #"\n".join(
                 #   f"Content: {doc['content']}\nSource: {doc['metadata']['name']}\nPath: {doc['metadata']['path']}\n --------------- \n"
                 #   for doc in transformed_results
                #)
                with st.expander("View Formatted Context"):
                    st.text_area("Context used for summarization", formatted_context, height=300, disabled=True)
                
                # Summarize the results
                st.subheader(f"Summary (in {selected_language}) from {st.session_state.vdb_dir} using tenant {st.session_state.tenant_id}")
                with st.spinner(f"Generating summary using {selected_llm} in {selected_language}..."):
                    start_time = time.time()
                    
                    # Store the selected language in session state for the graph workflow
                    if 'summary_language' in st.session_state:
                        st.session_state.selected_language = selected_language
                    
                    # Format the system prompt with the selected language
                    formatted_system_prompt = SUMMARIZER_SYSTEM_PROMPT.format(language=selected_language)
                    
                    # Pass the selected language to the summarizer
                    summary = source_summarizer_ollama(query, transformed_results, selected_language, formatted_system_prompt, selected_llm)
                    end_time = time.time()
                    
                    st.markdown(summary["content"])
                    st.info(f"Summary generated in {end_time - start_time:.2f} seconds using {selected_llm} in {selected_language}")
                    
            except Exception as e:
                st.error(f"Error during retrieval: {str(e)}")
    elif not st.session_state.vdb_dir and st.button("Perform Retrieval", key="perform_retrieval_disabled"):
        st.error("Please complete Step 2 to create a vector database first.")

# Add a footer
st.markdown("---")
st.markdown("*Embedding & Retrieval Testing App for RAG Researcher*")
