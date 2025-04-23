import streamlit as st
import streamlit_nested_layout
import warnings
import logging
import torch
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to Python path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Import ResearcherState directly for better type hinting and consistency
from src.assistant.v1_1.state_v1_1 import ResearcherState
from src.assistant.v1_1.graph_v1_1 import researcher_graph
from src.assistant.v1_1.utils_v1_1 import get_report_structures, process_uploaded_files, clear_cuda_memory
from src.assistant.v1_1.rag_helpers_v1_1 import similarity_search_for_tenant, transform_documents, source_summarizer_ollama
from src.assistant.v1_1.vector_db_v1_1 import get_or_create_vector_db, search_documents, get_embedding_model_path
from src.assistant.v1_1.prompts_v1_1 import SUMMARIZER_SYSTEM_PROMPT
# Use updated import path to avoid deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to original import if package is not installed
    from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Try to import pyperclip, but handle if it's not available
try:
    import pyperclip
    PYPERCLIP_AVAILABLE = True
except (ImportError, Exception):
    PYPERCLIP_AVAILABLE = False

load_dotenv()

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database")

# Define default tenant and collection settings
DEFAULT_TENANT_ID = '2025-04-22_15-41-10'
DEFAULT_COLLECTION_PREFIX = 'collection_'

# Special database configuration for automatic tenant/collection selection
SPECIAL_DB_CONFIG = {
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2--2000--400': {
        'tenant_id': '2025-04-22_15-41-10',
        'collection_name': '2025-04-22_15-41-10'  # Using the actual collection name without prefix
    }
}

# Set page config
st.set_page_config(
    page_title="RAG Deep Researcher",
    page_icon="🔍",
    layout="wide"
)

# Function to create a clean directory name from embedding model
def clean_model_name(model_name):
    return model_name.replace('/', '--').replace('\\', '--')

# Function to extract embedding model name from database directory
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

# Function to get embedding model
def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

def generate_workflow_visualization():
    # Create a custom Mermaid diagram for the simplified workflow
    # The simplified workflow: display_embedding_model_info -> detect_language -> generate_research_queries -> 
    # retrieve_rag_documents -> summarize_query_research -> generate_final_answer
    
    mermaid_representation = """
    graph TD
        START([START]) --> display_embedding_model_info["Display Embedding Model Info"]
        display_embedding_model_info --> detect_language["Detect Language"]
        detect_language --> generate_research_queries["Generate Research Queries"]
        generate_research_queries --> retrieve_rag_documents["Retrieve RAG Documents"]
        retrieve_rag_documents --> summarize_query_research["Summarize Query Research"]
        summarize_query_research --> generate_final_answer["Generate Final Answer"]
        generate_final_answer --> END([END])
        
        classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
        classDef active fill:#d4f1f9,stroke:#333,stroke-width:1px;
        classDef terminal fill:#e9e9e9,stroke:#333,stroke-width:1px;
        class START,END terminal;
    """
    
    # Return the mermaid representation for streamlit mermaid display
    return mermaid_representation

def generate_response(user_input, enable_web_search, report_structure, max_search_queries, report_llm, enable_quality_checker, quality_check_loops=1, use_ext_database=False, selected_database=None, k_results=3):
    """
    Generate response using the researcher agent and stream steps
    If use_ext_database is True, it will use an external database for document retrieval
    The original workflow is always enabled
    """
    # Clear CUDA memory before processing a new query
    clear_cuda_memory()
    
    # Initialize state for the researcher using ResearcherState structure
    initial_state: ResearcherState = {
        "user_query": user_input,
        "research_queries": [],
        "retrieved_documents": {},
        "search_summaries": {},
        "current_position": 0,
        "final_answer": "",
        "detected_language": "",  # Will be populated by language detection
        "additional_context": None  # Optional field for context from document retrieval
    }
    
    # Langgraph researcher config
    config = {"configurable": {
        "enable_web_search": enable_web_search,
        "report_structure": report_structure,
        "max_search_queries": max_search_queries,
        "llm_model": st.session_state.report_llm,  # General purpose LLM model (used for most tasks)
        "report_llm": st.session_state.report_llm,  # Specific LLM for report writing
        "summarization_llm": st.session_state.summarization_llm,  # Specific LLM for summarization
        "enable_quality_checker": enable_quality_checker,
        "quality_check_loops": quality_check_loops,
        "k_results": k_results,  # Number of results to retrieve for each query
        # Don't pass selected_language to force language detection in the graph
        # This ensures the language is detected from the user's query
    }}

    # Start timing the workflow
    start_time = time.time()
    
    # Create the status for the global "Researcher" process
    langgraph_status = st.status("**Researcher Running...**", state="running")

    # Create a placeholder for the elapsed time
    elapsed_time_placeholder = st.empty()
    
    # Define callback function to process and display the state during execution
    def update_callback(state):
        # Get the elapsed time since the start of the workflow
        elapsed_time = time.time() - start_time
        elapsed_time_placeholder.write(f"⏱️ Elapsed time: {elapsed_time:.1f} seconds")
        
        # Get the current step
        step = state.get('current_step', '💤 Idle')
        
        # Update the langgraph status
        langgraph_status.update(label=f"**Researcher Step: {step}**")
        
        # If there's a final_answer, update the status to complete
        if 'final_answer' in state and state['final_answer']:
            langgraph_status.update(state="complete", label="**Research Complete ✅**")
            
            # Display language detection result
            if 'detected_language' in state:
                st.success(f"🌐 Detected language: **{state['detected_language']}**")
            
            # Display the LangGraph workflow visualization
            st.subheader("LangGraph Workflow Visualization")
            try:
                st.markdown(generate_workflow_visualization())
            except Exception as e:
                st.error(f"Error generating workflow visualization: {str(e)}")
            
            # First display the final answer
            if 'final_answer' in state and state['final_answer']:
                st.subheader("Final Answer")
                st.markdown(state['final_answer'])
                
            # Display important research steps results using expanders
            # Use the field names from ResearcherState for consistency
            if 'research_queries' in state and 'retrieved_documents' in state and 'search_summaries' in state:
                st.subheader("Research Process Details")
                
                research_queries = state['research_queries']
                retrieved_documents = state['retrieved_documents']
                search_summaries = state['search_summaries']
                
                # Display each query with its retrieved documents and summary
                for i, query in enumerate(research_queries):
                    with st.expander(f"Research Query {i+1}: {query}"):
                        st.markdown(f"**Query:** {query}")
                        
                        # Display retrieved documents
                        documents = retrieved_documents.get(query, [])
                        st.markdown(f"### Retrieved Documents ({len(documents)})")
                        
                        if documents:
                            for j, doc in enumerate(documents):
                                with st.expander(f"Document {j+1}: {doc.metadata.get('source', 'Unknown')}"):
                                    st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                    st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                                    st.write(f"**Content:**\n{doc.page_content}")
                        else:
                            st.warning("No documents retrieved for this query.")
                        
                        # Display summary
                        st.markdown("### Summary of Retrieved Documents")
                        query_summaries = search_summaries.get(query, [])
                        if query_summaries:
                            for summary_doc in query_summaries:
                                st.markdown(summary_doc.page_content)
                        else:
                            st.warning("No summary available for this query.")
            
            # Display debugging information in expanders
            st.header("🔬 Research Process Details")
            
            # Display research queries in an expander
            with st.expander("1️⃣ Generated Research Queries"):
                research_queries = state.get("research_queries", [])
                if research_queries:
                    for i, query in enumerate(research_queries):
                        st.markdown(f"**Query {i+1}:** {query}")
                else:
                    st.warning("No research queries were generated.")
            
            # Display search summaries in an expander
            with st.expander("2️⃣ Search Summaries"):
                search_summaries = state.get("search_summaries", {})
                if search_summaries and len(search_summaries) > 0:
                    for query, summaries in search_summaries.items():
                        st.markdown(f"#### For Query: '{query}'")
                        for i, summary in enumerate(summaries):
                            st.markdown(f"**Summary {i+1}:**")
                            st.write(summary.page_content)
                            st.markdown("---")
                else:
                    st.warning("No search summaries were found.")
            
            # Display the final answer in an expander and in the main area
            with st.expander("3️⃣ Final Answer Generation"):
                st.info("Final answer was generated using all available research summaries.")
                st.code(state["final_answer"][:500] + "..." if len(state["final_answer"]) > 500 else state["final_answer"], language="markdown")
            
            # Display the final answer prominently in markdown format
            st.header("📝 Final Research Report")
            st.markdown(state["final_answer"])
                
            # Return the final state for further processing or display
            return {
                "steps": state,
                "final_answer": state["final_answer"]
            }
            
        # Return None to continue processing
        return None
    
    # If using external database, perform retrieval and summarization first
    if use_ext_database and selected_database:
        # Create the status for the retrieval process
        retrieval_status = st.status("**Document Retrieval...**", state="running")
        
        try:
            with retrieval_status:
                st.write("### Document Retrieval")
                
                # Display embedding model information
                embedding_model_name = extract_embedding_model(selected_database)
                st.info(f"**Selected Database:** {selected_database}")
                st.info(f"**Embedding Model:** {embedding_model_name}")
                
                # Get embedding model and update the Configuration to use this embedding model
                if embedding_model_name:
                    embed_model = get_embedding_model(embedding_model_name)
                    
                    # Update the global configuration to use this embedding model
                    from src.assistant.v1_1.configuration_v1_1 import update_embedding_model
                    update_embedding_model(embedding_model_name)
                    
                    # Print confirmation of embedding model update
                    st.write(f"✅ Updated embedding model to: {embedding_model_name}")
                
                # Get tenant ID from the database directory or use special configuration
                database_path = os.path.join(DATABASE_PATH, selected_database)
                
                # Check if this database has a special configuration
                if selected_database in SPECIAL_DB_CONFIG:
                    # Use the special configuration
                    tenant_id = SPECIAL_DB_CONFIG[selected_database]['tenant_id']
                    collection_name = SPECIAL_DB_CONFIG[selected_database]['collection_name']
                    st.write(f"**Using preconfigured tenant ID:** {tenant_id}")
                    st.write(f"**Using preconfigured collection:** {collection_name}")
                else:
                    # Use the standard approach of finding tenant directories
                    tenant_dirs = [d for d in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, d))]
                    
                    if not tenant_dirs:
                        st.error(f"No tenant directories found in {database_path}")
                        retrieval_status.update(state="error", label=f"**Error: No tenant directories found**")
                        return None
                    else:
                        st.write(f"**Tenant directories found:** {tenant_dirs}")
                        tenant_id = DEFAULT_TENANT_ID
                        collection_name = f"{DEFAULT_COLLECTION_PREFIX}{tenant_id}"
                        st.write(f"**Using tenant ID:** {tenant_id}")
                    
                    # Perform similarity search
                    st.write("### Retrieving Documents")
                    with st.spinner("Performing similarity search..."):
                        # Detect language for the query first
                        from src.assistant.v1_1.graph_v1_1 import detect_language
                        language_result = detect_language({"user_query": user_input}, {"configurable": {"llm_model": report_llm}})
                        detected_language = language_result.get("detected_language", "English")
                        
                        # Print detected language
                        st.write(f"ℹ️ Detected language: **{detected_language}**")
                        
                        # Pass the detected language to the similarity search
                        # If we have a specific collection name from special config, use it
                        if selected_database in SPECIAL_DB_CONFIG and 'collection_name' in SPECIAL_DB_CONFIG[selected_database]:
                            # Use the specific collection name from the configuration
                            results = similarity_search_for_tenant(
                                tenant_id=tenant_id,
                                embed_llm=embed_model,
                                persist_directory=database_path,
                                similarity="cosine",
                                normal=True,
                                query=user_input,
                                k=k_results,
                                language=detected_language,  # Pass the detected language
                                collection_name=SPECIAL_DB_CONFIG[selected_database]['collection_name']  # Use the specific collection
                            )
                        else:
                            # Use the default approach (tenant_id-based collection)
                            results = similarity_search_for_tenant(
                                tenant_id=tenant_id,
                                embed_llm=embed_model,
                                persist_directory=database_path,
                                similarity="cosine",
                                normal=True,
                                query=user_input,
                                k=k_results,
                                language=detected_language  # Pass the detected language
                            )
                        
                    # Transform documents
                    transformed_results = transform_documents(results)
                    st.write(f"Retrieved {len(transformed_results)} documents")
                    
                    # Display retrieved documents
                    st.subheader("Retrieved Documents")
                    for i, doc in enumerate(results):
                        with st.expander(f"Document {i+1}: {doc.metadata.get('source', 'Unknown')}"):
                            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                            st.write(f"**Content:**\n{doc.page_content}")
                    
                    # We already detected the language above, reuse it for summarization
                    # Make sure detected_language is set to the correct value (not 'en' but 'English')
                    detected_language = language_result.get("detected_language", "English")
                    
                    # Summarize the results using the summarization LLM
                    st.subheader("Document Summary")
                    with st.spinner(f"Generating summary using {st.session_state.summarization_llm} in {detected_language}..."):
                        start_time_summary = time.time()
                        # Format the system prompt with the detected language
                        formatted_system_prompt = SUMMARIZER_SYSTEM_PROMPT.format(language=detected_language)
                        summary = source_summarizer_ollama(user_input, transformed_results, detected_language, formatted_system_prompt, st.session_state.summarization_llm)
                        end_time_summary = time.time()
                        
                        st.markdown(summary["content"])
                        st.info(f"Summary generated in {end_time_summary - start_time_summary:.2f} seconds using {st.session_state.summarization_llm} in {detected_language}")
                    
                    # Update the user instructions with the summary to enhance the research
                    initial_state["additional_context"] = summary['content']
                    
                    # Update status to complete
                    retrieval_status.update(state="complete", label="**Document Retrieval Complete**")
                
        except Exception as e:
            # Update status to error
            if 'retrieval_status' in locals():
                retrieval_status.update(state="error", label=f"**Retrieval Error: {str(e)}**")
            st.error(f"Error during document retrieval: {str(e)}")
    
    # Continue with the original workflow regardless of whether external database was used
    try:
        # Display the workflow visualization
        with langgraph_status:
            st.write("### LangGraph Workflow Visualization")
            
            # Display embedding model information
            from src.assistant.v1_1.utils_v1_1 import get_configured_llm_model
            from src.assistant.v1_1.configuration_v1_1 import get_config_instance
            config = get_config_instance()
            embedding_model_name = config.embedding_model
            st.info(f"🤖 Using embedding model: **{embedding_model_name}**")
            
            # If we're using an external database with a specific embedding model, make sure it's displayed correctly
            if use_ext_database and selected_database and 'embedding_model_name' in locals() and embedding_model_name:
                # Check if the configuration's embedding model matches what we expect from the database
                if config.embedding_model != embedding_model_name:
                    st.warning(f"⚠️ Configuration embedding model ({config.embedding_model}) doesn't match database embedding model ({embedding_model_name}). Using database model.")
                    # Force update the embedding model again to ensure it's correct
                    from src.assistant.v1_1.configuration_v1_1 import update_embedding_model
                    update_embedding_model(embedding_model_name)
                    embedding_model = embedding_model_name
            
            # Get the embedding model from the configuration
            from src.assistant.v1_1.configuration_v1_1 import get_config_instance
            config_instance = get_config_instance()
            st.info(f"**Embedding Model:** {config_instance.embedding_model}")
            
            # Display the mermaid diagram
            st.markdown(f"```mermaid\n{generate_workflow_visualization()}\n```")
            
            # Display the actual langgraph visualization
            st.write("### LangGraph Workflow")
            # Just use the mermaid visualization we already have
            try:
                st.markdown(f"```mermaid\n{generate_workflow_visualization()}\n```")
            except Exception as e:
                st.error(f"Error generating graph visualization: {str(e)}")
            
            st.write("---")
            
            # Force order of expanders by creating them before iteration
            generate_queries_expander = st.expander("Generate Research Queries", expanded=False)
            search_queries_expander = st.expander("Search Queries", expanded=True)
            filter_summaries_expander = st.expander("Filter Summaries", expanded=False)
            rank_summaries_expander = st.expander("Rank Summaries", expanded=False)
            final_answer_expander = st.expander("Generate Final Answer", expanded=False)
            # Run the researcher graph and stream outputs 
            try:
                # Record the start time of the workflow
                st.session_state.workflow_start_time = time.time()
                
                # Create containers for displaying results as they come in
                results_container = st.container()
                
                # Initialize state tracking using ResearcherState structure as a template
                # This ensures we're using the same field names consistently
                current_state: ResearcherState = {
                    "user_query": initial_state["user_query"],
                    "research_queries": [],
                    "retrieved_documents": {},
                    "search_summaries": {},
                    "current_position": 0,
                    "final_answer": "",
                    "detected_language": "",
                    "additional_context": None
                }
                research_queries_displayed = False
                documents_displayed = {}  # Track which queries have had documents displayed
                summaries_displayed = {}  # Track which queries have had summaries displayed
                
                # Variable to store detected language
                language = ""
                
                # Initialize and run the graph with streaming
                researcher_instance = researcher_graph.compile()
                
                with results_container:
                    st.subheader("Research Steps Results")
                
                for output in researcher_instance.stream(
                    initial_state,
                    config=config
                ):
                    # Update elapsed time display
                    current_time = time.time()
                    elapsed_seconds = current_time - start_time
                    elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
                    elapsed_time_placeholder.info(f"⏱️ Elapsed time: {elapsed_time}")
                    
                    # Update current step in the status widget
                    if 'current_step' in output:
                        step = output['current_step']
                        langgraph_status.update(label=f"**Researcher Step: {step}**")
                    
                    # Display language detection result when it becomes available
                    if 'detected_language' in output:
                        with results_container:
                            st.success(f"Detected language: **{output['detected_language']}**")
                            # Store detected language for prompts
                            language = output['detected_language']
                    
                    # Display research queries as they become available
                    if 'research_queries' in output and not research_queries_displayed:
                        research_queries = output['research_queries']
                        with results_container:
                            for i, query in enumerate(research_queries):
                                with st.expander(f"Research Query {i+1}: {query}"):
                                    st.markdown(f"**Query:** {query}")
                                    st.info("Retrieving documents...")
                        research_queries_displayed = True
                    
                    # Merge the new output into the current state
                    current_state.update(output)
                    
                    # Display language detection result
                    # This comes from ResearcherState.detected_language field
                    if 'detected_language' in output:
                        with results_container:
                            st.success(f"🌐 Detected language: **{output['detected_language']}**")
                            # Store detected language for prompts
                            language = output['detected_language']
                            
                    # Display research queries as they become available
                    if 'research_queries' in output and not research_queries_displayed:
                        research_queries = output['research_queries']
                        with results_container:
                            for i, query in enumerate(research_queries):
                                with st.expander(f"Research Query {i+1}: {query}"):
                                    st.markdown(f"**Query:** {query}")
                                    st.info("Retrieving documents...")
                        research_queries_displayed = True
                    
                    # Display documents as they become available for each query
                    if 'all_query_documents' in output and 'research_queries' in current_state:
                        all_query_documents = output['all_query_documents']
                        research_queries = current_state['research_queries']
                        
                        # Display documents for each query that hasn't been displayed yet
                        for query in research_queries:
                            if query in all_query_documents and query not in documents_displayed:
                                documents = all_query_documents[query]
                                
                                # Find the expander for this query and update it
                                with results_container:
                                    for i, q in enumerate(research_queries):
                                        if q == query:
                                            with st.expander(f"Research Query {i+1}: {query}", expanded=True):
                                                st.markdown(f"**Query:** {query}")
                                                
                                                # Display retrieved documents
                                                st.markdown(f"### Retrieved Documents ({len(documents)})")
                                                
                                                if documents:
                                                    for j, doc in enumerate(documents):
                                                        with st.expander(f"Document {j+1}: {doc.metadata.get('source', 'Unknown')}"):
                                                            st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                                            st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                                                            st.write(f"**Content:**\n{doc.page_content}")
                                                else:
                                                    st.warning("No documents retrieved for this query.")
                                                    
                                                st.info("Generating summary...")
                                
                                # Mark this query's documents as displayed
                                documents_displayed[query] = True
                    
                    # Display summaries as they become available
                    if 'search_summaries' in output:
                        search_summaries = output['search_summaries']
                        research_queries = current_state.get('research_queries', [])
                        
                        # Create a mapping from query to summary for easier lookup
                        summary_map = {summary['query']: summary['content'] for summary in search_summaries}
                        
                        # Update expanders with summaries
                        with results_container:
                            for i, query in enumerate(research_queries):
                                if query in summary_map and query not in summaries_displayed:
                                    with st.expander(f"Research Query {i+1}: {query}", expanded=True):
                                        st.markdown(f"**Query:** {query}")
                                        
                                        # Show documents if we have them
                                        if query in documents_displayed:
                                            docs = current_state['all_query_documents'].get(query, [])
                                            st.markdown(f"### Retrieved Documents ({len(docs)})")
                                            
                                            if docs:
                                                for j, doc in enumerate(docs):
                                                    with st.expander(f"Document {j+1}: {doc.metadata.get('source', 'Unknown')}"):
                                                        st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}")
                                                        st.write(f"**Path:** {doc.metadata.get('path', 'Unknown')}")
                                                        st.write(f"**Content:**\n{doc.page_content}")
                                            else:
                                                st.warning("No documents retrieved for this query.")
                                        
                                        # Display summary
                                        st.markdown("### Summary of Retrieved Documents")
                                        st.markdown(summary_map[query])
                                    
                                    # Mark this query's summary as displayed
                                    summaries_displayed[query] = True
                    
                    # If there's a final answer, update the status to complete
                    if 'final_answer' in output:
                        langgraph_status.update(state="complete", label="**Research Complete ✅**")
                        
                        # Display LangGraph workflow visualization
                        with results_container:
                            st.subheader("LangGraph Workflow Visualization")
                            try:
                                st.markdown(generate_workflow_visualization())
                            except Exception as e:
                                st.error(f"Error generating workflow visualization: {str(e)}")
                
                # Return the final state when streaming completes
                result = {
                    "steps": current_state,
                    "final_answer": current_state.get("final_answer", "No final answer generated")
                }
                
                # Update elapsed time one final time
                elapsed_time = time.time() - start_time
                elapsed_time_placeholder.write(f"⏱️ Total elapsed time: {elapsed_time:.1f} seconds")
                
                # Return the result
                return result
            except Exception as e:
                st.error(f"Error during research: {str(e)}")
                langgraph_status.update(state="error", label=f"**Research Error: {str(e)}**")
                return {
                    "final_answer": f"Research encountered an error: {str(e)}",
                    "steps": {}
                }
    
    finally:
        # Final update of elapsed time
        if 'start_time' in locals():
            current_time = time.time()
            elapsed_seconds = current_time - start_time
            elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
            elapsed_time_placeholder.info(f"⏱️ Total elapsed time: {elapsed_time}")

def clear_chat():
    st.session_state.messages = []
    st.session_state.processing_complete = False
    st.session_state.uploader_key = 0
    if 'selected_database' in st.session_state:
        st.session_state.selected_database = None
    if 'use_ext_database' in st.session_state:
        st.session_state.use_ext_database = False
    if 'summarization_llm' in st.session_state:
        st.session_state.summarization_llm = "deepseek-r1:latest"
    if 'report_llm' in st.session_state:
        st.session_state.report_llm = "deepseek-r1:latest"

def copy_to_clipboard(text):
    """Safely copy text to clipboard if pyperclip is available"""
    if PYPERCLIP_AVAILABLE:
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False
    return False

def main():
    # Create header with two columns
    header_col1, header_col2 = st.columns([0.6, 0.4])
    with header_col1:
        st.title("RAG Deep Researcher")
    with header_col2:
        st.image("Header für Chatbot.png", use_container_width=True)

    # Initialize session states
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_report_structure" not in st.session_state:
        st.session_state.selected_report_structure = None
    if "max_search_queries" not in st.session_state:
        st.session_state.max_search_queries = 3  # Default value of 3
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False  # Tracks if files are uploaded but not processed
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "deepseek-r1:latest"  # Default LLM model
    if "enable_web_search" not in st.session_state:
        st.session_state.enable_web_search = True  # Default web search setting
    if "enable_quality_checker" not in st.session_state:
        st.session_state.enable_quality_checker = True  # Default quality checker setting
    if "workflow_start_time" not in st.session_state:
        st.session_state.workflow_start_time = None  # For tracking workflow elapsed time
    if "use_ext_database" not in st.session_state:
        st.session_state.use_ext_database = False  # Default external database setting
    if "selected_database" not in st.session_state:
        st.session_state.selected_database = None  # Default selected database
    if "k_results" not in st.session_state:
        st.session_state.k_results = 3  # Default number of results to retrieve
    if "summarization_llm" not in st.session_state:
        st.session_state.summarization_llm = "deepseek-r1:latest"  # Default summarization LLM
    if "report_llm" not in st.session_state:
        st.session_state.report_llm = "deepseek-r1:latest"  # Default report writing LLM
    if "detected_language" not in st.session_state:
        st.session_state.detected_language = ""  # Will be populated by language detection

    # Sidebar configuration
    st.sidebar.title("Research Settings")

    # Add Report LLM model selector to sidebar
    llm_models = ["deepseek-r1:latest", "deepseek-r1:70b", "qwq", "gemma3:27b", "mistral-small:latest", 
                 "deepseek-r1:1.5b", "llama3.1:8b-instruct-q4_0", "llama3.2", "llama3.3","gemma3:4b", "phi4-mini", 
                 "mistral:instruct", "mistrallite"]
    
    st.sidebar.subheader("LLM Models")
    
    # Report writing LLM
    st.session_state.report_llm = st.sidebar.selectbox(
        "Report Writing LLM",
        options=llm_models,
        index=llm_models.index(st.session_state.report_llm) if st.session_state.report_llm in llm_models else 0,
        help="Choose the LLM model to use for final report generation; good options: deepseek-r1:latest (fast), qwq or mistral-small:latest (medium), llama3.3 or deepseek-r1:70b (deep but slow)"
    )
    
    # Summarization LLM
    st.session_state.summarization_llm = st.sidebar.selectbox(
        "Summarization LLM",
        options=llm_models,
        index=llm_models.index(st.session_state.summarization_llm) if st.session_state.summarization_llm in llm_models else 0,
        help="Choose the LLM model to use for document summarization; good options: llama3.2 (fast and accurate), qwq (deep but slow)"
    )
    



    # Add report structure selector to sidebar
    st.sidebar.subheader("Report Structure")
    report_structures = get_report_structures()
    default_report = "standard report"

    selected_structure = st.sidebar.selectbox(
        "Select Report Structure",
        options=list(report_structures.keys()),
        index=list(map(str.lower, report_structures.keys())).index(default_report)
    )

    st.session_state.selected_report_structure = report_structures[selected_structure]

    # Maximum search queries input
    st.sidebar.subheader("Search Settings")
    st.session_state.max_search_queries = st.sidebar.number_input(
        "Max Number of Search Queries",
        min_value=1,
        max_value=10,
        value=st.session_state.max_search_queries,
        help="Set the maximum number of search queries to be made. (1-10)"
    )
    
    # Language will be automatically detected from the user's query
    # Remove the language selection dropdown to ensure automatic detection
    if "selected_language" not in st.session_state:
        st.session_state.selected_language = None  # Set to None to force language detection
    
    # Enable web search checkbox
    st.session_state.enable_web_search = st.sidebar.checkbox("Enable Web Search", value=st.session_state.enable_web_search)
    
    # Enable quality checker checkbox
    st.sidebar.subheader("Quality Control")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        st.session_state.enable_quality_checker = st.sidebar.checkbox("Enable Quality Checker", value=st.session_state.enable_quality_checker)
    
    # Add quality check loops input
    with col2:
        if "quality_check_loops" not in st.session_state:
            st.session_state.quality_check_loops = 1  # Default value
        st.session_state.quality_check_loops = st.sidebar.number_input(
            "Loops",
            min_value=1,
            max_value=5,
            value=st.session_state.quality_check_loops,
            help="Number of quality check improvement loops"
        )
    
    # Add Retrieval options
    st.sidebar.markdown("---")
    st.sidebar.subheader("Retrieval Options")
    
    # Enable external database checkbox
    st.session_state.use_ext_database = st.sidebar.checkbox("Use ext. Database", value=st.session_state.use_ext_database, 
                                                 help="Use an existing database for document retrieval")
    
    # Database selection
    if st.session_state.use_ext_database:
        # Get available databases
        database_dir = Path(DATABASE_PATH)
        database_options = [d.name for d in database_dir.iterdir() if d.is_dir()]
        
        if database_options:
            # Select database
            selected_db = st.sidebar.selectbox(
                "Select Database",
                options=database_options,
                index=database_options.index(st.session_state.selected_database) if st.session_state.selected_database in database_options else 0,
                help="Choose a database to use for retrieval"
            )
            st.session_state.selected_database = selected_db
            
            # Extract and update embedding model from database name
            embedding_model_name = extract_embedding_model(selected_db)
            if embedding_model_name:
                # Update the global configuration to use this embedding model
                from src.assistant.v1_1.configuration_v1_1 import update_embedding_model
                update_embedding_model(embedding_model_name)
                st.sidebar.info(f"Selected Database: {selected_db}")
                st.sidebar.success(f"Updated embedding model to: {embedding_model_name}")
            
            # Number of results to retrieve
            st.session_state.k_results = st.sidebar.slider(
                "Number of results to retrieve", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.k_results
            )
        else:
            st.sidebar.warning("No databases found. Please upload documents first.")
            st.session_state.use_ext_database = False
    
    # Clear chat button in a single column
    if st.button("Clear Chat", use_container_width=True):
        clear_chat()
        st.rerun()

    # Instructions dropdown below the clear chat button
    with st.expander("📝 How to use this app", expanded=False):
        st.markdown("""
        ### How to Use the RAG Deep Researcher
        
        1. **Choose RAG Method**:
           - **Option 1**: Upload your own documents using the sidebar
           - **Option 2**: Select an existing database from the dropdown
        
        2. **Configure Settings** (Optional):
           - Select an LLM model from the dropdown menu
           - Choose a report structure template
           - Set the maximum number of search queries (1-10)
           - Enable web search if needed
        
        3. **Ask Your Question**:
           - Type your research question in the chat input
           - Be specific and clear about what information you need
        
        4. **Review the Results**:
           - The system will retrieve relevant documents and summarize findings
           - A comprehensive final answer will be provided
           - You can copy the response using the clipboard button
        
        5. **Start a New Research**:
           - Click "Clear Chat" to start a new research session
        """)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=False)  # Use markdown for proper rendering

            # Show copy button only for AI messages at the bottom
            if message["role"] == "assistant" and PYPERCLIP_AVAILABLE:
                if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                    copy_to_clipboard(message["content"])

    # Chat input and response handling
    if user_input := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input, unsafe_allow_html=False)  # Use markdown for proper rendering

        # Generate and display assistant response
        report_structure = st.session_state.selected_report_structure["content"]
        # Execute the summarization search with callback
        # Pass parameters consistently based on ResearcherState structure
        result = generate_response(
            user_input=user_input,  # Maps to user_query in ResearcherState
            enable_web_search=st.session_state.enable_web_search,
            report_structure=report_structure,
            max_search_queries=st.session_state.max_search_queries,
            report_llm=st.session_state.report_llm,
            enable_quality_checker=st.session_state.enable_quality_checker,
            quality_check_loops=st.session_state.quality_check_loops,
            use_ext_database=st.session_state.use_ext_database,
            selected_database=st.session_state.selected_database if st.session_state.use_ext_database else None,
            k_results=st.session_state.k_results
        )

        # Store assistant message - only store the final answer part for the chat history
        st.session_state.messages.append({"role": "assistant", "content": result["final_answer"] if isinstance(result, dict) and "final_answer" in result else result})

        with st.chat_message("assistant"):
            try:
                # Get the final answer content - since we're now getting plain markdown text directly
                if isinstance(result, dict) and 'final_answer' in result:
                    final_answer = result['final_answer']
                else:
                    final_answer = str(result)
                
                # Display with markdown for proper formatting
                try:
                    # Check if final_answer is a dictionary with 'final_answer' key
                    if isinstance(final_answer, dict) and 'final_answer' in final_answer:
                        st.markdown(final_answer['final_answer'], unsafe_allow_html=False)
                    else:
                        # If it's a string or any other format, display directly
                        st.markdown(final_answer, unsafe_allow_html=False)
                except Exception as e:
                    st.error(f"Error displaying response: {str(e)}")
                    st.markdown(str(final_answer), unsafe_allow_html=False)
                
                # Add an expander to display the debug info (but hide it by default)
                with st.expander("🔍 Debug Information", expanded=False):
                    # If result is a dict with steps, display it nicely
                    if isinstance(result, dict) and 'steps' in result:
                        st.json(result['steps'])
                    else:
                        # Otherwise display the entire response
                        st.json(result)
            except Exception as e:
                st.error(f"Error displaying response: {str(e)}")
                st.markdown(str(result), unsafe_allow_html=False)

            # Copy button below the AI message
            if PYPERCLIP_AVAILABLE:
                if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                    copy_to_clipboard(result["final_answer"] if isinstance(result, dict) and "final_answer" in result else result)

    # Upload file logic
    if not st.session_state.use_ext_database:  # Only show upload option when not using external database
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload Documents")
        
        uploaded_files = st.sidebar.file_uploader(
            "Upload New Documents",
            type=["pdf", "txt", "csv", "md"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}"
        )

        # Check if files are uploaded but not yet processed
        if uploaded_files:
            st.session_state.files_ready = True  # Mark that files are available
            st.session_state.processing_complete = False  # Reset processing status

        # Display the "Process Files" button **only if files are uploaded but not processed**
        if st.session_state.files_ready and not st.session_state.processing_complete:
            process_button_placeholder = st.sidebar.empty()  # Placeholder for dynamic updates

            with process_button_placeholder.container():
                process_clicked = st.button("Process Files", use_container_width=True)

            if process_clicked:
                with process_button_placeholder:
                    with st.status("Processing files...", expanded=False) as status:
                        # Process files
                        if process_uploaded_files(uploaded_files):
                            st.session_state.processing_complete = True
                            st.session_state.files_ready = False  # Reset files ready flag
                            st.session_state.uploader_key += 1  # Reset uploader to allow new uploads

                        status.update(label="Files processed successfully!", state="complete", expanded=False)

        # Display green checkbox when processing is complete
        if st.session_state.processing_complete:
            st.sidebar.success("✔️ Files processed and ready to use")

if __name__ == "__main__":
    main()