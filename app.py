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

# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("streamlit").setLevel(logging.ERROR)

from src.assistant.graph import researcher, researcher_graph
from src.assistant.utils import get_report_structures, process_uploaded_files, clear_cuda_memory
from src.assistant.rag_helpers import load_embed, similarity_search_for_tenant, transform_documents, source_summarizer_ollama
from src.assistant.prompts import SUMMARIZER_SYSTEM_PROMPT
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
DEFAULT_DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "database")

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
    # to 'sentence-transformers/all-mpnet-base-v2'
    parts = db_dir_name.split('--')
    if len(parts) >= 2:
        return parts[0].replace('--', '/') + '/' + parts[1]
    return None

# Function to get embedding model
def get_embedding_model(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}
    )

def generate_workflow_visualization():
    """
    Generate a visualization of the langgraph workflow using mermaid
    """
    # Create a mermaid diagram representation of the graph
    mermaid_code = "graph TD\n"
    
    # Add nodes
    mermaid_code += "  A[Display Embedding Model] --> B[Detect Language]\n"
    mermaid_code += "  B --> C[Generate Research Queries]\n"
    mermaid_code += "  C --> D[Search Queries]\n"
    mermaid_code += "  D --> E[Search & Summarize Query]\n"
    mermaid_code += "  E -->|More Queries| D\n"
    mermaid_code += "  E -->|No More Queries| F[Filter Search Summaries]\n"
    mermaid_code += "  F --> G[Rank Search Summaries]\n"
    mermaid_code += "  G --> H[Generate Final Answer]\n"
    
    # Add subgraph for Search & Summarize Query with clear quality check connection
    mermaid_code += "  subgraph Search & Summarize Query\n"
    mermaid_code += "    E1[Retrieve RAG Documents] --> E2[Evaluate Documents]\n"
    mermaid_code += "    E2 -->|Relevant| E4[Summarize Research]\n"
    mermaid_code += "    E2 -->|Not Relevant| E3[Web Research]\n"
    mermaid_code += "    E3 --> E4\n"
    
    # Quality check process with loops
    mermaid_code += "    subgraph Quality Check Process\n"
    mermaid_code += "      E5{Quality Check Enabled?}\n"
    mermaid_code += "      E6[Quality Check]\n"
    mermaid_code += "      E7[Improve Summary]\n"
    mermaid_code += "      E5 -->|Yes| E6\n"
    mermaid_code += "      E6 -->|Needs Improvement & Loops Remaining| E7\n"
    mermaid_code += "      E7 --> E6\n"
    mermaid_code += "    end\n"
    
    mermaid_code += "    E4 --> E5\n"
    mermaid_code += "    E5 -->|No| E8[Return Summary]\n"
    mermaid_code += "    E6 -->|Sufficient or No Loops Remaining| E8\n"
    mermaid_code += "  end\n"
    
    return mermaid_code

def generate_langgraph_visualization():
    """
    Generate a visualization of the LangGraph workflow using NetworkX
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Define the main workflow nodes
        main_nodes = [
            "START",
            "display_embedding_model_info",
            "detect_language",
            "generate_research_queries",
            "search_queries",
            "search_and_summarize_query",
            "filter_search_summaries",
            "rank_search_summaries",
            "generate_final_answer",
            "END"
        ]
        
        # Add main nodes to the graph
        for node in main_nodes:
            G.add_node(node, type="main")
        
        # Define the subgraph nodes for query search
        subgraph_nodes = [
            "retrieve_rag_documents",
            "evaluate_retrieved_documents",
            "web_research",
            "summarize_query_research",
            "quality_check_summary",
            "improve_summary"
        ]
        
        # Add subgraph nodes to the graph
        for node in subgraph_nodes:
            G.add_node(node, type="subgraph")
        
        # Define main workflow edges
        main_edges = [
            ("START", "display_embedding_model_info"),
            ("display_embedding_model_info", "detect_language"),
            ("detect_language", "generate_research_queries"),
            ("generate_research_queries", "search_queries"),
            ("search_queries", "search_and_summarize_query"),
            ("search_and_summarize_query", "search_queries"),
            ("search_and_summarize_query", "filter_search_summaries"),
            ("filter_search_summaries", "rank_search_summaries"),
            ("rank_search_summaries", "generate_final_answer"),
            ("generate_final_answer", "END")
        ]
        
        # Add main edges to the graph
        G.add_edges_from(main_edges)
        
        # Define subgraph edges
        subgraph_edges = [
            ("retrieve_rag_documents", "evaluate_retrieved_documents"),
            ("evaluate_retrieved_documents", "web_research"),
            ("evaluate_retrieved_documents", "summarize_query_research"),
            ("web_research", "summarize_query_research"),
            ("summarize_query_research", "quality_check_summary"),
            ("quality_check_summary", "improve_summary"),
            ("improve_summary", "quality_check_summary")
        ]
        
        # Create a figure with two subplots - one for main workflow and one for subgraph
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), height_ratios=[2, 1])
        
        # Create a subgraph for the main workflow
        main_graph = G.subgraph(main_nodes)
        
        # Position nodes for main workflow
        main_pos = nx.spring_layout(main_graph, seed=42)
        
        # Draw main workflow
        nx.draw_networkx_nodes(main_graph, main_pos, ax=ax1, node_color='lightblue', node_size=2000)
        nx.draw_networkx_labels(main_graph, main_pos, ax=ax1, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(main_graph, main_pos, ax=ax1, arrows=True, arrowsize=20, edge_color='gray')
        
        # Add edge labels for conditional transitions in main workflow
        main_edge_labels = {
            ("search_and_summarize_query", "search_queries"): "More Queries",
            ("search_and_summarize_query", "filter_search_summaries"): "No More Queries"
        }
        nx.draw_networkx_edge_labels(main_graph, main_pos, edge_labels=main_edge_labels, ax=ax1, font_size=8)
        
        # Create a subgraph for the query search
        query_graph = G.subgraph(subgraph_nodes)
        
        # Position nodes for query search subgraph
        query_pos = nx.spring_layout(query_graph, seed=42)
        
        # Draw query search subgraph
        nx.draw_networkx_nodes(query_graph, query_pos, ax=ax2, node_color='lightgreen', node_size=2000)
        nx.draw_networkx_labels(query_graph, query_pos, ax=ax2, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(query_graph, query_pos, ax=ax2, arrows=True, arrowsize=20, edge_color='gray')
        
        # Add edge labels for conditional transitions in query search subgraph
        query_edge_labels = {
            ("evaluate_retrieved_documents", "web_research"): "Not Relevant",
            ("evaluate_retrieved_documents", "summarize_query_research"): "Relevant",
            ("quality_check_summary", "improve_summary"): "Needs Improvement"
        }
        nx.draw_networkx_edge_labels(query_graph, query_pos, edge_labels=query_edge_labels, ax=ax2, font_size=8)
        
        # Set titles for subplots
        ax1.set_title("Main LangGraph Workflow", fontsize=14, fontweight='bold')
        ax2.set_title("Query Search Subgraph", fontsize=14, fontweight='bold')
        
        # Remove axis
        ax1.axis('off')
        ax2.axis('off')
        
        # Add embedding model info
        from src.assistant.configuration import Configuration
        config = Configuration()
        embedding_model = config.embedding_model
        plt.figtext(0.5, 0.98, f"Embedding Model: {embedding_model}", ha="center", fontsize=12, 
                   bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save the figure
        plt.tight_layout()
        temp_file_path = "langgraph_workflow.png"
        plt.savefig(temp_file_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return temp_file_path
    except Exception as e:
        # If visualization fails, return the error
        st.error(f"Error generating visualization: {str(e)}")
        return None

def generate_response(user_input, enable_web_search, report_structure, max_search_queries, report_llm, enable_quality_checker, quality_check_loops=1, use_ext_database=False, selected_database=None, k_results=3):
    """
    Generate response using the researcher agent and stream steps
    If use_ext_database is True, it will use an external database for document retrieval
    The original workflow is always enabled
    """
    # Clear CUDA memory before processing a new query
    clear_cuda_memory()
    
    # Initialize state for the researcher
    initial_state = {
        "user_instructions": user_input,
        "quality_check_loops": quality_check_loops  # Add quality_check_loops to the initial state
    }
    
    # Langgraph researcher config
    config = {"configurable": {
        "enable_web_search": enable_web_search,
        "report_structure": report_structure,
        "max_search_queries": max_search_queries,
        "llm_model": report_llm,  # General purpose LLM model (used for most tasks)
        "report_llm": report_llm,  # Specific LLM for report writing
        "summarization_llm": st.session_state.summarization_llm,  # Specific LLM for summarization
        "enable_quality_checker": enable_quality_checker,
        "quality_check_loops": quality_check_loops,
        "k_results": k_results,  # Number of results to retrieve for each query
    }}

    # Start timing the workflow
    start_time = time.time()
    
    # Create the status for the global "Researcher" process
    langgraph_status = st.status("**Researcher Running...**", state="running")

    # Create a placeholder for the elapsed time
    elapsed_time_placeholder = st.empty()
    
    # If using external database, perform retrieval and summarization first
    if use_ext_database and selected_database:
        # Create the status for the retrieval process
        retrieval_status = st.status("**Document Retrieval...**", state="running")
        
        try:
            with retrieval_status:
                st.write("### Document Retrieval")
                
                # Display embedding model information
                embedding_model_name = extract_embedding_model(selected_database)
                st.info(f"**Embedding Model:** {embedding_model_name}")
                
                # Get embedding model and update the Configuration to use this embedding model
                embed_model = get_embedding_model(embedding_model_name)
                
                # Update the global configuration to use this embedding model
                from src.assistant.configuration import update_embedding_model
                update_embedding_model(embedding_model_name)
                
                # Get tenant ID from the database directory
                database_path = os.path.join(DATABASE_PATH, selected_database)
                tenant_dirs = [d for d in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, d))]
                
                if not tenant_dirs:
                    st.error(f"No tenant directories found in {database_path}")
                    retrieval_status.update(state="error", label=f"**Error: No tenant directories found**")
                else:
                    tenant_id = tenant_dirs[0]  # Use the first tenant directory
                    
                    st.write(f"**Using tenant ID:** {tenant_id}")
                    
                    # Perform similarity search
                    st.write("### Retrieving Documents")
                    with st.spinner("Performing similarity search..."):
                        results = similarity_search_for_tenant(
                            tenant_id=tenant_id,
                            embed_llm=embed_model,
                            persist_directory=database_path,
                            similarity="cosine",
                            normal=True,
                            query=user_input,
                            k=k_results
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
                    
                    # Summarize the results using the summarization LLM
                    st.subheader("Document Summary")
                    with st.spinner(f"Generating summary using {st.session_state.summarization_llm}..."):
                        start_time_summary = time.time()
                        summary = source_summarizer_ollama(user_input, transformed_results, SUMMARIZER_SYSTEM_PROMPT, st.session_state.summarization_llm)
                        end_time_summary = time.time()
                        
                        st.markdown(summary["content"])
                        st.info(f"Summary generated in {end_time_summary - start_time_summary:.2f} seconds using {st.session_state.summarization_llm}")
                    
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
            from src.assistant.configuration import get_config_instance
            embedding_model = get_config_instance().embedding_model
            st.info(f"**Embedding Model:** {embedding_model}")
            
            # Display the mermaid diagram
            st.markdown(f"```mermaid\n{generate_workflow_visualization()}\n```")
            
            # Display the actual langgraph visualization
            st.write("### Actual LangGraph Workflow")
            try:
                # Generate the visualization from the actual graph
                graph_image_path = generate_langgraph_visualization()
                
                # Display the image
                st.image(graph_image_path, caption="LangGraph Workflow (Generated from graph structure)", use_container_width=True)
            except Exception as e:
                st.error(f"Error generating graph visualization: {str(e)}")
            
            st.write("---")
            
            # Force order of expanders by creating them before iteration
            generate_queries_expander = st.expander("Generate Research Queries", expanded=False)
            search_queries_expander = st.expander("Search Queries", expanded=True)
            filter_summaries_expander = st.expander("Filter Summaries", expanded=False)
            rank_summaries_expander = st.expander("Rank Summaries", expanded=False)
            final_answer_expander = st.expander("Generate Final Answer", expanded=False)

            steps = []

            # Run the researcher graph and stream outputs
            for output in researcher.stream(initial_state, config=config):
                # Update elapsed time display directly in the main thread
                current_time = time.time()
                elapsed_seconds = current_time - start_time
                elapsed_time = str(timedelta(seconds=int(elapsed_seconds)))
                elapsed_time_placeholder.info(f"⏱️ Elapsed time: {elapsed_time}")
                
                for key, value in output.items():
                    expander_label = key.replace("_", " ").title()

                    if key == "generate_research_queries":
                        with generate_queries_expander:
                            st.write(value)

                    elif key.startswith("search_and_summarize_query"):
                        with search_queries_expander:
                            with st.expander(expander_label, expanded=False):
                                st.write(value)
                                
                    elif key == "filter_search_summaries":
                        with filter_summaries_expander:
                            # Display basic filtering information
                            filtered_count = len(value.get('filtered_summaries', []))
                            total_count = len(value.get('filtering_details', []))
                            st.write(f"Filtered {total_count} summaries down to {filtered_count} based on relevance to the original query")
                            
                            # Display detailed filtering information
                            if 'filtering_details' in value and value['filtering_details']:
                                st.write("### Filtering Details")
                                for detail in value['filtering_details']:
                                    # Create an icon based on relevance
                                    icon = "✔️" if detail['is_relevant'] else "❌"
                                    # Add a special icon if it was included anyway
                                    if not detail['is_relevant'] and detail.get('included_anyway', False):
                                        icon = "⚠️"
                                    
                                    # Create an expander for each summary's evaluation
                                    with st.expander(f"{icon} Summary {detail['summary_index']} (Confidence: {detail['confidence']:.2f})", expanded=False):
                                        st.write(f"**Preview:** {detail['summary_preview']}")
                                        st.write(f"**Justification:** {detail['justification']}")
                                        
                                        # If it was included despite being irrelevant, explain why
                                        if not detail['is_relevant'] and detail.get('included_anyway', False):
                                            st.write("**Note:** This summary was included despite being marked as irrelevant because all summaries were filtered out and this one had the highest confidence score.")

                    elif key == "rank_search_summaries":
                        with rank_summaries_expander:
                            # Display ranking information
                            if 'relevance_scores' in value and 'ranked_summaries' in value:
                                scores = value['relevance_scores']
                                summaries = value['ranked_summaries']
                                if scores and summaries:
                                    st.write("Summaries ranked by relevance:")
                                    for i, (score, summary) in enumerate(zip(scores, summaries)):
                                        with st.expander(f"Summary {i+1} (Relevance: {score}/10)", expanded=i==0):
                                            st.write(summary)

                    elif key == "generate_final_answer":
                        with final_answer_expander:
                            st.markdown(value, unsafe_allow_html=False)  # Use markdown for rendering links

                    steps.append({"step": key, "content": value})
        
        # Update status to complete
        langgraph_status.update(state="complete", label="**Research Completed**")
        
        # Return the final report
        return steps[-1]["content"] if steps else "No response generated"
    
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
        st.session_state.max_search_queries = 5  # Default value of 5
    if "files_ready" not in st.session_state:
        st.session_state.files_ready = False  # Tracks if files are uploaded but not processed
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "deepseek-r1:latest"  # Default LLM model
    if "enable_web_search" not in st.session_state:
        st.session_state.enable_web_search = False  # Default web search setting
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
        st.session_state.summarization_llm = "llama3.2"  # Default summarization LLM
    if "report_llm" not in st.session_state:
        st.session_state.report_llm = "deepseek-r1:latest"  # Default report writing LLM

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
            
            # Display embedding model
            embedding_model_name = extract_embedding_model(selected_db)
            if embedding_model_name:
                st.sidebar.info(f"Embedding Model: {embedding_model_name}")
            
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
        assistant_response = generate_response(
            user_input, 
            st.session_state.enable_web_search, 
            report_structure,
            st.session_state.max_search_queries,
            st.session_state.report_llm,
            st.session_state.enable_quality_checker,
            st.session_state.quality_check_loops,
            st.session_state.use_ext_database,
            st.session_state.selected_database,
            st.session_state.k_results
        )

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        with st.chat_message("assistant"):
            try:
                st.markdown(assistant_response['final_answer'], unsafe_allow_html=False)  # Use markdown for proper rendering
            except:
                st.markdown(assistant_response, unsafe_allow_html=False)

            # Copy button below the AI message
            if PYPERCLIP_AVAILABLE:
                if st.button("📋", key=f"copy_{len(st.session_state.messages)}"):
                    copy_to_clipboard(assistant_response)

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