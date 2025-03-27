import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader, PDFPlumberLoader

# Base path for vector database
VECTOR_DB_PATH = "database"
DEFAULT_TENANT_ID = "default"

def get_embedding_model():
    """Get the embedding model."""
    # Import here to avoid circular imports
    from src.assistant.configuration import Configuration
    
    # Get the embedding model from configuration
    embedding_model_name = Configuration().embedding_model
    
    emb_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    print('-------------------------')
    print(f"Using embedding model: {embedding_model_name}")
    print(emb_model)
    print('-------------------------')
    return emb_model

def get_embedding_model_path():
    """Get the sanitized embedding model name for use in paths."""
    # Import here to avoid circular imports
    from src.assistant.configuration import Configuration
    
    # Get the embedding model from configuration
    embedding_model_name = Configuration().embedding_model
    
    # Create a sanitized version of the model name for folder paths
    # Replace slashes with double hyphens
    sanitized_model_name = embedding_model_name.replace('/', '--')
    
    return sanitized_model_name

def get_vector_db_path():
    """Get the vector database path including the embedding model name."""
    sanitized_model_name = get_embedding_model_path()
    return os.path.join(VECTOR_DB_PATH, sanitized_model_name)

def get_or_create_vector_db():
    """Get or create the vector DB."""
    # Import here to avoid circular imports
    from src.assistant.rag_helpers import load_embed, get_tenant_vectorstore
    
    embeddings = get_embedding_model()
    
    # Use the default tenant ID
    tenant_id = DEFAULT_TENANT_ID
    
    # Get the vector DB path with embedding model name
    vector_db_path = get_vector_db_path()
    
    # Check if the vector DB exists
    tenant_vdb_dir = os.path.join(vector_db_path, tenant_id)
    if os.path.exists(tenant_vdb_dir) and os.listdir(tenant_vdb_dir):
        # Use the existing vector store with the default tenant
        vectorstore = get_tenant_vectorstore(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=vector_db_path,
            similarity="cosine",
            normal=True
        )
    else:
        # If no documents are loaded yet, create an empty directory structure
        os.makedirs(tenant_vdb_dir, exist_ok=True)
        vectorstore = get_tenant_vectorstore(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=vector_db_path,
            similarity="cosine",
            normal=True
        )
        
        # Check if there are files to load
        if os.path.exists("./files") and os.listdir("./files"):
            # Load documents and create a new vector store
            load_embed(
                folder="./files",
                vdbdir=vector_db_path,
                embed_llm=embeddings,
                similarity="cosine",
                c_size=2000,
                c_overlap=400,
                normal=True,
                clean=True,
                tenant_id=tenant_id
            )
    
    return vectorstore

def add_documents(documents):
    """Add new documents to the existing vector store."""
    # Import here to avoid circular imports
    from src.assistant.rag_helpers import load_embed
    
    embeddings = get_embedding_model()
    tenant_id = DEFAULT_TENANT_ID
    vector_db_path = get_vector_db_path()
    
    # Create a temporary directory to store the documents
    temp_dir = "./temp_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save the documents to the temporary directory
    for i, doc in enumerate(documents):
        file_path = os.path.join(temp_dir, f"document_{i}.txt")
        with open(file_path, "w") as f:
            f.write(doc.page_content)
    
    # Load and embed the documents
    load_embed(
        folder=temp_dir,
        vdbdir=vector_db_path,
        embed_llm=embeddings,
        similarity="cosine",
        c_size=2000,
        c_overlap=400,
        normal=True,
        clean=True,
        tenant_id=tenant_id
    )
    
    # Clean up the temporary directory
    for file in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)
    
    # Return the updated vector store
    return get_or_create_vector_db()

def search_documents(query, k=3):
    """Search for documents in the vector store."""
    # Import clear_cuda_memory here to avoid circular imports
    from src.assistant.utils import clear_cuda_memory
    from src.assistant.rag_helpers import similarity_search_for_tenant
    
    # Clear CUDA memory before embedding
    clear_cuda_memory()
    
    embeddings = get_embedding_model()
    tenant_id = DEFAULT_TENANT_ID
    vector_db_path = get_vector_db_path()
    
    try:
        # Use similarity_search_for_tenant to search for documents
        documents = similarity_search_for_tenant(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=vector_db_path,
            similarity="cosine",
            normal=True,
            query=query,
            k=k
        )
        
        # Clear CUDA memory after embedding
        clear_cuda_memory()
        
        return documents
    except Exception as e:
        print(f"Error searching for documents: {e}")
        # Clear CUDA memory in case of error
        clear_cuda_memory()
        return []