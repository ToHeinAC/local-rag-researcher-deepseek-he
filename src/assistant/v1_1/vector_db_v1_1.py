import os
# Use updated import path to avoid deprecation warning
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback to original import if package is not installed
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader, PDFPlumberLoader

# Base path for vector database
VECTOR_DB_PATH = "database"
DEFAULT_TENANT_ID = "2025-04-22_15-41-10"  # Updated to match the correct tenant ID

# Define the special database configuration
SPECIAL_DB_CONFIG = {
    'sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2--2000--400': {
        'tenant_id': '2025-04-22_15-41-10',
        'collection_name': 'collection_2025-04-22_15-41-10'  # Using collection name WITH prefix
    }
}

def get_embedding_model():
    """Get the embedding model."""
    # Import here to avoid circular imports
    from src.assistant.v1_1.configuration_v1_1 import get_config_instance
    
    # Get the embedding model from the global configuration instance
    embedding_model_name = get_config_instance().embedding_model
    
    emb_model = HuggingFaceEmbeddings(model_name=embedding_model_name, model_kwargs={'device': 'cpu'})
    print('-------------------------')
    print(f"Using embedding model: {embedding_model_name}")
    print(emb_model)
    print('-------------------------')
    return emb_model

def get_embedding_model_path():
    """Get the sanitized embedding model name for use in paths."""
    # Import here to avoid circular imports
    from src.assistant.v1_1.configuration_v1_1 import get_config_instance
    
    # Get the embedding model from the global configuration instance
    embedding_model_name = get_config_instance().embedding_model
    
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
    from src.assistant.v1_1.rag_helpers_v1_1 import load_embed, get_tenant_vectorstore
    
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
    from src.assistant.v1_1.rag_helpers_v1_1 import load_embed
    
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

def search_documents(query, k=3, language="English"):
    """Search for documents in the vector store."""
    # Import clear_cuda_memory here to avoid circular imports
    from src.assistant.v1_1.utils_v1_1 import clear_cuda_memory
    from src.assistant.v1_1.rag_helpers_v1_1 import similarity_search_for_tenant
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Clear CUDA memory before embedding
    clear_cuda_memory()
    
    # Get the configured embedding model
    embeddings = get_embedding_model()
    
    # Get the embedding model path (for accessing the DB)
    from src.assistant.v1_1.configuration_v1_1 import get_config_instance
    current_embedding_model = get_config_instance().embedding_model
    sanitized_model_name = current_embedding_model.replace('/', '--')
    
    # Use the module-level DATABASE_PATH constant instead
    import os
    # Use the global DATABASE_PATH but resolve it to an absolute path
    DATABASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../database'))
    
    
    # Always use the special configuration for sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 model
    # Since we know this is the model we need to use for German retrieval
    if 'paraphrase-multilingual-MiniLM-L12-v2' in current_embedding_model:
        # This is the model we need - always use the specific database configuration
        special_db_key = 'sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2--2000--400'
        logger.info(f"Detected multilingual model - using special database: {special_db_key}")
    else:
        # For other models, try to match the DB key
        special_db_key = None
        for key in SPECIAL_DB_CONFIG.keys():
            if sanitized_model_name in key or key in sanitized_model_name:
                special_db_key = key
                break
    
    if special_db_key:
        # Use the special configuration
        logger.info(f"Using special database configuration for: {special_db_key}")
        tenant_id = SPECIAL_DB_CONFIG[special_db_key]['tenant_id']
        collection_name = SPECIAL_DB_CONFIG[special_db_key]['collection_name']
        vector_db_path = os.path.join(DATABASE_PATH, special_db_key)
        logger.info(f"Using special DB path: {vector_db_path}, tenant: {tenant_id}, collection: {collection_name}")
    else:
        # Use default configuration
        tenant_id = DEFAULT_TENANT_ID
        collection_name = None  # Will be generated from tenant_id
        vector_db_path = get_vector_db_path()
        logger.info(f"Using default DB path: {vector_db_path}, tenant: {tenant_id}")
    
    try:
        # Use similarity_search_for_tenant to search for documents
        logger.info(f"Searching documents with language: {language}")
        documents = similarity_search_for_tenant(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=vector_db_path,
            similarity="cosine",
            normal=True,
            query=query,
            k=k,
            language=language,  # Pass the language parameter
            collection_name=collection_name  # Pass the collection name if we have one
        )
        
        # Clear CUDA memory after embedding
        clear_cuda_memory()
        
        return documents
    except Exception as e:
        print(f"Error searching for documents: {e}")
        # Clear CUDA memory in case of error
        clear_cuda_memory()
        return []
