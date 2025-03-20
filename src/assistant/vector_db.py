import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, CSVLoader, TextLoader, PDFPlumberLoader
from src.assistant.rag_helpers import load_embed, similarity_search_for_tenant, get_tenant_vectorstore

VECTOR_DB_PATH = "database"
DEFAULT_TENANT_ID = "default"

def get_embedding_model():
    """Get the embedding model."""
    return HuggingFaceEmbeddings()

def get_or_create_vector_db():
    """Get or create the vector DB."""
    embeddings = get_embedding_model()
    
    # Use the default tenant ID
    tenant_id = DEFAULT_TENANT_ID
    
    # Check if the vector DB exists
    tenant_vdb_dir = os.path.join(VECTOR_DB_PATH, tenant_id)
    if os.path.exists(tenant_vdb_dir) and os.listdir(tenant_vdb_dir):
        # Use the existing vector store with the default tenant
        vectorstore = get_tenant_vectorstore(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=VECTOR_DB_PATH,
            similarity="cosine",
            normal=True
        )
    else:
        # If no documents are loaded yet, create an empty directory structure
        os.makedirs(tenant_vdb_dir, exist_ok=True)
        vectorstore = get_tenant_vectorstore(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=VECTOR_DB_PATH,
            similarity="cosine",
            normal=True
        )
        
        # Check if there are files to load
        if os.path.exists("./files") and os.listdir("./files"):
            # Load documents and create a new vector store
            load_embed(
                folder="./files",
                vdbdir=VECTOR_DB_PATH,
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
    """
    Add new documents to the existing vector store.

    Args:
        documents: List of documents to add to the vector store
    """
    embeddings = get_embedding_model()
    tenant_id = DEFAULT_TENANT_ID
    
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
        vdbdir=VECTOR_DB_PATH,
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
    """
    Search for documents in the vector store.
    
    Args:
        query: The query to search for
        k: The number of documents to return
        
    Returns:
        List of documents
    """
    embeddings = get_embedding_model()
    tenant_id = DEFAULT_TENANT_ID
    
    try:
        # Use similarity_search_for_tenant to search for documents
        documents = similarity_search_for_tenant(
            tenant_id=tenant_id,
            embed_llm=embeddings,
            persist_directory=VECTOR_DB_PATH,
            similarity="cosine",
            normal=True,
            query=query,
            k=k
        )
        return documents
    except Exception as e:
        print(f"Error searching for documents: {e}")
        return []