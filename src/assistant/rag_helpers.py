import os, re
import fitz
from datetime import datetime
from langchain_core.runnables import chain
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
import nltk
from langchain_community.llms import Ollama
from langchain_core.messages import SystemMessage, HumanMessage

# Download nltk data if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import word_tokenize

# Define constants - must match the value in vector_db.py
VECTOR_DB_PATH = "database"

# Define functions
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def clean_(text):
    # Convert to lowercase
    #text = text.lower()
    
    # Remove unwanted characters but preserve . , : § $ % &
    text = re.sub(r'[^a-zA-Z0-9\s.,:§$%&€@-µ²³üöäß]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', 0)
        chunk.metadata['id'] = f"{source}:{page}:{i}"
    return chunks

def get_tenant_collection_name(tenant_id):
    return f"collection_{tenant_id}"

def get_tenant_vectorstore(tenant_id, embed_llm, persist_directory, similarity, normal=True):
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    
    # Create directory if it doesn't exist
    os.makedirs(tenant_vdb_dir, exist_ok=True)
    
    # Get collection name for tenant
    collection_name = get_tenant_collection_name(tenant_id)
    
    return Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm,
        collection_metadata={"hnsw:space": similarity, "normalize_embeddings": normal}
    )

def similarity_search_for_tenant(tenant_id, embed_llm, persist_directory, similarity, normal, query, k=2, language="English"):
    # Import clear_cuda_memory here to avoid circular imports
    from src.assistant.utils import clear_cuda_memory
    
    # Clear CUDA memory before search
    clear_cuda_memory()
    
    # Get tenant-specific directory
    tenant_vdb_dir = os.path.join(persist_directory, tenant_id)
    
    # Check if directory exists
    if not os.path.exists(tenant_vdb_dir):
        raise Exception(f"Vector database directory for tenant {tenant_id} does not exist at {tenant_vdb_dir}")
    
    # Get collection name for tenant
    collection_name = get_tenant_collection_name(tenant_id)
    
    # Initialize vectorstore for search
    vectorstore = Chroma(
        persist_directory=tenant_vdb_dir,
        collection_name=collection_name,
        embedding_function=embed_llm,
        collection_metadata={"hnsw:space": similarity, "normalize_embeddings": normal}
    )
    
    try:
        # Print language being used for retrieval
        print(f"Using language for retrieval: {language}")
        
        # Perform similarity search
        results = vectorstore.similarity_search(query, k=k)
        
        # Add language metadata to each document for downstream processing
        for doc in results:
            if "metadata" in doc.__dict__:
                doc.metadata["language"] = language
        
        # Clean up
        vectorstore._client = None
        del vectorstore
        
        # Clear CUDA memory after search
        clear_cuda_memory()
        
        return results
    except Exception as e:
        # Clean up in case of error
        if vectorstore:
            vectorstore._client = None
            del vectorstore
        
        # Clear CUDA memory in case of error
        clear_cuda_memory()
        
        # Re-raise the exception
        raise e

def load_embed(folder, vdbdir, embed_llm, similarity="cosine", c_size=1000, c_overlap=200, normal=True, clean=True, tenant_id=None):    
    # Import clear_cuda_memory here to avoid circular imports
    from src.assistant.utils import clear_cuda_memory
    
    # Clear CUDA memory before starting embedding process
    clear_cuda_memory()
    
    dirname = vdbdir
    # Now load and embed
    print(f"Step: Check for new data and embed new data to new vector DB '{dirname}'")
    # Load documents from the specified directory
    directory = folder
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            text = extract_text_from_pdf(pdf_path)
            documents.append(Document(page_content=text, metadata={'source': filename, 'path': pdf_path}))
        else:
            loader = DirectoryLoader(directory, exclude="**/*.pdf")
            loaded = loader.load()
            if loaded:
                # Add full path to metadata
                for doc in loaded:
                    if 'source' in doc.metadata:
                        doc.metadata['path'] = os.path.join(directory, doc.metadata['source'])
                documents.extend(loaded)
    
    docslen = len(documents)
    
    # multitenant
    if tenant_id is None:
        tenant_id = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')    
    print(f"Using tenant ID: {tenant_id}")
    vectorstore = get_tenant_vectorstore(tenant_id, embed_llm, persist_directory=dirname, similarity=similarity, normal=normal)
    print(f"Collection name: {vectorstore._collection.name}")
    print(f"Collection count before adding: {vectorstore._collection.count()}")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = []
    for document in documents:
        if clean:
            doc_chunks = text_splitter.create_documents([clean_(document.page_content)])
        else:
            doc_chunks = text_splitter.create_documents([document.page_content])
        for chunk in doc_chunks:
            chunk.metadata['source'] = document.metadata['source']
            chunk.metadata['page'] = document.metadata.get('page', 0)  # Assuming page metadata is available
            chunk.metadata['path'] = document.metadata.get('path', '')
        chunks.extend(doc_chunks)

    # Calculate human-readable chunk IDs
    chunks = calculate_chunk_ids(chunks)

    # Extract vector IDs from chunks
    vector_ids = [chunk.metadata['id'] for chunk in chunks]

    # Check for existing vector IDs in the database
    existing_ids = vectorstore.get()['ids']

    # Filter out chunks that are already in the database
    new_chunks = [chunk for chunk, vector_id in zip(chunks, vector_ids) if vector_id not in existing_ids]
    new_vector_ids = [vector_id for vector_id in vector_ids if vector_id not in existing_ids]

    newchunkslen = len(new_chunks)

    if new_chunks:
        # Clear CUDA memory before adding documents
        clear_cuda_memory()
        
        # Add the new chunks to the vector store with their embeddings
        vectorstore.add_documents(new_chunks, ids=new_vector_ids)
        print(f"Collection count after adding: {vectorstore._collection.count()}")
        vectorstore.persist()
        print(f"#{docslen} files embedded via #{newchunkslen} chunks in vector database.")
        
        # Clear CUDA memory after adding documents
        clear_cuda_memory()
    else:
        # Already existing
        print(f"Chunks already available, no new chunks added to vector database.")

    return dirname, tenant_id

# Function to transform documents
def transform_documents(documents):
    """
    Transforms a list of Document objects into a specific dictionary format.
    
    Args:
        documents (list): List of Document objects with metadata and page_content
        
    Returns:
        list: List of dictionaries with content and metadata in the required format
    """
    samples = []
    
    for doc in documents:
        transformed_doc = {
            "content": doc.page_content,
            "metadata": {
                "name": doc.metadata['id'],
                "path": doc.metadata['source']
            }
        }
        samples.append(transformed_doc)

    return samples

# Function to summarize sources using Ollama
def source_summarizer_ollama(query, context_documents, language, system_message, llm_model="deepseek-r1"):
    # Check if context_documents is already a formatted string
    if isinstance(context_documents, str):
        formatted_context = context_documents
    else:
        # Handle the case where context_documents is a list of dictionary objects
        try:
            formatted_context = "\n".join(
                f"Content: {doc['content']}\nSource: {doc['metadata']['name']}\nPath: {doc['metadata']['path']}"
                for doc in context_documents
            )
        except (TypeError, KeyError):
            # Fallback: try to use the documents as they are
            formatted_context = str(context_documents)
    #formatted_context = "\n".join(
    #    f"{str(doc)}"
    #    for doc in context_documents
    #)
    prompt = f"""
    Based on the user's query

    {query}

    , here are the documents to summarize:
    
    {formatted_context}
    
    Provide a deep summary in the language {language} with proper citations:
    """
    
    # Initialize ChatOllama with the specified model and temperature
    llm = Ollama(model=llm_model, temperature=0.1, repeat_penalty=1.2) 
    # For RAG systems like your summarizer, consider:
    #    Using lower temperatures (0.1-0.3) for factual accuracy
    #   Combining with repeat_penalty=1.1-1.3 to avoid redundant content
    #   Monitoring token usage with num_ctx for long documents
    
    # Format messages for LangChain
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=prompt)
    ]
    
    # Get response from the model
    response = llm.invoke(messages)
    
    # Extract content from response
    response_content = response
    
    # Clean markdown formatting if present
    try:
        final_content = re.sub(r"<think>.*?</think>", "", response_content, flags=re.DOTALL).strip()
    except:
        final_content = response_content.strip()

    # Extract metadata from all documents
    document_names = [doc['metadata']['name'] for doc in context_documents]
    document_paths = [doc['metadata']['path'] for doc in context_documents]

    return {
        "content": final_content,
        "metadata": {
            "name": document_names,
            "path": document_paths
        }
    }    
