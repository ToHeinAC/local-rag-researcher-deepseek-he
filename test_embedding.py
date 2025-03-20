import os
import shutil
from src.assistant.rag_helpers import load_embed
from src.assistant.vector_db import get_embedding_model, search_documents, VECTOR_DB_PATH, DEFAULT_TENANT_ID

def test_embedding_and_retrieval():
    # Create a test directory
    test_dir = "test_files"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test file
    test_file = os.path.join(test_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("This is a test document about artificial intelligence and machine learning.")
    
    # Get the embedding model
    embeddings = get_embedding_model()
    
    # Load and embed the test document
    print("Embedding test document...")
    load_embed(
        folder=test_dir,
        vdbdir=VECTOR_DB_PATH,
        embed_llm=embeddings,
        similarity="cosine",
        c_size=2000,
        c_overlap=400,
        normal=True,
        clean=True,
        tenant_id=DEFAULT_TENANT_ID
    )
    
    # Search for documents
    print("\nSearching for documents...")
    query = "What is artificial intelligence?"
    documents = search_documents(query, k=3)
    
    # Print the results
    print(f"\nQuery: {query}")
    print(f"Number of documents found: {len(documents)}")
    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Clean up
    shutil.rmtree(test_dir)
    
    return len(documents) > 0

if __name__ == "__main__":
    success = test_embedding_and_retrieval()
    print(f"\nTest {'succeeded' if success else 'failed'}")
