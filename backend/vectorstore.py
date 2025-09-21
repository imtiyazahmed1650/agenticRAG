# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from backend.config import PINECONE_API_KEY

# --- Environment setup ---
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Pinecone index name
INDEX_NAME = "langgraph-rag-index"

# --- Ensure Pinecone index exists ---
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Matches all-MiniLM-L6-v2 embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    print(f"Index '{INDEX_NAME}' created.")


# --- Retriever for querying ---
def get_retriever():
    """
    Returns a LangChain retriever for the Pinecone index.
    """
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    return vectorstore.as_retriever()


# --- Add / update documents ---
def add_document_to_vectorstore(text_content: str, file_name: str = "default_file"):
    """
    Adds a text document to Pinecone, splits into chunks, and safely replaces old vectors.
    
    Parameters:
    - text_content: str, raw document text
    - file_name: str, used as metadata to identify this document
    """
    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    # Split text into chunks
    documents = text_splitter.create_documents([text_content])

    # Attach metadata for deletion
    for doc in documents:
        doc.metadata["source_file"] = file_name

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    # Delete old vectors safely
    try:
        vectorstore.delete(filter={"source_file": file_name})
        print(f"Deleted old vectors for file '{file_name}'.")
    except Exception:
        print(f"[INFO] No existing vectors to delete for '{file_name}', continuing...")

    # Add new chunks
    vectorstore.add_documents(documents)
    print(f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'.")
