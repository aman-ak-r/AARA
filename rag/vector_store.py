from langchain_community.vectorstores import FAISS
from rag.embeddings import get_embeddings_model

def create_vector_store(text_chunks):
    """Creates a FAISS vector store from text chunks.
    
    Args:
        text_chunks (list): A list of text chunks extracted from PDFs.
        
    Returns:
        FAISS vector store index.
    """
    embeddings = get_embeddings_model()
    # Build FAISS index from the extracted chunks
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store
