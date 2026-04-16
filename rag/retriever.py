from langchain_community.tools import DuckDuckGoSearchRun

def retrieve_from_vector_store(vector_store, query, top_k=5):
    """Retrieves top_k relevant chunks from FAISS vector store."""
    docs = vector_store.similarity_search(query, k=top_k)
    return [doc.page_content for doc in docs]

def retrieve_from_web(query):
    """Retrieves web snippets using DuckDuckGo."""
    search = DuckDuckGoSearchRun()
    try:
        results = search.run(query)
        return results
    except Exception as e:
        return f"Web search failed: {e}"

def get_combined_context(vector_store, query):
    """Merges local PDF document chunks and DuckDuckGo search results."""
    local_chunks = []
    if vector_store:
        local_chunks = retrieve_from_vector_store(vector_store, query)
    
    web_snippet = retrieve_from_web(query)
    
    return {
        "local_chunks": local_chunks,
        "web_snippet": web_snippet
    }
