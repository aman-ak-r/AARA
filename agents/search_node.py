from rag.retriever import get_combined_context

def search_node(state):
    """Retrieves relevant web and local PDF chunks based on the query."""
    query = state.get("query")
    vector_store = state.get("vector_store")
    
    # Retrieve merged context
    context = get_combined_context(vector_store, query)
    
    # Append onto state safely
    state["local_chunks"] = context["local_chunks"]
    state["web_snippet"] = context["web_snippet"]
    
    return state
