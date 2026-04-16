def validation_node(state):
    """Filters duplicate chunks and weak web snippets before passing to summarization."""
    local_chunks = state.get("local_chunks", [])
    web_snippet = state.get("web_snippet", "")

    # Remove duplicate local document chunks while preserving retrieval order.
    unique_chunks = list(dict.fromkeys(local_chunks))

    # Validate web snippet strength
    if len(web_snippet) < 20 or "Web search failed" in web_snippet:
        web_snippet = "No reliable web results found for this query."

    state["local_chunks"] = unique_chunks
    state["web_snippet"] = web_snippet

    return state
