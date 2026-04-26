import json
import subprocess
import sys

from rag.embeddings import get_embeddings_model


def retrieve_from_vector_store(vector_store, query, top_k=10):
    """Retrieves top_k relevant chunks from the Pinecone index.

    Args:
        vector_store: A Pinecone Index object.
        query: The user's search query string.
        top_k: Number of top results to return (default raised to 10 for richer context).

    Returns:
        A list of (text, score) tuples from the most relevant chunks.
    """
    embeddings_model = get_embeddings_model()
    query_vector = embeddings_model.embed_query(query)

    results = vector_store.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    chunks = []
    for match in results.get("matches", []):
        text = match.get("metadata", {}).get("text", "")
        score = match.get("score", 0.0)
        if text:
            chunks.append({"text": text, "score": round(score, 4)})
    return chunks


def retrieve_from_web(query):
    """Retrieves web snippets using DuckDuckGo.

    The call is isolated in a subprocess because web-search dependencies can
    trigger native TLS/keychain panics on some macOS setups. Subprocess
    isolation ensures the main app survives and can continue with local context.
    """
    helper = (
        "import json\n"
        "import sys\n"
        "query = json.loads(sys.stdin.read() or '{}').get('query', '').strip()\n"
        "if not query:\n"
        "    print(json.dumps({'result': 'Web search failed: empty query'}))\n"
        "    raise SystemExit(0)\n"
        "try:\n"
        "    from ddgs import DDGS\n"
        "except Exception as import_err:\n"
        "    print(json.dumps({'error': f'Could not import DDGS client: {import_err}'}))\n"
        "    raise SystemExit(2)\n"
        "snippets = []\n"
        "try:\n"
        "    with DDGS() as ddgs:\n"
        "        for item in ddgs.text(query, max_results=5):\n"
        "            if not isinstance(item, dict):\n"
        "                continue\n"
        "            title = (item.get('title') or '').strip()\n"
        "            body = (item.get('body') or '').strip()\n"
        "            href = (item.get('href') or '').strip()\n"
        "            line = ' | '.join(part for part in (title, body, href) if part)\n"
        "            if line:\n"
        "                snippets.append(line)\n"
        "except Exception as search_err:\n"
        "    print(json.dumps({'error': f'DuckDuckGo search failed: {search_err}'}))\n"
        "    raise SystemExit(3)\n"
        "result = '\\n'.join(snippets).strip() or 'Web search failed: empty response'\n"
        "print(json.dumps({'result': result}))\n"
    )
    try:
        process = subprocess.run(
            [sys.executable, "-c", helper],
            input=json.dumps({"query": query}),
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if process.returncode != 0:
            details = process.stdout.strip() or process.stderr.strip() or "subprocess error"
            return f"Web search failed: {details}"

        payload = json.loads(process.stdout.strip() or "{}")
        return payload.get("result", "Web search failed: empty response")
    except Exception as e:
        return f"Web search failed: {e}"


def get_combined_context(vector_store, query):
    """Merges local PDF document chunks from Pinecone and DuckDuckGo search results."""
    local_chunks = []
    if vector_store is not None:
        local_chunks = retrieve_from_vector_store(vector_store, query)

    web_snippet = retrieve_from_web(query)

    return {
        "local_chunks": local_chunks,
        "web_snippet": web_snippet,
    }
