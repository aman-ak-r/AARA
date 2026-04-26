import re
from collections import Counter

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

_pipeline_instance = None


def get_llm():
    """Lazily load the Hugging Face pipeline wrapping a lightweight local LLM.
    We use google/flan-t5-base as a stable, lightweight baseline for the project requirements.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-base",
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
        )
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            truncation=True,
        )
        _pipeline_instance = HuggingFacePipeline(pipeline=hf_pipeline)
    return _pipeline_instance


def _split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text or "")
        if sentence.strip()
    ]


def _clean_chunk(text):
    """Clean up a chunk: remove leading dots/fragments, fix spacing."""
    text = text.strip()
    # Remove leading fragments like ".js?" or ". " at the start
    text = re.sub(r"^[.\s,;:]+", "", text)
    # Fix multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _generate_llm_summary(query, context_text):
    """Try to generate a brief summary using the LLM. Returns summary string."""
    prompt = f"""Summarize the following information about "{query}" in 2-3 clear sentences:

{context_text[:1500]}

Summary:"""
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        if response and len(response.strip()) > 20:
            return response.strip()
    except Exception:
        pass
    return None


def _generate_definition(query, chunk_texts):
    """Try to get a definition from the LLM."""
    prompt = f"""Based on the following context, define "{query}" in one clear sentence:

{' '.join(chunk_texts[:3])[:1000]}

Definition:"""
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
        if response and len(response.strip()) > 10:
            return response.strip()
    except Exception:
        pass
    return None


def _rank_sentences_by_relevance(query, sentences):
    """Rank sentences by relevance to the query, return sorted list."""
    query_terms = {
        token for token in re.findall(r"\w+", query.lower()) if len(token) > 2
    }
    ranked = []
    for sentence in sentences:
        tokens = re.findall(r"\w+", sentence.lower())
        # Score by query term overlap
        score = sum((token in query_terms) for token in tokens)
        # Bonus for longer, more informative sentences
        if len(tokens) > 8:
            score += 1
        # Penalty for very short sentences
        if len(tokens) < 5:
            score -= 2
        ranked.append((score, sentence))

    return [s for _, s in sorted(ranked, reverse=True)]


def _build_detailed_explanation(query, chunk_texts, web_snippet):
    """Build a comprehensive, well-structured explanation from retrieved chunks
    and web search results. This is the core of the report generation.
    """
    # Collect ALL sentences from chunks and web
    all_sentences = []
    for chunk in chunk_texts:
        cleaned = _clean_chunk(chunk)
        all_sentences.extend(_split_sentences(cleaned))

    if web_snippet and "Web search failed" not in web_snippet and "No reliable" not in web_snippet:
        all_sentences.extend(_split_sentences(web_snippet))

    if not all_sentences:
        return f"No relevant information was found for '{query}' in the uploaded documents or web search."

    # Remove duplicates while preserving order
    seen = set()
    unique_sentences = []
    for s in all_sentences:
        normalized = re.sub(r"\s+", " ", s.lower().strip())
        if normalized not in seen and len(s.strip()) > 15:
            seen.add(normalized)
            unique_sentences.append(s.strip())

    # Rank by relevance to query
    ranked = _rank_sentences_by_relevance(query, unique_sentences)

    # Take top sentences (up to 20) for a detailed report
    top_sentences = ranked[:20]

    # Group into paragraphs of ~4-5 sentences for readability
    paragraphs = []
    for i in range(0, len(top_sentences), 4):
        group = top_sentences[i : i + 4]
        paragraphs.append(" ".join(group))

    return "\n\n".join(paragraphs)


def _build_key_concepts(chunk_texts):
    """Extract key concepts/topics from the chunks as bullet points."""
    # Find recurring meaningful phrases
    all_text = " ".join(chunk_texts).lower()
    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    word_counts = Counter(words)

    # Filter out common stop-words
    stop_words = {
        "that", "this", "with", "from", "your", "have", "will", "been",
        "they", "their", "there", "what", "when", "which", "about", "would",
        "could", "should", "into", "also", "some", "more", "than", "then",
        "each", "just", "like", "make", "over", "such", "only", "very",
        "does", "used", "uses", "using", "http", "https", "localhost",
    }
    meaningful = [
        (word, count)
        for word, count in word_counts.most_common(30)
        if word not in stop_words and count >= 2
    ]

    if not meaningful:
        return ""

    # Format as bullet points
    bullets = []
    for word, count in meaningful[:10]:
        bullets.append(f"- **{word.title()}** (mentioned {count} times)")

    return "\n".join(bullets)


def summary_node(state):
    """Generates a comprehensive research report by synthesizing retrieved
    document chunks and web search results into a detailed, readable format.
    """
    query = state.get("query", "Default Query")
    local_chunks = state.get("local_chunks", [])  # list of {"text":..., "score":...}
    web_snippet = state.get("web_snippet", "")

    # Extract plain text from chunk dicts (backward-compatible with plain strings)
    if local_chunks and isinstance(local_chunks[0], dict):
        chunk_texts = [c["text"] for c in local_chunks]
        chunk_scores = [c.get("score", 0) for c in local_chunks]
    else:
        chunk_texts = list(local_chunks)
        chunk_scores = [0] * len(chunk_texts)

    # ── Generate components of the report ────────────────────────────────
    context_text = " ".join(chunk_texts)

    # 1. Try to get an LLM-generated overview
    llm_summary = _generate_llm_summary(query, context_text)
    llm_warning = ""
    if not llm_summary:
        llm_warning = (
            "The local LLM (Flan-T5) was unable to generate a summary. "
            "The report below is built directly from retrieved document content."
        )

    # 2. Try to get a definition
    definition = _generate_definition(query, chunk_texts)

    # 3. Build detailed explanation from chunks
    detailed_explanation = _build_detailed_explanation(query, chunk_texts, web_snippet)

    # 4. Extract key concepts
    key_concepts = _build_key_concepts(chunk_texts)

    # ── Assemble the Abstract ────────────────────────────────────────────
    abstract_parts = []
    if definition:
        abstract_parts.append(definition)
    if llm_summary and llm_summary != definition:
        abstract_parts.append(llm_summary)
    if not abstract_parts:
        abstract_parts.append(
            f"This report provides a comprehensive analysis of '{query}' "
            f"based on {len(chunk_texts)} relevant excerpts retrieved from "
            f"your uploaded documents and supplementary web search results."
        )
    abstract = " ".join(abstract_parts)

    # ── Assemble Key Findings (the main body) ────────────────────────────
    findings_parts = []
    if detailed_explanation:
        findings_parts.append(detailed_explanation)
    findings = "\n\n".join(findings_parts) if findings_parts else "No detailed findings available."

    # ── Sources section ──────────────────────────────────────────────────
    sources_list = []
    if chunk_texts:
        avg_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0
        sources_list.append(
            f"📄 **Uploaded PDFs** — {len(chunk_texts)} relevant excerpts "
            f"retrieved from Pinecone (avg relevance: {avg_score:.2f})"
        )
    if web_snippet and "Web search failed" not in web_snippet and "No reliable" not in web_snippet:
        sources_list.append("🌐 **DuckDuckGo Web Search** — supplementary web context")
    sources_text = "\n".join(f"- {s}" for s in sources_list) if sources_list else "No sources available."

    # ── Build the final report dict ──────────────────────────────────────
    report_dict = {
        "Title": query.title(),
        "Abstract": abstract,
        "Key Findings": findings,
    }

    if key_concepts:
        report_dict["Key Concepts"] = key_concepts

    report_dict["Sources"] = sources_text
    report_dict["Conclusion"] = (
        f"This report synthesized {len(chunk_texts)} document excerpts and web search "
        f"results to provide a comprehensive overview of '{query}'. The information "
        f"above is drawn directly from your uploaded research documents, ensuring "
        f"relevance and accuracy to your source material."
    )
    report_dict["Future Scope"] = (
        "For deeper analysis, consider uploading additional documents on this topic, "
        "or refining your query to focus on specific aspects of interest."
    )

    state["report_dict"] = report_dict
    if llm_warning:
        state["generation_warning"] = llm_warning
    return state
