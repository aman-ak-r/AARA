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
            local_files_only=True,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-base",
            local_files_only=True,
        )
        hf_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
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


def _fallback_summary(query, local_chunks, web_snippet):
    """Build a deterministic summary when no local LLM is available."""
    combined_text = " ".join(local_chunks + [web_snippet]).strip()
    if not combined_text:
        return (
            f"No local PDF context was available for '{query}', and live web retrieval "
            "did not produce usable results in this environment."
        )

    query_terms = {
        token
        for token in re.findall(r"\w+", query.lower())
        if len(token) > 2
    }
    ranked = []
    for sentence in _split_sentences(combined_text):
        tokens = re.findall(r"\w+", sentence.lower())
        score = sum((token in query_terms) for token in tokens)
        score += Counter(tokens).most_common(1)[0][1] if tokens else 0
        ranked.append((score, sentence))

    top_sentences = [sentence for _, sentence in sorted(ranked, reverse=True)[:3]]
    return " ".join(top_sentences) if top_sentences else combined_text[:500]


def summary_node(state):
    """Generates the main findings using a local Hugging Face LLM."""
    query = state.get("query", "Default Query")
    local_chunks = state.get("local_chunks", [])
    web_snippet = state.get("web_snippet", "")

    context_text = " ".join(local_chunks)

    prompt = f"""
Research Query: {query}
Source PDFs Context: {context_text[:1000]}
Web Search Context: {web_snippet[:500]}

Provide the key findings for this query in 3-4 sentences.
"""
    try:
        llm = get_llm()
        response = llm.invoke(prompt)
    except Exception as e:
        response = _fallback_summary(query, local_chunks, web_snippet)
        state["generation_warning"] = (
            "Used fallback summarization because the local LLM was unavailable: "
            f"{str(e)}"
        )

    report_dict = {
        "Title": query.title(),
        "Abstract": f"A generated research abstract assessing contextual findings for: '{query}'.",
        "Key Findings": response.strip(),
        "Sources": "Local Document Uploads, DuckDuckGo Search",
        "Conclusion": "The report combines the strongest available local and web context gathered during this run.",
        "Future Scope": "Further investigations should cross-reference additional documents and validate claims against primary sources.",
    }

    state["report_dict"] = report_dict
    return state
