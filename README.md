# Agentic AI Research Assistant

This is an AI-powered research assistant utilizing Retrieval-Augmented Generation (RAG) and LangGraph to answer research queries by combining local PDF knowledge and DuckDuckGo web search.

## Features
- **Upload Multiple PDFs**: Parses and chunks PDFs to a FAISS vector store.
- **RAG Pipeline**: Semantic search across your documents with `all-MiniLM-L6-v2`.
- **LangGraph Agent Workflow**: Implements a structured workflow using `google/flan-t5-base`.
- **Hybrid Search**: Combines web context from DuckDuckGo with local context.
- **Structured Export**: Download your final generated report as Markdown or PDF.

## Architecture

The workflow implemented using LangGraph operates as follows:

```
query_node → search_node → validation_node → summary_node → report_node
```

## Folder Structure
- `app.py`: Main Streamlit app.
- `agents/`: Contains LangGraph node definitions and overall workflow.
- `rag/`: Manages embeddings, FAISS vector store, and document retrieving.
- `utils/`: Helpers for PDF extraction and generating PDF exports.

## Local Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Future Improvements
- Upgrade the local LLM to support higher context lengths (e.g., Llama 3 or Mistral).
- Implement persistent memory across chat sessions.
- Deploy onto Hugging Face Spaces for continuous remote availability.
