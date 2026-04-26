import time

import streamlit as st
from dotenv import load_dotenv

from agents.graph import run_research_agent
from rag.vector_store import create_vector_store, delete_all_vectors
from utils.exporters import export_to_pdf
from utils.pdf_parser import extract_text_from_pdfs, get_text_chunks

load_dotenv()

st.set_page_config(page_title="Agentic AI Research Assistant", page_icon="🤖", layout="wide")

st.sidebar.title("🤖 Info / Setup")
st.sidebar.markdown(
    """
    **Upload Research Papers (PDF)**
    This dashboard implements a LangGraph-driven RAG architecture.
    It combines **Pinecone** vector search + DuckDuckGo to generate
    structured reports via a HuggingFace generative model.

    Uploaded documents are stored persistently in Pinecone so they
    remain available across sessions.
    """
)

st.sidebar.markdown("---")
uploaded_files = st.sidebar.file_uploader("1) Upload your PDFs", accept_multiple_files=True, type=["pdf"])

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "report_md" not in st.session_state:
    st.session_state.report_md = ""
if "generation_warning" not in st.session_state:
    st.session_state.generation_warning = ""

if st.sidebar.button("2) Process Knowledge Base"):
    if uploaded_files:
        with st.spinner("Extracting text and uploading embeddings to Pinecone..."):
            raw_text = extract_text_from_pdfs(uploaded_files)
            if raw_text.strip():
                chunks = get_text_chunks(raw_text)
                st.session_state.vector_store = create_vector_store(chunks)
                st.sidebar.success(f"✅ {len(chunks)} chunks upserted to Pinecone!")
            else:
                st.sidebar.error("Failed to extract sensible text from these PDFs.")
    else:
        st.sidebar.warning("Upload a PDF file first.")

if st.sidebar.button("🗑️ Clear Knowledge Base"):
    with st.spinner("Deleting all vectors from Pinecone..."):
        st.session_state.vector_store = delete_all_vectors()
        st.sidebar.success("Pinecone index cleared! ✅")

st.title("Agentic AI Research Assistant")
st.markdown("Using RAG, Pinecone vector DB, DuckDuckGo, and a Flan-T5-based summarization pipeline.")

query = st.text_input(
    "Enter your complex research question:",
    placeholder="E.g., What are the key findings on LLM emergent capabilities?",
)

if st.button("Generate Research Report", type="primary"):
    if not query:
        st.warning("Please enter a research question to begin the workflow.")
    else:
        with st.spinner("Executing Graph Nodes: Query -> Search + Web -> Validation -> Summary..."):
            try:
                time.sleep(1)

                final_state = run_research_agent(query, st.session_state.vector_store)
                st.session_state.report_md = final_state.get("final_report_md", "Error generating report.")
                st.session_state.generation_warning = final_state.get("generation_warning", "")

                st.success("Workflow executed successfully!")
            except Exception as e:
                st.session_state.generation_warning = ""
                st.error(f"Graph execution failed: {str(e)}")

if st.session_state.generation_warning:
    st.warning(st.session_state.generation_warning)

if st.session_state.report_md:
    st.markdown("---")
    st.markdown(st.session_state.report_md)

    st.markdown("---")
    st.markdown("### Export Artifact")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="📄 Download Markdown",
            data=st.session_state.report_md,
            file_name="research_output.md",
            mime="text/markdown",
        )
    with c2:
        try:
            pdf_bytes = export_to_pdf(st.session_state.report_md)
            st.download_button(
                label="📑 Download PDF",
                data=pdf_bytes,
                file_name="research_output.pdf",
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Failed to compile PDF Export: {str(e)}")
