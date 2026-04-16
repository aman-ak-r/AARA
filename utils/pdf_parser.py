import re

import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdfs(pdf_docs):
    """Extracts text from multiple uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            continue
    return clean_text(text)


def clean_text(text):
    """Cleans extra spaces and newlines."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_text_chunks(text, chunk_size=500, chunk_overlap=50):
    """Splits text into chunks with overlap for RAG."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    return chunks
