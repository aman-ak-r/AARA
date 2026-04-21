
try:
    import streamlit
    import langchain
    import langgraph
    import faiss
    import sentence_transformers
    import PyPDF2
    import reportlab
    import ddgs
    import torch
    import transformers
    print("All imports successful!")
except ImportError as e:
    print(f"Import error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
