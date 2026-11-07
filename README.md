# Enterprise AI Knowledge Assistant using RAG

An AI-powered document question-answering system built using
Retrieval-Augmented Generation (RAG).

The application allows users to upload PDF documents, ask
questions, retrieve relevant information, and receive
context-grounded answers with source references.

## Features

- Multiple PDF document uploads
- Text extraction and document chunking
- Semantic search using sentence-transformers
- Vector storage with ChromaDB
- Local LLM integration using Ollama
- Context-aware question answering
- Source references with PDF filenames and page numbers
- Document selection and deletion
- RAG evaluation dashboard
- LLM-as-a-Judge evaluation

## Tech Stack

- Python
- Streamlit
- LangChain
- Ollama
- Qwen 2.5
- ChromaDB
- Sentence Transformers
- Pandas

## Project Structure

```text
enterprise-rag-assistant/
├── app.py
├── evaluation.py
├── requirements.txt
├── README.md
├── data/
├── src/
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag_pipeline.py
└── tests/
    └── test_rag.py