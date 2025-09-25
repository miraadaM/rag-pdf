🚀 RAG PDF + Ollama Pipeline

This project implements a **lightweight Retrieval-Augmented Generation (RAG) pipeline** that lets you query PDFs using **FAISS vector search** and **Ollama LLMs** (e.g., `phi3`).  

It’s perfect for creating your own **local AI knowledge base** from research papers, reports, or textbooks.

---

## ✨ Features
- 📄 **PDF Extraction** with [PyMuPDF](https://pymupdf.readthedocs.io/)
- ✂️ **Smart Chunking** by custom markers (`$$$`) and token size
- 🧩 **Embeddings** via [Ollama Embeddings (`nomic-embed-text`)](https://ollama.ai/)
- 🔎 **Semantic Search** with [FAISS](https://github.com/facebookresearch/faiss)
- 🤖 **Context-Aware Q&A** powered by local Ollama models (tested with `phi3`)
- 🗂️ **Inspect FAISS Database** (metadata + vectors)

  ## ⚡ Quickstart

### 1. Clone Repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-pdf-ollama.git
cd rag-pdf-ollama
Install dependencies: pip install -r requirements.txt
Run the pipeline and plase your pdf to repo: python chunking.py
The program will:
Extract text from the PDF
Split into chunks
Build FAISS index
Allow interactive Q&A via terminal
🔧 Configuration
marker="$$$" → split by custom markers inside PDF text
max_tokens=500 → max token length per chunk
Default embedding model → nomic-embed-text
Default LLM → phi3
