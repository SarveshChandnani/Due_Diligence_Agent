# VC Due Diligence Research Agent

An intelligent, autonomous **AI-powered Due Diligence Agent** designed for Venture Capitalists. It analyzes startup documents (pitch decks, financial models, traction reports, etc.) using advanced RAG and LangGraph to deliver deep, structured insights.

---

## 📖 Project Journey

This project started as a simple **"Chat with Documents"** application. Through iterative development, it evolved into a **specialized Due Diligence Research Agent**.

### Evolution:
- **Phase 1**: Basic RAG with simple document upload and retrieval
- **Phase 2**: Session-based isolation using Chroma collections
- **Phase 3**: LangGraph integration for structured workflows
- **Phase 4**: Advanced **Hierarchical (Parent-Child) Chunking** for superior retrieval quality
- **Final Version**: Full agentic system with planning, retrieval, analysis, and transparent chunk visibility

The goal was to move beyond generic RAG and build something that actually thinks like a **VC analyst** — with proper context understanding, reasoning, and professional output.

---

## ✨ Key Features

- **Session-based Workspaces** — Each company has its own isolated document collection and chat history
- **Advanced Hierarchical RAG** — Parent-Child Chunking (small chunks for retrieval + large chunks for rich context)
- **LangGraph Agentic Workflow** — Multi-step reasoning with planning and analysis
- **Due Diligence Intelligence** — Understands VC frameworks (Team, Market, Product, Traction, Risks, Financials)
- **Full Transparency** — View retrieved child chunks, parent chunks, and final answer
- **Persistent Chat** — Switch between companies without losing conversation history
- **Clean Streamlit UI** — Professional and intuitive interface

---

## 🛠️ Tech Stack

- **Framework**: LangGraph + LangChain
- **Vector Database**: ChromaDB
- **Chunking Strategy**: ParentDocumentRetriever (Hierarchical Chunking)
- **Embeddings**: OpenAI `text-embedding-3-large`
- **LLM**: GPT-4o / GPT-4o-mini
- **Frontend**: Streamlit
- **Storage**: InMemoryStore (for parent chunks) + Chroma (for child chunks)

---

## 🚀 Quick Start

### 1. Clone & Setup

git clone https://github.com/SarveshChandnani/Due_Delligence_Agent

python -m venv .venv
.\.venv\Scripts\activate    # Windows

pip install -r requirements.txt

### 2. Environment Variables
Create .env file:
OPENAI_API_KEY=sk-...

### 2. Run
streamlit run ui/frontend.py

## 📋 How to Use
Create a new session with a company name
Upload documents (Pitch Deck, Financial Model, Technical Docs, etc.)
Ask due diligence questions like:
"Analyze the founding team"
"What are the major risks?"
"Give me a full investment memo summary"
"Is the traction sustainable?"


## 🎯 Future Roadmap

Web search integration for real-time research
Structured Investment Memo output (PDF export)
Reflection / Self-correction loops
Multi-agent system (specialized analysts)
Qdrant / PGVector migration
User authentication
