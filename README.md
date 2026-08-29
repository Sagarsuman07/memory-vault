# Memory Vault

Memory Vault is a multimodal personal AI memory assistant inspired by
modern AI memory systems such as OnePlus/OPPO Mind Space.

## Features

- Image memories
- Document memories
- Voice memories
- AI-generated summaries
- RAG-based memory search
- Memory-specific questions
- Global memory search
- Tool calling
- LangGraph agent
- Short-term conversational memory
- Grounded answers
- Source references
- LangSmith tracing and evaluation

## Architecture

Memory Vault uses:

- Python
- Streamlit
- LangChain
- LangGraph
- SQLite
- Chroma
- LLM
- Embeddings
- Speech-to-text
- LangSmith

## Development

Create virtual environment:

```bash
python -m venv mvenv

mvenv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py