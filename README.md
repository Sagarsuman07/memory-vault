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
- Supabase PostgreSQL (application data and LangGraph checkpoints)
- Qdrant (hybrid dense + BM25 search with RRF)
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
```

## Private demo access and low-memory deployment

Set `APP_ACCESS_PASSWORD` in Streamlit secrets or the environment before
starting the app. Every Streamlit page checks this password before connecting
to PostgreSQL, Qdrant, or the ingestion worker. The app intentionally fails
closed when the password is missing. This is a shared private-demo password,
not per-user authentication; the initial version still uses `DEMO_USER_ID`.

For small Streamlit deployments, ingestion and PDF OCR default to one worker.
Embeddings are generated in small batches and sent to Qdrant without retaining
the complete embedding matrix. Dashboard memory cards are fetched 20 at a time.
The worker is process-local and recovers interrupted jobs on startup; run only
one app replica. The `lease_expires_at` column is reserved for a future
multi-worker deployment and is not currently used for heartbeat recovery.

Use `QDRANT_COLLECTION_NAME` for the collection setting. The older
`QDRANT_COLLECTION` name remains a backward-compatible fallback.

## Supabase PostgreSQL configuration

Set `DATABASE_URL` to the connection string copied from **Supabase Dashboard → Connect**.

For a local development machine or hosting network that cannot reach IPv6, use
**Shared Pooler → Session mode** (port `5432`). Supabase direct database endpoints
are IPv6-only unless the project has the IPv4 add-on. Copy the exact pooler host,
username, and password from the dashboard; do not construct the pooler hostname
manually. Keep the connection string in a local `.env` file or Streamlit secrets,
not in source control. The application requires TLS for PostgreSQL connections.

Example shape only (replace every placeholder with values from the dashboard):

```dotenv
DATABASE_URL=postgresql://postgres.<PROJECT_REF>:<PASSWORD>@<POOLER_HOST>:5432/postgres
DB_POOL_MIN_SIZE=1
DB_POOL_MAX_SIZE=3
DB_CONNECT_TIMEOUT_SECONDS=8
DB_POOL_TIMEOUT_SECONDS=10
```

Never commit a real database password or connection string.
