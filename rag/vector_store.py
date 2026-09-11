from pathlib import Path
from typing import Callable, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import get_embeddings


# ============================================================
# Configuration
# ============================================================

CHROMA_DIR = Path(".chroma")

COLLECTION_NAME = "memory_vault"


# ============================================================
# Get Vector Store
# ============================================================

def get_vector_store():
    """
    Return the persistent Chroma vector store.
    """

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


# ============================================================
# Index Memory
# ============================================================

def index_memory(
    memory_id: str,
    user_id: str,
    memory_type: str,
    chunks: list[Document],
    progress_callback: Optional[
        Callable[[str], None]
    ] = None,
):
    """
    Index memory chunks into Chroma.

    Pipeline:

        Chunks
          ↓
        Embedding
          ↓
        Indexing
          ↓
        Chroma

    Phase 13 uses progress_callback so the UI can
    display the actual processing stage.
    """

    if not chunks:
        return 0

    if not memory_id or not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    # ========================================================
    # Get Vector Store
    # ========================================================

    vector_store = get_vector_store()

    # ========================================================
    # Remove Existing Vectors
    # ========================================================
    #
    # This makes re-indexing safe.
    #

    vector_store.delete(
        where={
            "memory_id": memory_id
        }
    )

    # ========================================================
    # Prepare Documents
    # ========================================================

    documents = []

    ids = []

    for index, chunk in enumerate(chunks):

        document = Document(
            page_content=chunk.page_content,
            metadata={
                "memory_id": memory_id,
                "user_id": user_id,
                "memory_type": memory_type,
                "chunk_index": index,
            },
        )

        documents.append(
            document
        )

        ids.append(
            f"{memory_id}_chunk_{index}"
        )

    # ========================================================
    # Extract Text
    # ========================================================

    texts = [
        document.page_content
        for document in documents
    ]

    try:

        # ====================================================
        # Stage 1: Embedding
        # ====================================================

        if progress_callback:

            progress_callback(
                "embedding"
            )

        embeddings = (
            get_embeddings()
            .embed_documents(
                texts
            )
        )

        if not embeddings:

            raise ValueError(
                "Failed to generate embeddings."
            )

        # ====================================================
        # Stage 2: Indexing
        # ====================================================

        if progress_callback:

            progress_callback(
                "indexing"
            )

        # ====================================================
        # Insert Directly into Chroma Collection
        # ====================================================
        #
        # We intentionally use the underlying Chroma
        # collection instead of Chroma.add_embeddings().
        #
        # This is compatible with the installed
        # langchain-chroma version where add_embeddings()
        # is not available.
        #

        vector_store._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                document.metadata
                for document in documents
            ],
        )

        return len(
            documents
        )

    except Exception:

        # ====================================================
        # Cleanup Partially Indexed Vectors
        # ====================================================

        try:

            vector_store.delete(
                where={
                    "memory_id": memory_id
                }
            )

        except Exception:
            pass

        raise