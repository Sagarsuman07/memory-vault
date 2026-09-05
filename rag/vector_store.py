from pathlib import Path

from langchain_chroma import Chroma

from langchain_core.documents import Document

from config.settings import settings
from rag.embeddings import get_embeddings


CHROMA_DIR = Path(".chroma")

COLLECTION_NAME = "memory_vault"


def get_vector_store():

    CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def index_memory(
    memory_id: str,
    user_id: str,
    memory_type: str,
    chunks: list[Document],
):

    vector_store = get_vector_store()

    # Remove existing chunks for this memory.
    # This makes re-indexing safe.
    vector_store.delete(
        where={
            "memory_id": memory_id
        }
    )

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

        documents.append(document)

        ids.append(
            f"{memory_id}_chunk_{index}"
        )

    if documents:

        vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

    return len(documents)