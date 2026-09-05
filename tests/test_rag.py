from memory.memory_service import list_memories
from rag.retriever import retrieve
from rag.chunker import split_text
from rag.vector_store import index_memory

from config.settings import settings


USER_ID = settings.DEMO_USER_ID


memories = list_memories(
    USER_ID
)


for memory in memories:

    print("\n" + "=" * 70)

    print(
        f"Memory: {memory['title']}"
    )

    print(
        f"Memory ID: {memory['id']}"
    )

    print(
        f"Type: {memory['memory_type']}"
    )

    text = memory["extracted_text"]

    chunks = split_text(
        text
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    count = index_memory(
        memory_id=memory["id"],
        user_id=memory["user_id"],
        memory_type=memory["memory_type"],
        chunks=chunks,
    )

    print(
        f"Chunks indexed: {count}"
    )


question = input(
    "\nEnter your question: "
)


results = retrieve(
    question=question,
    user_id=USER_ID,
    top_k=3,
)


print("\n" + "=" * 70)

print("RETRIEVED RESULTS")

print("=" * 70)


for rank, (document, distance) in enumerate(
    results,
    start=1
):

    print(
        f"\nResult {rank}"
    )

    print(
        f"Memory ID: "
        f"{document.metadata['memory_id']}"
    )

    print(
        f"Memory Type: "
        f"{document.metadata['memory_type']}"
    )

    print(
        f"Chunk: "
        f"{document.metadata['chunk_index']}"
    )

    print(
        f"Distance: "
        f"{distance}"
    )

    print(
        "\nContent:"
    )

    print(
        document.page_content
    )