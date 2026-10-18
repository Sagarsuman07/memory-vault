import sqlite3
from pathlib import Path
import chromadb

db_path = Path("memory_vault.db")
chroma_path = Path(".chroma")

if not db_path.exists():
    raise SystemExit(f"Database not found: {db_path.resolve()}")
if not chroma_path.exists():
    raise SystemExit(f"Chroma directory not found: {chroma_path.resolve()}")

# Read-only access to SQLite.
conn = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

client = chromadb.PersistentClient(path=str(chroma_path))
collection = client.get_collection("memory_vault")

markers = [
    "MV_PDF_TEST_ALPHA_7319",
    "MV_PDF_TEST_BETA_8426",
    "MV_PDF_TEST_GAMMA_9537",
    "MV_DOCX_OVERVIEW_1642",
    "MV_DOCX_TECHNICAL_2753",
    "MV_TABLE_3864",
    "MV_TABLE_4975",
]

print("\n=== 1. Find uploaded test memories ===")
rows = conn.execute(
    "SELECT id, extracted_text FROM memories"
).fetchall()

found_memory_ids = set()
for row in rows:
    content = row["extracted_text"] or ""
    matches = [marker for marker in markers if marker in content]
    if matches:
        found_memory_ids.add(str(row["id"]))
        print(f"\nMemory ID: {row['id']}")
        print("Markers found in extracted text:", ", ".join(matches))

print("\n=== 2. Inspect indexed chunks ===")
if not found_memory_ids:
    print("No test markers found in memories.extracted_text.")
    print("Check ingestion status and confirm the marker text was extracted.")
else:
    for memory_id in sorted(found_memory_ids):
        result = collection.get(
            where={"memory_id": memory_id},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        print(f"\nMemory ID: {memory_id} | Indexed chunks: {len(docs)}")
        for i, doc in enumerate(docs):
            meta = metadatas[i] or {}
            print(f"\nChunk {i + 1} | characters={len(doc or '')}")
            print("Metadata:", meta)
            print("Text preview:", (doc or "").replace("\n", " ")[:220])

conn.close()
