"""Safely patch the Phase 3 ingestion call to pass extraction metadata to chunking.

Run this script from the Memory Vault project root. It creates a one-time
backup of memory/memory_service.py before changing exactly one matching line.
"""
from pathlib import Path
import shutil


target = Path("memory/memory_service.py")
if not target.is_file():
    raise SystemExit("Could not find memory/memory_service.py. Run this from the project root.")

text = target.read_text(encoding="utf-8")
old = "        chunks = split_text(extracted_text)\n"
new = "        chunks = split_text(extracted_text, metadata=source_document.metadata)\n"

if new in text:
    print("Phase 4 memory_service patch is already applied; no changes made.")
    raise SystemExit(0)

count = text.count(old)
if count != 1:
    raise SystemExit(
        f"Expected exactly one target call in {target}, found {count}. "
        "No files were changed; inspect the file and patch manually if needed."
    )

backup = target.with_suffix(target.suffix + ".phase4.bak")
if backup.exists():
    raise SystemExit(
        f"Backup already exists at {backup}; no files were changed. "
        "Inspect it before choosing a different backup name."
    )

shutil.copy2(target, backup)
target.write_text(text.replace(old, new, 1), encoding="utf-8")
print(f"Updated {target}; backup created at {backup}.")
