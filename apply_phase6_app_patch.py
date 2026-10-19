"""Idempotently add background-worker startup to app.py.

Run from the project root: python apply_phase6_app_patch.py
Creates app.py.phase6.bak and refuses to overwrite an existing backup.
"""
from pathlib import Path
import shutil

path = Path("app.py")
backup = Path("app.py.phase6.bak")
if not path.is_file():
    raise SystemExit("app.py was not found. Run this script from the Memory Vault project root.")
text = path.read_text(encoding="utf-8")
marker = "from memory.background_worker import start_background_worker"
if marker in text and "start_background_worker()" in text:
    print("Phase 6 worker startup already appears to be installed; no changes made.")
    raise SystemExit(0)
if backup.exists():
    raise SystemExit(f"Refusing to overwrite existing backup: {backup}")
if "initialize_database()" not in text:
    raise SystemExit("Could not find initialize_database() call; app.py was not changed.")
# Validate all expected anchors before making a backup or changing the file.
insert_after = "from database.database import initialize_database"
if insert_after not in text:
    raise SystemExit("Could not find database initializer import; app.py was not changed.")
shutil.copy2(path, backup)
# Keep import alongside the other application imports, without duplicating it.
text = text.replace(insert_after, insert_after + "\nfrom memory.background_worker import start_background_worker", 1)
text = text.replace("initialize_database()", "initialize_database()\nstart_background_worker()", 1)
path.write_text(text, encoding="utf-8")
print(f"Updated {path}; original saved to {backup}.")
