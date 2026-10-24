# Phase 0 — Ingestion baseline and safety report

Captured on 2026-10-24 before the Phase 1 instrumentation commit.

## Repository safety baseline

- Repository: `Sagarsuman07/memory-vault`
- Working branch: `ingestion-optimisation`
- Baseline branch head: `735e689d540a963b5cb1c7d4b53d98d0c318fc76` — `Fixed vector store import issue`
- `main` / merge-base commit: `b1b4599d22e9af3ddc2efcceaab94ec27b43a4fc`
- Baseline comparison: 12 commits ahead, 0 behind.
- No production database or Chroma collection was modified for this baseline.
- The repository's `logs/ingestion.log` is tracked in Git. Treat it as an existing runtime artifact; do not append fresh test logs to it as part of instrumentation changes.

## Test status and limitations

A fresh baseline test run could not be completed from the available execution environment:

- A shell clone attempt failed because the environment could not resolve `github.com`.
- The GitHub Actions API reported zero workflow runs for `ingestion-optimisation` on 2026-10-24.
- Therefore, no test result in this report should be treated as a fresh run against baseline commit `735e689`.

The last test result previously reported for an earlier branch state was **67 passed, 4 failed** in approximately 5m22s; the focused logging tests had **8 passed**. One reported failure was the missing `logging` import in `rag/vector_store.py`, addressed in baseline commit `735e689` (`Fixed vector store import issue`). The other reported failures were:

1. `tests/test_phase6_contracts.py` — extraction error classification call-signature mismatch.
2. `tests/test_phase6_contracts.py` — indexing error classification call-signature mismatch.
3. `tests/test_settings.py` — test expected worker concurrency 1 while the configured default was 2.

Those three failures must be rerun and classified on the current branch before deciding whether they remain. They are not declared pre-existing solely from the historical output.

## Existing ingestion timing observations

These values were parsed from the branch's existing `logs/ingestion.log`. They are historical observations, not a controlled benchmark; sample counts are small and stage timing scopes may overlap.

| Event/stage | Samples | Average (s) | Maximum (s) |
|---|---:|---:|---:|
| PDF text extraction | 8 | 39.301 | 298.407 |
| Full extraction stage | 6 | 55.360 | 299.117 |
| Chunking | 6 | 0.088 | 0.392 |
| Embedding stage | 6 | 37.960 | 321.665 |
| Chroma indexing | 14 | 2.584 | 20.066 |
| Index-memory total | 6 | 103.726 | 335.511 |
| Final index verification | 14 | 0.666 | 1.620 |
| Successful ingestion job | 14 | 80.199 | 637.322 |
| Duplicate vector clone | 8 | 10.280 | 34.779 |
| Duplicate reuse job | 8 | 10.947 | 35.847 |
| Upload queued | 8 | 2.671 | 3.504 |
| Embedding batch | 26 | 9.332 | 23.193 |
| Chroma upsert batch | 23 | 0.877 | 2.324 |
| PDF page progress | 96 | 0.138 | 9.799 |

Notable samples:

- An 895-chunk PDF job took 637.322 seconds end-to-end; extraction was 299.117 seconds and embedding 321.665 seconds. Chunking was 0.392 seconds.
- An 85-page, 349-chunk PDF job took 260.311 seconds; embedding was 216.336 seconds, extraction 11.985 seconds, and Chroma indexing 20.066 seconds.
- A PDF page with OCR took 9.799 seconds in one logged sample. OCR is disabled by default, so this is not representative of all PDFs.
- Duplicate-clone timings varied widely; the cause needs measurement rather than assumption.

## Relevant current configuration

- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Embedding batch size: `32`
- Ingestion worker concurrency: `2`
- Embedding concurrency: `1`
- Chunk size / overlap: `1000 / 200`
- PDF OCR: disabled by default; configured DPI `144`; minimum page text threshold `20` characters.

## Phase 0 conclusion

The available log suggests that embedding and PDF extraction can dominate large ingestion jobs, while chunking is comparatively inexpensive. This is a hypothesis for Phase 2 benchmarking, not proof of a universal bottleneck. Queue wait, initialization time, lock wait/hold, OCR render time, and detailed Chroma operation timings were not fully measured in the historical logs.

A repeatable test-suite run and controlled benchmark should be captured when an execution environment is available. Phase 1 adds instrumentation only; performance changes are deferred to later phases.
