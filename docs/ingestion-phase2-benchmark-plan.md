# Phase 2 — Baseline analysis and low-risk benchmark plan

## Scope and safety

Phase 2 starts with measurement and a standalone embedding benchmark. No production defaults are changed in this commit. The benchmark harness calls the configured local embedding model only; it does not open SQLite, create a Chroma client, write vectors, or change chunking/indexing behavior.

- Repository: `Sagarsuman07/memory-vault`
- Branch: `ingestion-optimisation`
- `main` remains untouched.
- Do not use production DB or Chroma data for destructive benchmark runs.
- Do not enable startup warm-up until its startup cost, memory impact, and first-job benefit have been measured.

## Findings from the 2026-10-24 instrumented run

These are observations from one local session, not controlled before/after benchmark results.

| Job / stage | Duration | Interpretation |
|---|---:|---|
| 465-page PDF, 895 chunks, end-to-end | 962.706 s | Slowest observed fresh ingestion |
| Embedding for that PDF | 588.503 s | About 61.1% of end-to-end duration; first optimization candidate |
| PDF extraction stage | 293.995 s | About 30.5%; second major candidate |
| Chroma indexing | 69.350 s | About 7.2%; upsert itself took 65.081 s |
| Queue wait | 5.136 s | About 0.5%; not a primary bottleneck in this sample |
| Chunking | 0.311 s | Negligible relative to the major stages |
| Duplicate reuse, 895 chunks | 54.784 s | Much faster than fresh indexing, but cloning took 44.476 s |
| OCR on the 465-page PDF | 40/40 eligible pages succeeded | Provider calls totalled 122.841 s; page render totalled 9.975 s |

The embedding and extraction percentages are stage-to-job comparisons, not additive partitions: instrumentation scopes can overlap. Per-page progress summed to 289.087 s, while the measured text extraction, render, and vision-provider timers do not explain all of that wall time. If extraction is optimized later, add a more detailed breakdown before assuming OCR provider latency is the only cause.

Lock wait times were approximately zero in this single-user session. Long lock-hold durations corresponded to the work performed inside the critical section; do not remove or weaken locks based on these results alone.

Initialization observations: embedding model initialization was 12.239 s. The first Chroma wrapper construction was about 16.5 s; later wrapper constructions were usually much shorter. This log does not cleanly identify cold versus warm process starts, so repeat that measurement in fresh processes before deciding whether startup warm-up or client reuse is worthwhile.

## Embedding batch-size benchmark

A standalone script is available at `scripts/benchmark_embedding_batches.py`. It benchmarks embedding calls with candidate outer batch sizes, records per-run duration and throughput, and checks vector parity against batch size 32 (or the first supplied size if 32 is omitted). It does not index anything.

### Prepare the input

Create a local UTF-8 text file with one representative, already-chunked text per line. Use actual chunk text from a representative large document, ideally around 895 chunks, and do not commit private document content. Reuse exactly the same file for all candidate runs. Blank lines are ignored.

### Run on the same machine/environment as Memory Vault

```powershell
python scripts/benchmark_embedding_batches.py --input .\benchmark_chunks.txt --batch-sizes 16,32,64 --repeats 3 --output .\embedding-benchmark.json
```

Repeat the command in at least two fresh Python processes if comparing initialization costs. Save the output outside the repository or add only sanitized aggregate metrics to this report. Monitor peak memory in Windows Task Manager while each run executes; the script intentionally avoids a mandatory extra monitoring dependency.

### Interpretation rules

1. Compare medians and run-to-run spread for each batch size.
2. Check vector dimensions and parity against the reference. A timing win is not acceptable if embeddings materially differ.
3. Compare peak memory and whether the model or OS starts paging.
4. Keep `EMBEDDING_BATCH_SIZE=32` unless repeated measurements show a stable benefit with acceptable memory use and parity.
5. This script measures embedding only. The setting also influences Chroma upsert batch sizing, so full ingestion must be benchmarked separately in a disposable local test environment before changing the production default.

## Duplicate-index and initialization follow-up

Current log evidence shows a 44.476 s clone stage for 895 vectors, including 44.109 s reading source vectors under the Chroma lock. Before changing clone logic, repeat duplicate-upload benchmarks for at least a small and a large memory, verify source/target vector counts and content, and capture clone read/copy/write times separately. No clone algorithm change is included in this phase's initial commit.

For initialization, measure (a) first `get_embeddings()` access in a fresh process, (b) subsequent cached access, (c) first Chroma client construction in a fresh process, and (d) later constructions in the same process. Consider warm-up only if first-ingestion latency improves enough to justify the startup delay and memory cost.

## Current Phase 2 status

- [x] Analyze the latest instrumented log.
- [x] Add a non-destructive embedding batch-size benchmark harness.
- [ ] Run repeated batch-size measurements locally and attach the JSON results.
- [ ] Compare cold/warm initialization across fresh processes.
- [ ] Repeat duplicate-clone timing with small and large memories.
- [ ] Measure Chroma upsert separately for candidate batch sizes in a disposable test environment.
- [ ] Decide whether any runtime default change is justified by evidence.

No runtime optimization is enabled by this initial Phase 2 change. Parallel OCR and extraction/embedding pipelining remain out of scope; they belong to later phases and require separate implementation and regression validation.
