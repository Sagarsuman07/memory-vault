"""Benchmark embedding batch sizes without touching SQLite or Chroma.

Input format: UTF-8 text file with one already-chunked document per line.
Use the same extracted/chunked corpus for every run to make comparisons useful.

Example:
    python scripts/benchmark_embedding_batches.py --input benchmark_chunks.txt --repeats 3

This script only calls the configured local embedding model. It does not write to
the vector store or change application settings.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Sequence


def _parse_batch_sizes(value: str) -> list[int]:
    try:
        sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("Batch sizes must be comma-separated integers.") from error
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("Provide one or more positive batch sizes.")
    if len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("Batch sizes must not contain duplicates.")
    return sizes


def _embed_corpus(embedder, texts: Sequence[str], batch_size: int) -> tuple[list[list[float]], float]:
    vectors: list[list[float]] = []
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start:start + batch_size])
        vectors.extend(embedder.embed_documents(batch))
    elapsed = time.perf_counter() - started
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"Embedding count mismatch: expected {len(texts)}, received {len(vectors)}."
        )
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1 or not next(iter(dimensions), 0):
        raise RuntimeError("Embedding model returned empty or inconsistent vector dimensions.")
    if any(not math.isfinite(float(value)) for vector in vectors for value in vector):
        raise RuntimeError("Embedding model returned a non-finite vector value.")
    return [[float(value) for value in vector] for vector in vectors], elapsed


def _parity(reference: Sequence[Sequence[float]], candidate: Sequence[Sequence[float]]) -> dict:
    max_abs_diff = 0.0
    min_cosine = 1.0
    for left, right in zip(reference, candidate):
        if len(left) != len(right):
            return {"same_dimensions": False, "max_abs_diff": None, "min_cosine_similarity": None}
        max_abs_diff = max(max_abs_diff, max(abs(a - b) for a, b in zip(left, right)))
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
        min_cosine = min(min_cosine, cosine)
    return {
        "same_dimensions": True,
        "max_abs_diff": max_abs_diff,
        "min_cosine_similarity": min_cosine,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="UTF-8 text file with one existing chunk per line.")
    parser.add_argument("--batch-sizes", type=_parse_batch_sizes, default=_parse_batch_sizes("16,32,64"))
    parser.add_argument("--repeats", type=int, default=3,
                        help="Measured repetitions per batch size (default: 3).")
    parser.add_argument("--warmup-items", type=int, default=8,
                        help="Number of corpus items used to warm the model (default: 8).")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON output path; otherwise results print to stdout.")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if args.warmup_items < 0:
        parser.error("--warmup-items cannot be negative.")
    if not args.input.is_file():
        parser.error(f"Input file does not exist: {args.input}")

    texts = [line.strip() for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not texts:
        parser.error("Input file contains no non-empty chunk lines.")

    # Import after argument validation so usage errors do not initialize model dependencies.
    from config.settings import settings
    from rag.embeddings import get_embeddings

    init_started = time.perf_counter()
    embedder = get_embeddings()
    initialization_seconds = time.perf_counter() - init_started
    warm_started = time.perf_counter()
    _ = get_embeddings()
    warm_access_seconds = time.perf_counter() - warm_started

    warmup_count = min(args.warmup_items, len(texts))
    if warmup_count:
        _embed_corpus(embedder, texts[:warmup_count], max(1, min(32, warmup_count)))

    reference_size = 32 if 32 in args.batch_sizes else args.batch_sizes[0]
    results = []
    vectors_by_size: dict[int, list[list[float]]] = {}
    for batch_size in args.batch_sizes:
        durations = []
        representative_vectors = None
        for _ in range(args.repeats):
            vectors, elapsed = _embed_corpus(embedder, texts, batch_size)
            durations.append(elapsed)
            if representative_vectors is None:
                representative_vectors = vectors
        vectors_by_size[batch_size] = representative_vectors or []
        results.append({
            "batch_size": batch_size,
            "repeats": args.repeats,
            "seconds": [round(value, 4) for value in durations],
            "median_seconds": round(statistics.median(durations), 4),
            "min_seconds": round(min(durations), 4),
            "max_seconds": round(max(durations), 4),
            "chunks_per_second_median": round(len(texts) / statistics.median(durations), 3),
        })

    reference = vectors_by_size[reference_size]
    for result in results:
        result["parity_vs_reference"] = _parity(reference, vectors_by_size[result["batch_size"]])

    report = {
        "benchmark": "embedding_batch_size",
        "input_file": str(args.input),
        "chunk_count": len(texts),
        "embedding_model": settings.EMBEDDING_MODEL,
        "initialization_seconds_first_access": round(initialization_seconds, 4),
        "initialization_seconds_cached_access": round(warm_access_seconds, 6),
        "warmup_items": warmup_count,
        "reference_batch_size": reference_size,
        "results": results,
        "notes": [
            "No SQLite or Chroma operations are performed.",
            "Run on the same machine with the same input corpus and power mode.",
            "Compare medians and spread, not only the fastest run.",
            "Monitor peak process memory separately (for example, Task Manager on Windows).",
            "A faster embedding batch size does not prove a faster full ingestion; Chroma upsert must be measured separately.",
        ],
    }
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
