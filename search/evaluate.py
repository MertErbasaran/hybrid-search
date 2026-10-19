"""Compare BM25, vector and hybrid search on the SciFact test set.

    python -m search.evaluate

Prints a Markdown table and saves it to results.md.
"""

import argparse
import time
from pathlib import Path

from .data import download_scifact, load_qrels, load_queries
from .embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from .index import MODES, SearchIndex
from .metrics import mrr_at_k, ndcg_at_k, recall_at_k


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    k = args.k

    dataset_dir = download_scifact()
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, "test")
    query_ids = [q for q in qrels if q in queries]

    embedder = SentenceTransformerEmbedder(args.model)
    index = SearchIndex.load(args.index_dir, embedder)

    # Embed all queries in one batch instead of one call per query.
    query_vecs = embedder.encode([queries[q] for q in query_ids])

    lines = [
        f"Evaluation on SciFact test set ({len(query_ids)} queries), model `{args.model}`",
        "",
        f"| Method | nDCG@{k} | Recall@{k} | MRR@{k} | ms / query |",
        "|---|---|---|---|---|",
    ]
    for mode in MODES:
        ndcg = recall = mrr = 0.0
        start = time.perf_counter()
        for qid, qvec in zip(query_ids, query_vecs):
            ranked = index.doc_ids(index.ranking(queries[qid], mode, k, query_vec=qvec))
            relevant = qrels[qid]
            ndcg += ndcg_at_k(ranked, relevant, k)
            recall += recall_at_k(ranked, relevant, k)
            mrr += mrr_at_k(ranked, relevant, k)
        ms = (time.perf_counter() - start) * 1000 / len(query_ids)
        n = len(query_ids)
        lines.append(f"| {mode} | {ndcg / n:.3f} | {recall / n:.3f} | {mrr / n:.3f} | {ms:.1f} |")

    lines.append("")
    lines.append("ms / query excludes query embedding (queries are embedded in one batch).")
    report = "\n".join(lines)
    print(report)
    Path("results.md").write_text(report + "\n", encoding="utf-8")
    print("\nSaved to results.md")


if __name__ == "__main__":
    main()
