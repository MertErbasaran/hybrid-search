"""Build the index:  python -m search.build"""

import argparse
import time
from pathlib import Path

from .data import download_scifact, load_corpus
from .embedder import DEFAULT_MODEL, SentenceTransformerEmbedder
from .index import SearchIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SciFact and build the search index.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="sentence-transformers model name")
    parser.add_argument("--index-dir", type=Path, default=Path("data/index"))
    args = parser.parse_args()

    dataset_dir = download_scifact()
    docs = load_corpus(dataset_dir)
    print(f"Loaded {len(docs):,} documents. Embedding with {args.model} ...")

    start = time.perf_counter()
    SearchIndex.build(docs, SentenceTransformerEmbedder(args.model), args.index_dir)
    print(f"Index written to {args.index_dir} in {time.perf_counter() - start:.0f}s.")


if __name__ == "__main__":
    main()
