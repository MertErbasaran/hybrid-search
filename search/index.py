"""The search index: SQLite FTS5 for keyword search, a NumPy matrix for vectors.

Layout on disk (one folder):
    docs.sqlite      documents + FTS5 full-text index (BM25 ranking built in)
    embeddings.npy   one normalized vector per document, row i = rowid i+1
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import Document
from .embedder import Embedder
from .fusion import reciprocal_rank_fusion

MODES = ("bm25", "vector", "hybrid")
_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass
class Hit:
    doc_id: str
    title: str
    snippet: str
    score: float


def _fts_query(text: str) -> str | None:
    """Turn free user text into a safe FTS5 query.

    Raw input like `covid-19 (ACE2)?` would be a syntax error in FTS5, so every
    token is quoted and the tokens are OR-ed. BM25 then rewards documents that
    match more (and rarer) terms.
    """
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return None
    return " OR ".join(f'"{t}"' for t in tokens)


class SearchIndex:
    def __init__(self, conn: sqlite3.Connection, embeddings: np.ndarray, embedder: Embedder):
        self.conn = conn
        self.embeddings = embeddings
        self.embedder = embedder

    # ---------- building and loading ----------

    @classmethod
    def build(cls, docs: list[Document], embedder: Embedder, index_dir: Path) -> "SearchIndex":
        index_dir.mkdir(parents=True, exist_ok=True)
        db_path = index_dir / "docs.sqlite"
        db_path.unlink(missing_ok=True)

        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.executescript(
            """
            CREATE TABLE docs (
                rowid  INTEGER PRIMARY KEY,
                doc_id TEXT NOT NULL UNIQUE,
                title  TEXT,
                text   TEXT
            );
            -- External-content FTS5 table: the text is stored once in `docs`,
            -- the FTS table only holds the inverted index.
            -- Porter stemming lets "infections" match "infection".
            CREATE VIRTUAL TABLE docs_fts USING fts5(
                title, text,
                content='docs', content_rowid='rowid',
                tokenize='porter unicode61'
            );
            """
        )
        conn.executemany(
            "INSERT INTO docs (rowid, doc_id, title, text) VALUES (?, ?, ?, ?)",
            [(i + 1, d.doc_id, d.title, d.text) for i, d in enumerate(docs)],
        )
        conn.execute("INSERT INTO docs_fts(docs_fts) VALUES ('rebuild')")
        conn.commit()

        texts = [f"{d.title}. {d.text}" if d.title else d.text for d in docs]
        try:
            embeddings = embedder.encode(texts, show_progress=True)  # type: ignore[call-arg]
        except TypeError:
            embeddings = embedder.encode(texts)
        np.save(index_dir / "embeddings.npy", embeddings)
        return cls(conn, embeddings, embedder)

    @classmethod
    def load(cls, index_dir: Path, embedder: Embedder) -> "SearchIndex":
        db_path = index_dir / "docs.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"No index in {index_dir}. Run `python -m search.build` first.")
        conn = sqlite3.connect(db_path, check_same_thread=False)
        embeddings = np.load(index_dir / "embeddings.npy")
        return cls(conn, embeddings, embedder)

    # ---------- rankings (lists of rowids, best first) ----------

    def bm25_ranking(self, query: str, n: int) -> list[int]:
        fts = _fts_query(query)
        if fts is None:
            return []
        # bm25() returns lower = better; title matches weighted 2x over body.
        rows = self.conn.execute(
            "SELECT rowid FROM docs_fts WHERE docs_fts MATCH ? "
            "ORDER BY bm25(docs_fts, 2.0, 1.0) LIMIT ?",
            (fts, n),
        ).fetchall()
        return [r[0] for r in rows]

    def vector_ranking(self, query_vec: np.ndarray, n: int) -> list[int]:
        # Rows are normalized, so the dot product is the cosine similarity.
        sims = self.embeddings @ query_vec
        n = min(n, len(sims))
        top = np.argpartition(-sims, n - 1)[:n]  # top-n, unsorted, O(N)
        top = top[np.argsort(-sims[top])]  # sort only those n
        return [int(i) + 1 for i in top]  # array index -> rowid

    def hybrid_ranking(
        self, query: str, query_vec: np.ndarray, n: int, candidates: int = 100, rrf_k: int = 60
    ) -> list[int]:
        fused = reciprocal_rank_fusion(
            [self.bm25_ranking(query, candidates), self.vector_ranking(query_vec, candidates)],
            k=rrf_k,
        )
        return [rowid for rowid, _ in fused[:n]]

    def ranking(self, query: str, mode: str, n: int, query_vec: np.ndarray | None = None) -> list[int]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            return self.bm25_ranking(query, n)
        if query_vec is None:
            query_vec = self.embedder.encode([query])[0]
        if mode == "vector":
            return self.vector_ranking(query_vec, n)
        return self.hybrid_ranking(query, query_vec, n)

    # ---------- user-facing search ----------

    def doc_ids(self, rowids: list[int]) -> list[str]:
        if not rowids:
            return []
        marks = ",".join("?" * len(rowids))
        lookup = dict(self.conn.execute(f"SELECT rowid, doc_id FROM docs WHERE rowid IN ({marks})", rowids))
        return [lookup[r] for r in rowids]

    def search(self, query: str, mode: str = "hybrid", k: int = 10) -> list[Hit]:
        rowids = self.ranking(query, mode, k)
        if not rowids:
            return []
        marks = ",".join("?" * len(rowids))
        rows = {
            r[0]: r[1:]
            for r in self.conn.execute(
                f"SELECT rowid, doc_id, title, text FROM docs WHERE rowid IN ({marks})", rowids
            )
        }
        hits = []
        for rank, rowid in enumerate(rowids, start=1):
            doc_id, title, text = rows[rowid]
            snippet = text if len(text) <= 300 else text[:300].rsplit(" ", 1)[0] + " …"
            # Score shown to users: simple rank-based score, comparable across modes.
            hits.append(Hit(doc_id=doc_id, title=title, snippet=snippet, score=round(1 / rank, 4)))
        return hits
