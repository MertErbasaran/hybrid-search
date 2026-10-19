"""Web API and search page.

    uvicorn app.main:app --reload

Open http://127.0.0.1:8000 for the search page, /docs for the API docs.
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from search.embedder import SentenceTransformerEmbedder
from search.index import SearchIndex

INDEX_DIR = Path("data/index")
STATIC = Path(__file__).parent / "static"

state: dict = {}
lock = threading.Lock()  # one SQLite connection is shared, so serialize access


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["index"] = SearchIndex.load(INDEX_DIR, SentenceTransformerEmbedder())
    yield
    state.clear()


app = FastAPI(title="Hybrid Search", lifespan=lifespan)


class HitOut(BaseModel):
    doc_id: str
    title: str
    snippet: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    hits: list[HitOut]


@app.get("/api/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500),
    mode: Literal["bm25", "vector", "hybrid"] = "hybrid",
    k: int = Query(10, ge=1, le=50),
) -> SearchResponse:
    with lock:
        hits = state["index"].search(q, mode=mode, k=k)
    return SearchResponse(query=q, mode=mode, hits=[HitOut(**h.__dict__) for h in hits])


@app.get("/", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(STATIC / "index.html")
