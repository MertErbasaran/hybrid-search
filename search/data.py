"""Downloads and loads the SciFact dataset from the BEIR benchmark.

SciFact: ~5,200 scientific paper abstracts and 300 test queries (scientific
claims) with human relevance judgements. Small enough to index on a laptop,
but with real labels, so search quality can actually be measured.
"""

import csv
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path("data")
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str


def download_scifact(data_dir: Path = DATA_DIR) -> Path:
    """Download and unpack SciFact once. Returns the dataset folder."""
    target = data_dir / "scifact"
    if (target / "corpus.jsonl").exists():
        return target

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "scifact.zip"
    print(f"Downloading {SCIFACT_URL} ...")
    urllib.request.urlretrieve(SCIFACT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)
    zip_path.unlink()
    return target


def load_corpus(dataset_dir: Path) -> list[Document]:
    docs = []
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(Document(doc_id=str(row["_id"]), title=row.get("title", ""), text=row["text"]))
    return docs


def load_queries(dataset_dir: Path) -> dict[str, str]:
    queries = {}
    with open(dataset_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = row["text"]
    return queries


def load_qrels(dataset_dir: Path, split: str = "test") -> dict[str, dict[str, int]]:
    """Relevance judgements: query_id -> {doc_id: relevance}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(dataset_dir / "qrels" / f"{split}.tsv", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            relevance = int(row["score"])
            if relevance > 0:
                qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = relevance
    return qrels
