Evaluation on SciFact test set (300 queries), model `sentence-transformers/all-MiniLM-L6-v2`

| Method | nDCG@10 | Recall@10 | MRR@10 | ms / query |
|---|---|---|---|---|
| bm25 | 0.689 | 0.816 | 0.654 | 18.3 |
| vector | 0.648 | 0.788 | 0.607 | 1.0 |
| hybrid | 0.715 | 0.855 | 0.676 | 18.1 |

ms / query excludes query embedding (queries are embedded in one batch).
