# Movie Q&A — Retrieval-Augmented Generation

A from-scratch Retrieval-Augmented Generation (RAG) pipeline for answering natural-language questions about movies, built around the classic bi-encoder retrieval → cross-encoder re-ranking → grounded LLM generation architecture (in the spirit of Lewis et al., 2020 and standard dense-retrieval RAG setups), running entirely on local/open-source models — no hosted LLM APIs.

## Architecture

1. **Preprocessing** — the [Kaggle "The Movies Dataset"](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (`movies_metadata.csv` + `credits.csv`) is cleaned, deduplicated, and merged into a ~41K-movie corpus. Each movie is flattened into a single `document_text` block (title, year, genres, director, cast, overview).
2. **Retrieval** — documents are embedded with a Sentence-Transformers bi-encoder (`BAAI/bge-small-en-v1.5`) and indexed in FAISS (`IndexFlatIP` over L2-normalized vectors, i.e. exact cosine similarity).
3. **Re-ranking** — top candidates are re-scored with a CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), plus a title-aware boost to fix sequel/franchise title collisions (e.g. *Alien* vs *Alien³*, *Avatar* vs *My Avatar and Me*).
4. **Generation** — a small local instruction-tuned model (`Qwen/Qwen2.5-1.5B-Instruct`) generates a grounded answer from the retrieved context, with an explicit refusal path when the requested movie isn't in the corpus (avoids confidently hallucinating about movies like *Oppenheimer* that post-date the dataset).
5. **Evaluation** — retrieval metrics (Recall@5, Recall@10, MRR), a local RAGAS pass (Qwen2.5-7B-Instruct judge, 4-bit) for faithfulness / answer relevancy / context precision, and a custom keyword-overlap faithfulness fallback for when the local judge is unreliable.

## Public interface

```python
run_query(query: str) -> dict
# {"answer": str, "retrieved_titles": list[str]}
```

Wraps the full pipeline behind a single safe entrypoint — handles empty input, nonsense queries, non-English input, and unanswerable queries without raising.

## Ablations

- **A — bi-encoder size**: `bge-small` vs `bge-base` — identical retrieval quality, small model wins on cost.
- **B — top-k sweep**: candidate pool size before re-ranking (k = 5, 10, 20) vs quality/latency.
- **C — prompt strictness**: permissive vs strict+guarded generation prompt — strict prompt cuts hallucinations on unanswerable queries from 4/4 to 1/4.

Full results, interpretation, and a failure diary (retrieval misses, sequel confusion, corpus-absence hallucinations, with before/after fixes) are in the Analysis Report section at the end of the notebook.

## Stack

Python, pandas, FAISS, Sentence-Transformers, Hugging Face Transformers, Qwen2.5, RAGAS — developed and run on Google Colab (T4 GPU).

## Files

- [`movie_qa_rag.ipynb`](movie_qa_rag.ipynb) — full pipeline: preprocessing, indexing, retrieval, re-ranking, generation, ablations, evaluation, and the analysis report.

## Running it

1. Download [`movies_metadata.csv` and `credits.csv`](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle and place both files in `MyDrive/Data/` in your Google Drive (the notebook expects them at `/content/drive/MyDrive/Data/movies_metadata.csv` and `.../credits.csv`).
2. Open `movie_qa_rag.ipynb` in Google Colab (a GPU runtime, e.g. T4, is recommended), mount Drive when prompted, and run cells top to bottom.
3. All models (embedding, cross-encoder, generator, RAGAS judge) are pulled from Hugging Face at runtime — no API keys required.

See the notebook's "Run Evidence" section for the exact two-session (pipeline + RAGAS judge) runtime structure used to fit everything in a 16 GB T4.
