# Hybrid Semantic Legal Search System

This project now includes code for the main retrieval stages of a legal AI search system:

- BM25 retrieval
- Elasticsearch-backed BM25 integration
- Dense semantic search with Sentence Transformers and FAISS
- Hybrid retrieval pipeline
- Reciprocal Rank Fusion re-ranking
- Temporal filtering for recency-weighted legal results
- Benchmark evaluation with Precision@k, Recall@k, and NDCG@k

## Project Structure

- `main.py`: command-line demo
- `app.py`: local Flask web UI
- `evaluation.py`: benchmark metrics across graded legal query sets
- `legal_search/`: retrieval modules
- `data/legal_cases.json`: real-world judgment corpus plus imported official law/source records
- `data/applicable_laws.json`: intentionally empty in actual-data-only mode; manual law guidance is disabled
- `data/india_code_sources.json`: official source registry for Acts/sections, primarily India Code, plus regulator/government sources where appropriate
- `data/benchmark_queries.json`: benchmark query set with graded relevance labels
- `data/corpus_sources.md`: corpus source notes
- `tests/test_search.py`: retrieval and fusion tests

## Requirements

```bash
python -m pip install --user -r requirements.txt
```

## Run

```bash
python main.py "breach of contract damages"
```

## Web UI

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

The UI lets a user describe a real-life legal situation, choose BM25/vector fusion, enable temporal filtering, and view the final hybrid legal results.

Example situation queries:

```text
Can my landlord increase rent suddenly?
Do I need to mention my medical issue in insurance form?
How do I file a consumer complaint?
What happens if I break a contract?
Can someone use my photo without permission?
Is this agreement legally valid?
```

The system maps these questions to related legal areas such as tenancy law, insurance disclosure, consumer protection, contract law, privacy, copyright, and police/bail procedure. It then expands the retrieval query and returns related court judgments, statute sections, or official law source records.

Before searching, the local AI query converter rewrites the user's plain words into a legal-style query. For example:

```text
User input: What happens if I break a contract?
AI legal query: valid contract breach of agreement compensation and enforceability
```

The converted query is then passed into BM25, vector search, fusion, and temporal re-ranking.

The default demo runs the full project pipeline:

1. AI legal query conversion
2. BM25 keyword retrieval
3. Vector semantic search
4. Reciprocal Rank Fusion or weighted fusion
5. Temporal filtering to prefer recent precedents when relevance is close

You can tune or disable recency re-ranking:

```bash
python main.py --temporal-half-life-days 365 --temporal-boost 0.02 "environmental clearance mining pollution"
python main.py --disable-temporal "breach of contract damages"
```

Optional Elasticsearch run:

```bash
python main.py --use-elasticsearch --fusion rrf "environmental clearance mining pollution"
```

If Elasticsearch is not running locally, the demo falls back to the in-memory BM25 backend.

If your Sentence Transformers model is already cached locally, the dense retriever uses it without network access. To allow a fresh model download, run:

```bash
python main.py --allow-model-download "breach of contract damages"
```

## Test

```bash
python -m unittest discover -s tests
```

## Evaluate

```bash
python evaluation.py
```

The benchmark reports:

- `Precision@k`: how many retrieved documents are relevant
- `Recall@k`: how many known relevant documents were retrieved
- `NDCG@k`: ranking quality using graded relevance labels

You can change the cutoff:

```bash
python evaluation.py --top-k 3
```

To let the evaluation script download the dense model when needed:

```bash
python evaluation.py --allow-model-download
```

## Import Actual Legal Data

The project includes an importer for an actual public Indian legal judgments dataset hosted on Hugging Face:

```bash
python scripts/import_hf_legal_dataset.py --limit 300
```

This downloads structured judgment records and rebuilds `data/legal_cases.json` using actual dataset records. Existing `IN-LAW-*` official law/source records are preserved by default. The original curated corpus is backed up to `data/legal_cases_curated_backup.json` the first time the importer runs. To append the old curated anchors for controlled academic benchmarking, use `--include-curated`.

Import official law/source records from the source registry:

```bash
python scripts/import_india_code_statutes.py
```

This reads `data/india_code_sources.json`, fetches official India Code/regulator/government pages, discovers official PDFs where available, extracts source text, and appends searchable `IN-LAW-*` records to `data/legal_cases.json`. Result cards label these as `Statute / Law Section` or `Statute / Law Source`.

In actual-data mode, `data/applicable_laws.json` is empty and the UI does not show manually curated law advice. Legal references come from retrieved source-backed documents. To benchmark this mode properly, create relevance labels in `data/benchmark_queries.json` using document IDs from the imported dataset and `IN-LAW-*` records.

## Current Progress

- `BM25 via Elasticsearch`: implemented in code, requires a running Elasticsearch instance
- `Keyword-based retrieval`: implemented on the legal corpus
- `Semantic search`: implemented with Sentence Transformers and FAISS
- `Hybrid pipeline`: implemented
- `RRF ranking`: implemented
- `System tested`: baseline evaluation script and unit tests added
- `Temporal filtering`: implemented
- `Real-world corpus validation`: implemented with an imported Indian judgment dataset and official law/source records
- `Precision/Recall/NDCG benchmark`: implemented
