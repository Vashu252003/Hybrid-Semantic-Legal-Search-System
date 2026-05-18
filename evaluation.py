import json
import argparse
import math
import sys
from pathlib import Path

from legal_search import BM25Index, Document, HybridSearchPipeline, SentenceTransformerFaissIndex, TemporalFilter, VectorIndex


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "legal_cases.json"
BENCHMARK_FILE = BASE_DIR / "data" / "benchmark_queries.json"


def load_documents() -> list[Document]:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)
    return [Document(**item) for item in raw_documents]


def load_benchmark_queries() -> list[dict]:
    with BENCHMARK_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def filter_benchmark_queries(queries: list[dict], documents: list[Document]) -> list[dict]:
    document_ids = {document.id for document in documents}
    filtered_queries = []
    for item in queries:
        relevance = {
            doc_id: grade
            for doc_id, grade in item["relevance"].items()
            if doc_id in document_ids
        }
        if relevance:
            filtered_queries.append({**item, "relevance": relevance})
    return filtered_queries


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval backends on the sample legal corpus.")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Sentence Transformers to download a model if it is not already cached locally.",
    )
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--disable-temporal", action="store_true")
    parser.add_argument("--temporal-half-life-days", type=int, default=730)
    parser.add_argument("--temporal-boost", type=float, default=0.005)
    return parser.parse_args(argv)


def build_vector_backend(documents: list[Document], args: argparse.Namespace):
    try:
        return SentenceTransformerFaissIndex(
            documents,
            model_name=args.dense_model,
            allow_download=args.allow_model_download,
        )
    except Exception as error:  # pragma: no cover - depends on local model availability
        print(f"Dense vector evaluation fallback: {error}")
        return VectorIndex(documents)


def reciprocal_rank(position: int | None) -> float:
    if position is None:
        return 0.0
    return 1.0 / position


def precision_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = ranked_ids[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved if relevance.get(doc_id, 0) > 0)
    return relevant_retrieved / k


def recall_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    relevant_ids = {doc_id for doc_id, grade in relevance.items() if grade > 0}
    if not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def dcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    score = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        grade = relevance.get(doc_id, 0)
        gain = (2**grade) - 1
        score += gain / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    ideal_ids = [
        doc_id
        for doc_id, _grade in sorted(
            relevance.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    ideal_dcg = dcg_at_k(ideal_ids, relevance, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance, k) / ideal_dcg


def evaluate_backend(name: str, backend, queries: list[dict], top_k: int) -> dict[str, float]:
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    for item in queries:
        results = backend.search(item["query"], top_k=top_k)
        ranked_ids = [result.document.id for result in results]
        relevance = item["relevance"]

        precision_scores.append(precision_at_k(ranked_ids, relevance, top_k))
        recall_scores.append(recall_at_k(ranked_ids, relevance, top_k))
        ndcg_scores.append(ndcg_at_k(ranked_ids, relevance, top_k))

    return {
        "name": name,
        f"precision@{top_k}": sum(precision_scores) / max(len(precision_scores), 1),
        f"recall@{top_k}": sum(recall_scores) / max(len(recall_scores), 1),
        f"ndcg@{top_k}": sum(ndcg_scores) / max(len(ndcg_scores), 1),
    }


def main() -> None:
    args = parse_args(sys.argv[1:])
    documents = load_documents()
    raw_benchmark_queries = load_benchmark_queries()
    benchmark_queries = filter_benchmark_queries(raw_benchmark_queries, documents)
    bm25 = BM25Index(documents)
    vector = build_vector_backend(documents, args)
    temporal_filter = None
    if not args.disable_temporal:
        temporal_filter = TemporalFilter(
            half_life_days=args.temporal_half_life_days,
            boost_strength=args.temporal_boost,
        )
    weighted_hybrid = HybridSearchPipeline(
        documents,
        bm25_backend=bm25,
        vector_backend=vector,
        fusion_mode="weighted",
        temporal_filter=temporal_filter,
    )
    rrf_hybrid = HybridSearchPipeline(
        documents,
        bm25_backend=bm25,
        vector_backend=vector,
        fusion_mode="rrf",
        temporal_filter=temporal_filter,
    )

    metrics = [
        evaluate_backend("BM25", bm25, benchmark_queries, args.top_k),
        evaluate_backend("Vector", vector, benchmark_queries, args.top_k),
        evaluate_backend("Weighted Hybrid", weighted_hybrid, benchmark_queries, args.top_k),
        evaluate_backend("RRF Hybrid", rrf_hybrid, benchmark_queries, args.top_k),
    ]

    print("Legal Search Benchmark")
    print("----------------------")
    print(f"Corpus documents: {len(documents)}")
    print(f"Benchmark queries: {len(benchmark_queries)} active / {len(raw_benchmark_queries)} configured")
    print(f"Top-k: {args.top_k}")
    print(
        "Temporal filtering: "
        + (
            "disabled"
            if temporal_filter is None
            else f"enabled (half-life={args.temporal_half_life_days}, boost={args.temporal_boost:.3f})"
        )
    )
    for item in metrics:
        print(
            f"{item['name']}: "
            f"Precision@{args.top_k}={item[f'precision@{args.top_k}']:.2f}, "
            f"Recall@{args.top_k}={item[f'recall@{args.top_k}']:.2f}, "
            f"NDCG@{args.top_k}={item[f'ndcg@{args.top_k}']:.2f}"
        )


if __name__ == "__main__":
    main()
