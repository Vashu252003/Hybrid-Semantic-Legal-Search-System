import argparse
import json
import sys
from pathlib import Path

from legal_search import (
    BM25Index,
    Document,
    ElasticsearchBM25Index,
    HybridSearchPipeline,
    SentenceTransformerFaissIndex,
    TemporalFilter,
    VectorIndex,
    convert_to_legal_query,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "legal_cases.json"


def load_documents() -> list[Document]:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)
    return [Document(**item) for item in raw_documents]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid semantic legal search demo")
    parser.add_argument("query", nargs="*", help="Search query")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k")
    parser.add_argument("--fusion", choices=["weighted", "rrf"], default="rrf")
    parser.add_argument("--disable-temporal", action="store_true")
    parser.add_argument("--temporal-half-life-days", type=int, default=730)
    parser.add_argument("--temporal-boost", type=float, default=0.005)
    parser.add_argument("--use-elasticsearch", action="store_true")
    parser.add_argument("--es-host", default="http://localhost:9200")
    parser.add_argument("--es-index", default="legal_cases")
    parser.add_argument(
        "--dense-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow Sentence Transformers to download a model if it is not already cached locally.",
    )
    return parser.parse_args(argv)


def print_results(label: str, results) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    if not results:
        print("No matching legal documents found.")
        return

    for index, result in enumerate(results, start=1):
        document = result.document
        print(f"{index}. {document.title} [{document.id}]")
        print(f"   Court: {document.court}")
        print(f"   Date: {document.date}")
        print(f"   Score: {result.score:.4f}")


def print_guidance(conversion) -> None:
    guidance = conversion.guidance
    print("\nLegal Search Result")
    print("-------------------")
    print(f"Category: {guidance.legal_category}")
    print(f"Risk Level: {guidance.severity_level}")
    print(f"Problem Summary: {guidance.problem_summary}")

    print("\nApplicable Laws:")
    if guidance.applicable_laws:
        for index, law in enumerate(guidance.applicable_laws, start=1):
            print(f"{index}. {law.law} - {law.title}")
            print(f"   {law.when_applicable}")
    else:
        print(f"- {guidance.important_note}")

    if guidance.important_note and guidance.applicable_laws:
        print(f"\nImportant Note: {guidance.important_note}")

    print("\nSuggested Action:")
    for index, action in enumerate(guidance.suggested_actions, start=1):
        print(f"{index}. {action}")

    print("\nEvidence Required:")
    for item in guidance.evidence_required:
        print(f"- {item}")

    print(f"\nRecommended Authority: {', '.join(guidance.recommended_authority)}")
    print(f"Final Advice: {guidance.final_advice}")
    print(f"Disclaimer: {guidance.disclaimer}")


def build_bm25_backend(documents: list[Document], args: argparse.Namespace):
    if not args.use_elasticsearch:
        return BM25Index(documents), "In-memory BM25"

    try:
        backend = ElasticsearchBM25Index(
            index_name=args.es_index,
            hosts=[args.es_host],
        )
        if not backend.ping():
            raise ConnectionError(f"Elasticsearch is unreachable at {args.es_host}")
        backend.ensure_index()
        backend.index_documents(documents)
        return backend, "Elasticsearch BM25"
    except Exception as error:  # pragma: no cover - depends on local service
        print(f"Elasticsearch backend unavailable: {error}")
        print("Falling back to in-memory BM25 for this run.")
        return BM25Index(documents), "In-memory BM25"


def build_vector_backend(documents: list[Document], args: argparse.Namespace):
    try:
        backend = SentenceTransformerFaissIndex(
            documents,
            model_name=args.dense_model,
            allow_download=args.allow_model_download,
        )
        return backend, getattr(
            backend,
            "backend_label",
            f"Sentence Transformers Dense Search ({args.dense_model})",
        )
    except Exception as error:  # pragma: no cover - depends on local model availability
        print(f"Dense vector backend unavailable: {error}")
        print("Falling back to local vector retrieval for this run.")
        return VectorIndex(documents), "Local TF-IDF Vector Search"


def main() -> None:
    args = parse_args(sys.argv[1:])
    query = " ".join(args.query).strip() or "breach of contract damages"
    conversion = convert_to_legal_query(query)
    retrieval_query = conversion.search_query
    print("Loading legal corpus...", flush=True)
    documents = load_documents()

    print("Preparing BM25 backend...", flush=True)
    bm25_backend, bm25_label = build_bm25_backend(documents, args)
    print("Preparing vector backend...", flush=True)
    vector_backend, vector_label = build_vector_backend(documents, args)
    temporal_filter = None
    if not args.disable_temporal:
        temporal_filter = TemporalFilter(
            half_life_days=args.temporal_half_life_days,
            boost_strength=args.temporal_boost,
        )
    print("Running search...", flush=True)
    hybrid = HybridSearchPipeline(
        documents,
        bm25_backend=bm25_backend,
        vector_backend=vector_backend,
        fusion_mode=args.fusion,
        candidate_pool=max(args.top_k * 2, 10),
        temporal_filter=temporal_filter,
    )

    print("Hybrid Semantic Legal Search System")
    print(f"User Situation: {query}")
    print(f"Legal Areas: {', '.join(conversion.legal_areas)}")
    print(f"AI Converted Legal Query: {conversion.legal_query}")
    print(f"Expanded Retrieval Query: {retrieval_query}")
    print(f"BM25 Backend: {bm25_label}")
    print(f"Vector Backend: {vector_label}")
    print(f"Fusion: {args.fusion.upper()}")
    if temporal_filter is None:
        print("Temporal Filtering: disabled")
    else:
        print(
            "Temporal Filtering: enabled "
            f"(half-life={args.temporal_half_life_days} days, boost={args.temporal_boost:.3f})"
        )

    print_guidance(conversion)
    print_results("BM25 Results", bm25_backend.search(retrieval_query, top_k=args.top_k))
    print_results("Vector Results", vector_backend.search(retrieval_query, top_k=args.top_k))
    print_results(f"{args.fusion.upper()} Hybrid Results", hybrid.search(retrieval_query, top_k=args.top_k))


if __name__ == "__main__":
    main()
