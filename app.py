from argparse import Namespace

from flask import Flask, render_template, request

from legal_search import HybridSearchPipeline, TemporalFilter, convert_to_legal_query
from main import build_bm25_backend, build_vector_backend, load_documents


app = Flask(__name__)

DOCUMENTS = load_documents()
DEFAULT_ARGS = Namespace(
    use_elasticsearch=False,
    es_host="http://localhost:9200",
    es_index="legal_cases",
    dense_model="sentence-transformers/all-MiniLM-L6-v2",
    allow_model_download=False,
)

BM25_BACKEND, BM25_LABEL = build_bm25_backend(DOCUMENTS, DEFAULT_ARGS)
VECTOR_BACKEND, VECTOR_LABEL = build_vector_backend(DOCUMENTS, DEFAULT_ARGS)


def document_type(document):
    if document.id.startswith("IN-LAW-") and "-SEC-" in document.id:
        return "Statute / Law Section"
    if document.id.startswith("IN-LAW-"):
        return "Statute / Law Source"
    if document.id.startswith("HF-INDIAN-JUDGMENT-"):
        return "Court Judgment"
    if document.id.startswith("IN-SC-"):
        return "Supreme Court Case"
    if document.id.startswith("IN-REF-"):
        return "Legal / Statutory Reference"
    return "Legal Document"


def is_law_document(result):
    return result.document.id.startswith("IN-LAW-")


def is_court_judgment(result):
    return result.document.id.startswith(("HF-INDIAN-JUDGMENT-", "IN-SC-"))


def split_laws_and_judgments(results, top_k):
    laws = [result for result in results if is_law_document(result)]
    judgments = [result for result in results if is_court_judgment(result)]
    return laws[:top_k], judgments[:top_k]


def serialize_results(results):
    serialized = []
    for rank, result in enumerate(results, start=1):
        serialized.append(
            {
                "rank": rank,
                "id": result.document.id,
                "title": result.document.title,
                "court": result.document.court,
                "date": result.document.date,
                "text": result.document.text,
                "score": f"{result.score:.4f}",
                "method": result.method,
                "document_type": document_type(result.document),
                "breakdown": {
                    key: f"{value:.4f}"
                    for key, value in sorted(result.score_breakdown.items())
                    if isinstance(value, int | float)
                },
            }
        )
    return serialized


@app.route("/", methods=["GET", "POST"])
def index():
    query = "What happens if I break a contract?"
    top_k = 5
    fusion = "rrf"
    use_temporal = True
    temporal_half_life_days = 730
    temporal_boost = 0.005
    results = None

    if request.method == "POST":
        query = request.form.get("query", query).strip() or query
        top_k = int(request.form.get("top_k", top_k))
        fusion = request.form.get("fusion", fusion)
        use_temporal = request.form.get("use_temporal") == "on"
        temporal_half_life_days = int(
            request.form.get("temporal_half_life_days", temporal_half_life_days)
        )
        temporal_boost = float(request.form.get("temporal_boost", temporal_boost))

    temporal_filter = None
    if use_temporal:
        temporal_filter = TemporalFilter(
            half_life_days=temporal_half_life_days,
            boost_strength=temporal_boost,
        )

    conversion = convert_to_legal_query(query)
    retrieval_query = conversion.search_query

    hybrid = HybridSearchPipeline(
        DOCUMENTS,
        bm25_backend=BM25_BACKEND,
        vector_backend=VECTOR_BACKEND,
        fusion_mode=fusion,
        candidate_pool=max(top_k * 8, 40),
        temporal_filter=temporal_filter,
    )

    hybrid_candidates = hybrid.search(retrieval_query, top_k=max(top_k * 6, 30))
    law_results, judgment_results = split_laws_and_judgments(hybrid_candidates, top_k)

    results = {
        "laws": serialize_results(law_results),
        "judgments": serialize_results(judgment_results),
    }

    return render_template(
        "index.html",
        query=query,
        top_k=top_k,
        fusion=fusion,
        use_temporal=use_temporal,
        temporal_half_life_days=temporal_half_life_days,
        temporal_boost=temporal_boost,
        bm25_label=BM25_LABEL,
        vector_label=VECTOR_LABEL,
        corpus_size=len(DOCUMENTS),
        conversion=conversion,
        results=results,
    )


if __name__ == "__main__":
    app.run(debug=False, port=5000)
