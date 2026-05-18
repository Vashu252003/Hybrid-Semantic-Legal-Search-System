from collections import defaultdict

from .models import RankedResult


def reciprocal_rank_fusion(
    result_sets: list[list[RankedResult]],
    k: int = 60,
    top_k: int = 5,
) -> list[RankedResult]:
    fused_scores = defaultdict(float)
    documents = {}
    breakdown = defaultdict(dict)

    for result_set in result_sets:
        for rank, result in enumerate(result_set, start=1):
            doc_id = result.document.id
            documents[doc_id] = result.document
            fused_scores[doc_id] += 1.0 / (k + rank)
            breakdown[doc_id][f"{result.method}_rank"] = float(rank)
            breakdown[doc_id][f"{result.method}_score"] = result.score

    ranked_doc_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)

    return [
        RankedResult(
            document=documents[doc_id],
            score=fused_scores[doc_id],
            method="rrf",
            score_breakdown={
                **breakdown[doc_id],
                "rrf": fused_scores[doc_id],
            },
        )
        for doc_id in ranked_doc_ids[:top_k]
    ]
