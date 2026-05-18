from .bm25 import BM25Index
from .models import Document, RankedResult
from .rrf import reciprocal_rank_fusion
from .temporal_filter import TemporalFilter
from .vector_search import VectorIndex


class BasicHybridSearch:
    def __init__(self, documents: list[Document], bm25_weight: float = 0.5, vector_weight: float = 0.5):
        self.documents = documents
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.bm25 = BM25Index(documents)
        self.vector = VectorIndex(documents)

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        minimum = min(values)
        maximum = max(values)
        if maximum == minimum:
            return {
                doc_id: 1.0 if score > 0 else 0.0
                for doc_id, score in scores.items()
            }
        return {
            doc_id: (score - minimum) / (maximum - minimum)
            for doc_id, score in scores.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[RankedResult]:
        bm25_scores = self.bm25.score_all(query)
        vector_scores = self.vector.score_all(query)
        normalized_bm25 = self._normalize_scores(bm25_scores)
        normalized_vector = self._normalize_scores(vector_scores)

        combined_scores = {}
        for document in self.documents:
            combined_scores[document.id] = (
                self.bm25_weight * normalized_bm25.get(document.id, 0.0)
                + self.vector_weight * normalized_vector.get(document.id, 0.0)
            )

        ranked_documents = sorted(
            self.documents,
            key=lambda document: combined_scores[document.id],
            reverse=True,
        )
        top_documents = ranked_documents[:top_k]

        return [
            RankedResult(
                document=document,
                score=combined_scores[document.id],
                method="basic_hybrid",
                score_breakdown={
                    "bm25": bm25_scores.get(document.id, 0.0),
                    "vector": vector_scores.get(document.id, 0.0),
                    "hybrid": combined_scores[document.id],
                },
            )
            for document in top_documents
            if combined_scores[document.id] > 0
        ]


class HybridSearchPipeline:
    def __init__(
        self,
        documents: list[Document],
        bm25_backend=None,
        vector_backend=None,
        fusion_mode: str = "rrf",
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        candidate_pool: int = 10,
        temporal_filter: TemporalFilter | None = None,
    ):
        self.documents = documents
        self.bm25_backend = bm25_backend or BM25Index(documents)
        self.vector_backend = vector_backend or VectorIndex(documents)
        self.fusion_mode = fusion_mode
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.candidate_pool = max(candidate_pool, 1)
        self.temporal_filter = temporal_filter

    def _normalize_ranked_results(self, results: list[RankedResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [result.score for result in results]
        minimum = min(scores)
        maximum = max(scores)
        if maximum == minimum:
            return {
                result.document.id: 1.0 if result.score > 0 else 0.0
                for result in results
            }
        return {
            result.document.id: (result.score - minimum) / (maximum - minimum)
            for result in results
        }

    def _weighted_fusion(
        self,
        bm25_results: list[RankedResult],
        vector_results: list[RankedResult],
        top_k: int,
    ) -> list[RankedResult]:
        normalized_bm25 = self._normalize_ranked_results(bm25_results)
        normalized_vector = self._normalize_ranked_results(vector_results)

        bm25_lookup = {result.document.id: result for result in bm25_results}
        vector_lookup = {result.document.id: result for result in vector_results}
        documents_by_id = {document.id: document for document in self.documents}

        candidate_ids = set(normalized_bm25) | set(normalized_vector)
        combined_results = []
        for doc_id in candidate_ids:
            score = (
                self.bm25_weight * normalized_bm25.get(doc_id, 0.0)
                + self.vector_weight * normalized_vector.get(doc_id, 0.0)
            )
            if score <= 0:
                continue
            combined_results.append(
                RankedResult(
                    document=documents_by_id[doc_id],
                    score=score,
                    method="weighted_hybrid",
                    score_breakdown={
                        "bm25": bm25_lookup.get(doc_id, RankedResult(documents_by_id[doc_id], 0.0, "bm25")).score,
                        "vector": vector_lookup.get(doc_id, RankedResult(documents_by_id[doc_id], 0.0, "vector")).score,
                        "weighted_hybrid": score,
                    },
                )
            )

        combined_results.sort(key=lambda result: result.score, reverse=True)
        return combined_results[:top_k]

    def search(self, query: str, top_k: int = 5) -> list[RankedResult]:
        candidate_pool = max(top_k, self.candidate_pool)
        bm25_results = self.bm25_backend.search(query, top_k=candidate_pool)
        vector_results = self.vector_backend.search(query, top_k=candidate_pool)

        if self.fusion_mode == "weighted":
            fused_results = self._weighted_fusion(bm25_results, vector_results, candidate_pool)
        else:
            fused_results = reciprocal_rank_fusion(
                [bm25_results, vector_results],
                k=self.rrf_k,
                top_k=candidate_pool,
            )

        if self.temporal_filter is None:
            return fused_results[:top_k]
        return self.temporal_filter.rerank(fused_results, top_k=top_k)
