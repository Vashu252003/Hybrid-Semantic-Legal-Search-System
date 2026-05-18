from datetime import date

from .models import RankedResult


class TemporalFilter:
    def __init__(
        self,
        reference_date: date | None = None,
        half_life_days: int = 730,
        boost_strength: float = 0.005,
    ):
        self.reference_date = reference_date
        self.half_life_days = max(half_life_days, 1)
        self.boost_strength = max(boost_strength, 0.0)

    def _parse_date(self, value: str) -> date | None:
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None

    def _resolve_reference_date(self, results: list[RankedResult]) -> date:
        parsed_dates = [
            parsed_date
            for result in results
            if (parsed_date := self._parse_date(result.document.date)) is not None
        ]
        return self.reference_date or max(parsed_dates, default=date.today())

    def _recency_score(self, document_date: date | None, reference_date: date) -> float:
        if document_date is None:
            return 0.0
        age_days = max((reference_date - document_date).days, 0)
        return 0.5 ** (age_days / self.half_life_days)

    def rerank(self, results: list[RankedResult], top_k: int | None = None) -> list[RankedResult]:
        if not results or self.boost_strength == 0:
            return results[:top_k] if top_k is not None else list(results)

        reference_date = self._resolve_reference_date(results)
        reranked_results = []

        for result in results:
            document_date = self._parse_date(result.document.date)
            recency_score = self._recency_score(document_date, reference_date)
            temporal_multiplier = 1.0 + (self.boost_strength * recency_score)
            temporal_score = result.score * temporal_multiplier
            reranked_results.append(
                RankedResult(
                    document=result.document,
                    score=temporal_score,
                    method=f"{result.method}_temporal",
                    score_breakdown={
                        **result.score_breakdown,
                        "base_score": result.score,
                        "recency_score": recency_score,
                        "temporal_multiplier": temporal_multiplier,
                        "temporal_score": temporal_score,
                    },
                )
            )

        reranked_results.sort(
            key=lambda result: (
                result.score,
                result.score_breakdown.get("recency_score", 0.0),
            ),
            reverse=True,
        )
        return reranked_results[:top_k] if top_k is not None else reranked_results
