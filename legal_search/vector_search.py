import math
from collections import Counter

from .models import Document, RankedResult
from .tokenization import tokenize


class VectorIndex:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.doc_tokens = [tokenize(self._document_text(doc)) for doc in documents]
        self.document_frequencies = self._build_document_frequencies()
        self.idf = self._build_idf()
        self.document_vectors = [self._build_normalized_vector(tokens) for tokens in self.doc_tokens]

    def _document_text(self, document: Document) -> str:
        return f"{document.title} {document.text}"

    def _build_document_frequencies(self) -> Counter:
        doc_freq = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                doc_freq[term] += 1
        return doc_freq

    def _build_idf(self) -> dict[str, float]:
        total_docs = len(self.documents)
        idf_values = {}
        for term, doc_freq in self.document_frequencies.items():
            idf_values[term] = math.log((1 + total_docs) / (1 + doc_freq)) + 1
        return idf_values

    def _build_normalized_vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        if not counts:
            return {}

        max_tf = max(counts.values())
        vector = {
            term: (count / max_tf) * self.idf.get(term, 0.0)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {term: value / norm for term, value in vector.items()}

    def _cosine_similarity(self, left: dict[str, float], right: dict[str, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(term, 0.0) for term, value in left.items())

    def score_all(self, query: str) -> dict[str, float]:
        query_vector = self._build_normalized_vector(tokenize(query))
        scores = {}
        for document, vector in zip(self.documents, self.document_vectors):
            scores[document.id] = self._cosine_similarity(query_vector, vector)
        return scores

    def search(self, query: str, top_k: int = 5) -> list[RankedResult]:
        scores = self.score_all(query)
        ranked_documents = sorted(
            self.documents,
            key=lambda document: scores[document.id],
            reverse=True,
        )
        top_documents = ranked_documents[:top_k]
        return [
            RankedResult(
                document=document,
                score=scores[document.id],
                method="vector",
                score_breakdown={"vector": scores[document.id]},
            )
            for document in top_documents
            if scores[document.id] > 0
        ]
