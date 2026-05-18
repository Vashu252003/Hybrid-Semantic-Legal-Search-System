import math
from collections import Counter

from .models import Document, RankedResult
from .tokenization import tokenize


class BM25Index:
    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(self._document_text(doc)) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.document_frequencies = self._build_document_frequencies()
        self.idf = self._build_idf()

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
            numerator = total_docs - doc_freq + 0.5
            denominator = doc_freq + 0.5
            idf_values[term] = math.log(1 + (numerator / denominator))
        return idf_values

    def score_all(self, query: str) -> dict[str, float]:
        query_terms = tokenize(query)
        scores = {doc.id: 0.0 for doc in self.documents}

        for index, document in enumerate(self.documents):
            doc_length = self.doc_lengths[index]
            term_frequency = self.term_frequencies[index]
            score = 0.0

            for term in query_terms:
                if term not in term_frequency:
                    continue

                frequency = term_frequency[term]
                idf = self.idf.get(term, 0.0)
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1e-9))
                )
                score += idf * (numerator / denominator)

            scores[document.id] = score

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
                method="bm25",
                score_breakdown={"bm25": scores[document.id]},
            )
            for document in top_documents
            if scores[document.id] > 0
        ]
