from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from .models import Document, RankedResult

SentenceTransformer = None


class EmbeddingModel(Protocol):
    def encode(
        self,
        sentences: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ): ...


class _NumpyInnerProductIndex:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def search(self, query_embeddings: np.ndarray, top_k: int):
        queries = np.asarray(query_embeddings, dtype="float32")
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        if self.embeddings.size == 0 or top_k <= 0:
            empty_scores = np.empty((queries.shape[0], 0), dtype="float32")
            empty_indices = np.empty((queries.shape[0], 0), dtype="int64")
            return empty_scores, empty_indices

        scores = np.matmul(queries, self.embeddings.T).astype("float32", copy=False)
        candidate_count = min(top_k, self.embeddings.shape[0])
        indices = np.argsort(-scores, axis=1)[:, :candidate_count].astype("int64", copy=False)
        ranked_scores = np.take_along_axis(scores, indices, axis=1)
        return ranked_scores, indices


def _get_sentence_transformer():
    global SentenceTransformer
    if SentenceTransformer is not None:
        return SentenceTransformer

    try:
        from sentence_transformers import SentenceTransformer as sentence_transformer_class
    except ImportError as error:  # pragma: no cover - handled at runtime
        raise ImportError(
            "sentence-transformers is not installed. Install it before using dense retrieval."
        ) from error

    SentenceTransformer = sentence_transformer_class
    return SentenceTransformer


class SentenceTransformerFaissIndex:
    def __init__(
        self,
        documents: list[Document],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedder: EmbeddingModel | None = None,
        allow_download: bool = False,
    ):
        self.documents = documents
        self.model_name = model_name
        self.allow_download = allow_download
        self.backend_label = f"Sentence Transformers Dense Search ({self.model_name})"
        self.embedder = embedder or self._load_default_embedder(model_name, allow_download)
        self.document_texts = [self._document_text(document) for document in documents]
        self.document_embeddings = self._encode(self.document_texts)
        self.index = self._build_index(self.document_embeddings)

    def _load_default_embedder(self, model_name: str, allow_download: bool):
        sentence_transformer_class = _get_sentence_transformer()
        logger = logging.getLogger("sentence_transformers.SentenceTransformer")
        previous_level = logger.level
        try:
            if not allow_download:
                logger.setLevel(logging.ERROR)
            return sentence_transformer_class(
                model_name,
                local_files_only=not allow_download,
            )
        except Exception as error:
            mode = "local files only" if not allow_download else "model downloads enabled"
            raise RuntimeError(
                f"Unable to load dense model '{model_name}' with {mode}: {error}"
            ) from error
        finally:
            logger.setLevel(previous_level)

    def _document_text(self, document: Document) -> str:
        return f"{document.title}. {document.text}"

    def _encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        matrix = np.asarray(embeddings, dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def _build_index(self, embeddings: np.ndarray):
        # Normalized embeddings make a dot product equivalent to cosine similarity,
        # which is enough for this local demo without importing FAISS at startup.
        return _NumpyInnerProductIndex(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[RankedResult]:
        query_embedding = self._encode([query])
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(
                RankedResult(
                    document=self.documents[index],
                    score=float(score),
                    method="dense_vector",
                    score_breakdown={"dense_vector": float(score)},
                )
            )
        return results
