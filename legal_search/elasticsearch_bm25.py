from __future__ import annotations

from .models import Document, RankedResult

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
except ImportError:  # pragma: no cover - handled at runtime
    Elasticsearch = None
    bulk = None


class ElasticsearchBM25Index:
    def __init__(
        self,
        index_name: str = "legal_cases",
        hosts: list[str] | None = None,
        client=None,
    ):
        self.index_name = index_name
        self.hosts = hosts or ["http://localhost:9200"]
        self.client = client or self._build_client()

    def _build_client(self):
        if Elasticsearch is None:
            raise ImportError("elasticsearch is not installed. Install it before using this backend.")
        return Elasticsearch(self.hosts)

    def ensure_index(self, recreate: bool = False) -> None:
        if recreate and self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)

        if self.client.indices.exists(index=self.index_name):
            return

        self.client.indices.create(
            index=self.index_name,
            mappings={
                "properties": {
                    "id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "court": {"type": "keyword"},
                    "date": {"type": "date"},
                    "text": {"type": "text"},
                }
            },
        )

    def index_documents(self, documents: list[Document], refresh: bool = True) -> None:
        if bulk is None:
            raise ImportError("elasticsearch helpers are unavailable. Reinstall the package.")

        actions = [
            {
                "_index": self.index_name,
                "_id": document.id,
                "_source": {
                    "id": document.id,
                    "title": document.title,
                    "court": document.court,
                    "date": document.date,
                    "text": document.text,
                },
            }
            for document in documents
        ]
        bulk(self.client, actions)
        if refresh:
            self.client.indices.refresh(index=self.index_name)

    def ping(self) -> bool:
        return bool(self.client.ping())

    def search(self, query: str, top_k: int = 5) -> list[RankedResult]:
        response = self.client.search(
            index=self.index_name,
            query={
                "multi_match": {
                    "query": query,
                    "fields": ["title^2", "text", "court"],
                }
            },
            size=top_k,
        )

        results = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            document = Document(
                id=source["id"],
                title=source["title"],
                court=source["court"],
                date=source["date"],
                text=source["text"],
            )
            score = float(hit["_score"])
            results.append(
                RankedResult(
                    document=document,
                    score=score,
                    method="elasticsearch_bm25",
                    score_breakdown={"elasticsearch_bm25": score},
                )
            )
        return results
