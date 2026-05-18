import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app import document_type, split_laws_and_judgments
from evaluation import ndcg_at_k, precision_at_k, recall_at_k
from legal_search import (
    BM25Index,
    Document,
    HybridSearchPipeline,
    RankedResult,
    SentenceTransformerFaissIndex,
    TemporalFilter,
    VectorIndex,
    analyze_situation,
    convert_to_legal_query,
    reciprocal_rank_fusion,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "legal_cases.json"
APPLICABLE_LAWS_FILE = BASE_DIR / "data" / "applicable_laws.json"
INDIA_CODE_SOURCES_FILE = BASE_DIR / "data" / "india_code_sources.json"


def load_documents() -> list[Document]:
    with DATA_FILE.open("r", encoding="utf-8") as file:
        raw_documents = json.load(file)
    return [Document(**item) for item in raw_documents]


class MockSentenceTransformer:
    VOCABULARY = [
        "breach",
        "contract",
        "damages",
        "anticipatory",
        "bail",
        "fraud",
        "environmental",
        "clearance",
        "mining",
        "pollution",
    ]

    def encode(
        self,
        sentences: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        matrix = []
        for sentence in sentences:
            lowered = sentence.lower()
            matrix.append([float(lowered.count(term)) for term in self.VOCABULARY])
        embeddings = np.asarray(matrix, dtype="float32")

        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        if convert_to_numpy:
            return embeddings
        return embeddings.tolist()


class RecordingSentenceTransformer:
    last_init_kwargs = None

    def __init__(self, model_name_or_path: str, **kwargs):
        type(self).last_init_kwargs = kwargs
        self.model_name_or_path = model_name_or_path

    def encode(
        self,
        sentences: list[str],
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ):
        embeddings = np.ones((len(sentences), 4), dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embeddings = embeddings / norms

        if convert_to_numpy:
            return embeddings
        return embeddings.tolist()


class LegalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.documents = load_documents()

    def test_bm25_returns_contract_case(self) -> None:
        index = BM25Index(self.documents)
        results = index.search("breach of contract liquidated damages section 74", top_k=5)
        self.assertTrue(results)
        self.assertTrue(any("contract" in result.document.text.lower() for result in results))

    def test_document_type_labels(self) -> None:
        self.assertEqual(
            document_type(Document("IN-LAW-CONTRACT-ACT-SEC-10", "A", "Source", "1872-01-01", "text")),
            "Statute / Law Section",
        )
        self.assertEqual(
            document_type(Document("IN-LAW-CONTRACT-ACT-SOURCE", "A", "Source", "1872-01-01", "text")),
            "Statute / Law Source",
        )
        self.assertEqual(
            document_type(Document("HF-INDIAN-JUDGMENT-0001-X", "A", "Court", "2020-01-01", "text")),
            "Court Judgment",
        )
        self.assertEqual(
            document_type(Document("IN-SC-TEST", "A", "Court", "2020-01-01", "text")),
            "Supreme Court Case",
        )
        self.assertEqual(
            document_type(Document("IN-REF-TEST", "A", "Source", "2020-01-01", "text")),
            "Legal / Statutory Reference",
        )

    def test_vector_returns_bail_case(self) -> None:
        index = VectorIndex(self.documents)
        results = index.search("anticipatory bail economic offence", top_k=5)
        self.assertTrue(results)
        self.assertTrue(any("bail" in result.document.text.lower() for result in results))

    def test_dense_vector_returns_bail_case(self) -> None:
        index = SentenceTransformerFaissIndex(
            self.documents,
            embedder=MockSentenceTransformer(),
        )
        results = index.search("anticipatory bail fraud", top_k=1)
        self.assertIn("bail", results[0].document.text.lower())

    def test_rrf_fusion_keeps_environment_case_first(self) -> None:
        bm25 = BM25Index(self.documents)
        dense = SentenceTransformerFaissIndex(
            self.documents,
            embedder=MockSentenceTransformer(),
        )
        fused = reciprocal_rank_fusion(
            [
                bm25.search("environmental clearance mining pollution", top_k=3),
                dense.search("environmental clearance mining pollution", top_k=3),
            ],
            top_k=1,
        )
        self.assertIn("environment", fused[0].document.text.lower())

    def test_hybrid_pipeline_returns_environment_case(self) -> None:
        dense = SentenceTransformerFaissIndex(
            self.documents,
            embedder=MockSentenceTransformer(),
        )
        index = HybridSearchPipeline(
            self.documents,
            bm25_backend=BM25Index(self.documents),
            vector_backend=dense,
            fusion_mode="rrf",
        )
        results = index.search("environmental clearance mining pollution", top_k=3)
        self.assertTrue(any("environment" in result.document.text.lower() for result in results))

    def test_temporal_filter_promotes_recent_case_when_relevance_is_close(self) -> None:
        older = Document(
            id="OLD",
            title="Older Contract Damages Case",
            court="Delhi High Court",
            date="2020-01-01",
            text="breach of contract damages",
        )
        recent = Document(
            id="NEW",
            title="Recent Contract Damages Case",
            court="Supreme Court of India",
            date="2024-01-01",
            text="breach of contract damages",
        )
        results = [
            RankedResult(older, 1.0, "rrf", {"rrf": 1.0}),
            RankedResult(recent, 0.99, "rrf", {"rrf": 0.99}),
        ]

        reranked = TemporalFilter(boost_strength=0.25).rerank(results, top_k=2)

        self.assertEqual(reranked[0].document.id, "NEW")
        self.assertIn("recency_score", reranked[0].score_breakdown)

    def test_weighted_hybrid_keeps_single_result_queries(self) -> None:
        index = HybridSearchPipeline(
            self.documents,
            bm25_backend=BM25Index(self.documents),
            vector_backend=VectorIndex(self.documents),
            fusion_mode="weighted",
        )
        results = index.search("anticipatory bail financial fraud", top_k=3)
        self.assertTrue(any("bail" in result.document.text.lower() for result in results))

    def test_precision_recall_and_ndcg_metrics(self) -> None:
        ranked_ids = ["A", "B", "C"]
        relevance = {"A": 3, "C": 1, "D": 2}

        self.assertAlmostEqual(precision_at_k(ranked_ids, relevance, 3), 2 / 3)
        self.assertAlmostEqual(recall_at_k(ranked_ids, relevance, 3), 2 / 3)
        self.assertGreater(ndcg_at_k(ranked_ids, relevance, 3), 0.0)
        self.assertLessEqual(ndcg_at_k(ranked_ids, relevance, 3), 1.0)

    def test_situation_analyzer_maps_plain_language_questions(self) -> None:
        examples = {
            "Can my landlord increase rent suddenly?": "Landlord and tenant law",
            "Do I need to mention my medical issue in insurance form?": "Insurance disclosure and policyholder rights",
            "How do I file a consumer complaint?": "Consumer protection",
            "What happens if I break a contract?": "Contract law",
            "Can someone use my photo without permission?": "Privacy, image rights, and copyright",
            "Is this agreement legally valid?": "Contract law",
            "A girl is calling again and again and giving me mental stress.": "Repeated unwanted calls / harassment / mental harassment",
        }

        for query, expected_area in examples.items():
            with self.subTest(query=query):
                analysis = analyze_situation(query)
                self.assertIn(expected_area, analysis.legal_areas)
                self.assertGreater(len(analysis.expanded_query), len(query))

    def test_ai_query_converter_classifies_repeated_unwanted_calls_without_manual_laws(self) -> None:
        conversion = convert_to_legal_query(
            "I am a boy. A girl is calling again and again and giving me mental stress."
        )

        self.assertIn(
            "Repeated unwanted calls / harassment / mental harassment",
            conversion.legal_areas,
        )
        self.assertIn("harassment", conversion.legal_query)
        self.assertIn("repeatedly calls", conversion.simple_legal_meaning)
        self.assertEqual(
            conversion.guidance.legal_category,
            "Repeated unwanted calls / harassment / mental harassment",
        )
        self.assertEqual(conversion.guidance.severity_level, "Not manually classified")
        self.assertEqual(conversion.guidance.applicable_laws, [])
        self.assertEqual(conversion.guidance.evidence_required, [])

    def test_applicable_laws_json_is_empty_when_manual_data_disabled(self) -> None:
        with APPLICABLE_LAWS_FILE.open("r", encoding="utf-8") as file:
            law_rules = json.load(file)

        self.assertEqual(law_rules, {})

    def test_main_detected_categories_do_not_use_manual_applicable_laws(self) -> None:
        examples = [
            "What happens if I break a contract?",
            "How do I file a consumer complaint?",
            "Do I need to mention my medical issue in insurance form?",
            "Can my landlord increase rent suddenly?",
            "Can someone use my photo without permission?",
            "Police may arrest me, can I get bail?",
            "My land compensation is not paid.",
        ]

        for query in examples:
            with self.subTest(query=query):
                conversion = convert_to_legal_query(query)
                self.assertEqual(conversion.guidance.applicable_laws, [])

    def test_india_code_sources_remain_available_for_external_ingestion(self) -> None:
        with INDIA_CODE_SOURCES_FILE.open("r", encoding="utf-8") as file:
            source_rules = json.load(file)

        self.assertIn("Bharatiya Nyaya Sanhita, 2023", source_rules)
        self.assertIn("Information Technology Act, 2000", source_rules)

    def test_official_law_documents_are_imported_into_search_corpus(self) -> None:
        law_documents = [
            document for document in self.documents if document.id.startswith("IN-LAW-")
        ]

        self.assertTrue(law_documents)
        self.assertTrue(
            any("Indian Contract Act" in document.title for document in law_documents)
        )
        self.assertTrue(
            any(document.court == "Statute / Law Section" for document in law_documents)
        )

    def test_ui_splits_laws_and_court_judgments(self) -> None:
        case_results = [
            RankedResult(
                Document(f"HF-INDIAN-JUDGMENT-{index:04d}", "Case", "Court", "2020-01-01", "text"),
                1.0 - (index * 0.01),
                "rrf",
            )
            for index in range(5)
        ]
        law_result = RankedResult(
            Document(
                "IN-LAW-BHARATIYA-NYAYA-SANHITA-2023-SEC-351",
                "BNS Section 351",
                "Statute / Law Section",
                "2023-01-01",
                "criminal intimidation",
            ),
            0.7,
            "rrf",
        )

        laws, judgments = split_laws_and_judgments([*case_results, law_result], top_k=5)

        self.assertEqual(len(laws), 1)
        self.assertEqual(len(judgments), 5)
        self.assertTrue(all(result.document.id.startswith("IN-LAW-") for result in laws))
        self.assertTrue(
            all(result.document.id.startswith("HF-INDIAN-JUDGMENT-") for result in judgments)
        )

    def test_manual_severity_is_disabled(self) -> None:
        conversion = convert_to_legal_query(
            "A girl is calling again and again and giving threats and blackmail."
        )

        self.assertEqual(conversion.guidance.severity_level, "Not manually classified")

    def test_ai_query_converter_rewrites_user_words_to_legal_query(self) -> None:
        conversion = convert_to_legal_query("What happens if I break a contract?")

        self.assertIn("Contract law", conversion.legal_areas)
        self.assertIn("breach of agreement", conversion.legal_query)
        self.assertIn("liquidated damages", conversion.search_query)
        self.assertGreater(len(conversion.search_query), len(conversion.user_input))

    def test_situation_query_retrieves_actual_dataset_result(self) -> None:
        conversion = convert_to_legal_query("Can my landlord increase rent suddenly?")
        index = HybridSearchPipeline(
            self.documents,
            bm25_backend=BM25Index(self.documents),
            vector_backend=VectorIndex(self.documents),
            fusion_mode="rrf",
        )

        results = index.search(conversion.search_query, top_k=3)

        self.assertTrue(results)
        self.assertTrue(
            all(
                result.document.id.startswith(("HF-INDIAN-JUDGMENT-", "IN-LAW-"))
                for result in results
            )
        )

    def test_dense_index_uses_local_files_only_by_default(self) -> None:
        RecordingSentenceTransformer.last_init_kwargs = None
        with patch("legal_search.dense_vector_search.SentenceTransformer", RecordingSentenceTransformer):
            SentenceTransformerFaissIndex(
                self.documents[:2],
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            )

        self.assertIsNotNone(RecordingSentenceTransformer.last_init_kwargs)
        self.assertTrue(RecordingSentenceTransformer.last_init_kwargs["local_files_only"])

    def test_dense_index_can_enable_model_downloads(self) -> None:
        RecordingSentenceTransformer.last_init_kwargs = None
        with patch("legal_search.dense_vector_search.SentenceTransformer", RecordingSentenceTransformer):
            SentenceTransformerFaissIndex(
                self.documents[:2],
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                allow_download=True,
            )

        self.assertIsNotNone(RecordingSentenceTransformer.last_init_kwargs)
        self.assertFalse(RecordingSentenceTransformer.last_init_kwargs["local_files_only"])


if __name__ == "__main__":
    unittest.main()
