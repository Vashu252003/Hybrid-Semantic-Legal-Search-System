from dataclasses import dataclass

from .situation_analyzer import SituationAnalysis, analyze_situation


@dataclass(frozen=True)
class ApplicableLaw:
    law: str
    title: str
    when_applicable: str


@dataclass(frozen=True)
class LegalGuidance:
    legal_category: str
    problem_summary: str
    applicable_laws: list[ApplicableLaw]
    important_note: str
    severity_level: str
    suggested_actions: list[str]
    evidence_required: list[str]
    recommended_authority: list[str]
    final_advice: str
    disclaimer: str


@dataclass(frozen=True)
class LegalQueryConversion:
    user_input: str
    legal_query: str
    search_query: str
    legal_areas: list[str]
    search_terms: list[str]
    simple_legal_meaning: str
    guidance: LegalGuidance
    explanation: str


LEGAL_QUERY_TEMPLATES = {
    "Landlord and tenant law": "tenant rights and legality of sudden rent increase by landlord",
    "Insurance disclosure and policyholder rights": "insurance proposal form medical disclosure and claim rejection rules",
    "Consumer protection": "consumer complaint for defective goods deficiency in service and refund remedy",
    "Contract law": "valid contract breach of agreement compensation and enforceability",
    "Privacy, image rights, and copyright": "unauthorized use of photograph privacy rights copyright infringement and remedies",
    "Police, arrest, and bail": "police arrest FIR anticipatory bail personal liberty and criminal procedure",
    "Repeated unwanted calls / harassment / mental harassment": "unwanted contact harassment mental distress legal remedies",
    "Property and land acquisition": "property rights land possession acquisition compensation and statutory benefits",
    "General legal research": "legal rights statutory remedy court precedent and legal obligation",
}


def _build_legal_query(analysis: SituationAnalysis) -> str:
    template_parts = [
        LEGAL_QUERY_TEMPLATES.get(area, area.lower())
        for area in analysis.legal_areas
    ]
    legal_terms = " ".join(analysis.related_terms[:6])
    return " ".join([*template_parts, legal_terms]).strip()


def _build_default_guidance(analysis: SituationAnalysis) -> LegalGuidance:
    category = analysis.legal_areas[0] if analysis.legal_areas else "General legal research"
    return LegalGuidance(
        legal_category=category,
        problem_summary=analysis.simple_legal_meaning,
        applicable_laws=[],
        important_note=(
            "Manual law mappings are disabled. The system now relies on actual retrieved "
            "judgment records and source documents instead of curated applicable-law advice."
        ),
        severity_level="Not manually classified",
        suggested_actions=[],
        evidence_required=[],
        recommended_authority=[],
        final_advice="Review the retrieved legal results and consult a qualified legal professional for advice.",
        disclaimer=(
            "This system provides general legal information only. It is not a replacement "
            "for professional legal advice."
        ),
    )


def _build_guidance(analysis: SituationAnalysis) -> LegalGuidance:
    return _build_default_guidance(analysis)


def convert_to_legal_query(user_input: str) -> LegalQueryConversion:
    analysis = analyze_situation(user_input)
    legal_query = _build_legal_query(analysis)
    search_query = " ".join([analysis.original_query, legal_query]).strip()
    guidance = _build_guidance(analysis)

    return LegalQueryConversion(
        user_input=analysis.original_query,
        legal_query=legal_query,
        search_query=search_query,
        legal_areas=analysis.legal_areas,
        search_terms=analysis.related_terms,
        simple_legal_meaning=analysis.simple_legal_meaning,
        guidance=guidance,
        explanation=(
            "AI query conversion maps everyday words to legal issue terms, "
            "then the converted query is used for hybrid legal retrieval."
        ),
    )
