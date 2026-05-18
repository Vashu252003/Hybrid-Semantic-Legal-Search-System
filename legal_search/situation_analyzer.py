from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SituationAnalysis:
    original_query: str
    expanded_query: str
    legal_areas: list[str]
    related_terms: list[str]
    simple_legal_meaning: str
    note: str


LEGAL_INTENT_RULES = [
    {
        "area": "Landlord and tenant law",
        "keywords": ["landlord", "tenant", "rent", "eviction", "lease", "house owner"],
        "terms": [
            "tenancy agreement",
            "rent revision",
            "security deposit",
            "eviction process",
            "rent authority",
        ],
    },
    {
        "area": "Insurance disclosure and policyholder rights",
        "keywords": ["insurance", "policy", "claim", "medical", "disease", "health issue"],
        "terms": [
            "proposal form",
            "material information",
            "medical disclosure",
            "policyholder protection",
            "claim rejection",
        ],
    },
    {
        "area": "Consumer protection",
        "keywords": ["consumer", "complaint", "refund", "defective", "service", "seller", "ecommerce", "online order"],
        "terms": [
            "consumer complaint",
            "deficiency in service",
            "unfair trade practice",
            "district consumer commission",
            "product liability",
        ],
    },
    {
        "area": "Contract law",
        "keywords": ["contract", "agreement", "valid", "breach", "damages", "business deal", "signature"],
        "terms": [
            "valid agreement",
            "offer acceptance consideration",
            "free consent",
            "breach of contract",
            "liquidated damages",
        ],
    },
    {
        "area": "Privacy, image rights, and copyright",
        "keywords": ["photo", "image", "picture", "permission", "privacy", "instagram", "social media"],
        "terms": [
            "right to privacy",
            "informational privacy",
            "copyright photograph",
            "unauthorized use",
            "injunction damages",
        ],
    },
    {
        "area": "Police, arrest, and bail",
        "keywords": ["police", "arrest", "fir", "bail", "custody", "accused", "criminal"],
        "terms": [
            "anticipatory bail",
            "personal liberty",
            "arrest protection",
            "criminal procedure",
            "custodial interrogation",
        ],
        "meaning": (
            "If police action, arrest, FIR, or custody is involved, the issue may relate "
            "to criminal procedure and personal liberty. The seriousness depends on the "
            "facts, documents, and allegations."
        ),
    },
    {
        "area": "Repeated unwanted calls / harassment / mental harassment",
        "keywords": [
            "call",
            "calls",
            "calling",
            "again and again",
            "repeated",
            "harassment",
            "harass",
            "mental stress",
            "stress",
            "threat",
            "blackmail",
            "abuse",
            "abusive",
            "stalking",
            "stalk",
            "unwanted",
        ],
        "terms": [
            "repeated unwanted calls",
            "harassment",
            "mental harassment",
            "stalking",
            "criminal intimidation",
            "cyber harassment",
            "threatening messages",
            "privacy and personal liberty",
        ],
        "meaning": (
            "If someone repeatedly calls you even after you clearly told them not to, "
            "it may be treated as harassment. If the calls include threats, blackmail, "
            "abusive words, sexual content, or reputation damage, it can become a more "
            "serious legal issue."
        ),
    },
    {
        "area": "Property and land acquisition",
        "keywords": ["property", "land", "plot", "acquisition", "compensation", "possession"],
        "terms": [
            "land acquisition",
            "compensation",
            "possession",
            "statutory benefits",
            "landowner rights",
        ],
        "meaning": (
            "This may relate to property rights, possession, compensation, or land acquisition "
            "procedure depending on the facts and documents."
        ),
    },
]

DEFAULT_MEANINGS = {
    "Landlord and tenant law": (
        "This may relate to tenant rights, rent revision, deposit, eviction, or the terms "
        "of a tenancy agreement."
    ),
    "Insurance disclosure and policyholder rights": (
        "This may relate to disclosure of material information in an insurance proposal, "
        "claim processing, or policyholder remedies."
    ),
    "Consumer protection": (
        "This may relate to a consumer complaint for defective goods, service deficiency, "
        "refund, unfair trade practice, or product liability."
    ),
    "Contract law": (
        "This may relate to whether an agreement is valid, whether there is breach of "
        "contract, and what compensation or damages may follow."
    ),
    "Privacy, image rights, and copyright": (
        "This may relate to privacy, permission to use a photograph, copyright ownership, "
        "unauthorized publication, or remedies for misuse."
    ),
    "Police, arrest, and bail": (
        "This may relate to FIR, arrest, bail, personal liberty, and criminal procedure."
    ),
    "General legal research": (
        "The issue is broad, so the system is retrieving general legal rights, remedies, "
        "and court precedent. Add more facts for a more precise category."
    ),
}


def _matches_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def analyze_situation(query: str) -> SituationAnalysis:
    normalized_query = query.strip()
    lowered = normalized_query.lower()
    matched_areas = []
    related_terms = []
    meanings = []

    for rule in LEGAL_INTENT_RULES:
        if any(_matches_keyword(lowered, keyword) for keyword in rule["keywords"]):
            matched_areas.append(rule["area"])
            related_terms.extend(rule["terms"])
            meanings.append(rule.get("meaning") or DEFAULT_MEANINGS.get(rule["area"], ""))

    if not matched_areas:
        matched_areas = ["General legal research"]
        related_terms = [
            "legal rights",
            "court precedent",
            "statutory remedy",
            "legal obligation",
        ]
        meanings = [DEFAULT_MEANINGS["General legal research"]]

    deduped_terms = list(dict.fromkeys(related_terms))
    expanded_query = " ".join([normalized_query, *deduped_terms])
    simple_legal_meaning = " ".join(dict.fromkeys(meaning for meaning in meanings if meaning))

    return SituationAnalysis(
        original_query=normalized_query,
        expanded_query=expanded_query,
        legal_areas=matched_areas,
        related_terms=deduped_terms,
        simple_legal_meaning=simple_legal_meaning,
        note=(
            "This is legal information retrieval, not legal advice. "
            "The system maps your situation to related legal areas and retrieves relevant authorities."
        ),
    )
