from dataclasses import dataclass, field


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    court: str
    date: str
    text: str


@dataclass
class RankedResult:
    document: Document
    score: float
    method: str
    score_breakdown: dict[str, float] = field(default_factory=dict)
