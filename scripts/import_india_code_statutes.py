"""Import official law/source records into the searchable legal corpus.

The project keeps manually written legal advice disabled. This importer adds
source-backed statute records from the official source registry so laws can be
retrieved by the same BM25/vector/hybrid pipeline as judgments.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "legal_cases.json"
SOURCE_FILE = BASE_DIR / "data" / "india_code_sources.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/pdf,*/*",
}


@dataclass(frozen=True)
class SourceText:
    url: str
    text: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value.upper()).strip("-")
    return slug[:80] or "SOURCE"


def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    request = Request(url, headers=HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def strip_html(raw_html: str) -> str:
    cleaned = re.sub(r"<script.*?</script>", " ", raw_html, flags=re.I | re.S)
    cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def discover_pdf_links(page_url: str, raw_html: str) -> list[str]:
    links = []
    for match in re.finditer(r"href=[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']", raw_html, flags=re.I):
        links.append(urljoin(page_url, html.unescape(match.group(1))))

    preferred = []
    for link in links:
        lower = link.lower()
        if "/bitstream/" in lower and re.search(r"/a\d{4}", lower):
            preferred.append(link)
    return preferred or links


def extract_pdf_text(pdf_bytes: bytes, max_pages: int) -> str:
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages[:max_pages]:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def fetch_source_text(source_url: str, max_pdf_pages: int) -> SourceText:
    data = fetch_bytes(source_url)
    if data[:5] == b"%PDF-":
        return SourceText(source_url, extract_pdf_text(data, max_pdf_pages))

    raw_html = data.decode("utf-8", errors="ignore")
    pdf_links = discover_pdf_links(source_url, raw_html)
    for pdf_url in pdf_links[:2]:
        try:
            pdf_data = fetch_bytes(pdf_url)
            if pdf_data[:5] == b"%PDF-":
                pdf_text = extract_pdf_text(pdf_data, max_pdf_pages)
                if len(pdf_text.strip()) > 500:
                    return SourceText(pdf_url, pdf_text)
        except Exception:
            continue

    return SourceText(source_url, strip_html(raw_html))


def parse_year(source_name: str) -> str:
    match = re.search(r"(18|19|20)\d{2}", source_name)
    if match:
        return f"{match.group(0)}-01-01"
    return "1900-01-01"


def normalize_section(value: str) -> str:
    return re.sub(r"\s+", "", value.strip()).upper()


def clean_section_title(value: str) -> str:
    title = re.sub(r"\s+", " ", value).strip(" .-")
    title = re.split(r"\s*[. ]?[—-]\s*", title, maxsplit=1)[0].strip(" .-")
    title = title.replace("contrac t", "contract")
    return title[:140]


def find_section_chunk(text: str, section: str) -> tuple[str, str] | None:
    normalized = normalize_section(section)
    if not re.fullmatch(r"\d+[A-Z]?", normalized):
        return None

    heading_pattern = re.compile(
        rf"(?m)(?:^|\n)\s*{re.escape(normalized)}\.\s+([^\n]{{3,180}})"
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return None

    # The first occurrence is often the arrangement of sections. Later matches
    # usually contain the actual section text.
    start_match = matches[-1]
    start = start_match.start()
    next_heading = re.search(r"(?m)\n\s*\d+[A-Z]?\.\s+[^\n]{3,180}", text[start_match.end() :])
    end = start_match.end() + next_heading.start() if next_heading else start + 3000
    chunk = re.sub(r"\s+", " ", text[start:end]).strip()
    title = clean_section_title(start_match.group(1))
    return title, chunk[:3500]


def find_section_heading(text: str, section: str) -> tuple[str, str] | None:
    normalized = normalize_section(section)
    if not re.fullmatch(r"\d+[A-Z]?", normalized):
        return None

    pattern = re.compile(rf"Section\s+{re.escape(normalized)}\.\s+([^\.]+)\.", flags=re.I)
    match = pattern.search(text)
    if not match:
        return None
    title = clean_section_title(match.group(1))
    chunk = f"Official source section listing. Section {normalized}. {title}."
    return title, chunk


def build_law_documents(
    source_name: str,
    source_info: dict,
    source_text: SourceText,
) -> list[dict[str, str]]:
    source_slug = slugify(source_name)
    source_label = source_info.get("source", "Official source")
    source_type = source_info.get("source_type", "official_source")
    categories = ", ".join(source_info.get("used_for_categories", []))
    sections_used = source_info.get("sections_used", [])
    date = parse_year(source_name)
    text = source_text.text

    documents = [
        {
            "id": f"IN-LAW-{source_slug}-SOURCE",
            "title": f"{source_name} - Official Source",
            "court": "Statute / Law Source",
            "date": date,
            "text": (
                f"Official legal source: {source_name}. Source: {source_label}. "
                f"Source type: {source_type}. URL: {source_text.url}. "
                f"Relevant categories: {categories}. Sections/terms indexed: "
                f"{', '.join(sections_used)}. Extracted official text: {text[:9000]}"
            ),
        }
    ]

    for section in sections_used:
        section_match = find_section_chunk(text, section) or find_section_heading(text, section)
        if not section_match:
            continue
        title, chunk = section_match
        normalized = normalize_section(section)
        documents.append(
            {
                "id": f"IN-LAW-{source_slug}-SEC-{normalized}",
                "title": f"{source_name} - Section {normalized}: {title}",
                "court": "Statute / Law Section",
                "date": date,
                "text": (
                    f"Official statute section from {source_name}. Source: {source_label}. "
                    f"URL: {source_text.url}. Section {normalized}: {chunk}"
                ),
            }
        )

    return documents


def import_statutes(max_pdf_pages: int) -> list[dict[str, str]]:
    with SOURCE_FILE.open("r", encoding="utf-8") as file:
        sources = json.load(file)

    law_documents = []
    for source_name, source_info in sources.items():
        if not source_info.get("source_type", "").startswith("official"):
            continue
        source_url = source_info["url"]
        try:
            source_text = fetch_source_text(source_url, max_pdf_pages=max_pdf_pages)
        except Exception as error:
            print(f"Skipped {source_name}: {error}")
            continue

        docs = build_law_documents(source_name, source_info, source_text)
        law_documents.extend(docs)
        print(f"Imported {len(docs)} law document(s): {source_name}")

    with DATA_FILE.open("r", encoding="utf-8") as file:
        existing_documents = json.load(file)

    non_law_documents = [
        document for document in existing_documents if not document["id"].startswith("IN-LAW-")
    ]
    combined = non_law_documents + law_documents
    with DATA_FILE.open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=2, ensure_ascii=False)
        file.write("\n")

    return law_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import official law/source records into data/legal_cases.json."
    )
    parser.add_argument(
        "--max-pdf-pages",
        type=int,
        default=35,
        help="Maximum pages to extract from each official PDF.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    law_documents = import_statutes(max_pdf_pages=args.max_pdf_pages)
    print(f"Added {len(law_documents)} official law/source document(s) to the corpus.")


if __name__ == "__main__":
    main()
