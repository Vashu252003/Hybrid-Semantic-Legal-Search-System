import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LEGAL_CASES_FILE = DATA_DIR / "legal_cases.json"
BACKUP_FILE = DATA_DIR / "legal_cases_curated_backup.json"

HF_REPO = "aadityaJagdale/cleaned_legal_dataset"
HF_API_TREE = f"https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=1"
HF_RAW_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"


def fetch_json(url: str):
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def normalize_date(value: str) -> str:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(value.strip(), fmt).date().isoformat()
        except ValueError:
            continue
    return "1900-01-01"


def make_id(path: str, index: int) -> str:
    stem = Path(path).stem.upper()
    slug = re.sub(r"[^A-Z0-9]+", "-", stem).strip("-")
    return f"HF-INDIAN-JUDGMENT-{index:04d}-{slug[:48]}"


def document_title(record: dict) -> str:
    petitioner = (record.get("petitioner") or "Unknown Petitioner").strip()
    respondent = (record.get("respondent") or "Unknown Respondent").strip()
    return f"{petitioner} v. {respondent}"


def document_text(record: dict) -> str:
    judgement = (record.get("judgement") or record.get("judgment") or "").strip()
    citations = record.get("citations") or record.get("citation") or []
    tags = record.get("tags") or []

    if isinstance(citations, str):
        citations = [citations]
    if isinstance(tags, str):
        tags = [tags]

    parts = [
        judgement,
        "Citations: " + "; ".join(str(item) for item in citations if item),
        "Legal tags: " + "; ".join(str(item) for item in tags if item),
    ]
    return " ".join(part for part in parts if part and not part.endswith(": "))


def convert_record(record: dict, path: str, index: int) -> dict | None:
    text = document_text(record)
    if len(text) < 200:
        return None

    return {
        "id": make_id(path, index),
        "title": document_title(record),
        "court": "Indian Court Judgment Dataset",
        "date": normalize_date(record.get("date", "")),
        "text": text,
    }


def load_curated_documents() -> list[dict]:
    source_file = BACKUP_FILE if BACKUP_FILE.exists() else LEGAL_CASES_FILE
    if not source_file.exists():
        return []
    with source_file.open("r", encoding="utf-8") as file:
        current_documents = json.load(file)
    return current_documents


def load_current_law_documents() -> list[dict]:
    if not LEGAL_CASES_FILE.exists():
        return []
    with LEGAL_CASES_FILE.open("r", encoding="utf-8") as file:
        current_documents = json.load(file)
    return [
        document
        for document in current_documents
        if document.get("id", "").startswith("IN-LAW-")
    ]


def import_documents(limit: int) -> list[dict]:
    tree = fetch_json(HF_API_TREE)
    json_paths = [
        item["path"]
        for item in tree
        if item.get("type") == "file" and item.get("path", "").startswith("scraped/") and item["path"].endswith(".json")
    ]
    json_paths.sort()

    documents = []
    for path in json_paths:
        if len(documents) >= limit:
            break
        raw_url = f"{HF_RAW_BASE}/{quote(path)}"
        try:
            record = fetch_json(raw_url)
        except (json.JSONDecodeError, URLError, TimeoutError):
            continue

        document = convert_record(record, path, len(documents) + 1)
        if document is not None:
            documents.append(document)

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Import real Indian legal judgments from Hugging Face.")
    parser.add_argument("--limit", type=int, default=300, help="Maximum number of judgment records to import.")
    parser.add_argument(
        "--include-curated",
        action="store_true",
        help="Append the old curated statutory/case anchor records after importing actual judgments.",
    )
    parser.add_argument(
        "--skip-current-laws",
        action="store_true",
        help="Do not preserve already imported IN-LAW official law/source records.",
    )
    args = parser.parse_args()

    if LEGAL_CASES_FILE.exists() and not BACKUP_FILE.exists():
        shutil.copy2(LEGAL_CASES_FILE, BACKUP_FILE)

    imported_documents = import_documents(args.limit)
    curated_documents = load_curated_documents() if args.include_curated else []
    law_documents = [] if args.skip_current_laws else load_current_law_documents()
    combined = [*curated_documents, *imported_documents, *law_documents]

    with LEGAL_CASES_FILE.open("w", encoding="utf-8") as file:
        json.dump(combined, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print(f"Imported actual judgment records: {len(imported_documents)}")
    print(f"Kept curated statutory/case anchor records: {len(curated_documents)}")
    print(f"Kept official law/source records: {len(law_documents)}")
    print(f"Wrote corpus: {LEGAL_CASES_FILE}")
    if BACKUP_FILE.exists():
        print(f"Curated backup: {BACKUP_FILE}")


if __name__ == "__main__":
    main()
