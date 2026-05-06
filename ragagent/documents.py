"""Document loading and text-cleaning utilities.

Supports PDF, DOCX, XLSX/XLS files placed inside the `DOCS/` folder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable

import PyPDF2
import docx
import openpyxl


SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".xlsx", ".xls")


def clean_text(text: str) -> str:
    """Normalise whitespace, fix hyphenation across line breaks and strip junk."""
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"[^\w\s\-.,;:!?\"'\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_pdf(path: Path) -> str:
    reader = PyPDF2.PdfReader(str(path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""
    return clean_text(full_text)


def _extract_xlsx(path: Path) -> str:
    workbook = openpyxl.load_workbook(str(path), data_only=True)
    full_text = ""
    for sheet in workbook.worksheets:
        for row in sheet.iter_rows(values_only=True):
            row_text = " ".join(str(cell) for cell in row if cell is not None)
            full_text += row_text + " "
    return clean_text(full_text)


def _extract_docx(path: Path) -> str:
    document = docx.Document(str(path))
    full_text = " ".join(p.text for p in document.paragraphs)
    return clean_text(full_text)


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".xlsx": _extract_xlsx,
    ".xls": _extract_xlsx,
    ".docx": _extract_docx,
}


def extract_text(path: Path) -> str:
    """Dispatch to the right extractor based on file extension."""
    extension = path.suffix.lower()
    extractor = _EXTRACTORS.get(extension)
    if not extractor:
        return ""
    return extractor(path)


def list_supported_files(folder: Path) -> Iterable[Path]:
    """Yield every supported document inside *folder* (non-recursive)."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def load_corpus(folder: Path) -> Dict[str, str]:
    """Return a `{filename: cleaned_text}` mapping for every doc in *folder*."""
    corpus: Dict[str, str] = {}
    for path in list_supported_files(folder):
        text = extract_text(path)
        if text:
            corpus[path.name] = text
    return corpus
