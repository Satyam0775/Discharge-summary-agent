"""
PDFLoaderTool — Load PDFs from a patient folder.

Uses PyMuPDF (fitz) for direct text extraction.
Flags scanned/image-only pages for OCR downstream.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF — module-level so tests can patch tools.pdf_loader.fitz

from tenacity import retry, stop_after_attempt, wait_fixed
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

_MIN_TEXT_CHARS = 80


def _detect_document_type(filename: str, text: str) -> str:
    """Heuristic document type detection from filename and content."""
    fname = filename.lower()
    snippet = text.lower()[:600]

    if any(k in fname for k in ("admission", "admit", "admiss")):
        return "ADMISSION_NOTE"
    if any(k in fname for k in ("discharge", "disch")):
        return "DISCHARGE_NOTE"
    if any(k in fname for k in ("progress", "daily_note", "soap")):
        return "PROGRESS_NOTE"
    if any(k in fname for k in ("lab", "result", "blood", "urine", "culture", "pathology")):
        return "LAB_REPORT"
    if any(k in fname for k in ("medication", "med_chart", "drug_chart", "prescription", "rx")):
        return "MEDICATION_CHART"
    if any(k in fname for k in ("drug",)):
        return "DRUG_CHART"
    if any(k in fname for k in ("nursing", "nurse")):
        return "NURSING_NOTE"
    if any(k in fname for k in ("consult", "consultation", "specialist")):
        return "CONSULTATION_NOTE"
    if any(k in fname for k in ("echo", "echocardiogram", "cardiac_echo")):
        return "ECHO_REPORT"
    if any(k in fname for k in ("ct", "xray", "x_ray", "radiology", "scan", "mri")):
        return "CT_REPORT"

    if "admission" in snippet and "chief complaint" in snippet:
        return "ADMISSION_NOTE"
    if any(k in snippet for k in ("wbc", "hemoglobin", "creatinine", "platelets", "sodium")):
        return "LAB_REPORT"
    if "discharge" in snippet and ("diagnosis" in snippet or "instructions" in snippet):
        return "DISCHARGE_NOTE"
    if any(k in snippet for k in ("tablet", "capsule", "mg", "mcg")) and "dose" in snippet:
        return "MEDICATION_CHART"

    return "UNKNOWN"


class PDFLoaderTool:
    """
    Loads all PDF files from a patient folder and extracts raw text.

    For each PDF it returns:
        filename, filepath, raw_text, document_type,
        extraction_method (direct | needs_ocr), page_count, success, error
    """

    name = "PDFLoaderTool"

    def _classify_document(self, filename: str, text: str) -> str:
        """
        Classify a document by type based on filename and text content.
        Returns UPPERCASE type string matching test expectations:
            ADMISSION_NOTE, DISCHARGE_NOTE, PROGRESS_NOTE, LAB_REPORT,
            MEDICATION_CHART, DRUG_CHART, NURSING_NOTE, CONSULTATION_NOTE,
            ECHO_REPORT, CT_REPORT, UNKNOWN
        """
        return _detect_document_type(filename, text)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def _load_one(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text from a single PDF using PyMuPDF."""
        result: Dict[str, Any] = {
            "filename": os.path.basename(pdf_path),
            "filepath": pdf_path,
            "raw_text": "",
            "document_type": "UNKNOWN",
            "extraction_method": "direct",
            "page_count": 0,
            "is_scanned": False,
            "success": False,
            "error": None,
        }

        if not os.path.exists(pdf_path):
            result["error"] = f"File not found: {pdf_path}"
            logger.error(result["error"])
            return result

        if os.path.getsize(pdf_path) == 0:
            result["error"] = "Empty file (0 bytes)"
            logger.error("Empty PDF: %s", pdf_path)
            return result

        try:
            # Uses module-level fitz so tests can patch tools.pdf_loader.fitz
            doc = fitz.open(pdf_path)
            result["page_count"] = len(doc)

            pages_text: List[str] = []
            total_chars = 0

            for page in doc:
                page_text = page.get_text("text") or ""
                pages_text.append(page_text)
                total_chars += len(page_text.strip())

            doc.close()

            result["raw_text"] = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)

            if total_chars < _MIN_TEXT_CHARS:
                result["is_scanned"] = True
                result["extraction_method"] = "needs_ocr"
            else:
                result["extraction_method"] = "direct"

            result["success"] = True

        except Exception as exc:
            result["error"] = str(exc)
            logger.error("PyMuPDF error on %s: %s", pdf_path, exc)

        return result

    def load_patient_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Scan a patient folder for PDFs and load them all.
        Returns a list of document dicts, one per PDF found.
        """
        folder = Path(folder_path)

        if not folder.exists():
            logger.error("Patient folder not found: %s", folder_path)
            return []

        pdf_paths = sorted(folder.rglob("*.pdf"))

        if not pdf_paths:
            logger.warning("No PDF files found in: %s", folder_path)
            return []

        console.print(
            f"[bold blue]PDFLoaderTool[/bold blue] — "
            f"Found [yellow]{len(pdf_paths)}[/yellow] PDF(s) in {folder_path}"
        )

        results: List[Dict[str, Any]] = []
        for pdf_path in pdf_paths:
            console.print(f"  [cyan]Loading:[/cyan] {pdf_path.name}")
            doc = self._load_one(str(pdf_path))

            doc["document_type"] = self._classify_document(
                doc["filename"], doc["raw_text"]
            )

            status_icon = "[green]✓[/green]" if doc["success"] else "[red]✗[/red]"
            console.print(
                f"    {status_icon} type=[dim]{doc['document_type']}[/dim]  "
                f"method=[dim]{doc['extraction_method']}[/dim]  "
                f"chars=[dim]{len(doc['raw_text'])}[/dim]"
            )

            results.append(doc)

        scanned = sum(1 for d in results if d.get("is_scanned"))
        if scanned:
            console.print(
                f"  [yellow]⚠  {scanned} document(s) appear scanned — "
                f"OCRTool will process them.[/yellow]"
            )

        return results