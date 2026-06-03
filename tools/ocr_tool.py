"""
OCRTool — Optical Character Recognition for scanned PDFs.

Pipeline:  PDF → images (pdf2image / poppler)
                → OCR   (PaddleOCR)
                → text

Both pdf2image and PaddleOCR are imported lazily; the tool degrades
gracefully if either is missing rather than crashing the whole agent.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from tenacity import retry, stop_after_attempt, wait_fixed
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


# ─────────────────────────────────────────────
# Lazy import helpers
# ─────────────────────────────────────────────

def _try_import_pdf2image():
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except ImportError:
        logger.warning(
            "pdf2image not installed. Scanned PDFs cannot be OCR'd. "
            "Install: pip install pdf2image  (also requires poppler — see README)."
        )
        return None


def _try_init_paddleocr(lang: str = "en"):
    try:
        from paddleocr import PaddleOCR  # noqa: PLC0415
        ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        logger.info("PaddleOCR initialised (lang=%s)", lang)
        return ocr
    except ImportError:
        logger.warning(
            "paddleocr not installed. OCR unavailable. "
            "Install: pip install paddleocr paddlepaddle"
        )
        return None
    except Exception as exc:
        logger.error("PaddleOCR init failed: %s", exc)
        return None


# ─────────────────────────────────────────────
# OCRTool
# ─────────────────────────────────────────────

class OCRTool:
    """
    Runs OCR on one or more scanned PDFs.

    Usage
    -----
    tool = OCRTool()
    result = tool.ocr_pdf("/path/to/scanned.pdf")
    text = result["text"]
    """

    name = "OCRTool"

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._convert_from_path = _try_import_pdf2image()
        self._ocr: Optional[Any] = None           # lazy — only load when first needed
        self._poppler_path: Optional[str] = os.getenv("POPPLER_PATH", None)

    def _get_ocr(self):
        """Lazily initialise PaddleOCR on first use."""
        if self._ocr is None:
            self._ocr = _try_init_paddleocr(self._lang)
        return self._ocr

    # ─────────────────────────────────────────────
    # PDF → images
    # ─────────────────────────────────────────────

    def _pdf_to_images(self, pdf_path: str) -> List[Any]:
        """Convert a PDF to a list of PIL Image objects."""
        if self._convert_from_path is None:
            raise RuntimeError(
                "pdf2image is not installed. "
                "Run: pip install pdf2image  and install poppler (see README)."
            )
        kwargs: Dict[str, Any] = {"dpi": 200}
        if self._poppler_path:
            kwargs["poppler_path"] = self._poppler_path

        images = self._convert_from_path(pdf_path, **kwargs)
        return images  # type: ignore[return-value]

    # ─────────────────────────────────────────────
    # Single image → text
    # ─────────────────────────────────────────────

    @retry(stop=stop_after_attempt(2), wait=wait_fixed(2))
    def _ocr_image(self, image) -> str:
        """Run PaddleOCR on a single PIL image and return extracted text."""
        ocr = self._get_ocr()
        if ocr is None:
            raise RuntimeError("PaddleOCR is not available.")

        img_array = np.array(image)
        result = ocr.ocr(img_array, cls=True)

        if not result or not result[0]:
            return ""

        lines: List[str] = []
        for detection in result[0]:
            if detection and len(detection) > 1:
                text_val = detection[1][0]
                confidence = detection[1][1]
                if confidence >= 0.45:   # discard very low-confidence predictions
                    lines.append(text_val)

        return " ".join(lines)

    # ─────────────────────────────────────────────
    # Public: OCR a whole PDF
    # ─────────────────────────────────────────────

    def ocr_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        OCR a scanned PDF file.

        Returns
        -------
        dict with keys: filename, text, pages_processed,
                        extraction_method, success, error
        """
        result: Dict[str, Any] = {
            "filename": os.path.basename(pdf_path),
            "text": "",
            "pages_processed": 0,
            "extraction_method": "ocr",
            "success": False,
            "error": None,
        }

        try:
            console.print(
                f"  [yellow]OCRTool[/yellow] running on: [dim]{os.path.basename(pdf_path)}[/dim]"
            )

            images = self._pdf_to_images(pdf_path)
            page_texts: List[str] = []

            for idx, img in enumerate(images, start=1):
                console.print(f"    [dim]OCR page {idx}/{len(images)} …[/dim]")
                try:
                    page_text = self._ocr_image(img)
                    page_texts.append(page_text)
                    result["pages_processed"] += 1
                except Exception as exc:
                    logger.warning("OCR failed on page %d: %s", idx, exc)
                    page_texts.append(f"[OCR FAILED — PAGE {idx}]")

            result["text"] = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
            result["success"] = True
            console.print(
                f"    [green]✓[/green] OCR complete — "
                f"[dim]{result['pages_processed']}[/dim] pages, "
                f"[dim]{len(result['text'])}[/dim] chars extracted"
            )

        except RuntimeError as exc:
            # pdf2image or PaddleOCR not available
            result["error"] = str(exc)
            result["text"] = "[OCR NOT AVAILABLE — MISSING DEPENDENCY]"
            console.print(f"    [red]✗ OCR unavailable:[/red] {exc}")

        except Exception as exc:
            result["error"] = str(exc)
            result["text"] = f"[OCR FAILED: {exc}]"
            logger.error("OCR failed for %s: %s", pdf_path, exc)

        return result

    # ─────────────────────────────────────────────
    # Process list of scanned documents in-place
    # ─────────────────────────────────────────────

    def process_scanned_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        For any document where extraction_method == 'needs_ocr',
        run OCR and update raw_text in-place.
        """
        scanned = [d for d in documents if d.get("extraction_method") == "needs_ocr"]

        if not scanned:
            return documents

        console.print(
            f"[bold yellow]OCRTool[/bold yellow] — "
            f"Processing [yellow]{len(scanned)}[/yellow] scanned document(s)"
        )

        for doc in scanned:
            ocr_result = self.ocr_pdf(doc["filepath"])
            if ocr_result["success"] and ocr_result["text"].strip():
                doc["raw_text"] = ocr_result["text"]
                doc["extraction_method"] = "ocr"
            else:
                doc["extraction_method"] = "failed"
                doc["error_message"] = ocr_result.get("error", "OCR returned empty text")

        return documents
