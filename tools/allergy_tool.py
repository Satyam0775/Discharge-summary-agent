"""
AllergyExtractorTool — Extract allergy and adverse reaction information.

ROOT CAUSE FIX:
  No changes needed to production logic.
  The `generate_json(prompt, fallback=fallback)` call is correct for the real GeminiService.
  The fix is in tests/test_tools.py (_FakeGeminiService.generate_json now accepts fallback kwarg).
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

_ALLERGY_PROMPT = """\
You are a clinical safety reviewer extracting allergy information from hospital notes.

STRICT NO-FABRICATION RULES:
1. Extract ONLY allergies and adverse reactions that are EXPLICITLY documented.
2. If no allergies are documented, return "NOT DOCUMENTED" — NEVER assume "NKDA" unless the text says so.
3. Do NOT infer or guess reactions not stated.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "allergies_raw": "<exact text from document OR NOT DOCUMENTED>",
    "allergy_list": [
        {{
            "allergen": "",
            "reaction": "NOT DOCUMENTED",
            "severity": "NOT DOCUMENTED",
            "documented_in": ""
        }}
    ],
    "nkda_documented": false,
    "review_required": true,
    "safety_note": ""
}}
"""


class AllergyExtractorTool:
    """Extract allergy / adverse reaction information from clinical documents."""

    name = "AllergyExtractorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract allergy information. Returns NOT DOCUMENTED if absent."""
        console.print("[bold green]AllergyExtractorTool[/bold green] — extracting …")

        fallback: Dict[str, Any] = {
            "allergies_raw": "NOT DOCUMENTED",
            "allergy_list": [],
            "nkda_documented": False,
            "review_required": True,
            "safety_note": "Allergy information could not be extracted.",
        }

        if not text or len(text.strip()) < 30:
            console.print("  [yellow]⚠ No text — allergies NOT DOCUMENTED[/yellow]")
            return fallback

        try:
            prompt = _ALLERGY_PROMPT.format(text=text[:12_000])
            data = self.gemini.generate_json(prompt, fallback=fallback)

            allergies_raw = data.get("allergies_raw", "NOT DOCUMENTED")
            nkda = data.get("nkda_documented", False)
            review = data.get("review_required", True)

            if allergies_raw == "NOT DOCUMENTED" and not nkda:
                review = True

            data["review_required"] = review
            console.print(
                f"  [green]✓[/green] Allergies: [dim]{allergies_raw[:80]}[/dim]  "
                f"NKDA={nkda}  review={review}"
            )
            return data

        except Exception as exc:
            logger.error("AllergyExtractorTool error: %s", exc)
            fallback["error"] = str(exc)
            return fallback