"""
MedicationExtractorTool — Extract admission and discharge medications.

NO-FABRICATION: every unknown field becomes "NOT DOCUMENTED".
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

_MEDICATION_PROMPT = """\
You are a clinical pharmacist reviewing hospital documents.

STRICT NO-FABRICATION RULES:
1. Extract ONLY medications explicitly listed in the text.
2. For any sub-field (dose, frequency, route) not stated, use "NOT DOCUMENTED".
3. DO NOT guess drug doses, frequencies, or routes.
4. DO NOT invent medications not present in the text.
5. Separate admission medications from discharge medications where possible.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "admission_medications": [
        {{
            "name": "",
            "dose": "NOT DOCUMENTED",
            "frequency": "NOT DOCUMENTED",
            "route": "NOT DOCUMENTED",
            "indication": null,
            "prescriber": null
        }}
    ],
    "discharge_medications": [
        {{
            "name": "",
            "dose": "NOT DOCUMENTED",
            "frequency": "NOT DOCUMENTED",
            "route": "NOT DOCUMENTED",
            "indication": null,
            "prescriber": null
        }}
    ],
    "medications_source": "<which section/document the medications came from>",
    "notes": "",
    "review_required": true
}}

Return ONLY the JSON. If no medications are documented, return empty arrays.
"""


# ─────────────────────────────────────────────
# Tool class
# ─────────────────────────────────────────────

class MedicationExtractorTool:
    """Extract admission and discharge medication lists from clinical documents."""

    name = "MedicationExtractorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract medication information from combined document text.

        Returns
        -------
        dict with admission_medications, discharge_medications, review_required
        """
        console.print("[bold green]MedicationExtractorTool[/bold green] — extracting …")

        fallback: Dict[str, Any] = {
            "admission_medications": [],
            "discharge_medications": [],
            "medications_source": "NOT DOCUMENTED",
            "notes": "",
            "review_required": True,
        }

        if not text or len(text.strip()) < 30:
            console.print("  [yellow]⚠ No text — returning empty medication lists[/yellow]")
            return fallback

        try:
            prompt = _MEDICATION_PROMPT.format(text=text[:18_000])
            data = self.gemini.generate_json(prompt, fallback=fallback)

            # Normalise — ensure every entry has required keys
            for section in ("admission_medications", "discharge_medications"):
                meds: List[Dict[str, Any]] = data.get(section, [])
                for med in meds:
                    for field in ("name", "dose", "frequency", "route"):
                        if not med.get(field):
                            med[field] = "NOT DOCUMENTED"
                data[section] = meds

            adm = len(data.get("admission_medications", []))
            dis = len(data.get("discharge_medications", []))
            console.print(
                f"  [green]✓[/green] Admission meds: [dim]{adm}[/dim]  |  "
                f"Discharge meds: [dim]{dis}[/dim]"
            )
            return data

        except Exception as exc:
            logger.error("MedicationExtractorTool error: %s", exc)
            fallback["error"] = str(exc)
            return fallback
