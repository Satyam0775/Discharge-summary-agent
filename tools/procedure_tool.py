"""
ProcedureExtractorTool — Extract procedures and interventions performed during admission.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

_PROCEDURE_PROMPT = """\
You are a clinical information extractor reviewing hospital notes.

STRICT NO-FABRICATION RULES:
1. Extract ONLY procedures that are EXPLICITLY documented in the text.
2. Return "NOT DOCUMENTED" for any field that is absent.
3. Do NOT infer procedures from diagnoses (e.g., do not assume appendectomy from appendicitis).

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "procedures": [
        {{
            "name": "",
            "date": "NOT DOCUMENTED",
            "operator": "NOT DOCUMENTED",
            "indication": "NOT DOCUMENTED",
            "outcome": "NOT DOCUMENTED"
        }}
    ],
    "procedures_source": "<document/section where found>",
    "no_procedures_documented": false,
    "review_required": false
}}

If no procedures are documented, set "no_procedures_documented": true and return an empty procedures list.
"""


class ProcedureExtractorTool:
    """Extract procedures and interventions from clinical documents."""

    name = "ProcedureExtractorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract procedure information from combined document text."""
        console.print("[bold green]ProcedureExtractorTool[/bold green] — extracting …")

        fallback: Dict[str, Any] = {
            "procedures": [],
            "procedures_source": "NOT DOCUMENTED",
            "no_procedures_documented": True,
            "review_required": False,
        }

        if not text or len(text.strip()) < 30:
            console.print("  [yellow]⚠ No text — procedures NOT DOCUMENTED[/yellow]")
            return fallback

        try:
            prompt = _PROCEDURE_PROMPT.format(text=text[:15_000])
            data = self.gemini.generate_json(prompt, fallback=fallback)

            proc_count = len(data.get("procedures", []))
            console.print(
                f"  [green]✓[/green] Procedures found: [dim]{proc_count}[/dim]"
            )
            return data

        except Exception as exc:
            logger.error("ProcedureExtractorTool error: %s", exc)
            fallback["error"] = str(exc)
            return fallback
