"""
FollowupExtractorTool — Extract follow-up instructions and appointments from discharge notes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

_FOLLOWUP_PROMPT = """\
You are a clinical information extractor reviewing hospital discharge notes.

STRICT NO-FABRICATION RULES:
1. Extract ONLY follow-up instructions that are EXPLICITLY written in the notes.
2. If instructions are absent, return "NOT DOCUMENTED".
3. Do NOT generate generic follow-up advice not stated in the documents.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "followup_instructions": "<detailed instructions as documented OR NOT DOCUMENTED>",
    "followup_appointments": [
        {{
            "with": "NOT DOCUMENTED",
            "when": "NOT DOCUMENTED",
            "location": "NOT DOCUMENTED",
            "purpose": "NOT DOCUMENTED"
        }}
    ],
    "dietary_instructions": "NOT DOCUMENTED",
    "activity_restrictions": "NOT DOCUMENTED",
    "wound_care": "NOT DOCUMENTED",
    "return_precautions": "NOT DOCUMENTED",
    "review_required": false
}}
"""


class FollowupExtractorTool:
    """Extract follow-up instructions from clinical documents."""

    name = "FollowupExtractorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def extract(self, text: str) -> Dict[str, Any]:
        """Extract follow-up instructions from combined document text."""
        console.print("[bold green]FollowupExtractorTool[/bold green] — extracting …")

        fallback: Dict[str, Any] = {
            "followup_instructions": "NOT DOCUMENTED",
            "followup_appointments": [],
            "dietary_instructions": "NOT DOCUMENTED",
            "activity_restrictions": "NOT DOCUMENTED",
            "wound_care": "NOT DOCUMENTED",
            "return_precautions": "NOT DOCUMENTED",
            "review_required": True,
        }

        if not text or len(text.strip()) < 30:
            console.print("  [yellow]⚠ No text — follow-up NOT DOCUMENTED[/yellow]")
            return fallback

        try:
            prompt = _FOLLOWUP_PROMPT.format(text=text[:12_000])
            data = self.gemini.generate_json(prompt, fallback=fallback)

            followup_text = data.get("followup_instructions", "NOT DOCUMENTED")
            appts = len(data.get("followup_appointments", []))
            console.print(
                f"  [green]✓[/green] Follow-up: [dim]{followup_text[:60]}[/dim]  "
                f"appointments: {appts}"
            )
            return data

        except Exception as exc:
            logger.error("FollowupExtractorTool error: %s", exc)
            fallback["error"] = str(exc)
            return fallback
