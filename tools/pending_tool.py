"""
PendingResultDetectorTool — Detect pending lab results, imaging, or cultures.

Any result mentioned as 'pending', 'awaiting', 'ordered', or 'sent' is flagged
so the clinician knows to follow up rather than assume the result is in.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

_PENDING_PROMPT = """\
You are a clinical information extractor reviewing hospital notes.

YOUR TASK: Identify ANY results that were:
  - Ordered but not yet resulted
  - Explicitly described as "pending", "awaited", "sent out", "in progress"
  - Cultures, biopsies, or tests with results not yet documented

STRICT NO-FABRICATION RULES:
1. Only flag results explicitly mentioned as pending/outstanding in the text.
2. Do NOT assume a result is pending just because a test was ordered — only if stated.
3. If no pending results are found, return an empty list.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "pending_results": [
        {{
            "result_name": "",
            "result_type": "lab|imaging|culture|pathology|other",
            "ordered_date": "NOT DOCUMENTED",
            "ordered_by": "NOT DOCUMENTED",
            "expected_availability": "NOT DOCUMENTED",
            "notes": ""
        }}
    ],
    "has_pending_results": true,
    "clinician_action_required": true
}}
"""


class PendingResultDetectorTool:
    """Detect pending investigations and outstanding results."""

    name = "PendingResultDetectorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def detect(self, text: str) -> Dict[str, Any]:
        """Identify pending results from combined document text."""
        console.print("[bold green]PendingResultDetectorTool[/bold green] — scanning …")

        fallback: Dict[str, Any] = {
            "pending_results": [],
            "has_pending_results": False,
            "clinician_action_required": False,
        }

        if not text or len(text.strip()) < 30:
            return fallback

        try:
            prompt = _PENDING_PROMPT.format(text=text[:15_000])
            data = self.gemini.generate_json(prompt, fallback=fallback)

            count = len(data.get("pending_results", []))
            if count:
                console.print(f"  [yellow]⚠[/yellow] {count} pending result(s) detected")
            else:
                console.print("  [green]✓[/green] No pending results found")
            return data

        except Exception as exc:
            logger.error("PendingResultDetectorTool error: %s", exc)
            fallback["error"] = str(exc)
            return fallback
