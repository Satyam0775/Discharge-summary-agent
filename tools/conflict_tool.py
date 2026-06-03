"""
ConflictDetectorTool — Detect factual conflicts across clinical documents.

When two documents state different values for the same field (e.g., two different
discharge diagnoses), the tool flags the conflict for clinician resolution.
The agent NEVER auto-resolves conflicts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

_CONFLICT_PROMPT = """\
You are a clinical quality reviewer checking for inconsistencies across hospital documents.

TASK: Compare the following clinical notes and identify ALL factual conflicts where two
notes state different values for the same clinical field.

Fields to check:
  - Diagnosis (admission vs discharge vs progress notes)
  - Medication names, doses, or frequencies
  - Dates (admission, discharge, procedure dates)
  - Allergy information
  - Patient demographics (name, DOB)
  - Procedure descriptions

IMPORTANT: Only flag genuine contradictions — not missing information.

HOSPITAL NOTES (labelled by source document):
===============
{text}
===============

Return a single JSON object:

{{
    "conflicts": [
        {{
            "conflict_id": "C001",
            "field": "<field name>",
            "source_a": "<document/section name>",
            "value_a": "<what source A says>",
            "source_b": "<document/section name>",
            "value_b": "<what source B says>",
            "description": "<brief description of conflict>",
            "severity": "high|medium|low",
            "clinician_review_required": true
        }}
    ],
    "has_conflicts": false,
    "summary": ""
}}
"""


class ConflictDetectorTool:
    """Detect factual conflicts between documents in the patient record."""

    name = "ConflictDetectorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    def _build_labelled_text(self, documents: List[Dict[str, Any]]) -> str:
        """Build a text block where each section is labelled by its source document."""
        parts: List[str] = []
        for doc in documents:
            if doc.get("raw_text", "").strip():
                label = f"[DOCUMENT: {doc.get('filename', 'unknown')} | type: {doc.get('document_type', 'unknown')}]"
                parts.append(f"{label}\n{doc['raw_text'][:3_000]}")
        return "\n\n===\n\n".join(parts)

    def detect(
        self,
        documents: List[Dict[str, Any]],
        existing_conflicts: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Detect conflicts across all extracted documents.

        Parameters
        ----------
        documents : list of raw document dicts (from PDFLoaderTool)
        existing_conflicts : conflicts already found during extraction (e.g. from DiagnosisTool)
        """
        console.print("[bold green]ConflictDetectorTool[/bold green] — scanning …")

        fallback: Dict[str, Any] = {
            "conflicts": existing_conflicts or [],
            "has_conflicts": bool(existing_conflicts),
            "summary": "Conflict detection could not complete.",
        }

        if not documents:
            return fallback

        labelled_text = self._build_labelled_text(documents)
        if not labelled_text.strip():
            return fallback

        try:
            prompt = _CONFLICT_PROMPT.format(text=labelled_text[:20_000])
            data = self.gemini.generate_json(prompt, fallback=fallback)

            # Merge with any pre-existing conflicts from extraction tools
            all_conflicts: List[Dict[str, Any]] = list(existing_conflicts or [])
            new_conflicts: List[Dict[str, Any]] = data.get("conflicts", [])

            # Deduplicate by field + value_a + value_b
            seen = set()
            for c in new_conflicts:
                key = (c.get("field", ""), c.get("value_a", ""), c.get("value_b", ""))
                if key not in seen:
                    seen.add(key)
                    all_conflicts.append(c)

            data["conflicts"] = all_conflicts
            data["has_conflicts"] = bool(all_conflicts)

            if all_conflicts:
                console.print(
                    f"  [red]⚠ {len(all_conflicts)} conflict(s) detected — clinician review required[/red]"
                )
            else:
                console.print("  [green]✓[/green] No conflicts detected")

            return data

        except Exception as exc:
            logger.error("ConflictDetectorTool error: %s", exc)
            fallback["error"] = str(exc)
            return fallback
