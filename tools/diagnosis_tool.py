"""
DiagnosisExtractorTool — Extract diagnoses and hospital course from clinical notes.

NO-FABRICATION RULE: The prompt is designed so the model MUST return
"NOT DOCUMENTED" for any field it cannot find in the text.
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

_DIAGNOSIS_PROMPT = """\
You are a clinical information extractor reviewing hospital notes.

STRICT NO-FABRICATION RULES — READ CAREFULLY:
1. Extract ONLY information that is EXPLICITLY written in the notes below.
2. If a field is not present in the text, you MUST return "NOT DOCUMENTED".
3. NEVER guess, infer, or complete a diagnosis that is only partially stated.
4. NEVER invent clinical details, dates, or names.
5. If two notes disagree on a diagnosis, list BOTH and flag the conflict.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object with these exact keys:

{{
    "principal_diagnosis": "<string or NOT DOCUMENTED>",
    "secondary_diagnoses": ["<list of strings — empty if none found>"],
    "admission_diagnosis": "<string or NOT DOCUMENTED>",
    "discharge_diagnosis": "<string or NOT DOCUMENTED>",
    "diagnosis_sources": [
        {{"diagnosis": "", "source_document": "", "type": "admission|discharge|progress|other"}}
    ],
    "conflicts": [
        {{
            "field": "diagnosis",
            "value_a": "",
            "source_a": "",
            "value_b": "",
            "source_b": "",
            "description": ""
        }}
    ],
    "review_required": true,
    "notes": ""
}}
"""

_HOSPITAL_COURSE_PROMPT = """\
You are a clinical information extractor reviewing hospital notes.

STRICT NO-FABRICATION RULES:
1. Summarise the hospital course ONLY from what is explicitly written.
2. Do NOT invent clinical details or procedures not mentioned.
3. Return "NOT DOCUMENTED" for any field absent from the text.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "admission_date": "<string or NOT DOCUMENTED>",
    "discharge_date": "<string or NOT DOCUMENTED>",
    "length_of_stay": "<string or NOT DOCUMENTED>",
    "hospital_course": "<narrative from notes or NOT DOCUMENTED>",
    "discharge_condition": "<stable|improved|critical|deceased|NOT DOCUMENTED>",
    "review_required": true
}}
"""

_DEMOGRAPHICS_PROMPT = """\
You are a clinical information extractor.

STRICT NO-FABRICATION RULES:
1. Extract ONLY what is explicitly written.
2. Return "NOT DOCUMENTED" for missing fields.

HOSPITAL NOTES:
===============
{text}
===============

Return a single JSON object:

{{
    "patient_name": "<string or NOT DOCUMENTED>",
    "patient_id": "<MRN/patient ID or NOT DOCUMENTED>",
    "date_of_birth": "<DD/MM/YYYY or NOT DOCUMENTED>",
    "age": "<string or NOT DOCUMENTED>",
    "gender": "<Male|Female|Other|NOT DOCUMENTED>",
    "address": "<string or NOT DOCUMENTED>",
    "contact_number": "<string or NOT DOCUMENTED>",
    "next_of_kin": "<string or NOT DOCUMENTED>",
    "review_required": true
}}
"""


# ─────────────────────────────────────────────
# Tool class
# ─────────────────────────────────────────────

class DiagnosisExtractorTool:
    """
    Extracts from combined clinical text:
      - Patient demographics
      - Principal + secondary diagnoses
      - Admission / discharge diagnoses
      - Hospital course narrative
      - Discharge condition
      - Admission & discharge dates
      - Cross-document conflicts
    """

    name = "DiagnosisExtractorTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _run_prompt(self, template: str, text: str, fallback: dict) -> dict:
        prompt = template.format(text=text[:18_000])
        return self.gemini.generate_json(prompt, fallback=fallback)

    # ─────────────────────────────────────────────
    # Public extract
    # ─────────────────────────────────────────────

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Main extraction method.

        Parameters
        ----------
        text : combined text from all patient documents

        Returns
        -------
        dict with all extracted diagnosis / demographics fields
        """
        console.print("[bold green]DiagnosisExtractorTool[/bold green] — extracting …")

        result: Dict[str, Any] = {
            "demographics": {
                "patient_name": "NOT DOCUMENTED",
                "patient_id": "NOT DOCUMENTED",
                "date_of_birth": "NOT DOCUMENTED",
                "age": "NOT DOCUMENTED",
                "gender": "NOT DOCUMENTED",
                "address": "NOT DOCUMENTED",
                "contact_number": "NOT DOCUMENTED",
                "next_of_kin": "NOT DOCUMENTED",
                "review_required": True,
            },
            "principal_diagnosis": "NOT DOCUMENTED",
            "secondary_diagnoses": [],
            "admission_diagnosis": "NOT DOCUMENTED",
            "discharge_diagnosis": "NOT DOCUMENTED",
            "admission_date": "NOT DOCUMENTED",
            "discharge_date": "NOT DOCUMENTED",
            "length_of_stay": "NOT DOCUMENTED",
            "hospital_course": "NOT DOCUMENTED",
            "discharge_condition": "NOT DOCUMENTED",
            "conflicts": [],
            "review_required": True,
            "error": None,
        }

        if not text or len(text.strip()) < 30:
            result["error"] = "No usable text provided"
            console.print("  [red]✗ No text to extract from[/red]")
            return result

        # ── Demographics ─────────────────────────────────────────────────────
        try:
            demo = self._run_prompt(
                _DEMOGRAPHICS_PROMPT,
                text,
                fallback=result["demographics"],
            )
            result["demographics"] = demo
        except Exception as exc:
            logger.error("Demographics extraction failed: %s", exc)
            result["error"] = str(exc)

        # ── Diagnoses ────────────────────────────────────────────────────────
        try:
            diag_fallback: Dict[str, Any] = {
                "principal_diagnosis": "NOT DOCUMENTED",
                "secondary_diagnoses": [],
                "admission_diagnosis": "NOT DOCUMENTED",
                "discharge_diagnosis": "NOT DOCUMENTED",
                "diagnosis_sources": [],
                "conflicts": [],
                "review_required": True,
                "notes": "",
            }
            diag = self._run_prompt(_DIAGNOSIS_PROMPT, text, fallback=diag_fallback)

            result["principal_diagnosis"]  = diag.get("principal_diagnosis", "NOT DOCUMENTED")
            result["secondary_diagnoses"]  = diag.get("secondary_diagnoses", [])
            result["admission_diagnosis"]  = diag.get("admission_diagnosis", "NOT DOCUMENTED")
            result["discharge_diagnosis"]  = diag.get("discharge_diagnosis", "NOT DOCUMENTED")
            result["conflicts"]            = diag.get("conflicts", [])
            result["review_required"]      = diag.get("review_required", True)

        except Exception as exc:
            logger.error("Diagnosis extraction failed: %s", exc)
            result["error"] = str(exc)

        # ── Hospital course ──────────────────────────────────────────────────
        try:
            course_fallback: Dict[str, Any] = {
                "admission_date": "NOT DOCUMENTED",
                "discharge_date": "NOT DOCUMENTED",
                "length_of_stay": "NOT DOCUMENTED",
                "hospital_course": "NOT DOCUMENTED",
                "discharge_condition": "NOT DOCUMENTED",
                "review_required": True,
            }
            course = self._run_prompt(
                _HOSPITAL_COURSE_PROMPT, text, fallback=course_fallback
            )

            result["admission_date"]   = course.get("admission_date", "NOT DOCUMENTED")
            result["discharge_date"]   = course.get("discharge_date", "NOT DOCUMENTED")
            result["length_of_stay"]   = course.get("length_of_stay", "NOT DOCUMENTED")
            result["hospital_course"]  = course.get("hospital_course", "NOT DOCUMENTED")
            result["discharge_condition"] = course.get("discharge_condition", "NOT DOCUMENTED")

        except Exception as exc:
            logger.error("Hospital course extraction failed: %s", exc)

        console.print(
            f"  [green]✓[/green] Principal dx: [dim]{result['principal_diagnosis'][:60]}[/dim]"
        )
        return result
