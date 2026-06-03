"""
MedicationReconciliationTool — Compare admission vs discharge medications.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from rich.console import Console
from services.gemini_service import GeminiService

logger = logging.getLogger(__name__)
console = Console()

_RECONCILIATION_PROMPT = """\
You are a clinical pharmacist performing medication reconciliation.

TASK: Compare the admission medication list against the discharge medication list.
For each medication, classify the change.

ADMISSION MEDICATIONS:
{admission_meds}

DISCHARGE MEDICATIONS:
{discharge_meds}

For each medication, determine if it was:
  - "added"     : present at discharge but NOT at admission
  - "removed"   : present at admission but NOT at discharge
  - "modified"  : present at both but dose/frequency/route changed
  - "unchanged" : identical at both admission and discharge

For any "removed" or "modified" medication, check if a reason is documented.
If NO reason is documented, set "flag_for_review": true.

Return a single JSON object:

{{
    "added": [
        {{
            "medication_name": "",
            "name": "",
            "discharge_details": {{"dose":"","frequency":"","route":""}},
            "documented_reason": null,
            "flag_for_review": false,
            "flag_reason": null
        }}
    ],
    "removed": [
        {{
            "medication_name": "",
            "name": "",
            "admission_details": {{"dose":"","frequency":"","route":""}},
            "documented_reason": null,
            "flag_for_review": true,
            "flag_reason": "No documented reason for discontinuation"
        }}
    ],
    "modified": [
        {{
            "medication_name": "",
            "name": "",
            "admission_details": {{"dose":"","frequency":"","route":""}},
            "discharge_details": {{"dose":"","frequency":"","route":""}},
            "documented_reason": null,
            "flag_for_review": true,
            "flag_reason": "No documented reason for change"
        }}
    ],
    "unchanged": [
        {{
            "medication_name": "",
            "name": "",
            "details": {{"dose":"","frequency":"","route":""}}
        }}
    ],
    "reason_missing": ["<list medication names with no documented reason for change>"],
    "review_required": true,
    "notes": ""
}}
"""


class MedicationReconciliationTool:
    """
    Reconcile admission medications vs discharge medications.
    Uses local name-based diffing as primary logic, then enriches
    with Gemini analysis where available.
    """

    name = "MedicationReconciliationTool"

    def __init__(self, gemini: GeminiService) -> None:
        self.gemini = gemini

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_name(name: str) -> str:
        """Lowercase, strip whitespace for comparison."""
        return name.strip().lower()

    @staticmethod
    def _format_med_list(meds: List[Dict[str, Any]]) -> str:
        if not meds:
            return "NONE DOCUMENTED"
        lines: List[str] = []
        for m in meds:
            name  = m.get("name",      "Unknown")
            dose  = m.get("dose",      "NOT DOCUMENTED")
            freq  = m.get("frequency", "NOT DOCUMENTED")
            route = m.get("route",     "NOT DOCUMENTED")
            lines.append(
                f"  - {name} | dose: {dose} | freq: {freq} | route: {route}"
            )
        return "\n".join(lines)

    # ── Local diff (no LLM needed) ────────────────────────────────────────────

    def _local_diff(
        self,
        admission: List[Dict[str, Any]],
        discharge: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Name-based medication diff.
        Correctly identifies added, removed, modified, and unchanged meds
        without an LLM call — ensures tests and offline runs work.
        """
        adm_index: Dict[str, Dict[str, Any]] = {
            self._normalise_name(m.get("name", "")): m for m in admission
        }
        dis_index: Dict[str, Dict[str, Any]] = {
            self._normalise_name(m.get("name", "")): m for m in discharge
        }

        added:     List[Dict[str, Any]] = []
        removed:   List[Dict[str, Any]] = []
        modified:  List[Dict[str, Any]] = []
        unchanged: List[Dict[str, Any]] = []
        reason_missing: List[str] = []

        # Meds in discharge — added or modified/unchanged
        for norm_name, dis_med in dis_index.items():
            med_name = dis_med.get("name", norm_name)
            if norm_name not in adm_index:
                added.append({
                    "medication_name": med_name,
                    "name": med_name,
                    "discharge_details": {
                        "dose":      dis_med.get("dose", "NOT DOCUMENTED"),
                        "frequency": dis_med.get("frequency", "NOT DOCUMENTED"),
                        "route":     dis_med.get("route", "NOT DOCUMENTED"),
                    },
                    "documented_reason": None,
                    "flag_for_review": False,
                    "flag_reason": None,
                })
            else:
                adm_med = adm_index[norm_name]
                adm_sig = (
                    adm_med.get("dose", ""),
                    adm_med.get("frequency", ""),
                    adm_med.get("route", ""),
                )
                dis_sig = (
                    dis_med.get("dose", ""),
                    dis_med.get("frequency", ""),
                    dis_med.get("route", ""),
                )
                if adm_sig == dis_sig:
                    unchanged.append({
                        "medication_name": med_name,
                        "name": med_name,
                        "details": {
                            "dose":      dis_med.get("dose", "NOT DOCUMENTED"),
                            "frequency": dis_med.get("frequency", "NOT DOCUMENTED"),
                            "route":     dis_med.get("route", "NOT DOCUMENTED"),
                        },
                    })
                else:
                    modified.append({
                        "medication_name": med_name,
                        "name": med_name,
                        "admission_details": {
                            "dose":      adm_med.get("dose", "NOT DOCUMENTED"),
                            "frequency": adm_med.get("frequency", "NOT DOCUMENTED"),
                            "route":     adm_med.get("route", "NOT DOCUMENTED"),
                        },
                        "discharge_details": {
                            "dose":      dis_med.get("dose", "NOT DOCUMENTED"),
                            "frequency": dis_med.get("frequency", "NOT DOCUMENTED"),
                            "route":     dis_med.get("route", "NOT DOCUMENTED"),
                        },
                        "documented_reason": None,
                        "flag_for_review": True,
                        "flag_reason": "No documented reason for change",
                    })
                    reason_missing.append(med_name)

        # Meds in admission but not in discharge — removed
        for norm_name, adm_med in adm_index.items():
            if norm_name not in dis_index:
                med_name = adm_med.get("name", norm_name)
                removed.append({
                    "medication_name": med_name,
                    "name": med_name,
                    "admission_details": {
                        "dose":      adm_med.get("dose", "NOT DOCUMENTED"),
                        "frequency": adm_med.get("frequency", "NOT DOCUMENTED"),
                        "route":     adm_med.get("route", "NOT DOCUMENTED"),
                    },
                    "documented_reason": None,
                    "flag_for_review": True,
                    "flag_reason": "No documented reason for discontinuation",
                })
                reason_missing.append(med_name)

        return {
            "added":          added,
            "removed":        removed,
            "modified":       modified,
            "unchanged":      unchanged,
            "reason_missing": reason_missing,
            "review_required": bool(removed or modified or reason_missing),
            "notes": "",
        }

    # ── Public interface ──────────────────────────────────────────────────────

    def reconcile(
        self,
        admission_medications: List[Dict[str, Any]],
        discharge_medications: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Perform medication reconciliation.

        1. Always runs a local name-based diff first (reliable, no LLM).
        2. Attempts a Gemini-enriched reconciliation for narrative reasons.
        3. Falls back to the local diff if Gemini is unavailable.

        Returns dict with: added, removed, modified, unchanged,
                           reason_missing, review_required, notes.
        """
        console.print(
            "[bold green]MedicationReconciliationTool[/bold green] — reconciling …"
        )

        # ── Handle empty case ────────────────────────────────────────────────
        if not admission_medications and not discharge_medications:
            console.print("  [yellow]⚠ No medications to reconcile[/yellow]")
            return {
                "added": [],
                "removed": [],
                "modified": [],
                "unchanged": [],
                "reason_missing": [],
                "review_required": True,
                "notes": "No admission or discharge medications documented.",
            }

        # ── Step 1: local diff (always reliable) ─────────────────────────────
        local_result = self._local_diff(admission_medications, discharge_medications)

        # ── Step 2: try Gemini for enriched reasons ──────────────────────────
        adm_text = self._format_med_list(admission_medications)
        dis_text = self._format_med_list(discharge_medications)

        try:
            prompt = _RECONCILIATION_PROMPT.format(
                admission_meds=adm_text,
                discharge_meds=dis_text,
            )
            # NOTE: generate_json called WITHOUT fallback kwarg
            # so it is compatible with both real GeminiService and test stubs.
            gemini_data = self.gemini.generate_json(prompt)

            # Only accept the Gemini result if it has the expected keys
            if all(
                k in gemini_data
                for k in ("added", "removed", "modified", "unchanged")
            ):
                data = gemini_data
                # Merge reason_missing from local diff (more reliable)
                if not data.get("reason_missing"):
                    data["reason_missing"] = local_result["reason_missing"]
            else:
                data = local_result

        except Exception as exc:
            logger.warning(
                "Gemini reconciliation failed (%s) — using local diff.", exc
            )
            data = local_result

        # ── Report ───────────────────────────────────────────────────────────
        added    = len(data.get("added",    []))
        removed  = len(data.get("removed",  []))
        modified = len(data.get("modified", []))
        flagged  = len(data.get("reason_missing", []))

        console.print(
            f"  [green]✓[/green] Reconciliation: "
            f"[green]+{added}[/green] added  "
            f"[red]-{removed}[/red] removed  "
            f"[yellow]~{modified}[/yellow] modified  "
            f"[red]{flagged}[/red] flagged (no reason)"
        )

        return data