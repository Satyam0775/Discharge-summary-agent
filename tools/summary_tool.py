"""
SummaryGeneratorTool — Compile all extracted data into a structured discharge summary.

Produces both a JSON object and a formatted Markdown document.
The DRAFT watermark is always present.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()


class SummaryGeneratorTool:
    """
    Assembles the final discharge summary from all agent state fields.
    Does NOT make any additional LLM calls — it is purely a formatter.
    """

    name = "SummaryGeneratorTool"

    # ─────────────────────────────────────────────
    # JSON summary builder
    # ─────────────────────────────────────────────

    def build_json_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build the JSON discharge summary dict from agent state."""
        console.print("[bold green]SummaryGeneratorTool[/bold green] — building JSON …")

        demo = state.get("demographics", {}) or {}
        recon = state.get("reconciliation", {}) or {}
        conflict_data = state.get("conflict_data", {}) or {}
        pending_data = state.get("pending_data", {}) or {}
        allergy_data = state.get("allergy_data", {}) or {}
        procedure_data = state.get("procedure_data", {}) or {}
        followup_data = state.get("followup_data", {}) or {}
        review_flags = state.get("review_flags", []) or []

        summary = {
            "DRAFT_NOTICE": (
                "⚠️  THIS IS AN AI-GENERATED DRAFT FOR CLINICIAN REVIEW ONLY.  "
                "IT MUST NOT BE USED FOR CLINICAL PURPOSES WITHOUT VERIFICATION "
                "BY A QUALIFIED CLINICIAN.  ⚠️"
            ),
            "generated_at": datetime.utcnow().isoformat(),
            "agent_steps_taken": state.get("current_step", 0),
            "source_documents": [
                d.get("filename", "") for d in state.get("extracted_documents", [])
            ],
            "patient_demographics": {
                "patient_name":    demo.get("patient_name", "NOT DOCUMENTED"),
                "patient_id":      demo.get("patient_id", "NOT DOCUMENTED"),
                "date_of_birth":   demo.get("date_of_birth", "NOT DOCUMENTED"),
                "age":             demo.get("age", "NOT DOCUMENTED"),
                "gender":          demo.get("gender", "NOT DOCUMENTED"),
                "address":         demo.get("address", "NOT DOCUMENTED"),
                "contact_number":  demo.get("contact_number", "NOT DOCUMENTED"),
                "next_of_kin":     demo.get("next_of_kin", "NOT DOCUMENTED"),
            },
            "admission_date":         state.get("admission_date", "NOT DOCUMENTED"),
            "discharge_date":         state.get("discharge_date", "NOT DOCUMENTED"),
            "length_of_stay":         state.get("length_of_stay", "NOT DOCUMENTED"),
            "principal_diagnosis":    state.get("principal_diagnosis", "NOT DOCUMENTED"),
            "secondary_diagnoses":    state.get("secondary_diagnoses", []),
            "hospital_course":        state.get("hospital_course", "NOT DOCUMENTED"),
            "discharge_condition":    state.get("discharge_condition", "NOT DOCUMENTED"),
            "procedures":             procedure_data.get("procedures", []),
            "admission_medications":  state.get("admission_medications", []),
            "discharge_medications":  state.get("discharge_medications", []),
            "medication_reconciliation": {
                "added":          recon.get("added", []),
                "removed":        recon.get("removed", []),
                "modified":       recon.get("modified", []),
                "unchanged":      recon.get("unchanged", []),
                "reason_missing": recon.get("reason_missing", []),
                "review_required": recon.get("review_required", True),
                "notes":          recon.get("notes", ""),
            },
            "allergies": (
                allergy_data.get("allergies_raw", "NOT DOCUMENTED")
                if allergy_data.get("allergies_raw")
                else "NOT DOCUMENTED"
            ),
            "allergy_list": allergy_data.get("allergy_list", []),
            "followup_instructions": followup_data.get("followup_instructions", "NOT DOCUMENTED"),
            "followup_appointments": followup_data.get("followup_appointments", []),
            "dietary_instructions":  followup_data.get("dietary_instructions", "NOT DOCUMENTED"),
            "activity_restrictions": followup_data.get("activity_restrictions", "NOT DOCUMENTED"),
            "pending_results": pending_data.get("pending_results", []),
            "conflicts_detected": conflict_data.get("conflicts", []),
            "review_flags": review_flags,
            "missing_fields": state.get("missing_fields", []),
            "is_draft": True,
        }

        return summary

    # ─────────────────────────────────────────────
    # Markdown builder
    # ─────────────────────────────────────────────

    def build_markdown(self, summary: Dict[str, Any]) -> str:
        """Render the JSON summary as a formatted Markdown discharge summary."""
        console.print("[bold green]SummaryGeneratorTool[/bold green] — building Markdown …")
        now = summary.get("generated_at", datetime.utcnow().isoformat())
        demo = summary.get("patient_demographics", {})
        recon = summary.get("medication_reconciliation", {})
        conflicts = summary.get("conflicts_detected", [])
        flags = summary.get("review_flags", [])
        pending = summary.get("pending_results", [])
        adm_meds = summary.get("admission_medications", [])
        dis_meds = summary.get("discharge_medications", [])
        procedures = summary.get("procedures", [])
        followup_appts = summary.get("followup_appointments", [])

        def nd(val: Any, default: str = "NOT DOCUMENTED") -> str:
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return default
            return str(val)

        def med_rows(meds: List[Dict]) -> str:
            if not meds:
                return "_None documented_"
            rows = ["| Medication | Dose | Frequency | Route | Indication |",
                    "|-----------|------|-----------|-------|------------|"]
            for m in meds:
                rows.append(
                    f"| {nd(m.get('name'))} "
                    f"| {nd(m.get('dose'))} "
                    f"| {nd(m.get('frequency'))} "
                    f"| {nd(m.get('route'))} "
                    f"| {nd(m.get('indication', 'N/A'))} |"
                )
            return "\n".join(rows)

        md_lines: List[str] = [
            "# DISCHARGE SUMMARY DRAFT",
            "",
            "> **⚠️ FOR CLINICIAN REVIEW ONLY — NOT FOR CLINICAL USE WITHOUT VERIFICATION ⚠️**",
            "",
            f"_Generated: {now}_",
            "",
            "---",
            "",
            "## 1. Patient Demographics",
            "",
            f"- **Name:** {nd(demo.get('patient_name'))}",
            f"- **Patient ID:** {nd(demo.get('patient_id'))}",
            f"- **DOB:** {nd(demo.get('date_of_birth'))}",
            f"- **Age:** {nd(demo.get('age'))}",
            f"- **Gender:** {nd(demo.get('gender'))}",
            f"- **Address:** {nd(demo.get('address'))}",
            f"- **Contact:** {nd(demo.get('contact_number'))}",
            f"- **Next of Kin:** {nd(demo.get('next_of_kin'))}",
            "",
            "---",
            "",
            "## 2. Admission & Discharge Dates",
            "",
            f"- **Admission Date:** {nd(summary.get('admission_date'))}",
            f"- **Discharge Date:** {nd(summary.get('discharge_date'))}",
            f"- **Length of Stay:** {nd(summary.get('length_of_stay'))}",
            "",
            "---",
            "",
            "## 3. Diagnoses",
            "",
            f"### Principal Diagnosis",
            f"{nd(summary.get('principal_diagnosis'))}",
            "",
            "### Secondary Diagnoses",
        ]

        sec_diags = summary.get("secondary_diagnoses", [])
        if sec_diags:
            for d in sec_diags:
                md_lines.append(f"- {d}")
        else:
            md_lines.append("_None documented_")

        md_lines += [
            "",
            "---",
            "",
            "## 4. Hospital Course",
            "",
            nd(summary.get("hospital_course")),
            "",
            "---",
            "",
            "## 5. Discharge Condition",
            "",
            nd(summary.get("discharge_condition")),
            "",
            "---",
            "",
            "## 6. Procedures",
            "",
        ]

        if procedures:
            for proc in procedures:
                name = proc.get("name", "Unknown Procedure")
                date = proc.get("date", "NOT DOCUMENTED")
                md_lines.append(f"- **{name}** (Date: {date})")
        else:
            md_lines.append("_No procedures documented_")

        md_lines += [
            "",
            "---",
            "",
            "## 7. Medications",
            "",
            "### Admission Medications",
            "",
            med_rows(adm_meds),
            "",
            "### Discharge Medications",
            "",
            med_rows(dis_meds),
            "",
            "### Medication Reconciliation",
            "",
        ]

        # Reconciliation changes
        if recon.get("added"):
            md_lines.append("**Added at Discharge:**")
            for m in recon["added"]:
                flag = " 🚩" if m.get("flag_for_review") else ""
                md_lines.append(f"- {m.get('medication_name', 'Unknown')}{flag}")
        if recon.get("removed"):
            md_lines.append("")
            md_lines.append("**Stopped at Discharge:**")
            for m in recon["removed"]:
                flag = " 🚩 _(no documented reason)_" if m.get("flag_for_review") else ""
                md_lines.append(f"- {m.get('medication_name', 'Unknown')}{flag}")
        if recon.get("modified"):
            md_lines.append("")
            md_lines.append("**Modified:**")
            for m in recon["modified"]:
                flag = " 🚩 _(reason not documented)_" if m.get("flag_for_review") else ""
                md_lines.append(f"- {m.get('medication_name', 'Unknown')}{flag}")
        if not recon.get("added") and not recon.get("removed") and not recon.get("modified"):
            md_lines.append("_No changes between admission and discharge medications (or medications not documented)_")

        md_lines += [
            "",
            "---",
            "",
            "## 8. Allergies",
            "",
            nd(summary.get("allergies")),
            "",
            "---",
            "",
            "## 9. Pending Results",
            "",
        ]

        if pending:
            for pr in pending:
                md_lines.append(
                    f"- **{pr.get('result_name', 'Unknown')}** "
                    f"(Type: {pr.get('result_type', 'unknown')}) — "
                    f"⏳ AWAITING RESULT"
                )
        else:
            md_lines.append("_No pending results documented_")

        md_lines += [
            "",
            "---",
            "",
            "## 10. Follow-Up Instructions",
            "",
            nd(summary.get("followup_instructions")),
            "",
        ]

        if followup_appts:
            md_lines.append("**Scheduled Appointments:**")
            for appt in followup_appts:
                md_lines.append(
                    f"- With: {nd(appt.get('with'))}  |  "
                    f"When: {nd(appt.get('when'))}  |  "
                    f"Purpose: {nd(appt.get('purpose'))}"
                )

        md_lines += [
            "",
            "---",
            "",
            "## 11. ⚠️ Conflicts Detected",
            "",
        ]

        if conflicts:
            md_lines.append(
                "> **These conflicts MUST be resolved by the responsible clinician "
                "before this document is finalized.**"
            )
            md_lines.append("")
            for c in conflicts:
                md_lines += [
                    f"### Conflict — {c.get('field', 'Unknown')}",
                    f"- **Source A ({c.get('source_a', '?')}):** {c.get('value_a', '?')}",
                    f"- **Source B ({c.get('source_b', '?')}):** {c.get('value_b', '?')}",
                    f"- **Description:** {c.get('description', '')}",
                    f"- **Severity:** {c.get('severity', 'high').upper()}",
                    "",
                ]
        else:
            md_lines.append("_No conflicts detected_")

        md_lines += [
            "",
            "---",
            "",
            "## 12. 🚩 Review Flags",
            "",
        ]

        # Sort by severity
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        sorted_flags = sorted(
            flags, key=lambda f: severity_order.get(f.get("severity", "INFO"), 2)
        )

        if sorted_flags:
            for flag in sorted_flags:
                icon = "🔴" if flag.get("severity") == "CRITICAL" else "🟡" if flag.get("severity") == "WARNING" else "🔵"
                md_lines.append(
                    f"{icon} **{flag.get('severity', 'INFO')}** — "
                    f"**{flag.get('field', 'unknown')}**: "
                    f"{flag.get('reason', '')}"
                )
        else:
            md_lines.append("_No review flags_")

        md_lines += [
            "",
            "---",
            "",
            "_This discharge summary was generated by an AI agent and must be reviewed, "
            "corrected, and approved by the responsible clinician before clinical use._",
        ]

        return "\n".join(md_lines)

    # ─────────────────────────────────────────────
    # Save outputs
    # ─────────────────────────────────────────────

    def save_outputs(
        self,
        summary: Dict[str, Any],
        markdown: str,
        traces: List[Dict[str, Any]],
        output_dir: str,
        patient_id: str,
    ) -> Dict[str, str]:
        """
        Save discharge_summary.json, discharge_summary.md, and traces/trace.txt.

        Returns dict mapping label → file path.
        """
        pid_dir = os.path.join(output_dir, patient_id)
        traces_dir = os.path.join(pid_dir, "traces")
        os.makedirs(traces_dir, exist_ok=True)

        files: Dict[str, str] = {}

        # JSON
        json_path = os.path.join(pid_dir, "discharge_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        files["json"] = json_path

        # Markdown
        md_path = os.path.join(pid_dir, "discharge_summary.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        files["markdown"] = md_path

        # Trace
        trace_path = os.path.join(traces_dir, "trace.txt")
        with open(trace_path, "w", encoding="utf-8") as f:
            for t in traces:
                f.write(_format_trace(t))
                f.write("\n" + "─" * 60 + "\n\n")
        files["trace"] = trace_path

        console.print(
            f"  [green]✓[/green] Outputs saved to [dim]{pid_dir}[/dim]"
        )
        return files


def _format_trace(trace: Dict[str, Any]) -> str:
    """Format a single trace entry as readable text."""
    lines = [
        f"STEP {trace.get('step_number', '?')}",
        f"Timestamp : {trace.get('timestamp', '')}",
        f"Tool      : {trace.get('tool', '')}",
        f"",
        f"Reasoning :",
        f"  {trace.get('reasoning', '')}",
        f"",
        f"Input     :",
        f"  {trace.get('input_summary', '')}",
        f"",
        f"Output    :",
        f"  {trace.get('output_summary', '')}",
        f"",
        f"Decision  :",
        f"  {trace.get('decision', '')}",
        f"",
        f"Success   : {trace.get('success', True)}",
    ]
    if trace.get("error"):
        lines.append(f"Error     : {trace['error']}")
    return "\n".join(lines)
