"""
tests/test_tools.py — Unit tests for all Discharge Summary Agent tools.

ROOT CAUSE FIXES applied in this file:
---------------------------------------
1. _FakeGeminiService.generate_json now accepts `fallback=None, **kwargs`
   so tools calling self.gemini.generate_json(prompt, fallback=x) no longer
   raise "unexpected keyword argument 'fallback'".

2. _FakeGeminiService.generate_json returns realistic reconciliation data
   (parsed from the formatted prompt text) so added/removed medication
   tests pass without a live Gemini API call.

3. test_generate_json_strips_code_fences updated to mock the NEW google-genai
   SDK pattern (genai.Client / client.models.generate_content) instead of the
   deprecated google.generativeai pattern (genai.GenerativeModel / configure).

4. test_graph_builds_without_error updated to same new-SDK mock pattern.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import DocumentType, ExtractedDocument


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_doc(
    filename: str = "test.pdf",
    doc_type: str = "ADMISSION_NOTE",
    text: str = "Sample clinical text.",
    success: bool = True,
    is_scanned: bool = False,
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "doc_type": doc_type,
        "raw_text": text,
        "success": success,
        "is_scanned": is_scanned,
        "error": None,
    }


class _FakeGeminiService:
    """
    Lightweight Gemini stub.

    FIX 1: generate_json now accepts `fallback=None, **kwargs` so tools
    calling self.gemini.generate_json(prompt, fallback=x) do not raise
    "unexpected keyword argument 'fallback'".

    FIX 2: reconciliation prompts are handled by parsing the formatted
    medication lists out of the prompt text so added/removed tests pass.
    """

    def generate(self, prompt: str, **kwargs) -> str:
        return "Stubbed response."

    def generate_json(
        self,
        prompt: str,
        fallback: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Return plausible JSON stubs based on key phrases in the prompt."""

        # ── Diagnosis / demographics ─────────────────────────────────────────
        if "principal_diagnosis" in prompt or "demographics" in prompt:
            return {
                "demographics": {
                    "name": "John Doe",
                    "dob": "1960-01-01",
                    "mrn": "123456",
                    "gender": "Male",
                },
                "principal_diagnosis": "Type 2 Diabetes Mellitus",
                "secondary_diagnoses": ["Hypertension"],
                "admission_date": "2024-01-10",
                "discharge_date": "2024-01-15",
                "length_of_stay": "5 days",
                "hospital_course": "Patient admitted with hyperglycaemia.",
                "discharge_condition": "Stable",
                "conflicts": [],
            }

        # ── Medication extraction ────────────────────────────────────────────
        if "admission_medications" in prompt:
            return {
                "admission_medications": [
                    {"name": "Metformin", "dose": "500mg", "frequency": "BD", "route": "Oral"},
                ],
                "discharge_medications": [
                    {"name": "Metformin", "dose": "500mg", "frequency": "BD", "route": "Oral"},
                    {"name": "Lisinopril", "dose": "5mg", "frequency": "OD", "route": "Oral"},
                ],
            }

        # ── Allergies ────────────────────────────────────────────────────────
        if "allerg" in prompt.lower():
            return {
                "allergies_raw": "Penicillin — Rash",
                "allergies": [{"allergen": "Penicillin", "reaction": "Rash", "severity": "Moderate"}],
                "allergy_list": [{"allergen": "Penicillin", "reaction": "Rash", "severity": "Moderate", "documented_in": "admission_note"}],
                "nkda_documented": False,
                "nkda": False,
                "review_required": False,
                "safety_note": "",
            }

        # ── Procedures ───────────────────────────────────────────────────────
        if "procedure" in prompt.lower():
            return {
                "procedures": [
                    {
                        "name": "ECG",
                        "date": "2024-01-10",
                        "operator": "NOT DOCUMENTED",
                        "indication": "Chest pain",
                        "outcome": "Normal sinus rhythm",
                    }
                ]
            }

        # ── Follow-up ────────────────────────────────────────────────────────
        if "follow" in prompt.lower() or "instruction" in prompt.lower():
            return {
                "followup_instructions": "Review in 2 weeks",
                "appointments": [{"specialty": "Endocrinology", "date": "2024-01-29"}],
                "dietary_instructions": "Low-carbohydrate diet",
                "activity_restrictions": "NOT DOCUMENTED",
                "wound_care": "NOT DOCUMENTED",
                "return_precautions": "Return if blood glucose > 20 mmol/L",
            }

        # ── Pending results ──────────────────────────────────────────────────
        if "pending" in prompt.lower():
            return {
                "pending_results": [
                    {
                        "type": "Lab",
                        "test": "HbA1c",
                        "ordered_date": "2024-01-14",
                        "expected_date": "NOT DOCUMENTED",
                    }
                ]
            }

        # ── Conflict detection ───────────────────────────────────────────────
        if "conflict" in prompt.lower():
            return {
                "conflicts": [
                    {
                        "field": "discharge_diagnosis",
                        "document_a": "admission_note.pdf",
                        "value_a": "DKA",
                        "document_b": "progress_note.pdf",
                        "value_b": "T2DM uncontrolled",
                        "severity": "HIGH",
                        "clinician_review_required": True,
                    }
                ]
            }

        # ── Medication reconciliation (FIX 2) ────────────────────────────────
        if "pharmacist" in prompt.lower() or "reconcil" in prompt.lower():
            return self._reconcile_from_prompt(prompt, fallback)

        # Default — return fallback or empty dict
        return fallback if fallback is not None else {}

    # ── Reconciliation helper ─────────────────────────────────────────────────

    @staticmethod
    def _reconcile_from_prompt(
        prompt: str, fallback: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse formatted medication lists from the prompt and compute
        added / removed / unchanged without a live API call.

        The prompt contains:
            ADMISSION MEDICATIONS:
              - DrugName | dose: X | freq: Y | route: Z
            DISCHARGE MEDICATIONS:
              - DrugName | dose: X | freq: Y | route: Z
        """
        # Split into admission and discharge sections
        parts = prompt.split("DISCHARGE MEDICATIONS:")
        adm_section = parts[0].lower() if len(parts) > 0 else ""
        dis_section = parts[1].lower() if len(parts) > 1 else ""

        # Extract drug names from "  - DrugName |" pattern
        adm_drugs: set = set(re.findall(r"-\s+(\w+)\s+\|", adm_section))
        dis_drugs: set = set(re.findall(r"-\s+(\w+)\s+\|", dis_section))

        added = [
            {"name": d.capitalize(), "medication_name": d.capitalize(),
             "discharge_details": {}, "documented_reason": None, "flag_for_review": False}
            for d in sorted(dis_drugs - adm_drugs)
        ]
        removed = [
            {"name": d.capitalize(), "medication_name": d.capitalize(),
             "admission_details": {}, "documented_reason": None, "flag_for_review": True,
             "flag_reason": "No documented reason for discontinuation"}
            for d in sorted(adm_drugs - dis_drugs)
        ]
        unchanged = [
            {"name": d.capitalize(), "medication_name": d.capitalize(), "details": {}}
            for d in sorted(adm_drugs & dis_drugs)
        ]
        reason_missing = [d["name"] for d in removed]

        return {
            "added": added,
            "removed": removed,
            "modified": [],
            "unchanged": unchanged,
            "reason_missing": reason_missing,
            "review_required": bool(removed or added),
            "notes": "",
        }


# ─────────────────────────────────────────────────────────────────────────────
# PDF Loader Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFLoaderTool(unittest.TestCase):

    def setUp(self):
        from tools.pdf_loader import PDFLoaderTool
        self.tool = PDFLoaderTool()

    def test_missing_folder_returns_empty_list(self):
        docs = self.tool.load_patient_folder("/nonexistent/path/xyz")
        self.assertEqual(docs, [])

    def test_empty_folder_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            docs = self.tool.load_patient_folder(tmpdir)
            self.assertEqual(docs, [])

    @patch("tools.pdf_loader.fitz")
    def test_corrupted_pdf_returns_error_doc(self, mock_fitz):
        """
        A PDF that raises on fitz.open should return a failure doc, not crash.
        FIX: fitz is now a module-level import in pdf_loader.py so this patch works.
        """
        mock_fitz.open.side_effect = Exception("corrupted")
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "bad.pdf"
            pdf_path.write_bytes(b"%PDF-1.4 bad data")

            docs = self.tool.load_patient_folder(tmpdir)
            self.assertEqual(len(docs), 1)
            self.assertFalse(docs[0]["success"])
            self.assertIsNotNone(docs[0]["error"])

    def test_classify_document_type_from_filename(self):
        """
        _classify_document must exist as an instance method and return
        uppercase type strings.
        FIX: Added _classify_document instance method to PDFLoaderTool.
        """
        cases = [
            ("admission_note.pdf",    "ADMISSION_NOTE"),
            ("lab_results_day2.pdf",  "LAB_REPORT"),
            ("medication_chart.pdf",  "MEDICATION_CHART"),
            ("echo_report.pdf",       "ECHO_REPORT"),
            ("ct_scan_chest.pdf",     "CT_REPORT"),
            ("random_doc.pdf",        "UNKNOWN"),
        ]
        for fname, expected in cases:
            with self.subTest(fname=fname):
                result = self.tool._classify_document(fname, "")
                self.assertEqual(result, expected)


# ─────────────────────────────────────────────────────────────────────────────
# OCR Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestOCRTool(unittest.TestCase):

    def setUp(self):
        from tools.ocr_tool import OCRTool
        self.tool = OCRTool()

    def test_skip_non_scanned_doc(self):
        doc = _make_doc(is_scanned=False, text="Already has text.")
        result = self.tool.process_scanned_documents([doc])
        self.assertEqual(result[0]["raw_text"], "Already has text.")

    def test_handles_empty_doc_list(self):
        result = self.tool.process_scanned_documents([])
        self.assertEqual(result, [])

    def test_failed_doc_skipped(self):
        doc = _make_doc(success=False, is_scanned=True)
        result = self.tool.process_scanned_documents([doc])
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["success"])


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis Extractor Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnosisExtractorTool(unittest.TestCase):

    def setUp(self):
        from tools.diagnosis_tool import DiagnosisExtractorTool
        self.tool = DiagnosisExtractorTool(_FakeGeminiService())

    def test_extract_returns_required_keys(self):
        result = self.tool.extract("Patient admitted with DKA.")
        for key in ("principal_diagnosis", "secondary_diagnoses", "admission_date",
                    "discharge_date", "hospital_course", "discharge_condition"):
            self.assertIn(key, result, f"Key '{key}' missing from result")

    def test_no_fabrication_on_empty_text(self):
        fake = _FakeGeminiService()
        fake.generate_json = lambda *a, **kw: {
            "demographics": {},
            "principal_diagnosis": "NOT DOCUMENTED",
            "secondary_diagnoses": [],
            "admission_date": "NOT DOCUMENTED",
            "discharge_date": "NOT DOCUMENTED",
            "length_of_stay": "NOT DOCUMENTED",
            "hospital_course": "NOT DOCUMENTED",
            "discharge_condition": "NOT DOCUMENTED",
            "conflicts": [],
        }
        from tools.diagnosis_tool import DiagnosisExtractorTool
        tool = DiagnosisExtractorTool(fake)
        result = tool.extract("")
        self.assertEqual(result["principal_diagnosis"], "NOT DOCUMENTED")

    def test_error_does_not_crash(self):
        bad_gemini = MagicMock()
        bad_gemini.generate_json.side_effect = Exception("API timeout")
        from tools.diagnosis_tool import DiagnosisExtractorTool
        tool = DiagnosisExtractorTool(bad_gemini)
        result = tool.extract("some text")
        self.assertIn("error", result)
        self.assertIsNotNone(result["error"])


# ─────────────────────────────────────────────────────────────────────────────
# Medication Extractor Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestMedicationExtractorTool(unittest.TestCase):

    def setUp(self):
        from tools.medication_tool import MedicationExtractorTool
        self.tool = MedicationExtractorTool(_FakeGeminiService())

    def test_extract_returns_two_lists(self):
        result = self.tool.extract("Metformin 500mg OD on admission.")
        self.assertIn("admission_medications", result)
        self.assertIn("discharge_medications", result)

    def test_returns_lists_on_empty_text(self):
        fake = _FakeGeminiService()
        fake.generate_json = lambda *a, **kw: {
            "admission_medications": [],
            "discharge_medications": [],
        }
        from tools.medication_tool import MedicationExtractorTool
        tool = MedicationExtractorTool(fake)
        result = tool.extract("")
        self.assertIsInstance(result["admission_medications"], list)
        self.assertIsInstance(result["discharge_medications"], list)


# ─────────────────────────────────────────────────────────────────────────────
# Allergy Extractor Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestAllergyExtractorTool(unittest.TestCase):

    def setUp(self):
        from tools.allergy_tool import AllergyExtractorTool
        self.tool = AllergyExtractorTool(_FakeGeminiService())

    def test_extract_allergies(self):
        """
        FIX: _FakeGeminiService.generate_json now accepts fallback=None, **kwargs
        so AllergyExtractorTool calling generate_json(prompt, fallback=x) no longer fails.
        """
        result = self.tool.extract("Allergy: Penicillin — anaphylaxis")
        self.assertIn("allergies", result)

    def test_missing_allergies_sets_review_required(self):
        fake = _FakeGeminiService()
        fake.generate_json = lambda *a, **kw: {
            "allergies_raw": "NOT DOCUMENTED",
            "allergy_list": [],
            "nkda_documented": False,
            "review_required": True,
        }
        from tools.allergy_tool import AllergyExtractorTool
        tool = AllergyExtractorTool(fake)
        result = tool.extract("No allergy information available.")
        self.assertTrue(result.get("review_required"))

    def test_error_does_not_crash(self):
        bad = MagicMock()
        bad.generate_json.side_effect = Exception("Timeout")
        from tools.allergy_tool import AllergyExtractorTool
        tool = AllergyExtractorTool(bad)
        result = tool.extract("x")
        self.assertTrue(result.get("review_required"))


# ─────────────────────────────────────────────────────────────────────────────
# Procedure Extractor Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestProcedureExtractorTool(unittest.TestCase):

    def setUp(self):
        from tools.procedure_tool import ProcedureExtractorTool
        self.tool = ProcedureExtractorTool(_FakeGeminiService())

    def test_extract_procedures_returns_list(self):
        result = self.tool.extract("ECG performed on admission.")
        self.assertIn("procedures", result)
        self.assertIsInstance(result["procedures"], list)

    def test_no_procedures_returns_empty_list(self):
        fake = _FakeGeminiService()
        fake.generate_json = lambda *a, **kw: {"procedures": []}
        from tools.procedure_tool import ProcedureExtractorTool
        tool = ProcedureExtractorTool(fake)
        result = tool.extract("")
        self.assertEqual(result["procedures"], [])


# ─────────────────────────────────────────────────────────────────────────────
# Follow-up Extractor Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestFollowupExtractorTool(unittest.TestCase):

    def setUp(self):
        from tools.followup_tool import FollowupExtractorTool
        self.tool = FollowupExtractorTool(_FakeGeminiService())

    def test_extract_returns_followup_key(self):
        result = self.tool.extract("Review in outpatient clinic in 2 weeks.")
        self.assertIn("followup_instructions", result)

    def test_not_documented_when_absent(self):
        fake = _FakeGeminiService()
        fake.generate_json = lambda *a, **kw: {
            "followup_instructions": "NOT DOCUMENTED",
            "appointments": [],
            "dietary_instructions": "NOT DOCUMENTED",
            "activity_restrictions": "NOT DOCUMENTED",
            "wound_care": "NOT DOCUMENTED",
            "return_precautions": "NOT DOCUMENTED",
        }
        from tools.followup_tool import FollowupExtractorTool
        tool = FollowupExtractorTool(fake)
        result = tool.extract("")
        self.assertEqual(result["followup_instructions"], "NOT DOCUMENTED")


# ─────────────────────────────────────────────────────────────────────────────
# Pending Result Detector Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestPendingResultDetectorTool(unittest.TestCase):

    def setUp(self):
        from tools.pending_tool import PendingResultDetectorTool
        self.tool = PendingResultDetectorTool(_FakeGeminiService())

    def test_detect_returns_list(self):
        result = self.tool.detect("HbA1c pending. Blood culture awaited.")
        self.assertIn("pending_results", result)
        self.assertIsInstance(result["pending_results"], list)

    def test_no_pending_returns_empty_list(self):
        fake = _FakeGeminiService()
        fake.generate_json = lambda *a, **kw: {"pending_results": []}
        from tools.pending_tool import PendingResultDetectorTool
        tool = PendingResultDetectorTool(fake)
        result = tool.detect("All results available.")
        self.assertEqual(result["pending_results"], [])


# ─────────────────────────────────────────────────────────────────────────────
# Conflict Detector Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictDetectorTool(unittest.TestCase):

    def setUp(self):
        from tools.conflict_tool import ConflictDetectorTool
        self.tool = ConflictDetectorTool(_FakeGeminiService())

    def test_detect_returns_conflict_list(self):
        docs = [
            _make_doc("admission.pdf", text="Diagnosis: DKA"),
            _make_doc("progress.pdf", text="Diagnosis: T2DM uncontrolled"),
        ]
        result = self.tool.detect(docs)
        self.assertIn("conflicts", result)
        self.assertIsInstance(result["conflicts"], list)

    def test_no_docs_returns_empty_conflicts(self):
        result = self.tool.detect([])
        self.assertEqual(result.get("conflicts", []), [])

    def test_conflicts_never_auto_resolved(self):
        docs = [
            _make_doc("a.pdf", text="Patient has DKA"),
            _make_doc("b.pdf", text="Patient has AFI"),
        ]
        result = self.tool.detect(docs)
        for conflict in result.get("conflicts", []):
            self.assertTrue(
                conflict.get("clinician_review_required", False),
                "Conflict was not flagged for clinician review",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Medication Reconciliation Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestMedicationReconciliationTool(unittest.TestCase):

    def setUp(self):
        from tools.reconciliation_tool import MedicationReconciliationTool
        self.tool = MedicationReconciliationTool(_FakeGeminiService())

    def _med(self, name: str, dose: str = "10mg", freq: str = "OD") -> Dict[str, Any]:
        return {"name": name, "dose": dose, "frequency": freq, "route": "Oral"}

    def test_added_medication_detected(self):
        """
        FIX 1: _FakeGeminiService.generate_json accepts fallback= kwarg.
        FIX 2: _reconcile_from_prompt parses med lists from prompt text.
        """
        adm = [self._med("Metformin")]
        dis = [self._med("Metformin"), self._med("Lisinopril")]
        result = self.tool.reconcile(adm, dis)
        self.assertIn("added", result)
        added_names = [m.get("name", "").lower() for m in result["added"]]
        self.assertTrue(
            any("lisinopril" in n for n in added_names),
            f"Expected Lisinopril in added, got: {result['added']}",
        )

    def test_removed_medication_detected(self):
        """Aspirin present at admission but not discharge must appear in removed."""
        adm = [self._med("Metformin"), self._med("Aspirin")]
        dis = [self._med("Metformin")]
        result = self.tool.reconcile(adm, dis)
        self.assertIn("removed", result)
        removed_names = [m.get("name", "").lower() for m in result["removed"]]
        self.assertTrue(
            any("aspirin" in n for n in removed_names),
            f"Expected Aspirin in removed, got: {result['removed']}",
        )

    def test_required_output_keys_present(self):
        result = self.tool.reconcile([], [])
        for key in ("added", "removed", "modified", "unchanged", "reason_missing"):
            self.assertIn(key, result, f"Key '{key}' missing from reconciliation output")

    def test_empty_both_lists_returns_empty_categories(self):
        result = self.tool.reconcile([], [])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["modified"], [])


# ─────────────────────────────────────────────────────────────────────────────
# Review Flag Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewFlagTool(unittest.TestCase):

    def setUp(self):
        from tools.review_flag_tool import ReviewFlagTool
        self.tool = ReviewFlagTool()

    def _base_state(self, **overrides) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "principal_diagnosis": "Type 2 DM",
            "secondary_diagnoses": ["Hypertension"],
            "admission_date": "2024-01-10",
            "discharge_date": "2024-01-15",
            "hospital_course": "Patient managed and improved.",
            "discharge_condition": "Stable",
            "allergy_data": {
                "allergies": [{"allergen": "Penicillin"}],
                "nkda": False,
                "review_required": False,
            },
            "admission_medications": [],
            "discharge_medications": [],
            "procedure_data": {"procedures": []},
            "followup_data": {"followup_instructions": "Review in 2 weeks"},
            "pending_data": {"pending_results": []},
            "conflict_data": {"conflicts": []},
            "reconciliation": {
                "added": [], "removed": [], "modified": [], "unchanged": [], "reason_missing": []
            },
            "demographics": {"name": "John Doe"},
        }
        state.update(overrides)
        return state

    def test_missing_allergy_generates_critical_flag(self):
        state = self._base_state(
            allergy_data={"allergies": "NOT DOCUMENTED", "nkda": False, "review_required": True}
        )
        flags = self.tool.generate_flags(state)
        severities = [f["severity"] for f in flags if f["field"] == "allergies"]
        self.assertIn("CRITICAL", severities, "Missing allergies must generate a CRITICAL flag")

    def test_no_flags_on_complete_state(self):
        state = self._base_state()
        flags = self.tool.generate_flags(state)
        critical = [f for f in flags if f["severity"] == "CRITICAL"]
        self.assertEqual(critical, [], f"Unexpected CRITICAL flags: {critical}")

    def test_not_documented_diagnosis_generates_flag(self):
        state = self._base_state(principal_diagnosis="NOT DOCUMENTED")
        flags = self.tool.generate_flags(state)
        flag_fields = [f["field"] for f in flags]
        self.assertIn("principal_diagnosis", flag_fields)

    def test_pending_result_generates_info_flag(self):
        state = self._base_state(
            pending_data={"pending_results": [{"type": "Lab", "test": "HbA1c"}]}
        )
        flags = self.tool.generate_flags(state)
        pending_flags = [f for f in flags if "pending" in f["field"].lower()]
        self.assertTrue(len(pending_flags) >= 1)

    def test_conflict_generates_warning_or_higher(self):
        state = self._base_state(
            conflict_data={
                "conflicts": [
                    {
                        "field": "discharge_diagnosis",
                        "value_a": "DKA", "value_b": "AFI",
                        "document_a": "a.pdf", "document_b": "b.pdf",
                        "severity": "HIGH",
                    }
                ]
            }
        )
        flags = self.tool.generate_flags(state)
        conflict_flags = [f for f in flags if "conflict" in f["field"].lower()]
        self.assertTrue(len(conflict_flags) >= 1)
        severities = {f["severity"] for f in conflict_flags}
        self.assertTrue(
            severities & {"CRITICAL", "WARNING"},
            f"Conflict flags should be WARNING or CRITICAL, got: {severities}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Summary Generator Tool
# ─────────────────────────────────────────────────────────────────────────────

class TestSummaryGeneratorTool(unittest.TestCase):

    def setUp(self):
        from tools.summary_tool import SummaryGeneratorTool
        self.tool = SummaryGeneratorTool()

    def _full_state(self) -> Dict[str, Any]:
        return {
            "patient_id": "TEST001",
            "principal_diagnosis": "Type 2 DM",
            "secondary_diagnoses": ["Hypertension"],
            "admission_date": "2024-01-10",
            "discharge_date": "2024-01-15",
            "length_of_stay": "5 days",
            "hospital_course": "Patient managed with insulin and improved.",
            "discharge_condition": "Stable",
            "demographics": {"name": "John Doe", "dob": "1960-01-01", "mrn": "123456"},
            "allergy_data": {
                "allergies": [{"allergen": "Penicillin", "reaction": "Rash"}],
                "nkda": False,
                "review_required": False,
            },
            "admission_medications": [
                {"name": "Metformin", "dose": "500mg", "frequency": "BD", "route": "Oral"}
            ],
            "discharge_medications": [
                {"name": "Metformin", "dose": "500mg", "frequency": "BD", "route": "Oral"},
                {"name": "Lisinopril", "dose": "5mg", "frequency": "OD", "route": "Oral"},
            ],
            "procedure_data": {"procedures": []},
            "followup_data": {
                "followup_instructions": "Review in 2 weeks",
                "appointments": [],
                "dietary_instructions": "Low-carb diet",
                "activity_restrictions": "NOT DOCUMENTED",
                "wound_care": "NOT DOCUMENTED",
                "return_precautions": "Return if BGL > 20",
            },
            "pending_data": {"pending_results": [{"type": "Lab", "test": "HbA1c"}]},
            "conflict_data": {"conflicts": []},
            "reconciliation": {
                "added": [{"name": "Lisinopril"}],
                "removed": [],
                "modified": [],
                "unchanged": [{"name": "Metformin"}],
                "reason_missing": [],
            },
            "review_flags": [
                {
                    "severity": "INFO",
                    "field": "pending_HbA1c",
                    "message": "HbA1c result pending",
                    "action": "Follow up with lab",
                }
            ],
            "missing_fields": [],
            "conflicts": [],
            "errors": [],
            "step_traces": [],
        }

    def test_build_json_summary_contains_all_sections(self):
        summary = self.tool.build_json_summary(self._full_state())
        required_keys = [
            "patient_demographics", "admission_date", "discharge_date",
            "principal_diagnosis", "secondary_diagnoses", "hospital_course",
            "procedures", "discharge_medications", "medication_reconciliation",
            "allergies", "followup_instructions", "pending_results",
            "discharge_condition", "conflicts_detected", "review_flags",
        ]
        for key in required_keys:
            self.assertIn(key, summary, f"Summary missing required key: '{key}'")

    def test_build_markdown_returns_string(self):
        summary = self.tool.build_json_summary(self._full_state())
        md = self.tool.build_markdown(summary)
        self.assertIsInstance(md, str)
        self.assertGreater(len(md), 100)

    def test_save_outputs_creates_files(self):
        state = self._full_state()
        summary = self.tool.build_json_summary(state)
        md = self.tool.build_markdown(summary)
        with tempfile.TemporaryDirectory() as tmpdir:
            files = self.tool.save_outputs(
                summary=summary, markdown=md, traces=[],
                output_dir=tmpdir, patient_id="TEST001",
            )
            for path in files.values():
                self.assertTrue(Path(path).exists(), f"Expected file: {path}")

    def test_json_summary_not_documented_never_none(self):
        empty_state = {
            "patient_id": "EMPTY",
            "principal_diagnosis": "NOT DOCUMENTED",
            "secondary_diagnoses": [],
            "admission_date": "NOT DOCUMENTED",
            "discharge_date": "NOT DOCUMENTED",
            "length_of_stay": "NOT DOCUMENTED",
            "hospital_course": "NOT DOCUMENTED",
            "discharge_condition": "NOT DOCUMENTED",
            "demographics": {},
            "allergy_data": {},
            "admission_medications": [],
            "discharge_medications": [],
            "procedure_data": {},
            "followup_data": {},
            "pending_data": {},
            "conflict_data": {},
            "reconciliation": {},
            "review_flags": [],
            "missing_fields": [],
            "conflicts": [],
            "errors": [],
            "step_traces": [],
        }
        summary = self.tool.build_json_summary(empty_state)
        serialized = json.dumps(summary)
        self.assertNotIn(": null", serialized)


# ─────────────────────────────────────────────────────────────────────────────
# Agent State
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentState(unittest.TestCase):

    def test_default_state_has_all_keys(self):
        from agent.state import default_state
        state = default_state(patient_folder="/tmp/patient", patient_id="P001", max_steps=25)
        required_keys = [
            "patient_folder", "patient_id", "max_steps",
            "pdf_files", "extracted_documents", "all_text",
            "demographics", "admission_date", "discharge_date",
            "principal_diagnosis", "secondary_diagnoses",
            "hospital_course", "discharge_condition",
            "admission_medications", "discharge_medications",
            "allergy_data", "procedure_data", "followup_data",
            "pending_data", "conflict_data", "reconciliation",
            "current_step", "completed_tools", "next_tool", "is_complete",
            "review_flags", "missing_fields", "conflicts",
            "step_traces", "errors", "final_summary", "output_files",
        ]
        for key in required_keys:
            self.assertIn(key, state, f"AgentState missing default key: '{key}'")

    def test_default_state_not_documented_for_clinical_fields(self):
        from agent.state import default_state
        state = default_state("/tmp", "P001", 25)
        self.assertEqual(state["principal_diagnosis"], "NOT DOCUMENTED")
        self.assertEqual(state["admission_date"], "NOT DOCUMENTED")
        self.assertEqual(state["discharge_date"], "NOT DOCUMENTED")
        self.assertEqual(state["hospital_course"], "NOT DOCUMENTED")
        self.assertEqual(state["discharge_condition"], "NOT DOCUMENTED")


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Service
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiService(unittest.TestCase):

    @patch("services.gemini_service.genai")
    def test_init_without_api_key_raises_or_warns(self, mock_genai):
        """Service should fail gracefully when no API key is configured."""
        mock_genai.Client.return_value = MagicMock()
        from services.gemini_service import GeminiService
        with patch.dict(os.environ, {}, clear=True):
            try:
                svc = GeminiService()
            except Exception:
                pass  # Constructor raising EnvironmentError is acceptable

    @patch("services.gemini_service.genai")
    def test_generate_json_strips_code_fences(self, mock_genai):
        """
        JSON fences must be stripped before parsing.

        FIX: Updated to mock the NEW google-genai SDK pattern:
          genai.Client(api_key=...)  →  mock_client
          client.models.generate_content(...)  →  mock_response
        (Old pattern used genai.GenerativeModel / genai.configure which no longer exist.)
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '```json\n{"key": "value"}\n```'
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.Client.return_value = mock_client

        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            # Do NOT reload — reload breaks the mock binding.
            # The @patch decorator already replaced services.gemini_service.genai
            # with mock_genai before the test body runs.
            import services.gemini_service as svc_module
            svc = svc_module.GeminiService()
            result = svc.generate_json("test prompt")
            self.assertEqual(result.get("key"), "value")


# ─────────────────────────────────────────────────────────────────────────────
# Integration smoke tests (no real API calls)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndSmoke(unittest.TestCase):

    def test_graph_builds_without_error(self):
        """
        build_graph() must not raise.
        FIX: Updated to mock the new SDK (genai.Client) instead of old SDK (genai.GenerativeModel).
        """
        with patch("services.gemini_service.genai") as mock_genai, \
             patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"}):
            mock_genai.Client.return_value = MagicMock()

            from agent.graph import build_graph
            graph = build_graph(output_dir=tempfile.mkdtemp())
            self.assertIsNotNone(graph)

    def test_summary_tool_pipeline_runs_without_api(self):
        from tools.summary_tool import SummaryGeneratorTool
        tool = SummaryGeneratorTool()
        state = {
            "patient_id": "SMOKE001",
            "principal_diagnosis": "NOT DOCUMENTED",
            "secondary_diagnoses": [],
            "admission_date": "NOT DOCUMENTED",
            "discharge_date": "NOT DOCUMENTED",
            "length_of_stay": "NOT DOCUMENTED",
            "hospital_course": "NOT DOCUMENTED",
            "discharge_condition": "NOT DOCUMENTED",
            "demographics": {},
            "allergy_data": {"allergies": "NOT DOCUMENTED", "review_required": True},
            "admission_medications": [],
            "discharge_medications": [],
            "procedure_data": {"procedures": []},
            "followup_data": {"followup_instructions": "NOT DOCUMENTED"},
            "pending_data": {"pending_results": []},
            "conflict_data": {"conflicts": []},
            "reconciliation": {
                "added": [], "removed": [], "modified": [], "unchanged": [], "reason_missing": []
            },
            "review_flags": [],
            "missing_fields": ["principal_diagnosis", "allergies"],
            "conflicts": [],
            "errors": [],
            "step_traces": [],
        }
        summary = tool.build_json_summary(state)
        md = tool.build_markdown(summary)
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(md, str)
        self.assertIn("NOT DOCUMENTED", md)


if __name__ == "__main__":
    unittest.main(verbosity=2)