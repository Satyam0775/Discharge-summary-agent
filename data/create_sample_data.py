"""
data/create_sample_data.py — Generate synthetic patient document PDFs for testing.

Creates two patient folders:
  data/patient_001/   — complete, straightforward case (T2DM)
  data/patient_002/   — challenging case with conflicts, missing data, pending results

Usage:
    python data/create_sample_data.py
    python data/create_sample_data.py --output-dir /custom/path

No real patient data is used. All content is entirely synthetic.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Graceful import of reportlab ─────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Minimal plain-text fallback
# ─────────────────────────────────────────────────────────────────────────────

def _write_text_pdf_fallback(path: Path, title: str, body: str) -> None:
    """Write a minimal UTF-8 text file when reportlab is unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".txt").write_text(f"{title}\n{'=' * len(title)}\n\n{body}", encoding="utf-8")
    print(f"  [text fallback] {path.with_suffix('.txt')}")


# ─────────────────────────────────────────────────────────────────────────────
# ReportLab helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pdf(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    """
    Create a simple PDF with a title and a list of (heading, body) sections.

    Parameters
    ----------
    path      : full output path (.pdf)
    title     : document title shown at the top
    sections  : list of (section_heading, section_body_text) tuples
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not REPORTLAB_AVAILABLE:
        body = "\n\n".join(f"== {h} ==\n{b}" for h, b in sections)
        _write_text_pdf_fallback(path, title, body)
        return

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=14,
        spaceAfter=8,
    )
    heading_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
    )

    story = [Paragraph(title, title_style), Spacer(1, 6 * mm)]
    for heading, body in sections:
        story.append(Paragraph(heading, heading_style))
        # Preserve newlines as HTML breaks
        formatted = body.replace("\n", "<br/>")
        story.append(Paragraph(formatted, body_style))
        story.append(Spacer(1, 3 * mm))

    doc.build(story)
    print(f"  [pdf] {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Patient 001 — Clean T2DM case
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_001_DOCS = {
    "admission_note.pdf": (
        "ADMISSION NOTE — Patient 001",
        [
            ("Patient Demographics", (
                "Name: Margaret Chen\n"
                "Date of Birth: 15/03/1958\n"
                "MRN: 0012345\n"
                "Gender: Female\n"
                "Address: 42 Elm Street, Springfield\n"
                "Next of Kin: David Chen (husband) — 0400 111 222"
            )),
            ("Admission Details", (
                "Date of Admission: 10/01/2024\n"
                "Admitting Physician: Dr. Sarah Patel\n"
                "Ward: Medical Ward 3B\n"
                "Admission Source: Emergency Department"
            )),
            ("Presenting Complaint", (
                "67-year-old female with known Type 2 Diabetes Mellitus presenting with a 3-day "
                "history of polyuria, polydipsia, and fatigue. Random blood glucose on arrival "
                "was 22.4 mmol/L."
            )),
            ("Past Medical History", (
                "1. Type 2 Diabetes Mellitus — diagnosed 2009\n"
                "2. Hypertension — on treatment\n"
                "3. Hyperlipidaemia — on statin therapy\n"
                "4. No previous hospitalisations in the past 5 years"
            )),
            ("Medications on Admission", (
                "1. Metformin 500mg BD (Oral)\n"
                "2. Lisinopril 5mg OD (Oral)\n"
                "3. Atorvastatin 20mg nocte (Oral)\n"
                "4. Aspirin 100mg OD (Oral)"
            )),
            ("Allergies", (
                "Penicillin — causes urticarial rash (Moderate)\n"
                "Sulfonamides — causes anaphylaxis (Severe)"
            )),
            ("Examination Findings", (
                "Temperature: 37.1°C\n"
                "Blood Pressure: 148/92 mmHg\n"
                "Heart Rate: 88 bpm\n"
                "Respiratory Rate: 18/min\n"
                "SpO2: 97% on room air\n"
                "Weight: 78 kg\n"
                "BMI: 29.2\n"
                "General: Alert, oriented, mild dehydration"
            )),
            ("Assessment", (
                "1. Hyperglycaemia secondary to dietary non-compliance — Type 2 DM poorly controlled\n"
                "2. Hypertension — suboptimally controlled\n"
                "3. Hyperlipidaemia — stable on current therapy"
            )),
            ("Initial Management Plan", (
                "1. IV fluid resuscitation — Normal saline 1L over 4 hours\n"
                "2. Insulin sliding scale while inpatient\n"
                "3. Dietitian review\n"
                "4. Diabetes educator referral\n"
                "5. Monitor BSL 4-hourly\n"
                "6. Renal function panel and HbA1c ordered"
            )),
        ],
    ),
    "lab_results.pdf": (
        "LABORATORY RESULTS — Patient 001",
        [
            ("Biochemistry — 10/01/2024 (Admission)", (
                "Sodium: 138 mmol/L (Normal: 135–145)\n"
                "Potassium: 4.1 mmol/L (Normal: 3.5–5.0)\n"
                "Creatinine: 92 µmol/L (Normal: 60–110)\n"
                "eGFR: 62 mL/min/1.73m² (Normal >60)\n"
                "Random Blood Glucose: 22.4 mmol/L (HIGH)\n"
                "HbA1c: PENDING — sample collected 10/01/2024, result expected 12/01/2024\n"
                "Total Cholesterol: 5.2 mmol/L (Borderline)\n"
                "LDL: 3.1 mmol/L (Borderline high)"
            )),
            ("Haematology — 10/01/2024", (
                "Haemoglobin: 128 g/L (Low-normal for female)\n"
                "White Cell Count: 8.2 × 10⁹/L (Normal)\n"
                "Platelets: 245 × 10⁹/L (Normal)\n"
                "CRP: 12 mg/L (Mildly elevated)"
            )),
            ("ECG — 10/01/2024", (
                "Performed in ED on admission.\n"
                "Result: Normal sinus rhythm, rate 88 bpm, no acute ischaemic changes."
            )),
            ("Urinalysis — 10/01/2024", (
                "Glucose: 4+\n"
                "Ketones: 1+\n"
                "Protein: trace\n"
                "Leucocytes: negative\n"
                "Nitrites: negative"
            )),
        ],
    ),
    "progress_notes.pdf": (
        "PROGRESS NOTES — Patient 001",
        [
            ("Progress Note — 11/01/2024 (Day 2)", (
                "Dr. Patel — Medical Review\n"
                "Patient feeling better. Less polyuria. Eating well.\n"
                "BSL trending down — range 10–14 mmol/L over past 24h.\n"
                "BP 142/88 — still elevated. Increased Lisinopril to 10mg OD.\n"
                "Dietitian reviewed. Low-GI diet plan in place.\n"
                "Plan: Continue current management. Repeat BGL tomorrow morning."
            )),
            ("Progress Note — 12/01/2024 (Day 3)", (
                "Dr. Patel — Medical Review\n"
                "Patient significantly improved. BSL 8.2 mmol/L this morning.\n"
                "HbA1c result received: 9.4% — poorly controlled over past 3 months.\n"
                "Insulin sliding scale ceased. Transitioning back to oral medications.\n"
                "Added Sitagliptin 100mg OD to improve glycaemic control long-term.\n"
                "Discussed with patient re: medication compliance and diet.\n"
                "Plan: Discharge planning initiated. Endocrinology outpatient follow-up arranged."
            )),
            ("Nursing Note — 12/01/2024", (
                "Patient ambulating independently. Vitals stable.\n"
                "Blood glucose 7.8 mmol/L pre-dinner. Patient educated on self-monitoring.\n"
                "Discharge medications explained to patient and husband."
            )),
        ],
    ),
    "medication_chart.pdf": (
        "MEDICATION CHART — Patient 001",
        [
            ("Admission Medications (confirmed)", (
                "1. Metformin 500mg BD — Oral\n"
                "2. Lisinopril 5mg OD — Oral\n"
                "3. Atorvastatin 20mg nocte — Oral\n"
                "4. Aspirin 100mg OD — Oral"
            )),
            ("Inpatient Medications", (
                "1. IV Normal Saline 1L over 4h — 10/01/2024 only\n"
                "2. Insulin Actrapid — sliding scale — 10/01 to 12/01/2024\n"
                "3. Metformin 500mg BD — continued\n"
                "4. Lisinopril 10mg OD — increased 11/01/2024 (Dr. Patel, BP suboptimally controlled)"
            )),
            ("Discharge Medications", (
                "1. Metformin 1000mg BD — INCREASED (was 500mg BD; improved glycaemic control)\n"
                "2. Lisinopril 10mg OD — INCREASED (was 5mg; suboptimal BP control)\n"
                "3. Atorvastatin 20mg nocte — UNCHANGED\n"
                "4. Aspirin 100mg OD — UNCHANGED\n"
                "5. Sitagliptin 100mg OD — NEW (added 12/01/2024 for additional HbA1c reduction)"
            )),
        ],
    ),
    "discharge_summary_draft.pdf": (
        "DISCHARGE SUMMARY DRAFT — Patient 001",
        [
            ("Discharge Details", (
                "Discharge Date: 15/01/2024\n"
                "Length of Stay: 5 days\n"
                "Discharge Destination: Home\n"
                "Discharge Condition: Stable"
            )),
            ("Principal Diagnosis", "Type 2 Diabetes Mellitus — poorly controlled (HbA1c 9.4%)"),
            ("Secondary Diagnoses", (
                "1. Hypertension — suboptimally controlled, medication increased\n"
                "2. Hyperlipidaemia — stable"
            )),
            ("Follow-up Instructions", (
                "1. Endocrinology outpatient clinic — 29/01/2024 at 10:00 AM (Booked)\n"
                "2. GP review in 1 week — for BP and medication review\n"
                "3. Self-monitor blood glucose twice daily — fasting and 2h post-dinner\n"
                "4. Diet: Continue low-GI diet as per dietitian advice\n"
                "5. Return to ED immediately if blood glucose > 20 mmol/L or vomiting"
            )),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Patient 002 — Challenging case: conflicts, missing data, pending results
# ─────────────────────────────────────────────────────────────────────────────

PATIENT_002_DOCS = {
    "admission_note.pdf": (
        "ADMISSION NOTE — Patient 002",
        [
            ("Patient Demographics", (
                "Name: Robert Okafor\n"
                "Date of Birth: 22/07/1945\n"
                "MRN: 0098765\n"
                "Gender: Male\n"
                "Address: 7 Wellington Road, Riverside\n"
                "Next of Kin: NOT DOCUMENTED"
            )),
            ("Admission Details", (
                "Date of Admission: 05/02/2024\n"
                "Admitting Physician: Dr. James Nguyen\n"
                "Ward: Coronary Care Unit (CCU)\n"
                "Admission Source: Emergency Department — via ambulance"
            )),
            ("Presenting Complaint", (
                "78-year-old male with chest pain and dyspnoea. Onset 2 hours prior to arrival. "
                "Associated diaphoresis and nausea. Denies trauma."
            )),
            ("Past Medical History", (
                "1. Ischaemic Heart Disease — CABG 2015\n"
                "2. Type 2 Diabetes Mellitus\n"
                "3. Chronic Kidney Disease Stage 3\n"
                "4. Atrial Fibrillation — on anticoagulation\n"
                "NOTE: No prior records available. History taken from patient (poor historian)."
            )),
            ("Medications on Admission", (
                "Per patient report (unverified — no medication chart available on arrival):\n"
                "1. Warfarin — dose unknown\n"
                "2. 'A blood pressure tablet' — name/dose unknown\n"
                "3. 'A sugar tablet' — name/dose unknown\n"
                "Allergy status: NOT DOCUMENTED — patient unable to provide history"
            )),
            ("Examination Findings", (
                "Temperature: 36.8°C\n"
                "Blood Pressure: 160/95 mmHg\n"
                "Heart Rate: 102 bpm (irregular)\n"
                "Respiratory Rate: 22/min\n"
                "SpO2: 91% on 4L O2\n"
                "Weight: NOT DOCUMENTED\n"
                "JVP: elevated\n"
                "Bilateral basal crepitations"
            )),
            ("Assessment", (
                "1. Acute Coronary Syndrome — NSTEMI likely\n"
                "2. Acute decompensated heart failure\n"
                "3. Atrial Fibrillation with rapid ventricular response"
            )),
            ("Initial Management", (
                "1. Aspirin 300mg stat\n"
                "2. GTN infusion commenced\n"
                "3. Furosemide 40mg IV\n"
                "4. Cardiology review requested URGENTLY\n"
                "5. Troponin, BNP, full blood count, coagulation ordered"
            )),
        ],
    ),
    "consultation_notes.pdf": (
        "CONSULTATION NOTES — Patient 002",
        [
            ("Cardiology Consultation — 05/02/2024", (
                "Dr. Marina Volkov — Cardiology\n\n"
                "Referred for: Possible NSTEMI, decompensated HF, AF with RVR\n\n"
                "Review of ECG: AF with RVR, rate ~100, ST depression V4–V6. "
                "No ST elevation. Findings consistent with demand ischaemia.\n\n"
                "Echocardiogram performed (bedside):\n"
                "- EF approximately 35% (severely reduced)\n"
                "- Anterior wall hypokinesis\n"
                "- Moderate mitral regurgitation\n\n"
                "DIAGNOSIS (Cardiology): Acute decompensated heart failure secondary to "
                "AF with rapid ventricular response. Ischaemia contribution possible.\n\n"
                "CONFLICTING NOTE: Admitting team documented NSTEMI as primary. "
                "Cardiology assessment does not support NSTEMI as primary diagnosis — "
                "troponin rise likely demand-related.\n\n"
                "Plan:\n"
                "1. Rate control — Metoprolol 25mg BD commenced\n"
                "2. Increase diuresis — Furosemide 80mg BD\n"
                "3. Commence Ramipril 2.5mg OD once haemodynamically stable\n"
                "4. Repeat Echo in 6 weeks post-discharge\n"
                "5. Coronary angiography — deferred, high procedural risk given CKD"
            )),
            ("Renal Consultation — 06/02/2024", (
                "Dr. Aiko Tanaka — Nephrology\n\n"
                "CKD Stage 3 — eGFR 38 on admission bloods.\n"
                "Contrast-enhanced imaging CONTRAINDICATED.\n"
                "Monitor renal function closely with diuretic therapy.\n"
                "Metformin — HOLD, contraindicated with eGFR < 45.\n"
                "Warfarin to continue; INR monitoring essential.\n"
                "No RAAS initiation until creatinine stable."
            )),
        ],
    ),
    "lab_reports.pdf": (
        "LABORATORY REPORTS — Patient 002",
        [
            ("Troponin I — Serial Measurements", (
                "05/02/2024 00:30: 0.08 µg/L (Normal <0.04) — ELEVATED\n"
                "05/02/2024 06:30: 0.22 µg/L (Normal <0.04) — ELEVATED, rising\n"
                "05/02/2024 12:30: 0.31 µg/L (Normal <0.04) — ELEVATED, peak\n"
                "06/02/2024 08:00: 0.19 µg/L — trending down"
            )),
            ("Biochemistry — 05/02/2024", (
                "Sodium: 136 mmol/L\n"
                "Potassium: 4.8 mmol/L\n"
                "Creatinine: 178 µmol/L (HIGH — baseline CKD)\n"
                "eGFR: 38 mL/min/1.73m²\n"
                "BNP: 1840 pg/mL (CRITICALLY ELEVATED — heart failure)\n"
                "Random Glucose: 14.2 mmol/L\n"
                "HbA1c: PENDING — sample collected 05/02/2024"
            )),
            ("Coagulation — 05/02/2024", (
                "INR: 2.8 (Therapeutic range for AF: 2.0–3.0)\n"
                "APTT: 34 seconds (Normal)"
            )),
            ("Haematology — 05/02/2024", (
                "Haemoglobin: 101 g/L — ANAEMIA (microcytic, hypochromic picture)\n"
                "MCV: 71 fL (LOW — iron deficiency pattern)\n"
                "White Cell Count: 11.4 × 10⁹/L — mildly elevated\n"
                "Platelets: 189 × 10⁹/L\n"
                "Iron studies: PENDING — sample sent 05/02/2024"
            )),
            ("Blood Culture — 05/02/2024", (
                "Sample collected 05/02/2024 at 01:15\n"
                "PENDING — results expected 48–72 hours\n"
                "Clinical indication: Leukocytosis + fever query source"
            )),
        ],
    ),
    "progress_notes.pdf": (
        "PROGRESS NOTES — Patient 002",
        [
            ("Progress Note — 06/02/2024 (Day 2)", (
                "Dr. Nguyen — Cardiology Team\n"
                "Patient improving. Less dyspnoea. SpO2 96% on 2L O2.\n"
                "BP 138/84. HR 78 (irregular) — AF rate controlled.\n"
                "Diuresis: urine output 2.4L past 24h. Oedema reducing.\n"
                "Creatinine up slightly: 192 µmol/L — monitoring.\n"
                "Warfarin dose adjusted — INR 2.8 to INR target 2–3.\n"
                "Ramipril NOT yet started — renal team advising caution.\n"
                "DIAGNOSIS UPDATED: Acute decompensated cardiac failure "
                "secondary to AF with RVR. NSTEMI diagnosis withdrawn — "
                "troponin rise demand-related per cardiology."
            )),
            ("Progress Note — 07/02/2024 (Day 3)", (
                "Dr. Volkov — Cardiology\n"
                "Patient clinically stable. Tolerating oral medications.\n"
                "Blood culture results: NO GROWTH at 48h — preliminary report. "
                "Final result pending (72h).\n"
                "Iron studies pending.\n"
                "Plan: Discharge planning for 09/02/2024 if stable.\n"
                "Arrange: Repeat Echo 6 weeks, cardiology outpatient, GP review."
            )),
            ("Nursing Note — 08/02/2024", (
                "Patient voicing concerns about going home — lives alone.\n"
                "Social work referral placed for home supports.\n"
                "Patient able to self-administer medications with prompting.\n"
                "Medication reconciliation not completed — discharge medication "
                "list to be finalised by medical team before discharge."
            )),
        ],
    ),
    "medication_chart.pdf": (
        "MEDICATION CHART — Patient 002",
        [
            ("Admission Medications (unverified — patient report only)", (
                "1. Warfarin — dose NOT DOCUMENTED\n"
                "2. Antihypertensive — name/dose NOT DOCUMENTED\n"
                "3. Oral hypoglycaemic — name/dose NOT DOCUMENTED"
            )),
            ("Inpatient Medications", (
                "1. Aspirin 100mg OD (reduced from 300mg stat load)\n"
                "2. Metoprolol 25mg BD — commenced 05/02/2024\n"
                "3. Furosemide 80mg BD — IV, changed to oral 07/02/2024\n"
                "4. Warfarin — dose per INR (INR 2.8, target 2–3)\n"
                "5. Metformin — HELD per renal team (eGFR < 45)\n"
                "6. Ramipril — HELD pending renal function stabilisation"
            )),
            ("Discharge Medications", (
                "STATUS: INCOMPLETE — to be finalised by Dr. Nguyen before discharge\n\n"
                "Confirmed:\n"
                "1. Aspirin 100mg OD — Oral\n"
                "2. Metoprolol 25mg BD — Oral\n"
                "3. Furosemide 40mg OD — Oral (dose reduced from inpatient)\n"
                "4. Warfarin — dose to be confirmed with INR on day of discharge\n\n"
                "Pending decision:\n"
                "5. Ramipril — commence 2.5mg OD or defer? Awaiting renal review.\n"
                "6. Oral hypoglycaemic — which agent? Metformin held. Alternative required.\n"
                "7. Statin therapy — nil prescribed. Query oversight."
            )),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def create_patient_folder(
    base_dir: Path, patient_id: str, documents: dict[str, tuple[str, list]]
) -> None:
    folder = base_dir / patient_id
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating {patient_id} ({len(documents)} documents) → {folder}")
    for filename, (title, sections) in documents.items():
        _make_pdf(folder / filename, title, sections)


def main(output_dir: str = "data") -> None:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    if not REPORTLAB_AVAILABLE:
        print(
            "\n[WARNING] reportlab not installed — creating .txt files instead of PDFs.\n"
            "Install with: pip install reportlab\n"
        )

    create_patient_folder(base, "patient_001", PATIENT_001_DOCS)
    create_patient_folder(base, "patient_002", PATIENT_002_DOCS)

    print(
        f"\n✓ Sample data created in '{base}':\n"
        f"  patient_001 — Clean T2DM case (straightforward)\n"
        f"  patient_002 — Complex cardiac case (conflicts, missing data, pending results)\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic patient PDFs for discharge summary agent testing."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to write patient folders into (default: data/)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)
