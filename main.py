"""
main.py — Entry point for the Discharge Summary Agent.

CLI Usage
---------
# Process a patient folder directly
python main.py run --patient-folder data/patient_001 --patient-id P001

# Start the FastAPI server
python main.py serve

# Generate sample test data
python main.py sample-data

# Run tests
python main.py test

Environment
-----------
Copy .env.example → .env and set GROQ_API_KEY before running.
Get a free Groq key at: https://console.groq.com/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
from pathlib import Path

# ── Load .env early so all modules pick up env vars ──────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check_api_key() -> bool:
    """
    Accept GROQ_API_KEY (primary) or GEMINI_API_KEY (legacy fallback).
    Returns True if at least one key is present.
    """
    groq_key   = os.getenv("GROQ_API_KEY",   "").strip()
    gemini_key = os.getenv("GEMINI_API_KEY",  "").strip()

    if groq_key or gemini_key:
        return True

    print(
        "\n[ERROR] No API key found.\n"
        "  Set one of the following in your .env file:\n"
        "\n"
        "  GROQ_API_KEY=gsk_your_key_here   ← recommended (free, fast)\n"
        "  Get key at: https://console.groq.com/\n"
        "\n"
        "  — or —\n"
        "\n"
        "  GEMINI_API_KEY=your_key_here     ← legacy fallback\n"
        "  Get key at: https://aistudio.google.com/app/apikey\n"
    )
    return False


def _print_summary_preview(summary: dict) -> None:
    """Pretty-print key sections of the generated summary to the terminal."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    c = Console()

    c.print()
    c.print(Panel("[bold cyan]DISCHARGE SUMMARY — GENERATED DRAFT[/bold cyan]", expand=False))

    demo = summary.get("patient_demographics", {})
    c.print(f"  [bold]Patient:[/bold]   {demo.get('name', 'NOT DOCUMENTED')}")
    c.print(f"  [bold]MRN:[/bold]       {demo.get('mrn', 'NOT DOCUMENTED')}")
    c.print(f"  [bold]Admitted:[/bold]  {summary.get('admission_date', 'NOT DOCUMENTED')}")
    c.print(f"  [bold]Discharged:[/bold]{summary.get('discharge_date', 'NOT DOCUMENTED')}")
    c.print(f"  [bold]Principal Dx:[/bold] {summary.get('principal_diagnosis', 'NOT DOCUMENTED')}")
    c.print()

    flags = summary.get("review_flags", [])
    if flags:
        t = Table(title="Review Flags", show_header=True, header_style="bold yellow")
        t.add_column("Severity", style="bold")
        t.add_column("Field")
        t.add_column("Message")
        for f in flags:
            sev    = f.get("severity", "")
            colour = {"CRITICAL": "red", "WARNING": "yellow", "INFO": "cyan"}.get(sev, "white")
            t.add_row(
                f"[{colour}]{sev}[/{colour}]",
                f.get("field", ""),
                f.get("message", ""),
            )
        c.print(t)
    else:
        c.print("  [green]No review flags generated.[/green]")

    conflicts = summary.get("conflicts_detected", [])
    if conflicts:
        c.print(
            f"\n  [bold red]⚠  {len(conflicts)} conflict(s) detected "
            f"— clinician review required[/bold red]"
        )

    pending_raw = summary.get("pending_results", [])
    if isinstance(pending_raw, list):
        pending = pending_raw
    elif isinstance(pending_raw, dict):
        pending = pending_raw.get("pending_results", [])
    else:
        pending = []
    if pending:
        c.print(f"  [bold yellow]⏳  {len(pending)} pending result(s)[/bold yellow]")

    c.print()


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: run
# ─────────────────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    """Process a patient folder and generate a discharge summary."""
    if not _check_api_key():
        return 1

    patient_folder = Path(args.patient_folder).resolve()
    if not patient_folder.exists():
        print(f"[ERROR] Patient folder not found: {patient_folder}")
        return 1

    patient_id = args.patient_id or patient_folder.name
    output_dir = args.output_dir
    max_steps  = int(os.getenv("MAX_AGENT_STEPS", args.max_steps))

    from rich.console import Console
    console = Console()
    console.print(f"\n[bold]Discharge Summary Agent[/bold]")
    console.print(f"  Patient folder : {patient_folder}")
    console.print(f"  Patient ID     : {patient_id}")
    console.print(f"  Max steps      : {max_steps}")
    console.print(f"  Output dir     : {output_dir}\n")

    try:
        from agent.graph import build_graph
        from agent.state import default_state

        graph = build_graph(output_dir=output_dir)
        initial_state = default_state(
            patient_folder=str(patient_folder),
            patient_id=patient_id,
            max_steps=max_steps,
        )

        logger.info("Starting agent graph for patient: %s", patient_id)
        final_state = graph.invoke(initial_state)

        summary      = final_state.get("final_summary", {})
        output_files = final_state.get("output_files", {})

        _print_summary_preview(summary)

        console.print("[bold green]✓ Complete.[/bold green]  Output files:")
        for label, path in output_files.items():
            console.print(f"  [{label}] {path}")
        console.print()

        errors = final_state.get("errors", [])
        if errors:
            console.print(f"[yellow]Warnings / non-fatal errors ({len(errors)}):[/yellow]")
            for e in errors:
                console.print(f"  • {e}")

        return 0

    except Exception as exc:
        logger.exception("Agent failed: %s", exc)
        print(f"\n[ERROR] Agent run failed: {exc}")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: serve
# ─────────────────────────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    """Start the FastAPI server."""
    if not _check_api_key():
        return 1

    try:
        import uvicorn
    except ImportError:
        print("[ERROR] uvicorn is not installed. Run: pip install uvicorn")
        return 1

    host = os.getenv("API_HOST", args.host)
    port = int(os.getenv("API_PORT", args.port))

    print(f"\nStarting Discharge Summary API on http://{host}:{port}")
    print("  POST /generate-summary  — run agent on a patient folder")
    print("  GET  /health            — service health check")
    print("  Press Ctrl+C to stop.\n")

    uvicorn.run(
        "app.api:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level=LOG_LEVEL.lower(),
    )
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: sample-data
# ─────────────────────────────────────────────────────────────────────────────

def cmd_sample_data(args: argparse.Namespace) -> int:
    """Generate synthetic patient PDFs for testing."""
    output_dir = args.output_dir
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from data.create_sample_data import main as create_data
        create_data(output_dir=output_dir)
        return 0
    except ImportError as exc:
        print(f"[ERROR] Could not import sample data generator: {exc}")
        return 1
    except Exception as exc:
        logger.exception("Sample data creation failed: %s", exc)
        print(f"[ERROR] {exc}")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: test
# ─────────────────────────────────────────────────────────────────────────────

def cmd_test(args: argparse.Namespace) -> int:
    """Run the test suite."""
    try:
        import pytest
        exit_code = pytest.main(["tests/", "-v", "--tb=short", "--no-header"])
        return exit_code
    except ImportError:
        print("[ERROR] pytest is not installed. Run: pip install pytest")
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: demo
# ─────────────────────────────────────────────────────────────────────────────

def cmd_demo(args: argparse.Namespace) -> int:
    """Full demo: generate sample data then run agent on both patients."""
    from rich.console import Console
    console = Console()

    console.print("\n[bold cyan]═══ DISCHARGE SUMMARY AGENT DEMO ═══[/bold cyan]\n")

    console.print("[bold]Step 1: Generating synthetic patient data...[/bold]")
    rc = cmd_sample_data(argparse.Namespace(output_dir="data"))
    if rc != 0:
        return rc

    if not _check_api_key():
        return 1

    console.print("\n[bold]Step 2: Processing Patient 001 (straightforward T2DM case)...[/bold]")
    rc = cmd_run(argparse.Namespace(
        patient_folder="data/patient_001",
        patient_id="patient_001",
        output_dir="outputs",
        max_steps=25,
    ))

    console.print("\n[bold]Step 3: Processing Patient 002 (complex case — conflicts + missing data)...[/bold]")
    rc2 = cmd_run(argparse.Namespace(
        patient_folder="data/patient_002",
        patient_id="patient_002",
        output_dir="outputs",
        max_steps=25,
    ))

    console.print("\n[bold green]Demo complete.[/bold green]")
    console.print("  outputs/patient_001/discharge_summary.md")
    console.print("  outputs/patient_001/traces/trace.txt")
    console.print("  outputs/patient_002/discharge_summary.md")
    console.print("  outputs/patient_002/traces/trace.txt\n")

    return max(rc, rc2)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discharge-agent",
        description="Agentic AI Discharge Summary Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py sample-data
              python main.py run --patient-folder data/patient_001
              python main.py serve
              python main.py demo
              python main.py test
        """),
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Process a patient folder")
    run_p.add_argument("--patient-folder", required=True,
                       help="Path to the folder containing patient PDFs")
    run_p.add_argument("--patient-id", default=None,
                       help="Optional patient identifier (defaults to folder name)")
    run_p.add_argument("--output-dir", default="outputs",
                       help="Directory for generated summaries and traces (default: outputs/)")
    run_p.add_argument("--max-steps", type=int, default=25,
                       help="Hard agent step cap (default: 25)")

    # ── serve ─────────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the FastAPI HTTP server")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true",
                         help="Enable hot-reload (development only)")

    # ── sample-data ───────────────────────────────────────────────────────────
    sd_p = sub.add_parser("sample-data", help="Generate synthetic patient PDFs for testing")
    sd_p.add_argument("--output-dir", default="data",
                      help="Where to write patient folders (default: data/)")

    # ── test ──────────────────────────────────────────────────────────────────
    sub.add_parser("test", help="Run the test suite (requires pytest)")

    # ── demo ──────────────────────────────────────────────────────────────────
    sub.add_parser("demo", help="Generate sample data and run agent on both patients")

    return parser


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "run":         cmd_run,
        "serve":       cmd_serve,
        "sample-data": cmd_sample_data,
        "test":        cmd_test,
        "demo":        cmd_demo,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(handler(args))


if __name__ == "__main__":
    main()