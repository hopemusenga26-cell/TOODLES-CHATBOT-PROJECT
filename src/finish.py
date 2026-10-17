"""Final checks and evaluation, in order, after the human review sheet is filled in.

Steps: import the review sheet, check the book list, run the tests, check Ollama, write the
official report with Toodles Ultra on the test split, then put the human-review result into the
README table. Stops at the first step that fails. Takes about 6 minutes.

Usage:
    python -m src.finish
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import read_json, resolve  # noqa: E402

SHEET = "evaluation/human_review.xlsx"
REPORT = "evaluation/report.json"
README_ROW = "| Answer correctness (human review) |"


class StepFailed(Exception):
    pass


def run(*args: str) -> None:
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_VERBOSITY="error")
    if subprocess.run([sys.executable, *args], cwd=ROOT, env=env).returncode != 0:
        raise StepFailed(" ".join(args))


def check_ollama() -> None:
    if not shutil.which("ollama") or subprocess.run(["ollama", "list"], capture_output=True).returncode != 0:
        raise StepFailed("Ollama is not running. Start the Ollama app, then run this again.")


def percent(value) -> str:
    return "pending" if value is None else f"{value * 100:.1f}%"


def update_readme() -> None:
    """Fill the Toodles Ultra cell of the answer-correctness row."""
    metrics = read_json(resolve(REPORT), default={}).get("metrics", {})
    path = ROOT / "README.md"
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith(README_ROW):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # Columns: measure, v1, Toodles Instant, Toodles Ultra (bold), Transformers copy, target
            cells[3] = f"**{percent(metrics.get('answer_correctness'))}**"
            lines[i] = "| " + " | ".join(cells) + " |\n"
            path.write_text("".join(lines), encoding="utf-8")
            print(f"README updated: Toodles Ultra correctness {cells[3]}")
            return
    print("README results row not found; update it by hand.")


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    steps = [
        ("Importing the human review sheet",
         lambda: run("-m", "src.review_sheet", "import", SHEET) if resolve(SHEET).exists()
         else print(f"No {SHEET} found; correctness will stay pending.")),
        ("Checking the book list", lambda: run("-m", "src.manifest", "check")),
        ("Running the tests", lambda: run("-m", "pytest", "-q")),
        ("Checking that Ollama is running", check_ollama),
        ("Evaluating Toodles Ultra on the test questions (about 5 minutes)",
         lambda: run("-m", "src.evaluate", "--split", "test", "--backend", "ollama")),
        ("Updating the README results table", update_readme),
    ]
    for n, (title, step) in enumerate(steps, 1):
        print(f"\n[{n}/{len(steps)}] {title}", flush=True)
        try:
            step()
        except StepFailed as e:
            print(f"\nStopped: {e}\nFix it, then run this again.")
            return 1

    print("\nDone. Results:")
    run("-m", "src.evaluate", "--summary", REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
