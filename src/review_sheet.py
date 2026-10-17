"""Human review as a spreadsheet: export answers to Excel, import the verdicts back.

Reviewers fill in the yellow columns (Yes/No from a drop-down), then the import writes
evaluation/human_reviews.json, which the next evaluation run reads.

Usage:
    python -m src.review_sheet export evaluation/answers_for_review_test_v2_ollama.json
    python -m src.review_sheet import evaluation/human_review.xlsx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE  # noqa: E402

from src.config import load_config, read_json, resolve, write_json  # noqa: E402

SHEET = "evaluation/human_review.xlsx"
INFO = ["id", "type", "question", "reference_answer", "answer", "cited_passages"]
VERDICTS = ["correct", "grounded", "citations_support"]
HEADERS = {
    "id": "ID", "type": "Type", "question": "Question", "reference_answer": "Reference answer",
    "answer": "Chatbot answer", "cited_passages": "Passages it cited",
    "correct": "Correct?", "grounded": "Grounded?", "citations_support": "Citations support it?",
    "reviewer": "Reviewer", "notes": "Notes",
}
WIDTHS = {"id": 7, "type": 11, "question": 34, "reference_answer": 40, "answer": 55, "cited_passages": 70,
          "correct": 11, "grounded": 11, "citations_support": 13, "reviewer": 14, "notes": 30}
GUIDE = [
    ("Correct?", "Yes if the answer agrees with the reference answer and the books. For a question the "
                 "books don't cover (Answerable = No), Yes only if the chatbot refused."),
    ("Grounded?", "Yes if every claim in the answer is stated in the cited passages, with nothing invented."),
    ("Citations support it?", "Yes if each [n] points to a passage that backs the sentence it is attached to. "
                              "Leave blank for a refusal."),
    ("Reviewer", "Your name. Notes: anything wrong, missing or worth mentioning in the error analysis."),
    ("Blank cells", "Count as not reviewed yet. You can import a half-finished sheet at any time."),
]


def clean(text: str) -> str:
    # PDF text can carry control characters that Excel files cannot store.
    return ILLEGAL_CHARACTERS_RE.sub("", text)


def answer_text(row: dict) -> str:
    response = row["response"]
    if response["insufficient_evidence"]:
        return "[Refused: not enough evidence in the books]"
    return response["answer"]


def passages_text(row: dict) -> str:
    return "\n\n".join(f"[{n}] {p['citation']}\n{p['text']}" for n, p in enumerate(row["cited_passages"] or [], 1))


def export(source: Path, out: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    rows = read_json(source)
    old = read_json(resolve(load_config()["paths"]["human_reviews"]), default={})
    columns = INFO + VERDICTS + ["reviewer", "notes"]
    wb = Workbook()
    ws = wb.active
    ws.title = "Review"
    ws.append([HEADERS[c] for c in columns])
    for row in rows:
        values = {"id": row["id"], "type": row["type"] + ("" if row["answerable"] else " (not in books)"),
                  "question": row["question"], "reference_answer": row["reference_answer"] or "",
                  "answer": answer_text(row), "cited_passages": passages_text(row)}
        verdict = old.get(row["id"], {})
        for c in VERDICTS:
            values[c] = {True: "Yes", False: "No"}.get(verdict.get(c), "")
        values["reviewer"], values["notes"] = verdict.get("reviewer", ""), verdict.get("notes", "")
        ws.append([clean(values[c]) for c in columns])

    yellow = PatternFill("solid", fgColor="FFF2B3")
    for i, c in enumerate(columns, 1):
        letter = ws.cell(1, i).column_letter
        ws.column_dimensions[letter].width = WIDTHS[c]
        ws.cell(1, i).font = Font(bold=True)
        for r in range(2, ws.max_row + 1):
            cell = ws.cell(r, i)
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if c in VERDICTS + ["reviewer", "notes"]:
                cell.fill = yellow
    yes_no = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    ws.add_data_validation(yes_no)
    for i, c in enumerate(columns, 1):
        if c in VERDICTS:
            letter = ws.cell(1, i).column_letter
            yes_no.add(f"{letter}2:{letter}{ws.max_row}")
    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 180
    ws.freeze_panes = "C2"

    guide = wb.create_sheet("How to review")
    guide.column_dimensions["A"].width, guide.column_dimensions["B"].width = 24, 100
    guide.append(["Answers from", source.name])
    for item in GUIDE:
        guide.append(item)
    for row in guide.iter_rows():
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    print(f"Wrote {out} ({len(rows)} answers)")


def parse(value) -> bool | None:
    text = str(value or "").strip().lower()
    return True if text in ("yes", "y", "true", "1") else False if text in ("no", "n", "false", "0") else None


def import_sheet(sheet: Path) -> None:
    from openpyxl import load_workbook

    ws = load_workbook(sheet, read_only=True)["Review"]
    rows = ws.iter_rows(values_only=True)
    labels = {v: k for k, v in HEADERS.items()}
    columns = [labels.get(h) for h in next(rows)]
    path = resolve(load_config()["paths"]["human_reviews"])
    reviews = read_json(path, default={})
    count = 0
    for values in rows:
        row = dict(zip(columns, values))
        if not row.get("id"):
            continue
        verdict = {c: parse(row.get(c)) for c in VERDICTS}
        if all(v is None for v in verdict.values()):
            continue
        verdict.update(reviewer=str(row.get("reviewer") or ""), notes=str(row.get("notes") or ""))
        reviews[row["id"]] = verdict
        count += 1
    write_json(dict(sorted(reviews.items())), path)
    print(f"Saved {count} reviewed answers to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="write the review spreadsheet")
    ex.add_argument("answers", help="an evaluation/answers_for_review*.json file")
    ex.add_argument("--out", default=SHEET)
    im = sub.add_parser("import", help="save the verdicts to human_reviews.json")
    im.add_argument("sheet", nargs="?", default=SHEET)
    args = parser.parse_args()
    if args.command == "export":
        export(resolve(args.answers), resolve(args.out))
    else:
        import_sheet(resolve(args.sheet))


if __name__ == "__main__":
    main()
