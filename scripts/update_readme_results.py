"""Fill the generated blocks of README.md from saved result files (no numbers typed by hand).

Run with:  uv run python scripts/update_readme_results.py           (rewrite the blocks)
           uv run python scripts/update_readme_results.py --check   (assert, change nothing)

Blocks between <!-- BEGIN GENERATED: name --> and <!-- END GENERATED: name -->:
- results:     outputs/ml/results/final_test_table.md and detection_metrics.md, verbatim
- limitations: outputs/report/limitations_facts.md, verbatim (without its title)
- runtimes:    outputs/reproduce/pipeline_runtimes.tsv (scripts/reproduce_in_copy.sh)
--check asserts that every number in the README results tables equals report_numbers.json.
"""

import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "outputs" / "ml" / "results"
NUMBERS = ROOT / "outputs" / "ml" / "report_numbers.json"
LIMITATIONS = ROOT / "outputs" / "report" / "limitations_facts.md"
RUNTIMES = ROOT / "outputs" / "reproduce" / "pipeline_runtimes.tsv"
BLOCK = re.compile(r"(<!-- BEGIN GENERATED: (\w+) -->\n)(.*?)(<!-- END GENERATED: \2 -->)", re.S)


def demote(markdown: str) -> str:
    """Turn the copied file's '# Title' into '### Title' so it nests under README sections."""
    return re.sub(r"^# ", "### ", markdown.strip(), flags=re.M)


def results_block() -> str:
    return (
        "Source: `outputs/ml/results/final_test_table.md` (copied by "
        "`scripts/update_readme_results.py`). These are IMAGE-LEVEL classification metrics "
        "(firearm present yes/no) on the TEST split of split v2.\n\n"
        + demote((RESULTS / "final_test_table.md").read_text())
        + "\n\nSource: `outputs/ml/results/detection_metrics.md`. These are DETECTION metrics "
        "(boxes, mAP) and are not comparable with the image-level table.\n\n"
        + demote((RESULTS / "detection_metrics.md").read_text())
        + "\n"
    )


def limitations_block() -> str:
    lines = LIMITATIONS.read_text().strip().splitlines()
    body = "\n".join(line for line in lines if not line.startswith("# ")).strip()
    return "Source: `outputs/report/limitations_facts.md`.\n\n" + body + "\n"


def runtimes_block() -> str:
    if not RUNTIMES.exists():
        return "Not measured yet. Run `scripts/reproduce_in_copy.sh`.\n"
    rows = [line.split("\t") for line in RUNTIMES.read_text().splitlines() if "\t" in line]
    table = ["| step | seconds | exit code |", "|---|---|---|"]
    table += [f"| {name} | {seconds} | {code} |" for name, seconds, code in rows]
    return (
        "Measured on this laptop's CPU by `scripts/reproduce_in_copy.sh` (a fresh copy of the "
        "tracked files; source `outputs/reproduce/pipeline_runtimes.tsv`). Other jobs shared "
        "the CPU during this run, so the times are upper bounds. Colab training is not "
        "included.\n\n" + "\n".join(table) + "\n"
    )


def table_rows(markdown: str) -> list[dict]:
    """Parse the first Markdown table in a text into a list of {column: cell} dicts."""
    lines = [line for line in markdown.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    return [
        dict(zip(header, (cell.strip() for cell in line.strip("|").split("|")), strict=True))
        for line in lines[2:]
    ]


def same(cell: str, value) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return cell == ""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(cell), 4) == round(float(value), 4)
    return cell == str(value)


def check(readme: str) -> int:
    """Assert every README results cell equals report_numbers.json; return cells checked."""
    numbers = json.loads(NUMBERS.read_text())
    block = next(m.group(3) for m in BLOCK.finditer(readme) if m.group(2) == "results")
    test_part, detection_part = block.split("detection_metrics.md")
    checked = 0
    for part, rows in (
        (test_part, numbers["test_metrics"]["rows"]),
        (detection_part, numbers["detection_metrics"]["rows"]),
    ):
        shown = table_rows(part)
        assert len(shown) == len(rows), "row count differs"
        for cells, row in zip(shown, rows, strict=True):
            for column, cell in cells.items():
                assert column in row, f"column {column} missing in report_numbers.json"
                assert same(cell, row[column]), f"{column}: README {cell} != json {row[column]}"
                checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = README.read_text()
    if not args.check:
        makers = {"results": results_block, "limitations": limitations_block,
                  "runtimes": runtimes_block}  # fmt: skip
        text = BLOCK.sub(lambda m: m.group(1) + makers[m.group(2)]() + m.group(4), text)
        README.write_text(text)
    print(f"README results match report_numbers.json: {check(text)} cells checked")


if __name__ == "__main__":
    main()
