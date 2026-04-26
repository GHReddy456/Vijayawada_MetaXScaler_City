"""
Quick pass/fail gate for CrisisWorld training runs.

Usage:
  python backend/scripts/check_submission_gate.py \
    --report checkpoints/crisisworld-grpo-final/training_before_after.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Tuple


METRIC_PATTERNS = {
    "json_ok_rate": r"json_ok_rate\s+early=\s*([-\d.]+)%\s+late=\s*([-\d.]+)%",
    "rescue_success_%": r"rescue_success_%\s+early=\s*([-\d.]+)\s+late=\s*([-\d.]+)",
    "deaths (lower better)": r"deaths \(lower better\)\s+early=\s*([-\d.]+)\s+late=\s*([-\d.]+)",
    "panic (lower better)": r"panic \(lower better\)\s+early=\s*([-\d.]+)\s+late=\s*([-\d.]+)",
    "trust_score": r"trust_score\s+early=\s*([-\d.]+)\s+late=\s*([-\d.]+)",
    "coord_score_%": r"coord_score_%\s+early=\s*([-\d.]+)\s+late=\s*([-\d.]+)",
}


def parse_metrics(report_text: str) -> Dict[str, Tuple[float, float]]:
    values: Dict[str, Tuple[float, float]] = {}
    for metric, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, report_text)
        if m:
            values[metric] = (float(m.group(1)), float(m.group(2)))
    return values


def evaluate_gate(values: Dict[str, Tuple[float, float]]) -> Dict[str, bool]:
    required = [
        "json_ok_rate",
        "rescue_success_%",
        "deaths (lower better)",
        "panic (lower better)",
    ]
    missing = [k for k in required if k not in values]
    if missing:
        raise ValueError(f"Missing required metrics in report: {', '.join(missing)}")

    json_early, json_late = values["json_ok_rate"]
    rescue_early, rescue_late = values["rescue_success_%"]
    deaths_early, deaths_late = values["deaths (lower better)"]
    panic_early, panic_late = values["panic (lower better)"]

    return {
        "json_ok_rate": (json_late >= 92.0) and (json_late >= json_early - 1.0),
        "rescue_success_%": rescue_late >= rescue_early + 0.8,
        "deaths (lower better)": deaths_late <= deaths_early - 0.15,
        "panic (lower better)": panic_late <= panic_early - 1.0,
    }


def classify(result_flags: Dict[str, bool], values: Dict[str, Tuple[float, float]]) -> str:
    passed = sum(1 for ok in result_flags.values() if ok)
    if passed == 4:
        return "PASS"
    if passed <= 2:
        return "FAIL"

    trust = values.get("trust_score")
    coord = values.get("coord_score_%")
    if trust is None or coord is None:
        return "BORDERLINE"

    trust_early, trust_late = trust
    coord_early, coord_late = coord
    tie_break_ok = (trust_late >= trust_early + 2.0) and (coord_late >= coord_early + 8.0)
    return "PASS" if tie_break_ok else "BORDERLINE"


def main() -> int:
    parser = argparse.ArgumentParser(description="Check CrisisWorld submission gate.")
    parser.add_argument(
        "--report",
        default="checkpoints/crisisworld-grpo-final/training_before_after.txt",
        help="Path to training_before_after.txt",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"ERROR: report not found: {report_path}")
        return 2

    text = report_path.read_text(encoding="utf-8")
    values = parse_metrics(text)

    try:
        flags = evaluate_gate(values)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2

    decision = classify(flags, values)

    print("\n=== CrisisWorld Submission Gate ===")
    for metric, ok in flags.items():
        early, late = values[metric]
        mark = "PASS" if ok else "FAIL"
        print(f"- {metric}: early={early:.2f}, late={late:.2f} -> {mark}")

    if "trust_score" in values and "coord_score_%" in values:
        te, tl = values["trust_score"]
        ce, cl = values["coord_score_%"]
        print(f"- tie-break trust_score delta: {tl - te:+.2f}")
        print(f"- tie-break coord_score_% delta: {cl - ce:+.2f}")

    print(f"\nDecision: {decision}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
