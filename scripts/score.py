#!/usr/bin/env python3
"""
Lightweight scoring: compares extract.py JSON output against ground-truth JSON
in fixtures/{set}/ground-truth/. Emits a per-set markdown summary.

Usage:
  python scripts/score.py --set 01-1900-us-census
  python scripts/score.py --set 01-1900-us-census --output results/2026-05-04-baseline.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
FIXTURES_DIR = REPO_ROOT / "fixtures"
CRITICAL_FIELDS = ("vendor", "total", "vat", "date")


@dataclass
class SetMetrics:
    set_slug: str
    n_docs: int = 0
    passed_gate: int = 0
    bounced: int = 0
    field_correct: dict[str, int] = None
    field_seen: dict[str, int] = None
    silent_failures: int = 0  # auto-exported but wrong vs ground truth
    total_time_s: float = 0.0

    def __post_init__(self):
        if self.field_correct is None:
            self.field_correct = {f: 0 for f in CRITICAL_FIELDS}
        if self.field_seen is None:
            self.field_seen = {f: 0 for f in CRITICAL_FIELDS}

    @property
    def straight_through_rate(self):
        return self.passed_gate / self.n_docs if self.n_docs else 0.0

    @property
    def bounce_rate(self):
        return self.bounced / self.n_docs if self.n_docs else 0.0

    @property
    def silent_failure_rate(self):
        return self.silent_failures / self.passed_gate if self.passed_gate else 0.0


def score_set(set_slug: str) -> SetMetrics:
    results_dir = RESULTS_DIR / set_slug
    truth_dir = FIXTURES_DIR / set_slug / "ground-truth"
    if not results_dir.exists():
        raise FileNotFoundError(f"No results at {results_dir}. Run extract.py --set {set_slug} first.")

    metrics = SetMetrics(set_slug=set_slug)
    for result_file in sorted(results_dir.glob("*.json")):
        result = json.loads(result_file.read_text(encoding="utf-8"))
        metrics.n_docs += 1
        if result.get("gate_passed"):
            metrics.passed_gate += 1
        else:
            metrics.bounced += 1
        metrics.total_time_s += (result.get("completed_at", 0) - result.get("started_at", 0))

        # Compare against ground truth if available
        truth_file = truth_dir / result_file.name
        if not truth_file.exists():
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        truth_fields = {f["field_id"]: f["expected_value"] for f in truth.get("fields", [])}

        any_field_wrong = False
        for f in result.get("fields", []):
            fid = f["field_id"]
            if fid not in CRITICAL_FIELDS:
                continue
            expected = truth_fields.get(fid)
            if expected is None:
                continue
            metrics.field_seen[fid] += 1
            if str(f["parsed_value"]).strip().lower() == str(expected).strip().lower():
                metrics.field_correct[fid] += 1
            else:
                any_field_wrong = True

        if result.get("gate_passed") and any_field_wrong:
            metrics.silent_failures += 1

    return metrics


def render_markdown(all_metrics: list[SetMetrics]) -> str:
    lines = [
        f"# Benchmark — Document Ops extraction (proof-fixtures)",
        "",
        f"Sets scored: {len(all_metrics)}",
        "",
        "## Per-set summary",
        "",
        "| Set | N | Pass | Bounce | STR | Bounce % | Silent fail | Avg s/doc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in all_metrics:
        avg_t = m.total_time_s / m.n_docs if m.n_docs else 0
        lines.append(
            f"| {m.set_slug} | {m.n_docs} | {m.passed_gate} | {m.bounced} | "
            f"{m.straight_through_rate:.1%} | {m.bounce_rate:.1%} | "
            f"{m.silent_failures} ({m.silent_failure_rate:.1%}) | {avg_t:.1f} |"
        )

    lines += ["", "## Field-level accuracy (when not flagged)", "", "| Set | " + " | ".join(CRITICAL_FIELDS) + " |", "|---|" + "---|" * len(CRITICAL_FIELDS)]
    for m in all_metrics:
        cells = []
        for f in CRITICAL_FIELDS:
            seen = m.field_seen[f]
            correct = m.field_correct[f]
            cells.append(f"{correct}/{seen} ({correct/seen:.0%})" if seen else "—")
        lines.append(f"| {m.set_slug} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Targets at GA",
        "",
        "- Straight-through-rate ≥ 97%",
        "- Field-level-when-not-flagged ≥ 99%",
        "- Silent-failure-rate < 2%",
        "",
        "Run `python scripts/extract.py --set <slug>` then `python scripts/score.py --set <slug>` to refresh.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_slug", help="Score a single set; omit to score all.")
    ap.add_argument("--output", type=Path, help="Write markdown to this path; default stdout.")
    args = ap.parse_args()

    if args.set_slug:
        sets_to_score = [args.set_slug]
    else:
        sets_to_score = [d.name for d in RESULTS_DIR.iterdir() if d.is_dir()]

    metrics = []
    for s in sets_to_score:
        try:
            metrics.append(score_set(s))
        except FileNotFoundError as e:
            print(f"# WARN: {e}", file=sys.stderr)

    md = render_markdown(metrics)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
