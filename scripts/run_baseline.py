#!/usr/bin/env python3
"""v0.4.2 Step 9 — baseline run wrapper.

Walks all fixture sets matching --prefix, runs extract.py on each, then
score.py once at the end. Writes a single benchmark markdown report.

The default flags target the v0.4.1 floor pipeline (Tesseract-only, no
verifier, no LLM extract, no line items) so the run completes on a CPU-only
box without Ollama. Layer the heavier stages back in via --pipeline /
--with-verifier / --with-llm-extract once their deps are present.

Usage:
    # Floor baseline (Tesseract-only) on the full v0.4.2 r1 corpus
    python scripts/run_baseline.py --prefix synth-full-

    # Pilot subset (5 docs/family — saves OCR time)
    python scripts/run_baseline.py --prefix synth-full- --doc-limit 5

    # Full ensemble (RapidOCR + GLM-OCR Ollama verifier) — needs deps installed
    python scripts/run_baseline.py --prefix synth-full- \\
        --pipeline docops --with-verifier --with-llm-extract --with-line-items
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
RESULTS_DIR = REPO_ROOT / "results"


@dataclass(frozen=True)
class SetResult:
    set_slug: str
    requested: int
    extracted: int
    elapsed_s: float
    return_code: int


def discover_sets(prefix: str) -> list[str]:
    return sorted(
        d.name for d in FIXTURES_DIR.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and (d / "samples").exists()
    )


def stage_pilot_subset(set_slug: str, doc_limit: int) -> Path:
    """Copies first N docs from fixtures/{set}/ into a temp pilot set so
    extract.py only processes the subset. Returns the pilot set slug."""
    src = FIXTURES_DIR / set_slug
    pilot_slug = f"{set_slug}__pilot{doc_limit}"
    dst = FIXTURES_DIR / pilot_slug
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "samples").mkdir(parents=True)
    (dst / "ground-truth").mkdir(parents=True)

    pdfs = sorted((src / "samples").glob("*.pdf"))[:doc_limit]
    for pdf in pdfs:
        try:
            (dst / "samples" / pdf.name).hardlink_to(pdf)
        except OSError:
            shutil.copy2(pdf, dst / "samples" / pdf.name)
        gt = src / "ground-truth" / f"{pdf.stem}.json"
        if gt.exists():
            shutil.copy2(gt, dst / "ground-truth" / gt.name)
    return pilot_slug


def run_extract(set_slug: str, args: argparse.Namespace) -> SetResult:
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "extract.py"),
        "--set", set_slug,
        "--pipeline", args.pipeline,
    ]
    if not args.with_verifier:
        cmd.append("--no-verifier")
    if not args.with_llm_extract:
        cmd.append("--no-llm-extract")
    if not args.with_line_items:
        cmd.append("--no-line-items")

    requested = len(list((FIXTURES_DIR / set_slug / "samples").glob("*.pdf")))

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    extracted = len(list((RESULTS_DIR / set_slug).glob("*.json"))) if (RESULTS_DIR / set_slug).exists() else 0
    if proc.returncode != 0:
        sys.stderr.write(f"\n!! {set_slug} extract.py rc={proc.returncode}\n")
        sys.stderr.write(proc.stderr[-2000:] + "\n")
    return SetResult(set_slug, requested, extracted, elapsed, proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--prefix", default="synth-full-",
                    help="Fixture set prefix to baseline (default 'synth-full-')")
    ap.add_argument("--doc-limit", type=int, default=None,
                    help="If set, copy first N docs/family into a pilot subset and run on that")
    ap.add_argument("--pipeline", default="tesseract-only",
                    choices=("tesseract-only", "rapidocr-only", "docops"),
                    help="Pipeline preset (default tesseract-only — floor measurement)")
    ap.add_argument("--with-verifier", action="store_true",
                    help="Run Ollama verifier chain (needs ollama daemon + glm-ocr model)")
    ap.add_argument("--with-llm-extract", action="store_true",
                    help="Run Stage 5 LLM extract via Ollama (needs glm-ocr:q8_0 pulled)")
    ap.add_argument("--with-line-items", action="store_true",
                    help="Run Stage 4b line-items via RapidTable (needs rapid-table install)")
    ap.add_argument("--report-path", type=Path, default=None,
                    help="Markdown report path (default results/baseline-{ts}-{pipeline}.md)")
    args = ap.parse_args()

    sets = discover_sets(args.prefix)
    if not sets:
        sys.exit(f"No fixture sets matched prefix={args.prefix!r} under {FIXTURES_DIR}")

    # Pilot subset path
    if args.doc_limit:
        sets = [stage_pilot_subset(s, args.doc_limit) for s in sets]

    print(f"# run_baseline.py — {len(sets)} sets · pipeline={args.pipeline} "
          f"verifier={args.with_verifier} llm-extract={args.with_llm_extract} "
          f"line-items={args.with_line_items}")
    print(f"# fixtures: {[s for s in sets[:3]]}{'...' if len(sets) > 3 else ''}")
    print()

    t_start = time.monotonic()
    results: list[SetResult] = []
    for idx, slug in enumerate(sets, 1):
        print(f"[{idx:>2}/{len(sets)}] {slug:<46} ", end="", flush=True)
        r = run_extract(slug, args)
        results.append(r)
        rate = r.extracted / r.elapsed_s if r.elapsed_s > 0 else 0
        status = "OK " if r.return_code == 0 else "FAIL"
        print(f"{status} {r.extracted}/{r.requested} in {r.elapsed_s:5.1f}s ({rate:.2f}/s)")

    total_elapsed = time.monotonic() - t_start
    print(f"\nExtraction total: {sum(r.extracted for r in results)} docs in {total_elapsed:.1f}s")
    failures = [r for r in results if r.return_code != 0]
    if failures:
        print(f"FAILURES: {len(failures)}")
        for r in failures:
            print(f"  - {r.set_slug} rc={r.return_code} extracted={r.extracted}/{r.requested}")

    # Score everything in one shot
    if args.report_path is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.report_path = RESULTS_DIR / f"baseline-{ts}-{args.pipeline}.md"
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n# scoring -> {args.report_path}")
    score_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "score.py"),
        "--output", str(args.report_path),
    ]
    score_proc = subprocess.run(score_cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if score_proc.returncode != 0:
        sys.stderr.write(f"\n!! score.py rc={score_proc.returncode}\n{score_proc.stderr}\n")
        return score_proc.returncode
    print(score_proc.stdout)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
