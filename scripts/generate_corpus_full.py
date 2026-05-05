#!/usr/bin/env python3
"""v0.4.2 Step 6 — full balanced synth corpus run.

Wraps generate_corpus.py and loops over all 16 IE invoice template families,
producing N docs per family for a total of ~16*N. Each per-template subprocess
gets a derived seed so identical re-runs are reproducible while different
families don't collide on doc-id.

Default --per-template 32 yields 512 docs (PDR step 6 target ~500).

Usage:
    python scripts/generate_corpus_full.py --per-template 32 --master-seed 43 \\
        --out corpus/synth-full-v0.4.2-r1/

Run timing on Vega 8 / CPU-only WeasyPrint: ~1-2s per doc, ~10-15 min total
for 512-doc balanced run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from synth import content_engine  # noqa: E402

TEMPLATE_FAMILIES: list[str] = sorted(content_engine.BUILDERS.keys())


@dataclass(frozen=True)
class FamilyResult:
    family: str
    requested: int
    generated: int
    elapsed_s: float
    return_code: int


def run_one_family(
    family: str,
    count: int,
    seed: int,
    out_root: Path,
    validate: bool,
) -> FamilyResult:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_corpus.py"),
        "--template", family,
        "--count", str(count),
        "--seed", str(seed),
        "--out", str(out_root),
    ]
    if validate:
        cmd.append("--validate")

    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    family_dir = out_root if out_root.is_absolute() else REPO_ROOT / out_root
    family_dir = family_dir / family
    generated = len(list(family_dir.glob("*.pdf"))) if family_dir.exists() else 0

    if proc.returncode != 0:
        sys.stderr.write(f"\n!! {family} returned {proc.returncode}\n")
        sys.stderr.write(proc.stderr[-2000:] + "\n")
    return FamilyResult(family, count, generated, elapsed, proc.returncode)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-template", type=int, default=32,
                    help="Docs per template family (default 32 -> 512 total)")
    ap.add_argument("--master-seed", type=int, default=43,
                    help="Master seed; per-family seed = master*1000 + family_idx")
    ap.add_argument("--out", type=Path, default=Path("corpus/synth-full-v0.4.2-r1"),
                    help="Output dir relative to repo root")
    ap.add_argument("--families", nargs="+", default=None,
                    help="Subset of families (default: all 16)")
    ap.add_argument("--validate", action="store_true",
                    help="Validate every ground-truth JSON against output_schema.json")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="Optional path to write a JSON summary of per-family results")
    args = ap.parse_args()

    families = args.families or TEMPLATE_FAMILIES
    unknown = [f for f in families if f not in TEMPLATE_FAMILIES]
    if unknown:
        sys.exit(f"Unknown families: {unknown}\nAvailable: {TEMPLATE_FAMILIES}")

    print(f"# generate_corpus_full.py — {len(families)} families x {args.per_template} docs "
          f"= {len(families) * args.per_template} target")
    print(f"# master_seed={args.master_seed}  out={args.out}  validate={args.validate}")
    print()

    results: list[FamilyResult] = []
    t_start = time.monotonic()
    for idx, family in enumerate(families):
        family_seed = args.master_seed * 1000 + idx
        print(f"[{idx + 1:>2}/{len(families)}] {family:<32}  seed={family_seed}  ...", end=" ", flush=True)
        r = run_one_family(family, args.per_template, family_seed, args.out, args.validate)
        results.append(r)
        rate = r.generated / r.elapsed_s if r.elapsed_s > 0 else 0.0
        status = "OK " if r.return_code == 0 else "FAIL"
        print(f"{status} {r.generated}/{r.requested} in {r.elapsed_s:5.1f}s ({rate:.1f}/s)")

    total_elapsed = time.monotonic() - t_start
    total_requested = sum(r.requested for r in results)
    total_generated = sum(r.generated for r in results)
    failures = [r for r in results if r.return_code != 0]

    print()
    print(f"Total: {total_generated}/{total_requested} in {total_elapsed:.1f}s "
          f"({total_generated / total_elapsed:.1f}/s avg)")
    if failures:
        print(f"FAILURES: {len(failures)} family/families returned non-zero:")
        for r in failures:
            print(f"  - {r.family} rc={r.return_code} generated={r.generated}/{r.requested}")

    if args.summary_json:
        summary = {
            "master_seed": args.master_seed,
            "per_template": args.per_template,
            "out": str(args.out),
            "total_requested": total_requested,
            "total_generated": total_generated,
            "elapsed_s": total_elapsed,
            "results": [
                {
                    "family": r.family,
                    "requested": r.requested,
                    "generated": r.generated,
                    "elapsed_s": r.elapsed_s,
                    "return_code": r.return_code,
                }
                for r in results
            ],
        }
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"# summary written to {args.summary_json}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
