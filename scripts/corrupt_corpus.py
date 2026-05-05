#!/usr/bin/env python3
"""v0.4.2 Step 4 — bulk-corrupt a synth corpus through the 4-bucket Augraphy
pipeline.

Walks a source corpus dir, seeds bucket selection per-doc via
`bucket_for_seed()` (30/45/20/5 clean/light/heavy/very-heavy), writes
corrupted PDFs to dest. Ground-truth JSONs are copied unchanged but the
`corruption_bucket` field is rewritten to match the new bucket so downstream
scoring can stratify. Output structure mirrors source.

Usage:
    python scripts/corrupt_corpus.py \\
        --source corpus/synth-full-v0.4.2-r1 \\
        --dest   corpus/synth-full-corrupted-v0.4.2-r1 \\
        --master-seed 43

Run timing on Vega 8 / Augraphy CPU: ~0.5-2 s per doc for light bucket,
~3-5 s for heavy/very-heavy. ~10-15 min for 512 docs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from synth.corrupt import bucket_for_seed, corrupt  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, required=True,
                    help="Source corpus dir (relative to repo root unless absolute)")
    ap.add_argument("--dest", type=Path, required=True,
                    help="Dest corpus dir (relative to repo root unless absolute)")
    ap.add_argument("--master-seed", type=int, default=43,
                    help="Master seed; per-doc seed = master*1_000_000 + index")
    ap.add_argument("--dpi", type=int, default=200,
                    help="Render DPI (Augraphy works in pixel space)")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="Optional summary JSON path")
    args = ap.parse_args()

    src = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    dst = args.dest if args.dest.is_absolute() else REPO_ROOT / args.dest
    if not src.exists():
        sys.exit(f"--source dir not found: {src}")

    # Discover all family/doc.pdf pairs
    pdfs = sorted(src.glob("*/*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs under {src}/<family>/*.pdf")

    print(f"# corrupt_corpus.py — {len(pdfs)} docs · master_seed={args.master_seed}")
    print(f"# src={src}")
    print(f"# dst={dst}")
    print()

    bucket_counts: Counter[str] = Counter()
    failures: list[tuple[str, str]] = []
    t_start = time.monotonic()

    for idx, in_pdf in enumerate(pdfs):
        family = in_pdf.parent.name
        doc_seed = args.master_seed * 1_000_000 + idx
        bucket = bucket_for_seed(doc_seed)
        out_pdf = dst / family / in_pdf.name
        out_gt = dst / family / in_pdf.with_suffix(".gt.json").name.replace(".pdf.gt.json", ".gt.json")

        try:
            t_doc = time.monotonic()
            res = corrupt(in_pdf, bucket, out_pdf, seed=doc_seed, dpi=args.dpi)
            dt = time.monotonic() - t_doc
            bucket_counts[bucket] += 1

            # Copy + rewrite GT JSON (mark corruption_bucket)
            gt_in = in_pdf.with_name(in_pdf.stem + ".gt.json")
            if gt_in.exists():
                gt = json.loads(gt_in.read_text(encoding="utf-8"))
                gt["corruption_bucket"] = bucket
                # update source_path to reflect dest
                rel = out_pdf.resolve().relative_to(REPO_ROOT.resolve())
                gt["source_path"] = str(rel).replace("\\", "/")
                out_gt.parent.mkdir(parents=True, exist_ok=True)
                out_gt.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

            if (idx + 1) % 32 == 0 or (idx + 1) == len(pdfs):
                print(f"  {idx + 1:>4}/{len(pdfs)}  {family:<32}  bucket={bucket:<10}  {res.file_size_bytes // 1024:>4}KB  {dt:5.1f}s")
        except Exception as e:
            failures.append((str(in_pdf.relative_to(src)), repr(e)))
            sys.stderr.write(f"!! {in_pdf.name} ({bucket}): {e}\n")

    elapsed = time.monotonic() - t_start
    print()
    print(f"Total: {sum(bucket_counts.values())}/{len(pdfs)} corrupted in {elapsed:.1f}s "
          f"({sum(bucket_counts.values()) / elapsed:.1f}/s avg)")
    print(f"Bucket distribution: {dict(bucket_counts)}")
    if failures:
        print(f"FAILURES: {len(failures)}")
        for path, err in failures[:5]:
            print(f"  - {path}: {err[:120]}")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps({
            "src": str(args.source),
            "dst": str(args.dest),
            "master_seed": args.master_seed,
            "total_in": len(pdfs),
            "total_out": sum(bucket_counts.values()),
            "elapsed_s": elapsed,
            "buckets": dict(bucket_counts),
            "failures": [{"path": p, "error": e} for p, e in failures],
        }, indent=2), encoding="utf-8")
        print(f"# summary written to {args.summary_json}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
