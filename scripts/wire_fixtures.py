#!/usr/bin/env python3
"""Wire a synth corpus into fixtures/ for extract.py + score.py scoring.

extract.py expects:  fixtures/{set}/samples/*.pdf
score.py expects:    fixtures/{set}/ground-truth/*.json

This script bridges by HARDLINKING (no admin needed on Windows for hardlinks
within the same volume) the synth corpus outputs into fixtures/ folders
named `{prefix}{template_family}`.

Re-runnable. Idempotent — replaces existing links.

Usage:
    # Default: wire smoke corpus (synth-prototype/) into fixtures/synth-{family}/
    python scripts/wire_fixtures.py
    python scripts/wire_fixtures.py tradesman_rct

    # Wire full v0.4.2 r1 corpus into fixtures/synth-full-{family}/
    python scripts/wire_fixtures.py --source corpus/synth-full-v0.4.2-r1 \\
        --prefix synth-full-

Per Step 9 (PDR §7) prep: this is what `eval/run_baseline.py` will run before
calling extract.py on the synth corpus to measure baseline GLM-OCR Q8 field-EM.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = REPO_ROOT / "corpus" / "synth-prototype"
DEFAULT_FIXTURE_PREFIX = "synth-"
FIXTURES_DIR = REPO_ROOT / "fixtures"

# Module-level state set by main() so wire_template() can stay simple
CORPUS_DIR: Path = DEFAULT_CORPUS_DIR
FIXTURE_PREFIX: str = DEFAULT_FIXTURE_PREFIX


def _make_extract_groundtruth(synth_gt: dict) -> dict:
    """Translate v0.4.2 synth GT (output_schema.json) into extract.py's
    score.py-expected shape: {doc_id, fields: [{field_id, expected_value}, ...]}.

    Per Path A (D-V0.4.2-20-corrected): score.py CRITICAL_FIELDS = 4 on
    legacy fixtures, 6 on v0.4.2 corpus. Emit all 6 critical fields here;
    score.py will read its own CRITICAL_FIELDS to decide which to score.
    """
    gt_fields = synth_gt["ground_truth_fields"]
    return {
        "doc_id": synth_gt["doc_id"],
        "source_note": (
            f"Synth corpus v{synth_gt['contracts_version']} "
            f"template={synth_gt['template_family']} "
            f"bucket={synth_gt['corruption_bucket']} "
            f"seed={synth_gt['synthetic_seed']}"
        ),
        "fields": [
            {"field_id": "vendor",         "expected_value": gt_fields["vendor"]},
            {"field_id": "total",          "expected_value": gt_fields["total"]},
            {"field_id": "vat",            "expected_value": gt_fields["vat"]},
            {"field_id": "date",           "expected_value": gt_fields["date"]},
            {"field_id": "subtotal",       "expected_value": gt_fields["subtotal"]},
            {"field_id": "vendor_country", "expected_value": gt_fields["vendor_country"]},
        ],
    }


def wire_template(template_family: str) -> tuple[int, int]:
    """Wire one template's synth output into fixtures/synth-{family}/.
    Returns (n_pdfs_linked, n_gt_translated)."""
    src = CORPUS_DIR / template_family
    if not src.exists():
        raise FileNotFoundError(f"No synth corpus at {src}; run generate_corpus.py first")

    set_slug = f"{FIXTURE_PREFIX}{template_family}"
    samples_dir = FIXTURES_DIR / set_slug / "samples"
    gt_dir = FIXTURES_DIR / set_slug / "ground-truth"
    samples_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    n_pdfs = 0
    n_gt = 0
    for pdf in sorted(src.glob("*.pdf")):
        target = samples_dir / pdf.name
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            target.hardlink_to(pdf)  # Python 3.10+
        except OSError:
            shutil.copy2(pdf, target)  # cross-volume fallback
        n_pdfs += 1

    for gt_path in sorted(src.glob("*.gt.json")):
        synth_gt = json.loads(gt_path.read_text(encoding="utf-8"))
        translated = _make_extract_groundtruth(synth_gt)
        out_name = gt_path.name.replace(".gt.json", ".pdf").replace(".pdf", ".json")
        # extract.py / score.py convention: ground-truth filename matches result filename
        out_name = synth_gt["doc_id"] + ".json"
        out_path = gt_dir / out_name
        out_path.write_text(json.dumps(translated, indent=2, ensure_ascii=False), encoding="utf-8")
        n_gt += 1

    return n_pdfs, n_gt


def main() -> int:
    global CORPUS_DIR, FIXTURE_PREFIX
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("templates", nargs="*",
                    help="Subset of template family names (default: all dirs in --source)")
    ap.add_argument("--source", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="Synth corpus dir (relative to repo root unless absolute)")
    ap.add_argument("--prefix", default=DEFAULT_FIXTURE_PREFIX,
                    help="Fixture set prefix (e.g. 'synth-' or 'synth-full-')")
    args = ap.parse_args()

    CORPUS_DIR = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    FIXTURE_PREFIX = args.prefix

    if not CORPUS_DIR.exists():
        sys.exit(f"--source dir not found: {CORPUS_DIR}")

    templates = args.templates or sorted(
        d.name for d in CORPUS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")
    )

    total_pdfs = 0
    total_gt = 0
    for tpl in templates:
        try:
            n_pdfs, n_gt = wire_template(tpl)
            print(f"  OK {FIXTURE_PREFIX}{tpl:32s}  pdfs={n_pdfs}  gt={n_gt}")
            total_pdfs += n_pdfs
            total_gt += n_gt
        except FileNotFoundError as e:
            print(f"  XX {tpl:32s}  {e}", file=sys.stderr)

    print(f"\nWired {len(templates)} templates: {total_pdfs} PDFs + {total_gt} GT JSONs "
          f"into fixtures/{FIXTURE_PREFIX}*/")
    if templates:
        print(f"Run scoring: python scripts/score.py --set {FIXTURE_PREFIX}{templates[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
