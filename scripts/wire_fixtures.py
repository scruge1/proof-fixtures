#!/usr/bin/env python3
"""Wire synth-prototype corpus into fixtures/ for extract.py + score.py scoring.

extract.py expects:  fixtures/{set}/samples/*.pdf
score.py expects:    fixtures/{set}/ground-truth/*.json

This script bridges by HARDLINKING (no admin needed on Windows for hardlinks
within the same volume) the synth-prototype outputs into fixtures/ folders
named `synth-{template_family}`.

Re-runnable. Idempotent — replaces existing links.

Usage:
    python scripts/wire_fixtures.py                # wire all 16 templates
    python scripts/wire_fixtures.py tradesman_rct  # wire one template

Per Step 9 (PDR §7) prep: this is what `eval/run_baseline.py` will run before
calling extract.py on the synth corpus to measure baseline GLM-OCR Q8 field-EM.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "corpus" / "synth-prototype"
FIXTURES_DIR = REPO_ROOT / "fixtures"


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

    set_slug = f"synth-{template_family}"
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
    if len(sys.argv) > 1:
        templates = sys.argv[1:]
    else:
        templates = sorted([d.name for d in CORPUS_DIR.iterdir() if d.is_dir()])

    total_pdfs = 0
    total_gt = 0
    for tpl in templates:
        try:
            n_pdfs, n_gt = wire_template(tpl)
            print(f"  OK synth-{tpl:32s}  pdfs={n_pdfs}  gt={n_gt}")
            total_pdfs += n_pdfs
            total_gt += n_gt
        except FileNotFoundError as e:
            print(f"  XX {tpl:32s}  {e}", file=sys.stderr)

    print(f"\nWired {len(templates)} templates: {total_pdfs} PDFs + {total_gt} GT JSONs")
    print(f"Run scoring: python scripts/score.py --set synth-{templates[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
