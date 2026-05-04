#!/usr/bin/env python3
"""v0.4.2 synth corpus generator — Step 5 orchestrator (prototype scope).

Per PDR v0.2 §7 Step 5 + 6. Walks one or more template families, calls
content_engine.build() to produce content payloads, calls render.render_to_pdf()
to render them, writes ground-truth JSON alongside each PDF.

PROTOTYPE SCOPE (v0.4.2-step1-prototype): tradesman_rct ONLY, no Augraphy
corruption (clean bucket only), no MinHash leakage gate (only relevant when
both train + eval shards exist), no DVC add (added in next batch).

Usage:
    python scripts/generate_corpus.py --template tradesman_rct --count 5 \
        --seed 42 --out corpus/synth-prototype/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent  # proof-fixtures/
sys.path.insert(0, str(REPO_ROOT))

from synth import content_engine, render               # noqa: E402
from synth.contracts import CONTRACTS_VERSION          # noqa: E402

SCHEMA_VERSION = "0.4"
LICENSE_TAG = "Apache-2.0"


def _git_rev() -> str:
    """Returns current git commit SHA (40-char hex) of proof-fixtures, or 'pending'
    if git unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if len(out) == 40:
            return out
    except Exception:
        pass
    return "pending"


def _doc_id(template_family: str, master_seed: int, idx: int) -> str:
    """Deterministic 12-char hex doc-id matching output_schema regex
    `^synth-[a-f0-9]{12}$`."""
    h = hashlib.sha256(f"{template_family}|{master_seed}|{idx}".encode()).hexdigest()
    return f"synth-{h[:12]}"


def _build_ground_truth(
    doc_id: str,
    pdf_path: Path,
    template_family: str,
    seed: int,
    content: dict[str, Any],
    provenance: dict[str, Any],
    commit_hash: str,
) -> dict[str, Any]:
    """Assemble a ground-truth JSON record matching output_schema.json (step0.1)."""
    # Critical fields (6 — INVOICE_FIELDS step0.1)
    gt_fields = {
        "vendor":         content["vendor_name"],
        "total":          content["total"],
        "vat":            content["vat"],
        "date":           content["invoice_date"],
        "subtotal":       content["subtotal"],
        "vendor_country": content["vendor_country"],
    }

    # Header fields — all HEADER_FIELDS keys, falling back to "" when content omits
    header_keys = (
        "vendor_name", "vendor_address_line1", "vendor_address_line2",
        "vendor_city", "vendor_eircode", "vendor_country", "vendor_phone",
        "vendor_email", "vendor_website", "vendor_vat_number", "vendor_iban",
        "vendor_bic", "vendor_logo_present", "invoice_number", "invoice_date",
        "due_date", "po_reference",
    )
    gt_header = {k: content.get(k, "") for k in header_keys}

    return {
        "doc_id": doc_id,
        "source_path": str(pdf_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "schema_version": SCHEMA_VERSION,
        "contracts_version": CONTRACTS_VERSION,
        "template_family": template_family,
        "corruption_bucket": "clean",  # prototype: no Augraphy yet
        "ground_truth_fields": gt_fields,
        "ground_truth_line_items": content["line_items"],
        "ground_truth_header": gt_header,
        "ground_truth_provenance": provenance,
        "synthetic_seed": seed,
        "synthetic_generator_commit": commit_hash,
        "license_tag": LICENSE_TAG,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--template", required=True, choices=sorted(content_engine.BUILDERS.keys()),
                    help="Template family name (prototype: tradesman_rct only)")
    ap.add_argument("--count", type=int, default=5, help="Docs to generate (default 5 — smoke)")
    ap.add_argument("--seed", type=int, default=42, help="Master seed (default 42)")
    ap.add_argument("--out", type=Path, default=Path("corpus/synth-prototype"),
                    help="Output dir relative to repo root")
    ap.add_argument("--validate", action="store_true",
                    help="Validate each ground-truth JSON against output_schema.json")
    args = ap.parse_args()

    out_dir = REPO_ROOT / args.out / args.template
    out_dir.mkdir(parents=True, exist_ok=True)

    commit_hash = _git_rev()
    validator = _load_schema_validator() if args.validate else None

    print(f"# generate_corpus.py — template={args.template} count={args.count} "
          f"seed={args.seed}")
    print(f"# contracts_version={CONTRACTS_VERSION} commit={commit_hash}")
    print(f"# out_dir={out_dir}")

    for i in range(args.count):
        seed = args.seed * 1000 + i
        doc_id = _doc_id(args.template, args.seed, i)
        content = content_engine.build(args.template, seed)

        pdf_path = out_dir / f"{doc_id}.pdf"
        provenance = render.render_to_pdf(args.template, content, pdf_path)

        gt = _build_ground_truth(doc_id, pdf_path, args.template, seed, content,
                                 provenance, commit_hash)
        gt_path = out_dir / f"{doc_id}.gt.json"
        gt_path.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")

        if validator:
            errs = list(validator.iter_errors(gt))
            if errs:
                print(f"  XX {doc_id} schema-fail: {[e.message for e in errs[:3]]}", file=sys.stderr)
                return 2

        print(f"  OK{doc_id}  pdf={provenance['file_size_bytes']:>5}B  "
              f"items={len(content['line_items'])}  total=€{content['total']:.2f}")

    print(f"\nWrote {args.count} docs to {out_dir}")
    return 0


def _load_schema_validator():
    """Returns a jsonschema validator for output_schema.json, or None if jsonschema isn't installed."""
    try:
        import jsonschema
        schema_path = REPO_ROOT / "synth" / "contracts" / "output_schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return jsonschema.Draft7Validator(schema)
    except ImportError:
        print("# (validator skipped — jsonschema not installed)", file=sys.stderr)
        return None


if __name__ == "__main__":
    sys.exit(main())
