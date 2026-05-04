"""Slot/placeholder taxonomy referenced by templates + content engine + ground truth.

Per PDR v0.2 §7 Step 0: locking slot names BEFORE templates is required so
the content engine + render + ground truth all agree. Adding a new slot
requires bumping CONTRACTS_VERSION in __init__.py.
"""
from __future__ import annotations

from typing import Final


# Header-level fields (always present)
HEADER_FIELDS: Final[tuple[str, ...]] = (
    "vendor_name",
    "vendor_address_line1",
    "vendor_address_line2",
    "vendor_city",
    "vendor_eircode",
    "vendor_country",        # ISO-3166-1 alpha-2 (default "IE"); critical for foreign_currency_mixed_vat + cross-border RCT
    "vendor_phone",
    "vendor_email",
    "vendor_website",
    "vendor_vat_number",
    "vendor_iban",
    "vendor_bic",
    "vendor_logo_present",  # bool
    "invoice_number",
    "invoice_date",
    "due_date",
    "po_reference",
)

# Critical fields = LABEL TARGET for v0.4.2 synth corpus + eval scoring.
# v0.4.2-step0.1: extended from 4 → 6 (added subtotal, vendor_country) per
# Adam review 2026-05-04 + path A (D-V0.4.2-20-corrected).
#
# IMPORTANT — divergence is intentional:
#   - INVOICE_FIELDS (here)              = 6 fields, the ground-truth target
#   - extract.py CRITICAL_FIELDS         = 4 fields, the v0.4.1 verifier baseline
#     (lives at proof-fixtures/scripts/extract.py:57, NOT document-ops-portal)
#   - score.py CRITICAL_FIELDS           = 4 fields on legacy fixtures, 6 on v0.4.2
#     (split logic added in step 9 baseline-run scaffold)
# The v0.4.2 thesis IS that LoRA training closes the 4 → 6 gap. Lock-step
# bump would let prompt-engineering, not training, account for the lift —
# preempting the experiment. Sync layer (extract.py 4 → 6 + extractor
# logic + cross-field subtotal+vat≈total check + ground-truth retrofit)
# DEFERRED to v0.4.3 per D-V0.4.2-23 (also depends on D-V0.4.2-18 verifier
# refactor for vision-crop path).
INVOICE_FIELDS: Final[tuple[str, ...]] = (
    "vendor",          # vendor_name normalized
    "total",           # gross including VAT, EUR
    "vat",             # VAT amount, EUR
    "date",            # invoice_date in DD/MM/YYYY
    "subtotal",        # net before VAT, EUR — required for cross-field check (subtotal + vat ≈ total ±0.01)
    "vendor_country",  # ISO-3166-1 alpha-2; "IE" domestic, anything else = foreign_currency_mixed_vat path
)

# Line-item-level fields
LINE_ITEM_FIELDS: Final[tuple[str, ...]] = (
    "description",
    "quantity",
    "unit_price",
    "vat_rate_pct",
    "vat_letter_code",  # for retailers that use per-letter codes (A/B/C/Z)
    "amount_net",
    "amount_vat",
    "amount_gross",
)

# Provenance fields (mirrors extract.py IngestProvenance v0.4.1)
PROVENANCE_FIELDS: Final[tuple[str, ...]] = (
    "original_form",       # digital_native | image_only_pdf | paper_scan | photo
    "source_engine",
    "file_sha256",
    "file_size_bytes",
    "ingest_timestamp_utc",
    "page_count",
    "pdf_metadata",
    "vatca_attestation",
)

# Ground-truth output schema — what generate_corpus.py emits per doc
# (parallel to a real extract.py output, used for self-supervised training)
GROUND_TRUTH_KEYS: Final[tuple[str, ...]] = (
    "doc_id",
    "source_path",
    "schema_version",
    "contracts_version",
    "template_family",
    "corruption_bucket",
    "ground_truth_fields",  # dict with all INVOICE_FIELDS values
    "ground_truth_line_items",  # list of LineItem dicts
    "ground_truth_header",  # dict with all HEADER_FIELDS values
    "ground_truth_provenance",  # dict mirroring IngestProvenance
    "synthetic_seed",
    "synthetic_generator_commit",
    "license_tag",
)
