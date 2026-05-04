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

# Critical fields (the four extract.py CRITICAL_FIELDS — MUST match v0.4.1 schema)
INVOICE_FIELDS: Final[tuple[str, ...]] = (
    "vendor",       # vendor_name normalized
    "total",        # gross including VAT, EUR
    "vat",          # VAT amount, EUR
    "date",         # invoice_date in DD/MM/YYYY
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
