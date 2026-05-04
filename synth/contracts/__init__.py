"""v0.4.2 synth corpus contracts — schema + taxonomy + VAT constants + manifest.

Per CALLMEIE-DOCAI-V0.4.2-CORPUS-BOOTSTRAP-PDR.md §7 Step 0 (Codex §B
addition). All downstream synth-gen / public-fetch / training / eval steps
reference these contracts. Locking these BEFORE templates / content engine
is the precondition for clean parallelism between synth (steps 1-5) and
public corpus fetch (step 7).

Versioning: increment CONTRACTS_VERSION on any breaking change to the
ground-truth output schema or layout-slot taxonomy. Eval and training
runs pin the version they were built against.
"""
from __future__ import annotations

CONTRACTS_VERSION = "0.4.2-step0"

from .vat_constants import IE_VAT_RATES, IE_PER_LETTER_VAT, IE_VAT_NUMBER_REGEX
from .placeholder_taxonomy import (
    INVOICE_FIELDS,
    LINE_ITEM_FIELDS,
    PROVENANCE_FIELDS,
    HEADER_FIELDS,
)
from .layout_slots import LAYOUT_SLOTS, TEMPLATE_FAMILIES
from .manifest import build_dvc_manifest, build_license_manifest
from .leakage_gate import MinHashGate

__all__ = [
    "CONTRACTS_VERSION",
    "IE_VAT_RATES",
    "IE_PER_LETTER_VAT",
    "IE_VAT_NUMBER_REGEX",
    "INVOICE_FIELDS",
    "LINE_ITEM_FIELDS",
    "PROVENANCE_FIELDS",
    "HEADER_FIELDS",
    "LAYOUT_SLOTS",
    "TEMPLATE_FAMILIES",
    "build_dvc_manifest",
    "build_license_manifest",
    "MinHashGate",
]
