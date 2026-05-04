"""Layout slot taxonomy — per-template slot positions for 16 IE seed layouts.

Each template family declares which slots it surfaces + their positional
hints. Templates are HTML+CSS (rendered via WeasyPrint) referencing these
slot names as `<span data-slot="vendor_name">`. The content engine fills
slots; the renderer rasterizes; the ground-truth emitter records the
slot-bbox map for downstream training.

Per PDR v0.2 §5 + Adam review 2026-05-04 — 16 templates × 4× augmentation
= 64 layout-pose permutations BEFORE corruption. Adding a new template
family requires bumping CONTRACTS_VERSION in __init__.py.

v0.4.2-step0.1 added: cafe_receipt, gp_medical, vet, solicitor_loe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class LayoutSlot:
    """One filled-in slot on a rendered document. Position hints are
    fractional (0-1) of page width/height; renderer applies absolute
    coordinates."""
    name: str                                 # e.g. "vendor_name"
    required: bool                            # must this template fill it?
    hint_x: tuple[float, float]               # min, max x fraction of page width
    hint_y: tuple[float, float]               # min, max y fraction of page height
    multiline: bool = False
    overflow_strategy: str = "truncate"       # truncate | wrap | shrink


@dataclass(frozen=True)
class TemplateFamily:
    name: str
    description: str
    slots: tuple[LayoutSlot, ...]
    page_size: str  # "A4" | "thermal_80mm"
    has_line_items: bool
    multi_vat_rate: bool
    typical_line_item_count: tuple[int, int]  # min, max range
    license_tag: str = "Apache-2.0"           # all self-generated synth = Apache-2.0


# Common slot definitions reused across templates
_VENDOR_HEADER_SLOTS = (
    LayoutSlot("vendor_logo_present", False, (0.05, 0.30), (0.02, 0.12)),
    LayoutSlot("vendor_name",         True,  (0.05, 0.50), (0.05, 0.15)),
    LayoutSlot("vendor_address_line1", False, (0.05, 0.50), (0.10, 0.18)),
    LayoutSlot("vendor_vat_number",   True,  (0.05, 0.95), (0.12, 0.22), False),
    LayoutSlot("invoice_number",      True,  (0.55, 0.95), (0.05, 0.15)),
    LayoutSlot("invoice_date",        True,  (0.55, 0.95), (0.10, 0.20)),
)

_FOOTER_TOTAL_SLOTS = (
    LayoutSlot("subtotal_amount",     False, (0.55, 0.95), (0.70, 0.85)),
    LayoutSlot("vat",                 True,  (0.55, 0.95), (0.75, 0.90)),
    LayoutSlot("total",               True,  (0.55, 0.95), (0.80, 0.95)),
    LayoutSlot("vendor_iban",         False, (0.05, 0.95), (0.85, 0.95), True),
)


TEMPLATE_FAMILIES: Final[tuple[TemplateFamily, ...]] = (
    TemplateFamily(
        name="tradesman_rct",
        description="Tradesman invoice with RCT principal-contractor reverse-charge wording",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("rct_clause", True, (0.05, 0.95), (0.60, 0.75), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=False,
        typical_line_item_count=(2, 8),
    ),
    TemplateFamily(
        name="restaurant_thermal",
        description="Restaurant receipt thermal-strip 80mm, multi-VAT (food=9%, alcohol=23%)",
        slots=(
            LayoutSlot("vendor_name",      True, (0.10, 0.90), (0.02, 0.10)),
            LayoutSlot("vendor_address_line1", True, (0.10, 0.90), (0.08, 0.16)),
            LayoutSlot("invoice_date",     True, (0.10, 0.90), (0.16, 0.22)),
            LayoutSlot("vat",              True, (0.10, 0.90), (0.85, 0.92)),
            LayoutSlot("total",            True, (0.10, 0.90), (0.92, 0.98)),
        ),
        page_size="thermal_80mm",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(3, 15),
    ),
    TemplateFamily(
        name="supermarket_per_letter_vat",
        description="Tesco/Dunnes/SuperValu per-letter VAT codes (A/B/C/D/Z)",
        slots=(
            LayoutSlot("vendor_name", True, (0.10, 0.90), (0.02, 0.08)),
            LayoutSlot("invoice_date", True, (0.10, 0.90), (0.10, 0.16)),
            LayoutSlot("vat", True, (0.10, 0.90), (0.88, 0.94)),
            LayoutSlot("total", True, (0.10, 0.90), (0.94, 0.99)),
        ),
        page_size="thermal_80mm",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(5, 25),
    ),
    TemplateFamily(
        name="professional_services",
        description="A4 letterhead, no line items, single fee + VAT",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("service_description", True, (0.05, 0.95), (0.40, 0.65), True),
        ),
        page_size="A4",
        has_line_items=False,
        multi_vat_rate=False,
        typical_line_item_count=(1, 1),
    ),
    TemplateFamily(
        name="utility_bill",
        description="Multi-period utility bill (ESB / Bord Gáis / Eir / Virgin Media)",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("billing_period", True, (0.05, 0.50), (0.20, 0.30), True),
            LayoutSlot("usage_table", True, (0.05, 0.95), (0.35, 0.60), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(2, 6),
    ),
    TemplateFamily(
        name="bank_statement",
        description="AIB/BOI/PTSB/Revolut/Wise/N26 transaction list, NO VAT (financial exempt)",
        slots=(
            LayoutSlot("vendor_name", True, (0.05, 0.50), (0.02, 0.10)),
            LayoutSlot("vendor_iban", True, (0.05, 0.95), (0.10, 0.18), True),
            LayoutSlot("invoice_date", True, (0.55, 0.95), (0.05, 0.15)),
            LayoutSlot("billing_period", True, (0.55, 0.95), (0.10, 0.18)),
            LayoutSlot("transaction_table", True, (0.05, 0.95), (0.25, 0.85), True),
            LayoutSlot("balance_total", True, (0.55, 0.95), (0.88, 0.95)),
        ),
        page_size="A4",
        has_line_items=True,  # transactions = line items
        multi_vat_rate=False,  # bank statements are VAT-exempt
        typical_line_item_count=(8, 40),
    ),
    TemplateFamily(
        name="mixed_rate_retailer",
        description="Multiple VAT rates per invoice with sub-totals per rate",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("vat_breakdown_table", True, (0.05, 0.95), (0.65, 0.82), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(4, 12),
    ),
    TemplateFamily(
        name="construction_supplier",
        description="Materials breakdown + delivery note + RCT line (Chadwicks / Heitons pattern)",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("delivery_note_ref", False, (0.05, 0.50), (0.20, 0.28)),
            LayoutSlot("materials_table", True, (0.05, 0.95), (0.30, 0.70), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=False,
        typical_line_item_count=(3, 10),
    ),
    TemplateFamily(
        name="credit_note",
        description="Credit note — negative amounts, references original invoice",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("references_invoice", True, (0.05, 0.95), (0.18, 0.28)),
            LayoutSlot("credit_reason", True, (0.05, 0.95), (0.30, 0.45), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=False,
        typical_line_item_count=(1, 4),
    ),
    TemplateFamily(
        name="photographed_receipt",
        description="Thermal-strip rendered then heavy-corrupted as photo (perspective + lens)",
        slots=(
            LayoutSlot("vendor_name", True, (0.10, 0.90), (0.02, 0.10)),
            LayoutSlot("invoice_date", True, (0.10, 0.90), (0.10, 0.16)),
            LayoutSlot("vat", True, (0.10, 0.90), (0.85, 0.92)),
            LayoutSlot("total", True, (0.10, 0.90), (0.92, 0.98)),
        ),
        page_size="thermal_80mm",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(2, 10),
    ),
    TemplateFamily(
        name="foreign_currency_mixed_vat",
        description="EUR + GBP cross-border with reverse-charge",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("currency_breakdown", True, (0.05, 0.95), (0.60, 0.75), True),
            LayoutSlot("reverse_charge_clause", True, (0.05, 0.95), (0.78, 0.86), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(2, 6),
    ),
    TemplateFamily(
        name="handwritten_override",
        description="Printed receipt with handwritten total + signature (GNHK glyph overlay)",
        slots=(
            LayoutSlot("vendor_name", True, (0.10, 0.90), (0.02, 0.10)),
            LayoutSlot("invoice_date", True, (0.10, 0.90), (0.10, 0.16)),
            LayoutSlot("printed_total", True, (0.10, 0.90), (0.70, 0.78)),
            LayoutSlot("handwritten_total", True, (0.10, 0.90), (0.80, 0.92)),  # GNHK overlay
            LayoutSlot("handwritten_signature", False, (0.50, 0.95), (0.92, 0.99)),
        ),
        page_size="thermal_80mm",
        has_line_items=True,
        multi_vat_rate=False,
        typical_line_item_count=(1, 6),
    ),
    # ── v0.4.2-step0.1 additions (Adam review 2026-05-04) ──
    TemplateFamily(
        name="cafe_receipt",
        description="Cafe / coffee-shop thermal receipt — single-VAT 9% (food-and-drink consumed on premises)",
        slots=(
            LayoutSlot("vendor_name", True, (0.10, 0.90), (0.02, 0.10)),
            LayoutSlot("vendor_address_line1", False, (0.10, 0.90), (0.08, 0.14)),
            LayoutSlot("invoice_date", True, (0.10, 0.90), (0.14, 0.20)),
            LayoutSlot("invoice_number", False, (0.10, 0.90), (0.18, 0.24)),
            LayoutSlot("vat", True, (0.10, 0.90), (0.85, 0.92)),
            LayoutSlot("total", True, (0.10, 0.90), (0.92, 0.98)),
        ),
        page_size="thermal_80mm",
        has_line_items=True,
        multi_vat_rate=False,
        typical_line_item_count=(1, 6),
    ),
    TemplateFamily(
        name="gp_medical",
        description="GP / medical-practice fee receipt — exempt VAT (medical services 0% per VATCA Sched 1)",
        slots=_VENDOR_HEADER_SLOTS + (
            LayoutSlot("service_description", True, (0.05, 0.95), (0.35, 0.55), True),
            LayoutSlot("subtotal_amount", False, (0.55, 0.95), (0.65, 0.78)),
            LayoutSlot("total", True, (0.55, 0.95), (0.78, 0.90)),
            LayoutSlot("vatca_attestation", False, (0.05, 0.95), (0.90, 0.97), True),
        ),
        page_size="A4",
        has_line_items=False,
        multi_vat_rate=False,  # medical exempt, no VAT line at all
        typical_line_item_count=(1, 1),
    ),
    TemplateFamily(
        name="vet",
        description="Veterinary practice — mixed VAT (consultation 23%, livestock medicines 4.8%, food 0%)",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("vat_breakdown_table", True, (0.05, 0.95), (0.60, 0.78), True),
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=True,
        typical_line_item_count=(2, 8),
    ),
    TemplateFamily(
        name="solicitor_loe",
        description="Solicitor letter-of-engagement — A4 letterhead with PA fee + outlay split, single 23% VAT",
        slots=_VENDOR_HEADER_SLOTS + _FOOTER_TOTAL_SLOTS + (
            LayoutSlot("service_description", True, (0.05, 0.95), (0.30, 0.55), True),
            LayoutSlot("rct_clause", False, (0.05, 0.95), (0.55, 0.65), True),  # outlay disclosure
        ),
        page_size="A4",
        has_line_items=True,
        multi_vat_rate=False,
        typical_line_item_count=(2, 6),
    ),
)


# Quick-lookup map for content engine
LAYOUT_SLOTS: Final[dict[str, TemplateFamily]] = {
    f.name: f for f in TEMPLATE_FAMILIES
}


# Sanity assertions at import time
assert len(TEMPLATE_FAMILIES) == 16, \
    f"Expected 16 IE seed templates per PDR v0.2 §5 + step0.1; got {len(TEMPLATE_FAMILIES)}"
assert all(f.license_tag == "Apache-2.0" for f in TEMPLATE_FAMILIES), \
    "All synth templates must be Apache-2.0 (no GPL/CC-BY-NC)"
assert len(LAYOUT_SLOTS) == len(TEMPLATE_FAMILIES), \
    f"TemplateFamily.name collision detected — LAYOUT_SLOTS has {len(LAYOUT_SLOTS)} entries vs {len(TEMPLATE_FAMILIES)} families"
