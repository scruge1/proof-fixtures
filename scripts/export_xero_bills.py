#!/usr/bin/env python3
"""Document Ops -> Xero supplier-bills CSV mapper.

Closes round 3 PDR D-LIFE-18: ONE mapper that turns Document Ops Extraction
output into a CSV the customer pastes straight into Xero's "Manual Bills
+ Credit Notes" import (Xero IE region, supplier-bill flow). Sage IE
Quick-Entry + Surf Accounts companion outputs are TODO until first paying
customer on each.

Live verification 2026-05-07: Adam's CallMeIE Xero tenant
(go.xero.com/!tV0dM) has only the 4 default tax rates ("Sales Tax on
Imports", "Tax Exempt", "Tax on Purchases", "Tax on Sales"). Real customer
tenants will have the IE-specific VAT rates configured ("VAT on Expenses",
"Zero Rated Expenses", etc. per Xero IE chart of accounts setup).

Schema (Xero "Manual Bills + Credit Notes" CSV — IE / UK region):
    ContactName, EmailAddress, POAddressLine1, POAddressLine2,
    POAddressLine3, POAddressLine4, POCity, PORegion, POPostalCode,
    POCountry, InvoiceNumber, Reference, InvoiceDate, DueDate, Total,
    InventoryItemCode, Description, Quantity, UnitAmount, Discount,
    AccountCode, TaxType, TaxAmount, TrackingName1, TrackingOption1,
    TrackingName2, TrackingOption2, Currency, BrandingTheme

Date format: DD/MM/YYYY (must match Xero org's regional locale; Irish org
default).

Per-customer config in `proof-fixtures/customers/{slug}/xero-config.json`:
    {
        "default_account_code": "404",                    # Office Expenses
        "default_tax_type": "VAT on Expenses",            # 23% IE Standard
        "default_due_days": 30,
        "default_currency": "EUR",
        "tax_rate_map": {                                  # extracted -> Xero
            "23": "VAT on Expenses",
            "13.5": "VAT 13.5% on Expenses",
            "9": "VAT 9% on Expenses",
            "4.8": "VAT 4.8% on Expenses",
            "0": "Zero Rated Expenses",
            "RC": "Reverse Charge Expenses (20%)"
        }
    }

When this file is missing, mapper falls back to canonical defaults below
and prints a warning. Customer's first run = blocking gate to fill in
xero-config.json with their actual TaxType strings + AccountCodes.

Usage:
    # All this month's clean rows for one customer
    python scripts/export_xero_bills.py --tenant <slug>

    # Specific extraction IDs (e.g. for re-export after correction)
    python scripts/export_xero_bills.py --tenant <slug> \\
        --extraction-ids 0a1b2c... 0d3e4f...

    # Smoke test against synth corpus (uses fake customer config)
    python scripts/export_xero_bills.py --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# ----------------------------------------------------------------------------
# Canonical Xero IE strings (defaults if customer didn't supply xero-config.json)
# ----------------------------------------------------------------------------

XERO_BILLS_HEADER = [
    "ContactName", "EmailAddress",
    "POAddressLine1", "POAddressLine2", "POAddressLine3", "POAddressLine4",
    "POCity", "PORegion", "POPostalCode", "POCountry",
    "InvoiceNumber", "Reference", "InvoiceDate", "DueDate", "Total",
    "InventoryItemCode", "Description", "Quantity", "UnitAmount", "Discount",
    "AccountCode", "TaxType", "TaxAmount",
    "TrackingName1", "TrackingOption1", "TrackingName2", "TrackingOption2",
    "Currency", "BrandingTheme",
]

CANONICAL_DEFAULTS: dict[str, Any] = {
    # Sensible Xero IE defaults a freshly-onboarded Irish business gets after
    # selecting "Ireland" + "Limited Company" tax setup. Customer overrides
    # via xero-config.json.
    "default_account_code": "404",        # 404 Office Expenses (Xero IE preset)
    "default_tax_type": "VAT on Expenses",  # 23% IE Standard purchases
    "default_due_days": 30,
    "default_currency": "EUR",
    "tax_rate_map": {
        "23":   "VAT on Expenses",
        "13.5": "VAT 13.5% on Expenses",
        "9":    "VAT 9% on Expenses",
        "4.8":  "VAT 4.8% on Expenses",
        "0":    "Zero Rated Expenses",
        # Construction RCT — bookkeeper still adds the manual journal per
        # _research/2026-05-06-docops-accounting-csv-gotchas.md Q4. CSV row
        # uses Reverse Charge code for the bill itself.
        "RC":   "Reverse Charge Expenses (20%)",
        # EU intra-community
        "EC":   "EC Acquisitions (Zero Rated)",
        # Catch-all when extractor returned no VAT label
        "":     "Zero Rated Expenses",
    },
}


# ----------------------------------------------------------------------------
# Data shapes
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class XeroConfig:
    default_account_code: str
    default_tax_type: str
    default_due_days: int
    default_currency: str
    tax_rate_map: dict[str, str]
    used_defaults: bool

    @classmethod
    def load(cls, path: Path | None) -> "XeroConfig":
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                default_account_code=str(data.get(
                    "default_account_code", CANONICAL_DEFAULTS["default_account_code"])),
                default_tax_type=str(data.get(
                    "default_tax_type", CANONICAL_DEFAULTS["default_tax_type"])),
                default_due_days=int(data.get(
                    "default_due_days", CANONICAL_DEFAULTS["default_due_days"])),
                default_currency=str(data.get(
                    "default_currency", CANONICAL_DEFAULTS["default_currency"])),
                tax_rate_map={**CANONICAL_DEFAULTS["tax_rate_map"], **data.get("tax_rate_map", {})},
                used_defaults=False,
            )
        return cls(
            default_account_code=CANONICAL_DEFAULTS["default_account_code"],
            default_tax_type=CANONICAL_DEFAULTS["default_tax_type"],
            default_due_days=CANONICAL_DEFAULTS["default_due_days"],
            default_currency=CANONICAL_DEFAULTS["default_currency"],
            tax_rate_map=dict(CANONICAL_DEFAULTS["tax_rate_map"]),
            used_defaults=True,
        )


# ----------------------------------------------------------------------------
# Mapping logic
# ----------------------------------------------------------------------------

def _format_date(raw: str | None) -> str:
    """Coerce extracted date string to DD/MM/YYYY (Xero IE locale)."""
    if not raw:
        return ""
    raw = str(raw).strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%d/%m/%Y")
        except ValueError:
            continue
    return raw  # leave alone if we can't parse — Xero will reject and bookkeeper sees the raw value


def _due_date(invoice_date_dmy: str, default_days: int) -> str:
    if not invoice_date_dmy:
        return ""
    try:
        dt = datetime.strptime(invoice_date_dmy, "%d/%m/%Y")
        return (dt + timedelta(days=default_days)).strftime("%d/%m/%Y")
    except ValueError:
        return ""


def _resolve_tax_type(vat_rate_pct: Any, country_hint: str | None, cfg: XeroConfig) -> str:
    """Map our extracted vat-rate label to a Xero TaxType string."""
    if vat_rate_pct is None:
        return cfg.tax_rate_map.get("", cfg.default_tax_type)
    key = str(vat_rate_pct).strip().rstrip("%").rstrip(".0").rstrip(".")
    # Special cases
    if country_hint and country_hint.upper() not in {"IE", "EI", "IRELAND"}:
        # EU intra-community: customer is in another EU member state — RC code
        if country_hint.upper() in {"GB", "UK", "DE", "FR", "ES", "IT", "NL", "BE", "DK", "SE", "FI", "PL", "PT", "AT"}:
            return cfg.tax_rate_map.get("EC", cfg.default_tax_type)
    return cfg.tax_rate_map.get(key, cfg.default_tax_type)


def extraction_to_row(extraction: dict[str, Any], cfg: XeroConfig) -> dict[str, str]:
    """Map ONE Document Ops Extraction record (the JSON the portal stores)
    to a Xero supplier-bill CSV row. Pure function — easy to test."""
    payload = extraction.get("payload") or {}
    vendor = str(payload.get("vendor") or "").strip()
    total = payload.get("total")
    vat = payload.get("vat")
    subtotal = payload.get("subtotal")
    invoice_no = str(payload.get("invoice_number") or extraction.get("doc_id") or "").strip()
    raw_date = payload.get("date") or payload.get("invoice_date")
    invoice_dmy = _format_date(raw_date)
    due_dmy = _due_date(invoice_dmy, cfg.default_due_days)
    vendor_country = str(payload.get("vendor_country") or "").strip()
    vat_rate = payload.get("vat_rate") or payload.get("vat_rate_pct")
    tax_type = _resolve_tax_type(vat_rate, vendor_country, cfg)

    # If extractor gave subtotal (net) use it as UnitAmount; else fallback to total - vat
    unit_amount = subtotal
    if unit_amount is None and total is not None:
        try:
            unit_amount = float(total) - (float(vat) if vat is not None else 0.0)
        except (TypeError, ValueError):
            unit_amount = total

    return {
        "ContactName":      vendor or "Unknown supplier",
        "EmailAddress":     "",
        "POAddressLine1":   "",
        "POAddressLine2":   "",
        "POAddressLine3":   "",
        "POAddressLine4":   "",
        "POCity":           "",
        "PORegion":         "",
        "POPostalCode":     "",
        "POCountry":        vendor_country,
        "InvoiceNumber":    invoice_no,
        "Reference":        f"DocOps:{extraction.get('doc_id', '')}",
        "InvoiceDate":      invoice_dmy,
        "DueDate":          due_dmy,
        "Total":            _money(total),
        "InventoryItemCode": "",
        "Description":      str(payload.get("description") or vendor or "Supplier invoice"),
        "Quantity":         "1",
        "UnitAmount":       _money(unit_amount),
        "Discount":         "",
        "AccountCode":      cfg.default_account_code,
        "TaxType":          tax_type,
        "TaxAmount":        _money(vat),
        "TrackingName1":    "",
        "TrackingOption1":  "",
        "TrackingName2":    "",
        "TrackingOption2":  "",
        "Currency":         str(payload.get("currency") or cfg.default_currency),
        "BrandingTheme":    "",
    }


def _money(v: Any) -> str:
    """Format a money value the way Xero accepts: bare number, no currency
    symbol, no thousand separator, dot decimal."""
    if v is None or v == "":
        return ""
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        s = str(v).replace("€", "").replace("£", "").replace(",", "").strip()
        try:
            return f"{float(s):.2f}"
        except ValueError:
            return s


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _smoke_test() -> int:
    """Exercise the mapper on 8 synth corpus extractions across template
    families. Pure-Python; no DB required. Output goes to stdout."""
    samples = [
        # tradesman_rct: standard 23%
        {"doc_id": "synth-aaaa11", "payload": {
            "vendor": "Murphy Plumbing & Heating",
            "vendor_country": "IE",
            "invoice_number": "INV-RCT-2026-0142",
            "date": "12/03/2026",
            "subtotal": 850.00, "vat": 195.50, "total": 1045.50,
            "vat_rate": "23",
            "description": "Boiler service + replacement valve",
            "currency": "EUR",
        }},
        # cafe_receipt: 9% food
        {"doc_id": "synth-bbbb22", "payload": {
            "vendor": "Boyle's Cafe Limerick",
            "vendor_country": "IE",
            "invoice_number": "RCP-90218",
            "date": "2026-04-04",
            "subtotal": 18.34, "vat": 1.65, "total": 19.99,
            "vat_rate": "9",
        }},
        # gp_medical: VAT exempt
        {"doc_id": "synth-cccc33", "payload": {
            "vendor": "Dr. Smith Health Centre",
            "vendor_country": "IE",
            "invoice_number": "GP-2026-318",
            "date": "01/05/2026",
            "subtotal": 60.00, "vat": 0.00, "total": 60.00,
            "vat_rate": "0",
        }},
        # foreign_currency_mixed_vat: GBP + EU intra-community reverse charge
        {"doc_id": "synth-dddd44", "payload": {
            "vendor": "MacInerney Limited",
            "vendor_country": "GB",
            "invoice_number": "INV-INT-2026-091",
            "date": "10/04/2026",
            "subtotal": 1200.00, "vat": 0.00, "total": 1200.00,
            "vat_rate": "0",
            "currency": "GBP",
        }},
        # construction_supplier: 13.5% materials
        {"doc_id": "synth-eeee55", "payload": {
            "vendor": "Chadwicks Builders Merchants",
            "vendor_country": "IE",
            "invoice_number": "CHA-2026-44217",
            "date": "15/03/2026",
            "subtotal": 2200.00, "vat": 297.00, "total": 2497.00,
            "vat_rate": "13.5",
        }},
        # restaurant_thermal: mixed-VAT (food 9 + alcohol 23)
        {"doc_id": "synth-ffff66", "payload": {
            "vendor": "Clancys on Grand Parade",
            "vendor_country": "IE",
            "invoice_number": "REST-887",
            "date": "21/03/2026",
            "subtotal": 87.40, "vat": 14.62, "total": 102.02,
            "vat_rate": "23",
        }},
        # credit_note: negative
        {"doc_id": "synth-gggg77", "payload": {
            "vendor": "Optident Limerick",
            "vendor_country": "IE",
            "invoice_number": "CN-0091",
            "date": "20/04/2026",
            "subtotal": -150.00, "vat": -34.50, "total": -184.50,
            "vat_rate": "23",
            "description": "Credit note: returned dental supplies",
        }},
        # mixed_rate_retailer: blended
        {"doc_id": "synth-hhhh88", "payload": {
            "vendor": "Mulqueen Hardware",
            "vendor_country": "IE",
            "invoice_number": "MUL-72209",
            "date": "11/04/2026",
            "subtotal": 412.00, "vat": 75.00, "total": 487.00,
            "vat_rate": "23",
        }},
    ]
    cfg = XeroConfig.load(None)
    sys.stderr.write(
        f"[smoke] using canonical defaults (no xero-config.json) — "
        f"customer org must have these TaxType strings configured: "
        f"{sorted(set(cfg.tax_rate_map.values()))}\n"
    )

    writer = csv.DictWriter(sys.stdout, fieldnames=XERO_BILLS_HEADER)
    writer.writeheader()
    for ext in samples:
        writer.writerow(extraction_to_row(ext, cfg))
    return 0


def _export_for_tenant(tenant_slug: str, extraction_ids: list[str] | None,
                       cfg_path: Path | None, out_path: Path | None) -> int:
    """Read extractions from the Doc Ops Postgres for one tenant, emit CSV."""
    try:
        from sqlmodel import Session, select
    except ImportError:
        sys.exit("sqlmodel not installed — pip install sqlmodel before running --tenant mode")

    portal_root = REPO_ROOT.parent / "document-ops-portal"
    if not portal_root.exists():
        sys.exit(f"document-ops-portal repo not found at {portal_root}; cannot read extractions")
    sys.path.insert(0, str(portal_root))

    from app.db import engine  # type: ignore
    from app.models import Extraction, Tenant  # type: ignore

    cfg = XeroConfig.load(cfg_path)
    if cfg.used_defaults:
        sys.stderr.write(
            f"[warn] no xero-config.json at {cfg_path} — using canonical Xero IE defaults. "
            f"Customer should override TaxType strings + AccountCode for their org.\n"
        )

    with Session(engine) as session:
        tenant = session.exec(select(Tenant).where(Tenant.slug == tenant_slug)).first()
        if not tenant:
            sys.exit(f"tenant slug {tenant_slug!r} not found")

        q = select(Extraction).where(Extraction.tenant_id == tenant.id).where(Extraction.gate_passed == True)  # noqa: E712
        if extraction_ids:
            q = q.where(Extraction.id.in_(extraction_ids))
        rows = list(session.exec(q.order_by(Extraction.created_at)).all())

    if not rows:
        sys.stderr.write(f"[info] no clean extractions for tenant {tenant_slug}; emitting header-only CSV\n")

    fh = open(out_path, "w", encoding="utf-8", newline="") if out_path else sys.stdout
    try:
        writer = csv.DictWriter(fh, fieldnames=XERO_BILLS_HEADER)
        writer.writeheader()
        for ext in rows:
            d = {
                "doc_id": str(ext.id),
                "payload": ext.payload or {},
            }
            writer.writerow(extraction_to_row(d, cfg))
    finally:
        if out_path:
            fh.close()
            print(f"wrote {len(rows)} rows to {out_path}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--smoke", action="store_true",
                    help="Run mapper against 8 hand-built sample extractions; print CSV to stdout")
    ap.add_argument("--tenant", default=None,
                    help="Tenant slug to export this month's clean rows for")
    ap.add_argument("--extraction-ids", nargs="*", default=None,
                    help="Optional list of extraction UUIDs (for re-export after correction)")
    ap.add_argument("--config", type=Path, default=None,
                    help="Path to per-customer xero-config.json (default: customers/{slug}/xero-config.json)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV path (default: stdout)")
    args = ap.parse_args()

    if args.smoke:
        return _smoke_test()
    if not args.tenant:
        ap.error("Pass --tenant <slug> or --smoke")

    cfg_path = args.config or (REPO_ROOT / "customers" / args.tenant / "xero-config.json")
    return _export_for_tenant(args.tenant, args.extraction_ids, cfg_path, args.out)


if __name__ == "__main__":
    sys.exit(main())
