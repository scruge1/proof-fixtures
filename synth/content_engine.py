"""Content engine — fills template placeholders with realistic IE data.

Per PDR v0.2 §7 Step 2. Uses:
  - Faker `en_IE` locale (MIT)         — vendor name, address, eircode, phone, email
  - schwifty (MIT)                     — IE IBAN/BIC validation + generation
  - Custom IE VAT-number regex-gen     — IE\\d{7}[A-Z]{1,2} (mod-23 simplified per vat_constants.py)
  - Custom service-pool                — IE tradesman line-items (plumbing/electrical/carpentry/...)

Step 1 prototype scope = tradesman_rct ONLY. Other 15 template families
get their own build_<family>() in subsequent batches.

The build_*() functions return a dict whose keys MUST match the
template's `<span data-slot="...">` markers AND whose ground-truth
fields match `placeholder_taxonomy.INVOICE_FIELDS` (6) +
`HEADER_FIELDS` + `LINE_ITEM_FIELDS`.
"""
from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from faker import Faker
from schwifty import IBAN, BIC


# ─── IE-specific service catalogues ──────────────────────────────────────────

# Tradesman service descriptions — used by tradesman_rct template family.
# Real-world pricing in EUR (2026 IE market rates; Codex-reviewed for plausibility).
TRADESMAN_SERVICES: tuple[tuple[str, float, float], ...] = (
    # (description, min_unit_price_EUR, max_unit_price_EUR)
    ("Plumbing — leaking pipe repair",                       80.0, 220.0),
    ("Plumbing — boiler annual service",                    120.0, 250.0),
    ("Plumbing — replace bathroom tap set",                 180.0, 380.0),
    ("Electrical — fault diagnostic + repair",              110.0, 290.0),
    ("Electrical — consumer unit upgrade",                  450.0, 1200.0),
    ("Electrical — additional double socket installation",   75.0, 145.0),
    ("Carpentry — kitchen door re-hang",                     60.0, 140.0),
    ("Carpentry — skirting board replacement (per metre)",   25.0,  55.0),
    ("Carpentry — wardrobe assembly + fitting",             140.0, 320.0),
    ("Painting — interior room (labour only)",              250.0, 480.0),
    ("Painting — exterior render touch-up",                 380.0, 850.0),
    ("Roofing — slate replacement (per slate)",              35.0,  85.0),
    ("Roofing — gutter clearance + check",                  120.0, 240.0),
    ("Heating — radiator power-flush (per radiator)",        45.0,  95.0),
    ("Heating — thermostat installation",                   180.0, 320.0),
    ("Materials — copper pipe + fittings (lump sum)",        45.0, 280.0),
    ("Materials — paint + sundries",                         60.0, 180.0),
    ("Call-out + diagnostic charge",                         65.0,  95.0),
)


# ─── IE eircode + city pairs ─────────────────────────────────────────────────
# (Faker en_IE postcode generator does not always emit valid eircodes;
#  use this curated pool for headline IE cities + plausible routing keys.)

IE_CITY_EIRCODE_ROUTES: tuple[tuple[str, str], ...] = (
    ("Limerick",   "V94"),
    ("Limerick",   "V42"),
    ("Cork",       "T12"),
    ("Cork",       "T23"),
    ("Galway",     "H91"),
    ("Galway",     "H53"),
    ("Dublin 2",   "D02"),
    ("Dublin 8",   "D08"),
    ("Dublin 24",  "D24"),
    ("Waterford",  "X91"),
    ("Kilkenny",   "R95"),
    ("Sligo",      "F91"),
    ("Ennis",      "V95"),
    ("Tralee",     "V92"),
)


# ─── Faker per-locale instance (cached) ──────────────────────────────────────


def _faker(seed: int) -> Faker:
    """Returns a deterministic en_IE Faker. Seed locks all generated content
    so the same input seed always produces the same vendor + line items."""
    f = Faker("en_IE")
    Faker.seed(seed)
    f.seed_instance(seed)
    return f


# ─── Helpers ─────────────────────────────────────────────────────────────────


def gen_ie_vat_number(rng: random.Random) -> str:
    """Generate a regex-matching IE VAT number. Format IE + 7 digits + 1-2 letters.
    Per `synth/contracts/vat_constants.py` IE_VAT_NUMBER_REGEX. Mod-23 checksum
    is the placeholder simplified version (per v0.4.2-step0.1 — full impl v0.4.3).
    """
    digits = "".join(rng.choice(string.digits) for _ in range(7))
    suffix_len = rng.choice([1, 2])
    suffix = "".join(rng.choice(string.ascii_uppercase) for _ in range(suffix_len))
    return f"IE{digits}{suffix}"


def gen_ie_iban_bic(faker: Faker, rng: random.Random) -> tuple[str, str]:
    """Returns (iban, bic) — schwifty-validated IE IBAN + matching BIC.

    IE IBAN structure (per ISO 13616): IE + 2-digit checksum + 4-letter bank
    + 6-digit branch (sort code) + 8-digit account = 22 chars total.
    schwifty 2026.3.0 takes bank_code (4), branch_code (6), account_code (8)
    as SEPARATE args; passing the concat as account_code overflows the 8-char
    cap.

    Bank pool covers AIB / BOI / PTSB / Revolut / N26 (BWIC-issued IE codes).
    """
    bank_codes = ["AIBK", "BOFI", "IPBS", "REVO", "NTSB"]
    bank = rng.choice(bank_codes)
    branch = "".join(rng.choice(string.digits) for _ in range(6))
    account = "".join(rng.choice(string.digits) for _ in range(8))
    iban = IBAN.generate("IE", bank_code=bank, branch_code=branch, account_code=account)
    bic = str(iban.bic) if iban.bic else f"{bank}IE2D"
    return str(iban), bic


def gen_invoice_number(rng: random.Random, issue_date: date) -> str:
    """Format INV-YYYYMMDD-NNN per IE SME convention (3-digit running)."""
    n = rng.randint(1, 999)
    return f"INV-{issue_date.strftime('%Y%m%d')}-{n:03d}"


def gen_recent_date(rng: random.Random, max_days_back: int = 90) -> date:
    """Date within last N days. Uses RNG for determinism (NOT today())."""
    epoch_2026 = date(2026, 1, 1).toordinal()
    today_ord = date(2026, 5, 4).toordinal()  # locked anchor for reproducibility
    chosen = rng.randint(today_ord - max_days_back, today_ord)
    return date.fromordinal(chosen)


# ─── Builder: tradesman_rct ──────────────────────────────────────────────────


@dataclass(frozen=True)
class LineItem:
    description: str
    quantity: float
    unit_price: float
    vat_rate_pct: float       # 0.0 for RCT — multi_vat_rate=False
    vat_letter_code: str | None
    amount_net: float
    amount_vat: float
    amount_gross: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "vat_rate_pct": self.vat_rate_pct,
            "vat_letter_code": self.vat_letter_code,
            "amount_net": self.amount_net,
            "amount_vat": self.amount_vat,
            "amount_gross": self.amount_gross,
        }


def build_tradesman_rct(seed: int) -> dict[str, Any]:
    """Build a single tradesman_rct document content payload.

    Returns a dict suitable for both:
      - Jinja2 template fill (matches `<span data-slot="...">` keys), AND
      - Ground-truth JSON emission (matches placeholder_taxonomy schema).

    RCT semantics: VAT amount = 0 (reverse-charge applies), so total = subtotal.
    """
    rng = random.Random(seed)
    faker = _faker(seed)

    # Vendor identity
    vendor_name = faker.company()
    if not any(suffix in vendor_name for suffix in ("Ltd", "Limited", "& Sons", "& Co")):
        vendor_name = f"{vendor_name} Ltd"
    vendor_address_line1 = faker.street_address()
    # faker en_IE doesn't ship secondary_address(); use IE-style apt/unit fallback.
    vendor_address_line2 = (
        f"{rng.choice(['Unit', 'Apt', 'Suite'])} {rng.randint(1, 24)}"
        if rng.random() < 0.4 else ""
    )
    city, eircode_route = rng.choice(IE_CITY_EIRCODE_ROUTES)
    eircode_suffix = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(4))
    vendor_eircode = f"{eircode_route} {eircode_suffix}"
    vendor_phone = f"+353 {rng.randint(80, 89)}{rng.randint(0,9)} {rng.randint(100,999)} {rng.randint(1000,9999)}"
    vendor_email = faker.company_email()
    vendor_vat_number = gen_ie_vat_number(rng)
    vendor_iban, vendor_bic = gen_ie_iban_bic(faker, rng)

    # Invoice meta
    issue_date = gen_recent_date(rng)
    invoice_number = gen_invoice_number(rng, issue_date)
    due_date = (issue_date + timedelta(days=rng.choice([14, 21, 28, 30]))).strftime("%d/%m/%Y") \
        if rng.random() < 0.65 else ""
    po_reference = f"PO/{rng.randint(1000, 99999)}" if rng.random() < 0.40 else ""

    # Line items — 2-8 per layout_slots.tradesman_rct.typical_line_item_count
    n_items = rng.randint(2, 8)
    line_items: list[LineItem] = []
    for _ in range(n_items):
        desc, p_min, p_max = rng.choice(TRADESMAN_SERVICES)
        qty = float(rng.randint(1, 4))
        unit_price = round(rng.uniform(p_min, p_max), 2)
        amount_net = round(qty * unit_price, 2)
        line_items.append(LineItem(
            description=desc,
            quantity=qty,
            unit_price=unit_price,
            vat_rate_pct=0.0,           # RCT — reverse charge
            vat_letter_code=None,       # tradesman doesn't use per-letter codes
            amount_net=amount_net,
            amount_vat=0.0,
            amount_gross=amount_net,
        ))

    # Totals — RCT: vat=0, total=subtotal
    subtotal = round(sum(li.amount_net for li in line_items), 2)
    vat_amount = 0.0
    total = subtotal

    return {
        # Template-fill keys (Jinja2 placeholders)
        "vendor_name":             vendor_name,
        "vendor_address_line1":    vendor_address_line1,
        "vendor_address_line2":    vendor_address_line2,
        "vendor_city":             city,
        "vendor_eircode":          vendor_eircode,
        "vendor_phone":            vendor_phone,
        "vendor_email":            vendor_email,
        "vendor_vat_number":       vendor_vat_number,
        "vendor_iban":             vendor_iban,
        "vendor_bic":              vendor_bic,
        "vendor_country":          "IE",
        "invoice_number":          invoice_number,
        "invoice_date":            issue_date.strftime("%d/%m/%Y"),
        "due_date":                due_date,
        "po_reference":            po_reference,
        "line_items":              [li.to_dict() for li in line_items],
        "subtotal":                subtotal,
        "vat":                     vat_amount,
        "total":                   total,
    }


# ─── Builder registry ────────────────────────────────────────────────────────

BUILDERS: dict[str, Any] = {
    "tradesman_rct": build_tradesman_rct,
}


def build(template_family: str, seed: int) -> dict[str, Any]:
    """Dispatch to the right builder by template family name."""
    if template_family not in BUILDERS:
        raise ValueError(
            f"No builder for template family {template_family!r}. "
            f"Available: {sorted(BUILDERS.keys())}. "
            f"Step 1 prototype scope = tradesman_rct ONLY; other 15 added in subsequent batches."
        )
    return BUILDERS[template_family](seed)
