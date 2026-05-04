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

# ─── Cafe receipt items (cafe_receipt template) ───────────────────────────────
# IE cafe / coffee-shop pricing 2026, single 9% VAT (food-and-drink on premises).
CAFE_ITEMS: tuple[tuple[str, float, float], ...] = (
    # (description, min_price_EUR, max_price_EUR)
    ("Americano",                       3.20, 4.40),
    ("Flat white",                      3.80, 4.80),
    ("Cappuccino",                      3.60, 4.70),
    ("Latte",                           3.80, 4.80),
    ("Espresso (single)",               2.60, 3.40),
    ("Pot of tea",                      3.20, 4.20),
    ("Hot chocolate",                   3.80, 4.80),
    ("Sparkling water 330ml",           2.80, 3.80),
    ("Croissant",                       3.20, 4.20),
    ("Pain au chocolat",                3.40, 4.40),
    ("Scone with butter + jam",         3.80, 4.80),
    ("Carrot cake slice",               4.50, 5.80),
    ("Brown bread + smoked salmon",     8.50, 12.00),
    ("Eggs Benedict",                   9.50, 13.50),
    ("Avocado toast",                   8.00, 11.50),
    ("Toasted ham + cheese sandwich",   6.50,  9.50),
    ("Soup of the day + bread",         6.50,  8.80),
    ("Fresh-baked sourdough loaf",      4.80,  6.20),
)

# IE cafe vendor naming pool — appended/customized via Faker en_IE.
CAFE_VENDOR_SUFFIXES: tuple[str, ...] = (
    "Coffee Roasters", "Coffee Co.", "Espresso Bar", "Coffee House",
    "Cafe", "Bakery", "Patisserie", "Tea Rooms", "Coffee + Brunch",
)


# ─── GP / medical practice (gp_medical template) ──────────────────────────────
# IE GP fees 2026 — services exempt from VAT per VATCA Sched 1 (medical).
GP_SERVICES: tuple[tuple[str, float, float], ...] = (
    ("Standard GP consultation fee",                60.00, 80.00),
    ("Repeat consultation",                         45.00, 65.00),
    ("Out-of-hours consultation",                   90.00, 130.00),
    ("Vaccination administration fee",              25.00, 40.00),
    ("Travel vaccination consultation + jab",       70.00, 95.00),
    ("Medical certificate / cert renewal",          25.00, 40.00),
    ("Driver medical examination (D509)",           80.00, 120.00),
    ("Insurance medical exam",                     150.00, 220.00),
    ("Wound dressing / suture removal",             40.00, 65.00),
    ("ECG + report",                                75.00, 110.00),
)

GP_PRACTICE_SUFFIXES: tuple[str, ...] = (
    "Family Practice", "Medical Centre", "GP Surgery", "Family Doctors",
    "Health Centre", "Primary Care Clinic", "GP Practice",
)


# ─── Veterinary practice (vet template) ──────────────────────────────────────
# IE vet pricing 2026 — multi-VAT: consultation 23%, livestock medicines 4.8%, pet food 0%.
# (description, min_EUR, max_EUR, vat_rate_pct)
VET_PROCEDURES: tuple[tuple[str, float, float, float], ...] = (
    ("Standard consultation",                        45.00,  75.00, 23.0),
    ("Extended consultation (complex case)",         80.00, 130.00, 23.0),
    ("Annual vaccination booster (cat)",             55.00,  85.00, 23.0),
    ("Annual vaccination booster (dog)",             65.00,  95.00, 23.0),
    ("Microchip implant + registration",             35.00,  55.00, 23.0),
    ("Neuter procedure (cat, female)",              140.00, 220.00, 23.0),
    ("Neuter procedure (dog, female)",              250.00, 480.00, 23.0),
    ("Dental scale + polish (general anaesthetic)", 280.00, 480.00, 23.0),
    ("X-ray (per region)",                           90.00, 160.00, 23.0),
    ("Blood panel (in-house)",                       75.00, 140.00, 23.0),
    # Livestock medicine — 4.8%
    ("Livestock anti-parasitic (per dose)",          12.50,  28.00,  4.8),
    ("Livestock antibiotic injection (per dose)",    18.00,  42.00,  4.8),
    ("Cattle pour-on wormer (per litre)",            38.00,  72.00,  4.8),
    # Pet food — 0% (zero-rated)
    ("Hill's Science Diet 7kg bag (prescription)",   42.00,  68.00,  0.0),
    ("Royal Canin urinary care 4kg bag",             38.00,  58.00,  0.0),
)

VET_PRACTICE_SUFFIXES: tuple[str, ...] = (
    "Veterinary Clinic", "Vets", "Animal Hospital", "Veterinary Surgery",
    "Pet Care Centre", "Veterinary Practice",
)


# ─── Solicitor letter-of-engagement (solicitor_loe template) ──────────────────
# IE solicitor fees 2026 — single 23% VAT on professional fee, outlay 0% (pass-through).
# (description, min_EUR, max_EUR, vat_rate_pct, is_outlay)
SOLICITOR_ITEMS: tuple[tuple[str, float, float, float, bool], ...] = (
    ("Conveyancing — residential purchase (legal fee)",  1200.00, 2500.00, 23.0, False),
    ("Conveyancing — residential sale (legal fee)",       950.00, 1800.00, 23.0, False),
    ("Will drafting (standard)",                          250.00,  450.00, 23.0, False),
    ("Probate application (legal fee)",                   850.00, 2200.00, 23.0, False),
    ("Pre-nuptial agreement drafting",                    650.00, 1400.00, 23.0, False),
    ("Lease review + advice (commercial)",                450.00,  950.00, 23.0, False),
    ("Letter of engagement preparation",                  150.00,  280.00, 23.0, False),
    # Outlay (pass-through, no VAT — solicitor advances on client behalf)
    ("Land Registry — Form 17 outlay",                     80.00,  130.00,  0.0, True),
    ("Stamp duty (1% on €260K residential)",             1500.00, 4000.00,  0.0, True),
    ("Property Registration Authority search",             40.00,   80.00,  0.0, True),
    ("Companies Registration Office filing fee",           30.00,   60.00,  0.0, True),
    ("Sworn affidavit commissioner outlay",                25.00,   50.00,  0.0, True),
)

SOLICITOR_FIRM_SUFFIXES: tuple[str, ...] = (
    "& Co Solicitors", "& Partners Solicitors", "Solicitors LLP",
    "Legal Services", "& Associates Solicitors",
)


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


# ─── Builder helpers (shared across builders) ────────────────────────────────


def _gen_vendor_identity(rng: random.Random, faker: Faker, suffix_pool: tuple[str, ...],
                         include_eircode: bool = True) -> dict[str, Any]:
    """Common vendor-identity fields. Caller may override vendor_name post-call."""
    base_name = faker.last_name()
    vendor_name = f"{base_name} {rng.choice(suffix_pool)}"
    address = faker.street_address()
    address2 = (
        f"{rng.choice(['Unit', 'Apt', 'Suite'])} {rng.randint(1, 24)}"
        if rng.random() < 0.30 else ""
    )
    city, eircode_route = rng.choice(IE_CITY_EIRCODE_ROUTES)
    eircode = (
        f"{eircode_route} {''.join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(4))}"
        if include_eircode else ""
    )
    phone = f"+353 {rng.randint(80, 89)}{rng.randint(0,9)} {rng.randint(100,999)} {rng.randint(1000,9999)}"
    email = faker.company_email()
    vat_num = gen_ie_vat_number(rng)
    iban, bic = gen_ie_iban_bic(faker, rng)
    return {
        "vendor_name": vendor_name,
        "vendor_address_line1": address,
        "vendor_address_line2": address2,
        "vendor_city": city,
        "vendor_eircode": eircode,
        "vendor_country": "IE",
        "vendor_phone": phone,
        "vendor_email": email,
        "vendor_vat_number": vat_num,
        "vendor_iban": iban,
        "vendor_bic": bic,
    }


# ─── Builder: cafe_receipt (thermal 80mm, single 9% VAT, food-and-drink) ──────


def build_cafe_receipt(seed: int) -> dict[str, Any]:
    """Cafe / coffee-shop thermal receipt. Single 9% VAT (food-and-drink on
    premises per Revenue.ie reduced rate). 1-6 line items per
    layout_slots.cafe_receipt.typical_line_item_count."""
    rng = random.Random(seed)
    faker = _faker(seed)

    vendor = _gen_vendor_identity(rng, faker, CAFE_VENDOR_SUFFIXES, include_eircode=False)
    issue_date = gen_recent_date(rng)
    invoice_number = f"R{rng.randint(10000, 99999)}"  # cafe receipts use short ref

    n_items = rng.randint(1, 6)
    line_items: list[LineItem] = []
    for _ in range(n_items):
        desc, p_min, p_max = rng.choice(CAFE_ITEMS)
        qty = float(rng.randint(1, 3))
        unit_price = round(rng.uniform(p_min, p_max), 2)
        amount_gross = round(qty * unit_price, 2)  # cafe prices are GROSS (VAT-inclusive)
        amount_net = round(amount_gross / 1.09, 2)
        amount_vat = round(amount_gross - amount_net, 2)
        line_items.append(LineItem(
            description=desc, quantity=qty, unit_price=unit_price,
            vat_rate_pct=9.0, vat_letter_code=None,
            amount_net=amount_net, amount_vat=amount_vat, amount_gross=amount_gross,
        ))

    # Totals — cafe uses gross prices; subtotal = sum(net), vat = sum(vat), total = sum(gross)
    total = round(sum(li.amount_gross for li in line_items), 2)
    vat_amount = round(sum(li.amount_vat for li in line_items), 2)
    subtotal = round(total - vat_amount, 2)

    return {
        **vendor,
        "invoice_number": invoice_number,
        "invoice_date": issue_date.strftime("%d/%m/%Y"),
        "due_date": "",
        "po_reference": "",
        "line_items": [li.to_dict() for li in line_items],
        "subtotal": subtotal,
        "vat": vat_amount,
        "total": total,
    }


# ─── Builder: gp_medical (A4, exempt VAT, single fee) ─────────────────────────


def build_gp_medical(seed: int) -> dict[str, Any]:
    """GP / medical-practice fee receipt. Exempt VAT (medical services 0% per
    VATCA Sched 1). Single fee or 1 line item. has_line_items=False per
    layout_slots.gp_medical.typical_line_item_count=(1,1)."""
    rng = random.Random(seed)
    faker = _faker(seed)

    vendor = _gen_vendor_identity(rng, faker, GP_PRACTICE_SUFFIXES, include_eircode=True)
    # GP practices typically have "Dr. [Surname]" prefix
    if rng.random() < 0.6:
        vendor["vendor_name"] = f"Dr. {faker.first_name()} {vendor['vendor_name']}"
    else:
        vendor["vendor_name"] = vendor["vendor_name"].replace(faker.last_name(),
                                                              f"{faker.last_name()} {faker.last_name()}")

    issue_date = gen_recent_date(rng)
    invoice_number = f"GP-{issue_date.strftime('%Y%m')}-{rng.randint(100, 999)}"

    desc, p_min, p_max = rng.choice(GP_SERVICES)
    fee = round(rng.uniform(p_min, p_max), 2)
    line_items = [LineItem(
        description=desc, quantity=1.0, unit_price=fee,
        vat_rate_pct=0.0, vat_letter_code="EXEMPT",
        amount_net=fee, amount_vat=0.0, amount_gross=fee,
    )]

    subtotal = fee
    vat_amount = 0.0
    total = fee

    return {
        **vendor,
        "invoice_number": invoice_number,
        "invoice_date": issue_date.strftime("%d/%m/%Y"),
        "due_date": (issue_date + timedelta(days=14)).strftime("%d/%m/%Y") if rng.random() < 0.50 else "",
        "po_reference": "",
        "service_description": desc,
        "line_items": [li.to_dict() for li in line_items],
        "subtotal": subtotal,
        "vat": vat_amount,
        "total": total,
    }


# ─── Builder: vet (A4, mixed-VAT consultation 23%/livestock-med 4.8%/food 0%) ──


def build_vet(seed: int) -> dict[str, Any]:
    """Veterinary practice — mixed VAT. consultation+procedure 23%, livestock
    medicines 4.8%, pet food 0%. 2-8 line items per layout_slots.vet."""
    rng = random.Random(seed)
    faker = _faker(seed)

    vendor = _gen_vendor_identity(rng, faker, VET_PRACTICE_SUFFIXES, include_eircode=True)
    issue_date = gen_recent_date(rng)
    invoice_number = gen_invoice_number(rng, issue_date)

    n_items = rng.randint(2, 8)
    line_items: list[LineItem] = []
    for _ in range(n_items):
        desc, p_min, p_max, vat_pct = rng.choice(VET_PROCEDURES)
        qty = float(rng.randint(1, 3))
        unit_price = round(rng.uniform(p_min, p_max), 2)
        amount_net = round(qty * unit_price, 2)
        amount_vat = round(amount_net * (vat_pct / 100.0), 2)
        amount_gross = round(amount_net + amount_vat, 2)
        line_items.append(LineItem(
            description=desc, quantity=qty, unit_price=unit_price,
            vat_rate_pct=vat_pct, vat_letter_code=None,
            amount_net=amount_net, amount_vat=amount_vat, amount_gross=amount_gross,
        ))

    subtotal = round(sum(li.amount_net for li in line_items), 2)
    vat_amount = round(sum(li.amount_vat for li in line_items), 2)
    total = round(subtotal + vat_amount, 2)

    # VAT breakdown by rate for the breakdown table
    vat_breakdown: dict[float, dict[str, float]] = {}
    for li in line_items:
        b = vat_breakdown.setdefault(li.vat_rate_pct, {"net": 0.0, "vat": 0.0})
        b["net"] = round(b["net"] + li.amount_net, 2)
        b["vat"] = round(b["vat"] + li.amount_vat, 2)
    vat_breakdown_rows = [
        {"rate_pct": rate, "net": data["net"], "vat": data["vat"]}
        for rate, data in sorted(vat_breakdown.items())
    ]

    return {
        **vendor,
        "invoice_number": invoice_number,
        "invoice_date": issue_date.strftime("%d/%m/%Y"),
        "due_date": (issue_date + timedelta(days=21)).strftime("%d/%m/%Y") if rng.random() < 0.55 else "",
        "po_reference": "",
        "line_items": [li.to_dict() for li in line_items],
        "vat_breakdown_rows": vat_breakdown_rows,
        "subtotal": subtotal,
        "vat": vat_amount,
        "total": total,
    }


# ─── Builder: solicitor_loe (A4 letterhead, fee + outlay split, 23% on fees) ──


def build_solicitor_loe(seed: int) -> dict[str, Any]:
    """Solicitor letter-of-engagement. A4 letterhead. Single 23% VAT on
    professional fees; outlay 0% (pass-through). 2-6 line items per
    layout_slots.solicitor_loe."""
    rng = random.Random(seed)
    faker = _faker(seed)

    vendor = _gen_vendor_identity(rng, faker, SOLICITOR_FIRM_SUFFIXES, include_eircode=True)
    # Solicitor firms often "[Surname1] [Surname2]" prefix
    if rng.random() < 0.5:
        vendor["vendor_name"] = f"{faker.last_name()} {vendor['vendor_name']}"

    issue_date = gen_recent_date(rng)
    invoice_number = gen_invoice_number(rng, issue_date)

    n_items = rng.randint(2, 6)
    line_items: list[LineItem] = []
    has_fee = False
    for _ in range(n_items):
        desc, p_min, p_max, vat_pct, is_outlay = rng.choice(SOLICITOR_ITEMS)
        qty = 1.0
        unit_price = round(rng.uniform(p_min, p_max), 2)
        amount_net = unit_price
        amount_vat = round(amount_net * (vat_pct / 100.0), 2)
        amount_gross = round(amount_net + amount_vat, 2)
        line_items.append(LineItem(
            description=desc + (" [Outlay]" if is_outlay else ""),
            quantity=qty, unit_price=unit_price,
            vat_rate_pct=vat_pct,
            vat_letter_code="OUTLAY" if is_outlay else None,
            amount_net=amount_net, amount_vat=amount_vat, amount_gross=amount_gross,
        ))
        if not is_outlay:
            has_fee = True

    # Guarantee at least one professional fee item — re-roll if all outlay
    if not has_fee:
        desc, p_min, p_max, vat_pct, is_outlay = SOLICITOR_ITEMS[0]
        unit_price = round(rng.uniform(p_min, p_max), 2)
        amount_vat = round(unit_price * (vat_pct / 100.0), 2)
        line_items.insert(0, LineItem(
            description=desc, quantity=1.0, unit_price=unit_price, vat_rate_pct=vat_pct,
            vat_letter_code=None, amount_net=unit_price, amount_vat=amount_vat,
            amount_gross=round(unit_price + amount_vat, 2),
        ))

    subtotal = round(sum(li.amount_net for li in line_items), 2)
    vat_amount = round(sum(li.amount_vat for li in line_items), 2)
    total = round(subtotal + vat_amount, 2)

    return {
        **vendor,
        "invoice_number": invoice_number,
        "invoice_date": issue_date.strftime("%d/%m/%Y"),
        "due_date": (issue_date + timedelta(days=30)).strftime("%d/%m/%Y") if rng.random() < 0.65 else "",
        "po_reference": f"PO/{rng.randint(1000, 99999)}" if rng.random() < 0.30 else "",
        "service_description": "Letter of Engagement — see itemised fees + outlay below",
        "line_items": [li.to_dict() for li in line_items],
        "subtotal": subtotal,
        "vat": vat_amount,
        "total": total,
    }


# ─── Builder registry ────────────────────────────────────────────────────────

BUILDERS: dict[str, Any] = {
    "tradesman_rct":  build_tradesman_rct,
    "cafe_receipt":   build_cafe_receipt,
    "gp_medical":     build_gp_medical,
    "vet":            build_vet,
    "solicitor_loe":  build_solicitor_loe,
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
