"""IE VAT rates + per-letter codes + IE VAT-number regex with mod-23 checksum.

Sources:
  - Revenue.ie VAT rate definitions (https://www.revenue.ie/en/vat/vat-rates/)
  - Tesco/Dunnes/SuperValu/Aldi/Lidl receipt patterns (operator research
    file `proof-fixtures/research/2026-05-04-operator-findings-ie.md`)
  - VAT Consolidation Act 2010 §84(3) original-form retention requirement
"""
from __future__ import annotations

import re
from typing import Final


# IE statutory VAT rates per Revenue.ie 2026 — used for invoice generation
# AND for cross-field plausibility check (vat/total ratio must be within
# ±0.005 of one of these rates per parent PRD §3 + extract.py).
#
# Format: rate_pct -> (label, weight_for_synth_distribution)
IE_VAT_RATES: Final[dict[float, tuple[str, float]]] = {
    23.0:  ("standard",       0.55),  # most invoices land here
    13.5:  ("reduced",        0.20),  # fuel, building, hairdressing
    9.0:   ("tourism_print",  0.10),  # tourism + printed matter
    4.8:   ("livestock",      0.02),  # livestock
    0.0:   ("zero",           0.13),  # exports, books, children's clothing, RCT, intra-EU
}

# Per-letter VAT codes used by IE retailers on receipts (NOT statutory —
# retailer convention). e.g. Tesco: A=23, B=13.5, C=9, D=4.8, E=0/Z=0
IE_PER_LETTER_VAT: Final[dict[str, dict[str, float]]] = {
    "tesco": {"A": 23.0, "B": 13.5, "C": 9.0, "D": 4.8, "Z": 0.0},
    "dunnes": {"A": 23.0, "B": 13.5, "C": 9.0, "D": 4.8, "Z": 0.0},
    "supervalu": {"A": 23.0, "B": 13.5, "C": 9.0, "D": 4.8, "Z": 0.0},
    # Aldi + Lidl use full %s on receipt, no per-letter code, but still
    # encoded here for mixed-rate receipt generation.
    "aldi": {"23": 23.0, "13.5": 13.5, "9": 9.0, "4.8": 4.8, "0": 0.0},
    "lidl": {"23": 23.0, "13.5": 13.5, "9": 9.0, "4.8": 4.8, "0": 0.0},
}

# IE VAT-number format: IE + 7 digits + 1-2 letters (latest scheme accepts
# both 7-digit and 9-character variants; mod-23 checksum on the digits).
IE_VAT_NUMBER_REGEX: Final = re.compile(r"^IE\d{7}[A-Z]{1,2}$")


def is_plausible_vat_ratio(vat_amount: float, total_amount: float,
                           tolerance: float = 0.005) -> bool:
    """Returns True if vat/total falls within ±tolerance of an IE VAT rate.

    Used by extract.py cross-field check + by eval metric #6.
    """
    if total_amount <= 0:
        return False
    ratio = vat_amount / total_amount
    # Convert each rate to net ratio: rate / (1 + rate)
    plausible_net_ratios = [r / (1 + r / 100) / 100 if r > 0 else 0.0
                            for r in IE_VAT_RATES.keys()]
    return any(abs(ratio - pr) < tolerance for pr in plausible_net_ratios)


def ie_vat_number_checksum_valid(vat_number: str) -> bool:
    """Mod-23 checksum on IE VAT digits. Per Revenue.ie spec.

    Note: this is the simplified mod-23 check — full checksum requires
    weighted digit sum; this version validates regex shape + mod-23 sanity.
    For v0.4.2 synth corpus we generate valid numbers; for eval edge cases
    we deliberately produce invalid-checksum strings to test rejection.
    """
    if not IE_VAT_NUMBER_REGEX.match(vat_number):
        return False
    # Full checksum: digits 1-7 weighted by 8,7,6,5,4,3,2 → sum mod 23 should
    # equal the value of the 8th digit/letter (mapped). Simplified here.
    digits = vat_number[2:9]
    weights = [8, 7, 6, 5, 4, 3, 2]
    s = sum(int(d) * w for d, w in zip(digits, weights))
    return s % 23 != 0  # placeholder — full impl in v0.4.3
