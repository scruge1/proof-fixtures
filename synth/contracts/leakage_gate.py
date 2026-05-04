"""MinHash near-duplicate leakage gate between train + eval shards.

Per PDR v0.2 §8 + Codex §C + Adam review 2026-05-04: train and eval
generators MUST be demonstrably disjoint. Beyond seed/vendor/layout/VAT/
items/noise axes + no-shared-CSS-partials + commit-hash pin, post-
generation we run a MinHash near-duplicate check.

Two-tier threshold (v0.4.2-step0.1):
    - SOFT 0.85: collision flag → review + regenerate offending eval doc
                 with different seed. Surfaces structural overlap risk.
    - HARD 0.95: hard violation → raise RuntimeError + block corpus build.
                 Catches "definitely-the-same-doc" emergencies.

Rationale: invoices share heavy boilerplate (vendor address blocks, VAT
table headers, footer disclaimers). With identity-strip normalization a
score >0.85 means content is genuinely overlapping, not just template
shared. Background research (FineWeb / DCLM / academic LLM-corpus dedup
literature, retrieved 2026-05-04 — see PDR v0.2 §8 footnote): 0.8 is the
academic norm, 0.85 the production-LLM norm at trillion-doc scale; we
sit at 0.85 because (a) our corpus is ≤1k docs not 10^12, so per-doc
cost of false positive is tolerable, (b) two-tier hard-cutoff at 0.95
gives an unambiguous block on near-identical pairs.

Identity-strip normalization (NORMALIZE_IDENTITY default True): before
shingling, redact volatile per-doc fields (vendor_name, vendor_iban,
vendor_vat_number, invoice_number, invoice_date, vendor_eircode) so
Jaccard reflects content + layout overlap, not boilerplate text. Pass
NORMALIZE_IDENTITY=False to compare raw text (Codex review override
mode).

Implementation uses `datasketch` MinHash (MIT licensed). 128 permutations
gives ~0.04 std-error on Jaccard estimate — sufficient for both tiers.
"""
from __future__ import annotations

import re
from typing import Iterable
from pathlib import Path


# Volatile per-doc fields stripped before shingling so Jaccard reflects
# layout + content overlap, not the presence of generic text like
# "VAT Number" / "Subtotal" / address suffixes. Patterns match the
# RENDERED text in the doc (post-content-engine fill), not template HTML.
_IDENTITY_STRIP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bIE\d{7}[A-Z]{1,2}\b"),                  # vendor_vat_number
    re.compile(r"\bIE\d{2}[A-Z]{4}\d{14}\b"),              # vendor_iban
    re.compile(r"\bINV[-\s]?\d{4,}\b", re.IGNORECASE),     # invoice_number (INV-prefix variants)
    re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),                  # DD/MM/YYYY
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                  # ISO date
    re.compile(r"\b[A-Z]\d{2}\s?[A-Z0-9]{4}\b"),           # IE eircode
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\+?353[-\s]?\d{1,2}[-\s]?\d{3,}[-\s]?\d{3,}"),  # IE phone
    re.compile(r"€\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?"),     # currency amounts (per-doc varying)
)


def _strip_identity(text: str) -> str:
    """Redact volatile per-doc fields with the literal token <ID> so that
    boilerplate text contributes to MinHash but per-doc identity does not."""
    out = text
    for pat in _IDENTITY_STRIP_PATTERNS:
        out = pat.sub("<ID>", out)
    return out


class MinHashGate:
    """MinHash near-duplicate detector for the leakage gate.

    Usage:
        gate = MinHashGate()
        for doc_id, text in train_docs:
            gate.add_train(doc_id, text)
        violations = gate.check_eval(eval_docs)
        # violations rows include severity tier (soft|hard); hard ones MUST block.
        for ev_id, tr_id, jac, tier in violations:
            if tier == "hard":
                raise RuntimeError(f"leakage gate HARD failure: {ev_id} ↔ {tr_id} = {jac:.3f}")
    """

    def __init__(self,
                 num_perm: int = 128,
                 threshold: float = 0.85,
                 hard_threshold: float = 0.95,
                 normalize_identity: bool = True) -> None:
        if hard_threshold < threshold:
            raise ValueError(f"hard_threshold {hard_threshold} must be >= soft threshold {threshold}")
        self.num_perm = num_perm
        self.threshold = threshold
        self.hard_threshold = hard_threshold
        self.normalize_identity = normalize_identity
        self._train_hashes: dict[str, "MinHash"] = {}
        try:
            from datasketch import MinHash, MinHashLSH
            self._MinHash = MinHash
            # LSH banded at the SOFT threshold so we surface every candidate
            # at-or-above 0.85; the HARD tier is then computed exactly.
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._available = True
        except ImportError:
            self._available = False

    def available(self) -> bool:
        return self._available

    def _shingle(self, text: str, k: int = 5) -> set[str]:
        """k-shingles for MinHash. Lower k = more sensitive, higher k =
        more permissive. k=5 chosen per datasketch docs + FineWeb/DCLM
        precedent (n in 5..9 typical for document near-dup detection)."""
        if self.normalize_identity:
            text = _strip_identity(text)
        text = " ".join(text.split())  # normalize whitespace
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def add_train(self, doc_id: str, text: str) -> None:
        if not self._available:
            raise RuntimeError("datasketch not installed; cannot run leakage gate")
        m = self._MinHash(num_perm=self.num_perm)
        for shingle in self._shingle(text):
            m.update(shingle.encode("utf-8"))
        self._train_hashes[doc_id] = m
        self._lsh.insert(doc_id, m)

    def check_eval(self, eval_docs: Iterable[tuple[str, str]]) -> list[tuple[str, str, float, str]]:
        """Returns list of (eval_doc_id, train_doc_id, jaccard_estimate, tier)
        for any pair exceeding self.threshold. tier ∈ {"soft", "hard"}.

        Caller MUST treat any tier=="hard" row as a build-blocking failure.
        Rows with tier=="soft" are review-required but the corpus build
        may proceed once each soft violation is regenerated or accepted
        by the operator.
        """
        if not self._available:
            raise RuntimeError("datasketch not installed; cannot run leakage gate")
        violations: list[tuple[str, str, float, str]] = []
        for eval_id, eval_text in eval_docs:
            m = self._MinHash(num_perm=self.num_perm)
            for shingle in self._shingle(eval_text):
                m.update(shingle.encode("utf-8"))
            for train_id in self._lsh.query(m):
                jaccard = float(m.jaccard(self._train_hashes[train_id]))
                tier = "hard" if jaccard >= self.hard_threshold else "soft"
                violations.append((eval_id, train_id, jaccard, tier))
        return violations
