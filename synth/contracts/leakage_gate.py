"""MinHash near-duplicate leakage gate between train + eval shards.

Per PDR v0.2 §8 + Codex §C: train and eval generators MUST be
demonstrably disjoint. Beyond seed/vendor/layout/VAT/items/noise axes
+ no-shared-CSS-partials + commit-hash pin, post-generation we run a
MinHash near-duplicate check.

Threshold: Jaccard similarity ≥0.85 between any train doc and any eval
doc = collision flag. Action: regenerate the offending eval doc with
different seed, OR move the train doc out of training.

Implementation uses `datasketch` MinHash (MIT licensed). 128 permutations
gives ~0.04 std-error on Jaccard estimate — sufficient for our threshold.
"""
from __future__ import annotations

from typing import Iterable
from pathlib import Path


class MinHashGate:
    """MinHash near-duplicate detector for the leakage gate.

    Usage:
        gate = MinHashGate()
        for doc_id, text in train_docs:
            gate.add_train(doc_id, text)
        violations = gate.check_eval(eval_docs)
        if violations:
            raise RuntimeError(f"leakage gate failed: {violations}")
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self._train_hashes: dict[str, "MinHash"] = {}
        try:
            from datasketch import MinHash, MinHashLSH
            self._MinHash = MinHash
            self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
            self._available = True
        except ImportError:
            self._available = False

    def available(self) -> bool:
        return self._available

    def _shingle(self, text: str, k: int = 5) -> set[str]:
        """k-shingles for MinHash. Lower k = more sensitive, higher k =
        more permissive. k=5 chosen as default per datasketch docs."""
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

    def check_eval(self, eval_docs: Iterable[tuple[str, str]]) -> list[tuple[str, str, float]]:
        """Returns list of (eval_doc_id, train_doc_id, jaccard_estimate)
        for any pair exceeding self.threshold."""
        if not self._available:
            raise RuntimeError("datasketch not installed; cannot run leakage gate")
        violations: list[tuple[str, str, float]] = []
        for eval_id, eval_text in eval_docs:
            m = self._MinHash(num_perm=self.num_perm)
            for shingle in self._shingle(eval_text):
                m.update(shingle.encode("utf-8"))
            for train_id in self._lsh.query(m):
                jaccard = m.jaccard(self._train_hashes[train_id])
                violations.append((eval_id, train_id, float(jaccard)))
        return violations
