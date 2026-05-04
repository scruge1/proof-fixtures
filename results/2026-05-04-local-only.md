# Benchmark — Local-Only Stack v0.3 (2026-05-04)

**Pipeline:** Tesseract 5.5 + RapidOCR 3.8.1 (ONNX) → TwoEngineVoter (semantic-match) → GLM-OCR Q8 (Ollama) verifier on disagreement → schema validation → cross-field consistency.

**License posture:** All Apache-2.0 / MIT. Zero paid-API dependency. Intra-EEA only (no third-country flow).

**Hardware:** AMD Ryzen 5 3500U / 30GB RAM / Vega 8 (no CUDA). Windows 11 + Python 3.10.

**Verifier model:** Ollama `glm-ocr:q8_0` (1.6 GB on disk). Settings: `temperature=0.1`, `repeat_penalty=1.2` per Mhjorleifsson recipe.

## Architecture (v0.3)

```
PDF / image
  └─ pdf2image(dpi=300) + Poppler 25.07          [or cv2.imread for image]
  └─ preprocess: 3.5x Lanczos upscale + Hough deskew + adaptive Gaussian threshold
  └─ Tesseract 5.5 (--oem 1 --psm 6) ──┐
  └─ RapidOCR 3.8.1 (ONNX)             ├── TwoEngineVoter (semantic value match)
                                        │
                              voter unanimous + ocr_conf ≥ 0.85?
                                        ├── YES → trust consensus, skip verifier
                                        └── NO  → GLM-OCR Q8 text-mode verifier
                                                    on disagreement-only fields
  └─ regex stub field extraction (vendor / total / vat / date)         [v0.4: structured-LLM prompt]
  └─ schema validation (regex / range / date parse)
  └─ cross-field consistency (VAT/total ratio matches IE 23%/13.5%/9%)
  └─ composite confidence + 0.98 gate
```

## Run — synthetic Chadwicks invoice

| Field | Ground truth | Extracted | Voter | Schema | Confidence |
|---|---|---|---|---|---|
| vendor | CHADWICKS LIMITED | CHADWICKS LIMITED | ✓ unanimous | ✓ | 1.00 |
| total | 283.52 | 283.52 | ✓ unanimous | ✓ | 1.00 |
| vat | 53.02 | 53.02 | ✓ unanimous | ✓ | 1.00 |
| date | 04/05/2026 | 04/05/2026 | ✓ unanimous | ✓ | 1.00 |

**Cross-field check:** VAT/total ratio = 0.187 → matches IE 23% net-of-gross rate (0.23/1.23 = 0.187). Plausible. ✓

**Gate:** PASS. **Latency:** 5.1 s (cold-start → extraction → schema → gate). **Verifier invoked:** 0 calls (voter unanimous, skipped).

## Delta from v0.2 baseline (2026-05-03)

| Metric | v0.2 (Mistral verifier) | v0.3 (local-only) | Delta |
|---|---|---|---|
| Per-doc cost | €0.001 (paid API) | €0.000 | −100% |
| Sub-processor list | Mistral SAS + Google Cloud IE | none external | DPF/SCCs removed |
| 0.98 gate pass on test-invoice | bounced (silent total=0%) | passes (4/4) | +100% |
| Latency cold | 3.1 s + API round-trip | 5.1 s end-to-end | +2 s (CPU-only) |
| Schrems II tail risk | active (DPF reliance) | gone | eliminated |
| GDPR Art. 28 sub-processor count | 2 | 0 | flat |
| OCR engine count | 2 (Tesseract + RapidOCR) | 2 (same) | unchanged |
| Verifier substrate | paid Mistral Small 3.1 | local GLM-OCR Q8 | model swap |
| License | Mistral commercial DPA + GCP DPA | Apache-2.0 only | clean |

## What's still v0.3 stub (v0.4 fix list)

1. **Regex field extraction** — handle-tuned per-pattern, fragile on every new vendor format. v0.4 → structured-LLM extraction (Pydantic schema + Ollama JSON-mode).
2. **Single-page only** — `first_page=1, last_page=1`. Real invoices span 2-5 pages with continuation totals.
3. **No table extraction** — line items are tabular; we extract totals only. Adam's bookkeeping clients post line-item-by-line-item for VATCA records.
4. **No vendor disambiguation** — "Chadwicks", "Chadwicks Ltd", "CHADWICKS LIMITED" should resolve to the same supplier. Sentence-transformers + vector DB.
5. **No active learning** — bounce → Adam corrects in spreadsheet → never lands in training set. v0.4 builds the flywheel.
6. **No vision-crop verifier** — verifier currently text-only (sees OCR text, not image pixels). v0.4 routes uncertain fields to glm-ocr image-mode for direct visual confirmation.
7. **No PEPPOL UBL XML reader** — locked D18 commit Q3-2026 still pending.
8. **No drift detection** — model regression on real corpus would be silent.

These constitute the v0.4 PRD scope (research in progress, agent `a8d72aa4955f1b4f9`).

## Files touched in v0.3 ship

- `proof-fixtures/scripts/extract.py` — full refactor (~720 lines): TwoEngineVoter + GLMOCRVerifier + preprocess + schema + cross-field.
- `~/.claude/hooks/pretool-dep-probe.js` — added `PY_DEP_REQUIRES_BIN` map (catches transitive native-dep gaps like `pdf2image → poppler`).
- `~/.claude/SYSTEM-INVENTORY.md` — refreshed (Poppler 25.07 now present).
- Poppler installed via `winget install --id oschwartz10612.Poppler -e`. Path auto-detected by `extract.py:_detect_poppler()`.
- `proof-fixtures/results/test-invoice-001.json` — first PASS under local-only stack.

## Citations (research basis for v0.3)

- `proof-fixtures/research/2026-05-04-local-ocr-deep-research.md` — engine-by-engine bake-off
- GLM-OCR ranks #1 OmniDocBench V1.5 at 94.62 — `https://ollama.com/library/glm-ocr`
- Two-engine fusion paper SROIE 96.5% — `https://www.sciencedirect.com/science/article/pii/S111001682500657X`
- VLM-as-verifier pattern — `https://www.f22labs.com/blogs/ocr-vs-vlm-vision-language-models-key-comparison/`
- Mistral OCR 72.2% real-world — themanmaran HN — `https://news.ycombinator.com/item?id=43282905`
