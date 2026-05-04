# Synthetic Invoice Generation for IE Document AI — Bootstrap Corpus Research

**Date:** 2026-05-04
**Author:** Claude (Opus 4.7, 1M ctx) research subagent
**Audience:** v0.4.2 PRD authors, Adam (Limerick sole-trader bookkeeper persona), proof-fixtures architects
**Purpose:** Bootstrap a 500-invoice training corpus for fine-tuning Qwen2.5-VL-3B (vision tower frozen) BEFORE first paying customer arrives — license-clean, IE-flavoured, transferable to real-world performance.
**Companions (read-first, do not duplicate):** `2026-05-04-vendor-architectures.md`, `2026-05-04-active-learning-flywheel.md`, `2026-05-04-hybrid-stack-libraries.md`, `2026-05-04-operator-findings-ie.md`, `2026-05-04-local-ocr-deep-research.md`.

---

## §0 One-line summary

Generate 500 synthetic IE invoices/receipts via a layered stack — Faker+reportlab/WeasyPrint for content, SynthDoG-style HTML→PDF→PNG rendering, then Augraphy + scan-simulator for corruption — at a 30 % clean / 50 % light corruption / 20 % heavy corruption split, two-stage curriculum (synth pre-train → fine-tune on first 50 real customer corrections), with explicit IE patterns (23/13.5/9/4.8/0 % VAT, RCT reverse-charge wording, supermarket per-letter VAT codes, AIB/BOI/Revolut bank statement layouts) baked into the content generator, not the corruption layer.

---

## §1 Executive recommendations (numbers up front)

The literature converges on five concrete settings. Each is pulled from a peer-reviewed paper or production write-up cited inline; the 500-doc corpus mapping is mine.

1. **Corruption ratio: 30 / 50 / 20.** ~30 % clean digital (synthetic PDF rendered, no degradation), ~50 % light corruption (light blur / 1-3° rotation / JPEG q=70-85 / mild lighting gradient), ~20 % heavy corruption (BadPhotoCopy / DirtyDrum / Faxify / 5-10° skew / JPEG q=25-50 / fold marks). Source: scrambledtext (arXiv:2409.19735) [11] showed CER-stratified buckets (0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4 each at 25 %) gave best transfer; ritw.dev compliance write-up [9] reported 30→85 % accuracy lift on real scans when augmentation pipeline was applied vs not. For 500 docs: 150 clean / 250 light / 100 heavy.
2. **Augmentation enlargement factor ≈ 4×, not 8×.** Tatusch et al. (arXiv:2209.14970) [13] showed 4× augmentation reduced CER 14.92 pp on a 15 % subset, and 4×→8× yielded only marginal gains. The cited synthetic-data post-OCR study (arXiv:2408.02253) [14] independently reports the same 1×→4× effective, 4×→8× marginal pattern. For 500 originals (125 unique seed templates × 4 augmented variants), this is the natural target.
3. **Two-stage curriculum, not one big mixed dump.** Nanonets-OCR2-3B [referenced in vendor-architectures.md §2.4] uses Stage 1 (synthetic pre-train) → Stage 2 (real corrections fine-tune). DocILE (arXiv:2302.05658) [1] showed `RoBERTa+SYNTH` and `LayoutLMv3+SYNTH` baselines beat their no-synth counterparts on every key-information task. Mind-the-Gap (arXiv:2405.03243) [10] showed final 2 layers carry the synth→real gap — pre-train all, fine-tune last 2 layers on real (≥1/8 of real data closes the gap). Translation: pre-train on 500 synth → freeze most layers → fine-tune on first 50-100 real customer corrections.
4. **CER / WER target on synth: median around the real distribution's median.** scrambledtext [11] found best transfer when synthetic corruption matches the real-world distribution's median CER (their NCSE corpus had median CER 0.17; best synth model was trained on CER ≈ 0.20). For IE invoices: Adam's real distribution skews clean (most invoices are PDF-native), so the bulk of the synth corpus should also lean clean. This ratifies the 30/50/20 split.
5. **Layout diversity > content diversity.** ritw.dev [9] and Provectus whitepaper [3] both note that a model trained on identical layouts with varied content fails on real data; the reverse (varied layouts with the same content distribution) generalizes. Translation: if you can only afford to vary one axis at scale, vary the **layout** (column orders, header positions, line-item table styles, font families, logo placements). Use 8-12 distinct base layouts × 40-60 variants each = 320-720 docs. Clusters tied to real IE vendor families (Tesco-style supermarket receipt, Revenue-style RCT-flagged subcontractor invoice, Aldi/Lidl per-VAT-rate summary, AIB statement, plain trade invoice).

---

## §2 What peer-reviewed papers + open-source projects actually use

### §2.1 DocILE — the gold-standard published synthetic pipeline (Rossum + Czech Tech U)

DocILE [1] is the only large-scale public synthetic invoice corpus with KILE/LIR labels: **100 000 synthetic documents** generated from layout annotations of 100 manually annotated seed invoices. The pipeline (their §3.5):

1. **Cluster real layouts** — annotate 100 seed invoices into layout clusters, capturing bbox + semantic category + text per element.
2. **Rule-based synthesizer fills slots** — content generators (names, emails, addresses, bank account numbers) drawn from the **Mimesis library** [2] (Apache-2.0; richer than Faker for structured EU data) plus copied keys from the seed.
3. **Style generator varies look** — font family, font size, border style, content shifts.
4. **Render HTML → PDF.** Source HTML shared with the dataset for reproducibility / re-rendering.

Headline result: pre-training RoBERTa or LayoutLMv3 on the 100 K synth subset improved KILE/LIR F1 in nearly every condition; `RoBERTa+SYNTH` was the best non-augmented-with-extra-data baseline. The synth pipeline itself is **proprietary** (Rossum has not open-sourced it), but the *dataset* (synthetic + real + unlabeled) is downloadable from Hugging Face for research use [1, github.com/rossumai/docile].

**Direct adoption value for v0.4.2:** Replicate the **layout-clusters-from-seeds** pattern — pick 8-12 IE-flavoured seed layouts (one per vendor family below), annotate them once, drive a synthesizer over the slot bboxes. Mimesis or Faker (with Irish locale `en_IE` and a Mimesis Eire address provider) is the content engine.

### §2.2 SynthDoG — Donut's open-source generator (NAVER Clova)

SynthDoG [4, 6] is **MIT-licensed** and powers the synthdog-en/zh/ja/ko datasets used to pre-train Donut (clovaai/donut, ECCV 2022) [5]. Implementation: thin wrapper over **synthtiger** [pypi:synthtiger]; YAML config for page size / font size / rotation / background; renders text on a paper-textured background with simple paragraph layout. Designed for **text-reading pre-training**, not structured-field extraction — limitations called out by the authors themselves: no native LaTeX, no tables, no structured documents [GH issue #325]. SynthDoG-RTL [7] extends it to Arabic/Urdu/Persian/Hebrew with libraqm shaping.

**Direct adoption value for v0.4.2:** Use SynthDoG/synthtiger as the **rendering + corruption shell** (rotation, blur, background paper textures, font variation), but feed it our own structured invoice HTML rather than its bare-text paragraphs. The library's PDF/PNG render path is solid; its content generator is too thin for invoices.

### §2.3 TextRecognitionDataGenerator (TRDG) — Belval

TRDG [8] is **MIT-licensed**, 4 K stars, generates word/line-level OCR training images — not full invoice pages. Useful as a *crop-level* augmenter for fine-tuning the OCR engines underneath the VLM (TwoEngineVoter in v0.4.1 = Tesseract+RapidOCR; both can be quietly improved with TRDG-generated EUR-symbol / Irish-postcode / RCT-statement crops). It supports skewing (`-k`), Gaussian blur (`-bl`), 4 background types (Gaussian noise / white / quasicrystal / image), and an experimental handwritten mode. Not the page-level tool we need but an honest add-on for the OCR layer.

### §2.4 SynthID — layout-preserving content replacement (arXiv:2508.03754)

SynthID [3a] (paper 2025, BevinV/Synthetic_Invoice_Generation [3b] is the GitHub) introduces a **3-stage pipeline**: OCR a real seed invoice → LLM rewrites field content (preserving type — date stays a date, EUR amount stays an EUR amount) → image inpainting renders the new text in the original glyph positions. Output: visually realistic invoice + perfectly aligned JSON labels. The trick: layout fidelity comes free from the seed; only content is novel. License posture: paper is publicly available; repo license unspecified at time of writing — treat as restricted until verified.

**Direct adoption value for v0.4.2:** This is the **fastest path to 500 docs** if Adam can supply ~20-50 anonymized real seed invoices (the v0.3 proof-fixtures collection plus anything Callmeie can legally use under the DPA). LLM (GLM-OCR via Ollama, free, already wired) does the field rewrite step. Inpainting is the only new dependency (HuggingFace `lama-cleaner` MIT license, or `simple-lama-inpainting` MIT). Risk: inherits seed-vendor bias; layout diversity is bounded by the seeds on hand.

### §2.5 Provectus synthetic invoice whitepaper (industry, 2021)

Provectus [16] published a templating approach: 5-12 base layouts × Faker content variation × stochastic field placement. Reported privacy-preserving training data quality "adequate for ML training" — no headline accuracy number, but the *method* is the cleanest small-shop reference (they explicitly target the same regime: privacy-blocked from real customer data, need bootstrap corpus before launch). Compatible with our §2.1 DocILE-derived recipe.

### §2.6 Donut / SROIE / CORD training-data prep — what works in practice

Two production write-ups [17, 18] document Donut fine-tuning on SROIE (1 000 receipts; ICDAR 2019) and CORD (1 000 Indonesian receipts). The mechanical lessons:

- **Dataset format is rigid.** Image + `metadata.jsonl` with `{"file_name": "xx.png", "ground_truth": "{\"gt_parse\": {...}}"}`. Donut interprets *every* task as JSON prediction. Same for Qwen2.5-VL fine-tunes that we're doing.
- **Image size matters more than you think.** Schmid [17] downsized SROIE images from 1 920 × 2 560 → 720 × 960 to fit memory; OK for receipts, lossy for full A4 invoices with small line-item fonts. Keep ≥ 1 280 × 1 920 for IE A4 invoices.
- **Even 100 images can fine-tune Donut.** Desaraju [18] reported usable results with **100 SROIE images** (out of 626) on a single Colab T4. This is the empirical floor — 500 synthetic + 50-100 real corrections is well above it.
- **`gt_parse` schema must match what you'll query at inference.** Sloppy schemas during training poison the production output. Lock the JSON schema in v0.4.0/v0.4.1 (already done in `extract.py` Pydantic models) and emit synth labels to *that exact schema*.

### §2.7 LayoutLM / LayoutLMv3 training data prep

LayoutLMv3 (Microsoft, ACL 2022, MIT license on the repo) is fed **(image, OCR words, OCR word bboxes, label per word)** — synthetic generation needs to emit not just the rendered image but also the word-level bboxes. Two clean ways to do this:

1. **Render via HTML → PDF → pdfplumber/PyMuPDF**, then re-extract word bboxes from the digital text layer (cheap, exact, only works on un-corrupted PDFs).
2. **Render → run OCR (RapidOCR) → align OCR boxes to known content** (works on corrupted images but introduces OCR errors into the labels — only do this if your OCR engine is the same one used at inference time, which gives an honest end-to-end training signal).

For Qwen2.5-VL (our target), neither bbox source is strictly required — the VLM ingests raw pixels. But maintaining bbox provenance is cheap insurance for later layoutLM-style baselines and for the active-learning correction UI (described in active-learning-flywheel.md §3.1).

### §2.8 Variation strategies catalogued across the literature

| Axis | What to vary | How papers vary it | Notes for IE |
|---|---|---|---|
| **Vendor name** | Faker `en_IE` / Mimesis `Eire` Address provider | DocILE: Mimesis [1, 2] | Add IE-flavoured corpus: pub names, supermarket chains, agri-suppliers, building/trade |
| **Layout** | 8-12 base templates, varied per render | DocILE 100 layout clusters | Anchor to real IE archetypes (§4.1 below) |
| **Font** | TTF rotation: Arial, Liberation Sans, DejaVu, Noto Sans, Inter, Source Sans | SynthDoG [4]; TRDG [8] (random font pool) | Avoid Inter (AI-tell, see ~/.claude/rules/common/self-check.md). Use commodity desktop / receipt-printer fonts. |
| **Font size** | 8-14 pt body, 14-24 pt header | SynthDoG; TRDG `-f` flag | IE bookkeeping receipts run small (8-9 pt body) |
| **Line-item count** | 1-25 lines, log-uniform | DocILE LIR statistics | Sole-trader median is 3-7 lines; weight accordingly |
| **VAT-rate distribution** | Single-rate, multi-rate, rate-summary footer | Adevinta blog § hybrid; Aldi-style summary [operator-findings-ie.md §2.4] | IE-specific: 23/13.5/9/4.8/0 — never US-style |
| **Date format** | DD/MM/YYYY, DD-MM-YYYY, "12 March 2026" | Faker with locale | IE always DD/MM/YYYY in commerce; `Y_m_d` is database-style and rare on invoices |
| **Currency formatting** | "€1,234.56", "EUR 1.234,56", " €1234.56" | Faker locale | IE: comma thousands, dot decimal — but expect either order; receipts often omit thousand-comma |
| **Logo placement** | None, top-left, top-right, header banner | DocILE layout clusters | ~30 % of sole-trader invoices have no logo at all |
| **Field labels** | "Invoice no.", "Inv. No.", "Number", "Reference" | DocILE keys | Capture this lexical variation in the seed templates |
| **Total reconciliation** | Subtotal+VAT=Total; Total alone; gross-with-VAT-included | Mindee invoice schemas | Cross-field validation in v0.4.0 already encodes this |

---

## §3 Corruption / augmentation strategies — bridging the synth → real gap

### §3.1 Augraphy (Apache-2.0) — the document-AI augmentation library

**Augraphy** (sparkfish/augraphy, arXiv:2208.14558) [19, 20] is the only major augmentation library purpose-built for paper-document distortions (vs. albumentations / imgaug, which are camera-image-centric). Apache-2.0; PyPI `augraphy`; 26-35 augmentations across three phases:

- **Ink phase** — InkBleed, BleedThrough, Letterpress, LowInkPeriodicLines, LowInkRandomLines, Dithering
- **Paper phase** — PaperFactory (paper-texture overlay), ColorPaper, BrightnessTexturize, NoiseTexturize, WaterMark
- **Post phase** — Geometric (rotate / perspective), Folding, BadPhotoCopy, DirtyDrum, DirtyRollers, Faxify, JPEG, Markup, Scribbles, ShadowCast, LightingGradient, ReflectedLight, SubtleNoise

This library is the **default pick for v0.4.2 corruption**. It exposes rate parameters (`Jpeg(quality_range=(25, 95))`, `Geometric(rotate_range=(-5, 5))`) so the 30/50/20 split maps cleanly onto a single AugraphyPipeline with weighted phases. It also handles **mask/keypoint/bbox transforms** — the geometry is applied identically to ground-truth bboxes, which preserves label fidelity post-rotation.

### §3.2 scan-simulator (license unspecified — verify before adoption)

`s1mb1o/scan-simulator` [21] (April 2026) — 25 transforms across 6 presets (`scan-clean`, `scan-heavy`, `photo-indoor`, `photo-outdoor`, `photocopy`, `archive`). Adds **camera-photo** primitives Augraphy lacks: chromatic aberration, lens defocus, motion blur, moire, scanner-lid shadow, book-binding curvature. License **needs verification** before commit — the README does not state one. If MIT/Apache, layer on top of Augraphy for the heavy-photo bucket; if GPL or absent, skip and use Augraphy + manual OpenCV for photo effects.

### §3.3 SynthDoG / synthtiger augmentations [4, 6]

YAML-controlled rotation, blur, background paper textures, font randomization. Light-touch; useful for the SynthDoG-as-renderer path. Apache-2.0.

### §3.4 The papers on which corruption recipes work

- **Augraphy (arXiv:2208.14558)** [20] — three-phase pipeline rationale; documents that 26 augmentations × paper-factory composition fabricates printer/scanner/fax artifacts.
- **scrambledtext (arXiv:2409.19735)** [11] — synthetic CER buckets (0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4 at 25 % each) achieved 55 % CER reduction / 32 % WER reduction on a 19th-century newspaper LLM-correction task; **non-uniform character corruption beats uniform corruption**.
- **scrambledtext heuristic 4** [11] — train on under-corrupted data > over-corrupted data. (Translation: 30/50/20 split, not 20/40/40, despite the temptation to weight heavy.)
- **Tatusch et al. (arXiv:2209.14970)** [13] — 4× augmentation enlargement gives -14.92 pp CER and -18.19 pp WER on Tesseract LSTM; 4×→8× is marginal.
- **Post-OCR synthetic methods comparison (arXiv:2408.02253, ACL 2024)** [14] — glyph-similarity-based corruption (visually-confusable substitutions) outperforms uniform character noise; ByT5 + 4× augmentation reduced CER 31-48 % across English/Russian/Spanish/Frisian. **Important for IE**: include glyph-confusable substitutions in the corruption layer (€↔E, l↔I↔1, O↔0, S↔5, B↔8) — operator-findings-ie.md §2.3 already flagged EasyOCR's `$↔8` and `€↔E` failures.
- **Mind the Gap (arXiv:2405.03243)** [10] — final 2 model layers carry most of the synth→real gap; pre-train all but last 2 with synth, fine-tune last 2 on real. ~1/8 real data ≈ 7 pp drop vs full real; no synth pre-training ≈ 20 pp drop. **Pre-training is worth the engineering.**
- **GaFi (arXiv:2305.10118)** [22] — three post-processing techniques (Dynamic Sample Filtering, Dynamic Dataset Recycle, Expansion Trick) reduce real-vs-synth accuracy gap to 2-4 pp on classification tasks. Worth knowing exists; not v0.4.2 priority unless the 50-real-doc fine-tune ceiling proves too low.
- **Effective Synthetic Data (ACL 2024 / aclanthology.org/2024.emnlp-main.862)** [15] — defines "effective" CER thresholds for synthetic data; SCN-TTA (test-time adaptation) reduced CER 68.67 % without manual annotation. Directional input for v0.5; out of scope for v0.4.2.
- **Synthetic Mixed Training (arXiv:2603.23562)** [23] — mixing synthetic-QA + synthetic-document at 1:1 with 10 % real data hits log-linear scaling up to 700 M tokens, surpassing RAG +2.6 %. Different domain (long-context QA) but the **1:1 mix + 10 % real** anchor is robust across literature.

### §3.5 Recommended augmentation parameter ranges (concrete numbers for v0.4.2)

Pulled from Augraphy default complex pipeline [19, 20] and the Donut/SROIE production write-ups [17]:

```python
# Light corruption bucket (~50 % of corpus, 250 docs)
ink_phase = [
    InkBleed(intensity_range=(0.1, 0.3), kernel_size=(3,3), p=0.33),
]
paper_phase = [
    PaperFactory(p=0.33),
    BrightnessTexturize(p=0.33),
]
post_phase = [
    Geometric(rotate_range=(-2, 2), p=0.5),  # ≤ 2 degrees
    Jpeg(quality_range=(70, 95), p=0.5),      # mild JPEG
    LightingGradient(max_brightness=255, min_brightness=180, p=0.33),
    SubtleNoise(p=0.33),
]

# Heavy corruption bucket (~20 % of corpus, 100 docs)
ink_phase = [
    InkBleed(intensity_range=(0.5, 0.8), p=0.5),
    BleedThrough(intensity_range=(0.1, 0.4), ksize=(17,17), alpha=0.2, p=0.33),
    Letterpress(p=0.33),
]
paper_phase = [
    PaperFactory(p=0.66),
    ColorPaper(hue_range=(0,255), saturation_range=(10, 40), p=0.33),
    NoiseTexturize(p=0.33),
]
post_phase = [
    Geometric(rotate_range=(-7, 7), p=0.66),
    Jpeg(quality_range=(25, 60), p=0.66),
    OneOf([
        BadPhotoCopy(noise_iteration=(1,3), noise_size=(1,3), p=1),
        DirtyDrum(p=1),
        Faxify(p=1),
    ], p=0.5),
    Folding(fold_count=random.randint(1, 4), p=0.33),
    LightingGradient(max_brightness=255, min_brightness=80, p=0.5),
    ShadowCast(p=0.33),
]

# Clean bucket (~30 % of corpus, 150 docs) — render PDF → PNG, no Augraphy at all
```

Glyph-confusable substitutions (for the **OCR-engine** retrain track only — *not* the VLM track, since VLMs see pixels and glyph confusables map back trivially) per §3.4 [14]:

```
€ → E, l → I → 1, O → 0, S → 5, B → 8, G → 6, Z → 2
```

---

## §4 IE-specific patterns — what must be modelled

Note: §4.1-4.4 below are content-generator templates, not corruption parameters. Each is anchored to a real IE document family. Most are corroborated in `operator-findings-ie.md` already; numbers below cross-cite where new.

### §4.1 IE VAT invoice (Revenue.ie reference) [24, 25]

**Required fields** (Revenue.ie, 2026-03-16) [24]:

- Date of issue
- Unique sequential number
- Supplier full name, address, VAT number
- Customer name + address (B2B)
- Customer VAT number (reverse-charge or intra-Community supply only)
- Description of goods/services + quantity + unit price ex-VAT
- VAT rate(s) and total VAT
- Breakdown by VAT rate (multi-rate invoices)
- Date of supply if different from issue date
- Foreign-currency: EUR equivalent at Central Bank rate

**IE VAT rates** (2026): standard 23 %, reduced 13.5 %, reduced 9 %, super-reduced 4.8 %, zero 0 %. Revenue 2026 budget made some sector-specific rate moves (effective 2026-01-01 and 2026-07-01) [25]. Generator must support multi-rate invoices (≥ 2 rates appears on ~30 % of B2B invoices empirically — Adam's data point, cross-check with §6 below).

**Special-case wording** [24]:

- Reverse charge: "reverse charge applies"
- Intra-Community supply: "intra-Community supply of goods"
- Triangulation: "EC triangulation simplification"
- Margin scheme (second-hand): "Margin scheme — second-hand goods"
- Auctioneer scheme: "Margin scheme — auction goods"
- Construction services subject to RCT: "VAT on this supply to be accounted for by the Principal Contractor"

**Simplified invoice** allowed when total ≤ EUR 100 [25].

### §4.2 RCT (Relevant Contracts Tax) — construction subcontractor invoice format [26, 27, 28, 29]

**Pattern (Revenue Part 18-02-04)** [26]:

- Subcontractor invoice **omits VAT rate and VAT amount** (reverse charge applies)
- Invoice **must include** the wording: "VAT on this supply to be accounted for by the Principal Contractor"
- Subcontractor's VAT number **still appears** on invoice (the subcontractor is VAT-registered; just doesn't charge VAT on this supply)
- RCT rate (0 / 20 / 35 %) and RCT deduction **do not appear on the invoice** — they live on Revenue's eRCT deduction authorisation, separate document
- Hand-writing the reverse-charge narrative on a printed invoice is **not acceptable** unless the whole invoice is hand-written (Teagasc supplier policy [28])

Generator template should produce ~5-10 % of synth corpus as RCT subcontractor invoices to make sure the model learns "VAT-number-present-but-no-VAT-rate" is a valid state, not a bug.

### §4.3 Supermarket per-letter VAT codes (Tesco / SuperValu / Aldi / Lidl / Dunnes) [30, 31, 32, 33]

Each chain uses a different per-line letter code to indicate VAT treatment. Customers must **request** a VAT receipt at customer service to get the breakdown — the till receipt alone is **insufficient for input-VAT reclaim** in most cases [30, 32].

| Chain | Letter code | Meaning |
|---|---|---|
| **Tesco IE** | A, B, Z (varies by product) | A = standard 23 %, B = reduced 13.5 %, Z = zero 0 %. Footer prints VAT-rate summary by code. |
| **Aldi IE** | per-line VAT breakdown printed at footer | per-rate summary block — easy parse [operator-findings-ie.md §2.4] |
| **Lidl IE** | per-line VAT breakdown printed at footer | per-rate summary block — easy parse |
| **SuperValu** | per-line VAT code (retailer-specific letter scheme) | Must collect from customer service for VAT-compliant receipt |
| **Dunnes** | per-line letter (often `*` indicates VATable) | Customer-service VAT receipt required for compliance |
| **B&Q / Argos** | Often VAT number on **back** of receipt; starred items = VATable [30] | Two-page receipt pattern |

ASDA UK convention (informative, not IE) [33]: D = VAT not payable, V = VAT payable, M = VAT payable kiosk/tobacco, no indicator = VAT not payable.

**IE-specific invariants:**

- Per-letter code per line + footer-summary block = 60-70 % of supermarket receipts
- Long thermal-receipt format: 80 mm wide × variable length (200-1500 mm)
- Often missing customer-services VAT block (just till receipt)
- Star (`*`) sometimes indicates VATable items (B&Q UK pattern, occasionally seen on IE retailers' rebadges)

### §4.4 Bank statement layouts (AIB / BOI / PTSB / Revolut / Wise / N26) [34, 35, 36]

bankstatementlab.com [34] is direct: **"There is no ISO standard for bank statement layout."** Each institution's PDF generation stack evolved independently. Concrete failure modes documented:

- Column order varies (date / desc / debit / credit / balance vs. date / desc / amount / balance with sign embedded)
- Multi-page running totals handled inconsistently → naive parsers double-count or miss rows at page breaks
- "Digital" PDFs may embed text as custom glyph mappings instead of Unicode → extracted "€1,234.56" comes out garbled
- Some banks render transaction tables as PDF table structures; others place each cell as standalone X/Y-positioned text objects with **no structural row relationship**
- Anchor-string drift: "Date" → "Value Date" between statement template versions breaks regex extractors

**Per-bank notes:**

- **AIB** [35]: PDF-only download (no native CSV); EUR currency; current/business/savings statement variants
- **Revolut** [36]: PDF uses complex vector graphics + embedded fonts (defeats naive copy-paste); multi-currency statements with per-currency sections; currency-exchange rows span two visually-linked rows; ISO-8601 dates available
- **BOI / PTSB**: similar PDF-only patterns; column layouts differ per product
- **Wise / N26**: digital-native EU banks; PDFs cleaner but multi-currency
- **Strip / GoCardless**: settlement statements not technically bank statements but extracted by the same pipeline

For synth: generate 5-7 archetypal bank-statement layouts. Each archetype = a different (column-order, header-style, row-format) combination. Page-break handling is the **nasty edge case** — synth-generate multi-page statements with running-balance footer to teach the model the page-stitch pattern (operator-findings-ie.md §2.5 § "5. Bank statement reconciliation across page breaks").

### §4.5 Section 84(3) original-form attestation requirements

VAT Consolidation Act 2010, s.84(3) requires a VAT-registered trader to retain invoices and credit notes for 6 years from the date of supply [37]. Practically, a "Section 84(3) attestation" sometimes appears as a watermark or stamped footer on receipt-bank-style scans certifying the document is a true copy of the original. Low-frequency in primary corpus (sub-2 %), but the watermark pattern is worth one synth template variant so the model doesn't over-key on it as a signal.

### §4.6 Practical IE-flavour content corpus (Faker / Mimesis input)

- **Vendor names**: Irish surnames (Murphy, Kelly, O'Brien, Walsh, Byrne) + commercial suffixes (Ltd, Limited, & Co., Trading As, T/A) + sector words (Plumbing, Construction, Veterinary, Catering, Agri, Solicitors)
- **Addresses**: Eircode format (`A65 F4E2` — letter+digit+digit+space+letter+digit+letter+digit; ~14 routing keys); county names (Limerick, Cork, Galway, Dublin, etc.); common city/town names per county
- **Phone**: `+353 nn nnn nnnn` or `0nn nnn nnnn`
- **VAT numbers**: `IE` + 7-8 digits + 1-2 letters (e.g., `IE1234567T`, `IE1234567TW`) — Revenue VIES validator pattern
- **Bank account format**: IBAN (`IE` + 2 check + 4 BIC + 6 sort + 8 account = 22 chars total)
- **Currency lexicon**: `EUR`, `€`, `Euro`, `e` (informal, rare)
- **Pub names + supplier mix**: pub catalogue for Limerick/Munster from Wikipedia / OSM is a free corpus — one-liner scrape via Overpass API

Mimesis `Address` provider supports Eire as `eire-en` locale; Faker has `en_IE`. Both Apache-2.0/MIT.

---

## §5 Corruption-percentage curves — recommended split (with citations)

Cross-paper synthesis on the synth → real ratio:

| Source | Recommended split | Notes |
|---|---|---|
| **scrambledtext (arXiv:2409.19735)** [11] | 4 buckets at 25 % each (CER 0-0.1, 0.1-0.2, 0.2-0.3, 0.3-0.4) | Cumulative: 25 % clean-ish, 50 % medium, 25 % heavy. Best transfer when median synth CER ≈ median real CER. |
| **Synthetic Mixed Training (arXiv:2603.23562)** [23] | 1:1 synth-QA : synth-doc + 10 % real | Different task (long-context QA), but **10 % real** anchor recurs in literature. |
| **Tatusch et al. (arXiv:2209.14970)** [13] | 4× augmented dataset over 1× clean | Translation: 25 % clean / 75 % corrupted of *some* form. |
| **Mind the Gap (arXiv:2405.03243)** [10] | All-but-last-2-layers synth, last-2-layers real (≥1/8 of full real) | Translation: ratio is *layer-wise*, not row-wise. Compatible with any row-split. |
| **ritw.dev compliance write-up (production)** [9] | "Apply visual noise across the board" | 30→85 % accuracy lift on real scans when augmentation enabled vs disabled. |
| **DocILE pre-training results** [1] | 100k synth + 5.7k real, *all synth corrupted to varying degrees* | RoBERTa+SYNTH and LayoutLMv3+SYNTH outperform baselines without synth pre-train. |

**My recommendation for v0.4.2 — 30 / 50 / 20 split (clean / light / heavy):**

- ≈ scrambledtext 25/50/25 with 5 pp shifted toward clean — IE invoices are dominantly digital-PDF (PEPPOL clock starts Nov 2028, but most B2B is already structured-PDF email today per `operator-findings-ie.md` §1)
- Compatible with Tatusch's 4× rule (each clean-PDF master → 1 clean rendered + 2-3 augmented variants)
- Hits the "median synth corruption ≈ median real corruption" target from scrambledtext
- Concrete: 500 docs = 150 clean / 250 light / 100 heavy

**Knock-on:** generate **125 clean masters**, render each to PDF→PNG once at clean (= 125 clean), then sample 4× augmentation per master → 250 light + 100 heavy + 25 retained as clean replicas (round to 150 clean). The 125 master count is also right at DocILE's 100-layout-cluster footprint [1], which empirically supports 100k-doc downstream synth — generous headroom for 500.

---

## §6 Python libraries — concrete picks

Beyond Faker + reportlab, these open-source tools materially reduce v0.4.2 build time. License-checked.

### §6.1 Content + structured generation

| Library | License | Role | Notes |
|---|---|---|---|
| **Faker** [pypi.org/project/Faker] | MIT | Names, addresses, dates, phones, IBANs, EUR amounts | `en_IE` locale; standard. |
| **Mimesis** [pypi.org/project/mimesis] | MIT | Same surface as Faker, Eire address provider | DocILE [1, 2] anchor library; richer EU coverage than Faker for some fields. |
| **schwifty** [pypi.org/project/schwifty] | MIT | IBAN / BIC validation + generation | IE-IBAN-correct (22 chars, country IE). |
| **eircode** [GitHub: johnthechip/eircode-py] | MIT | Eircode format generator | One-liner; verify license before commit. |
| **Pendulum** [pypi.org/project/pendulum] | MIT | Datetime formatting in IE locale | `format("DD/MM/YYYY")` etc. |

### §6.2 Rendering (clean PDF → image)

| Library | License | Role | Notes |
|---|---|---|---|
| **reportlab** [pypi.org/project/reportlab] | BSD-3-Clause | Programmatic PDF (low-level: canvas, table, paragraph) | Heavy lifting for invoices with consistent margin/spacing. |
| **WeasyPrint** [pypi.org/project/weasyprint] | BSD-3-Clause | HTML/CSS → PDF | Easier styling + DocILE-style HTML-first templates [1]. |
| **fpdf2** [pypi.org/project/fpdf2] | LGPL-3.0 | Lightweight PDF | LGPL — link compatible (dynamic) but **prefer reportlab/WeasyPrint** for proof-fixtures' Apache-only floor. |
| **pdfplumber** [pypi.org/project/pdfplumber] | MIT | Re-extract bboxes from rendered PDF | Round-trip for `metadata.jsonl` ground truth. |
| **pypdfium2** [pypi.org/project/pypdfium2] | Apache-2.0 / BSD-3-Clause | PDF → PNG raster (already in v0.4 stack) | Use existing pipeline. |

### §6.3 Augmentation / corruption

| Library | License | Role | Notes |
|---|---|---|---|
| **Augraphy** [pypi.org/project/augraphy] | Apache-2.0 / MIT | Document-AI corruption (35 augs, 3 phases) | **First pick.** Mask/keypoint/bbox-aware. |
| **synthtiger** [pypi.org/project/synthtiger] | MIT (per Donut repo) | SynthDoG renderer + corrupter | Backup if Augraphy heavy-rotation insufficient; layered with SynthDoG. |
| **TRDG** [pypi.org/project/trdg] | MIT | Word/line-level OCR text-image generator | OCR-engine retrain only; not page-level. |
| **albumentations** [pypi.org/project/albumentations] | MIT | Generic image augmentations | Camera noise / lens blur if scan-simulator license is ambiguous. |
| **scan-simulator** [GitHub: s1mb1o/scan-simulator] | **License TBD — verify** | Camera-photo-style augmentations | 6 presets ready-to-use; only adopt if license clears. |
| **lama-cleaner** / **simple-lama-inpainting** [pypi] | MIT (lama-cleaner) | Inpainting for SynthID-style content rewrite | Optional — only for §2.4 layout-preserving content replacement. |

### §6.4 Active-learning / labeling integration

Already covered in active-learning-flywheel.md §2 — Label Studio CE (Apache-2.0) is the chosen capture surface. Synth-generated metadata.jsonl can be batch-imported into Label Studio for in-line correction during the synth → real bridge phase.

---

## §7 v0.4.2 implementation order

Ten steps, sized to chat-exchange units (per `~/.claude/rules/common/self-check.md` §9 — rounds take minutes, not days).

**Phase A — Seed templates (1-2 rounds)**

1. **Annotate 8-12 IE seed layouts.** Hand-draw bbox + slot semantics for: (a) plain trade B2B invoice, (b) sole-trader simplified ≤€100 invoice, (c) RCT subcontractor reverse-charge invoice, (d) Tesco-style supermarket receipt with per-letter codes, (e) Aldi/Lidl per-VAT-rate footer summary, (f) AIB current-account statement (1-page), (g) AIB current-account statement (multi-page with running-balance), (h) Revolut multi-currency statement, (i) intra-Community supply invoice, (j) margin-scheme second-hand-goods invoice. Output: `proof-fixtures/synth/templates/{layout_id}.html` × 8-12 + `bbox_schema.json`.

**Phase B — Content engine (1 round)**

2. **Build IE-flavour content engine** (`proof-fixtures/synth/generators/content_ie.py`): Faker(`en_IE`) + Mimesis Eire + schwifty IBAN + eircode generator + IE pub/supplier corpus + IE VAT-rate distribution sampler (sole-trader weighted). Wires into the bbox slot map from §A1.

**Phase C — Renderer (1 round)**

3. **HTML → PDF → PNG render path.** WeasyPrint (HTML→PDF) + pypdfium2 (PDF→PNG). Sidecar emit per-doc `metadata.jsonl` matching v0.4 Pydantic schema (= zero schema drift between synth pre-train and real-correction fine-tune).

**Phase D — Corruption layer (1 round)**

4. **Augraphy pipeline** with three named profiles: `clean` (no Augraphy), `light` (50 % bucket, parameters in §3.5), `heavy` (20 % bucket, parameters in §3.5). Sampler script: weighted draw 30/50/20.

**Phase E — Generate the 500-doc corpus (1 round)**

5. **Generate corpus** at 125 unique-master × 4 augmentations + 25 retained-clean replicas = ~525 docs. Audit: validate every metadata.jsonl row reconciles (subtotal + VAT = total or RCT-marker present); reject + regenerate failures.

**Phase F — Pre-train (1-2 rounds)**

6. **Stage 1 pre-train Qwen2.5-VL-3B** with vision tower frozen (per Qwen2.5-VL fine-tuning guide [38, 39, 40, 41]; LoRA r=16-32, alpha=32-64, target_modules `["q_proj", "v_proj", "o_proj", "up_proj", "down_proj"]` on **language layers only**, not vision). 4-bit QLoRA via bitsandbytes if VRAM ≤ 16 GB. Train 2-3 epochs, lr 1e-4-2e-4, batch 1-2 with grad-accum. Plan A: Adam's ZBook RTX 8 GB → 4-8 hrs. Plan B: RunPod RTX 4090 burst $0.74/hr × 2-4 hrs ≈ $2-4 per pre-train run.

**Phase G — Bridge to real (deferred until first 50 customer corrections — not v0.4.2 scope)**

7. **Stage 2 fine-tune** on first 50 real corrections from Label Studio. Per Mind-the-Gap [10]: freeze all layers except final 2, train 1 epoch, lr 5e-5. Closes most of the synth→real gap.
8. **Quarterly re-pre-train** as IE-vendor mix shifts (per active-learning-flywheel.md §1).

**Phase H — Evaluate (1 round, ongoing)**

9. **Eval against `00-test` real-fixture set** (20 docs in v0.4.0). Track per-field F1 + structural exact-match. Goal: synth-only model ≥ 70 % F1 on real fixtures (literature floor [9, 10]); post-stage-2 model ≥ 90 % F1 on real fixtures.

**Phase I — Crystallize (deferred)**

10. **Promote synth pipeline** into a routed workflow package (per `~/.claude/CLAUDE.md` § "Workflow Crystallization") once 2nd customer corpus runs cleanly through it.

**Build-first / build-second priority order if cycle time is tight:**

- Build first: **§A1 templates, §A2 content engine, §A3 renderer, §A5 corpus generation** (no GPU needed; produces the 500 docs that unblock everything else).
- Build second: **§A4 corruption layer** (only matters once you have clean masters to corrupt). Augraphy install is 15 minutes; corpus regeneration is reversible.
- Build third (only after corpus is green): **§A6 pre-train run** (GPU-bounded; can move to RunPod burst).
- Defer: **§A7 bridge, §A8 quarterly retrain, §A10 crystallization.**

---

## §8 License posture summary (Apache-2.0 / MIT only — no GPL/CC-BY-NC blockers)

| Tool | License | Usable? |
|---|---|---|
| Augraphy | Apache-2.0 / MIT | ✓ Yes |
| SynthDoG / synthtiger | MIT (per Donut clovaai repo) | ✓ Yes |
| TRDG | MIT | ✓ Yes |
| Faker | MIT | ✓ Yes |
| Mimesis | MIT | ✓ Yes |
| schwifty | MIT | ✓ Yes |
| reportlab | BSD-3-Clause | ✓ Yes (BSD-3 is permissive, compatible) |
| WeasyPrint | BSD-3-Clause | ✓ Yes |
| pypdfium2 | Apache-2.0 / BSD-3-Clause | ✓ Yes (already in stack) |
| pdfplumber | MIT | ✓ Yes |
| albumentations | MIT | ✓ Yes |
| lama-cleaner | Apache-2.0 | ✓ Yes |
| **fpdf2** | **LGPL-3.0** | ✗ Avoid for proof-fixtures Apache-only floor (LGPL link rules add ambiguity) |
| **scan-simulator** | **License TBD** | ⚠ Verify before use; do not commit until license declared |
| **DocILE dataset** | Research-use only (CC-BY-NC adjacent terms in some files) | ✗ Do not redistribute; OK to read-only-research the methodology only |
| **SynthID inpainting repo (BevinV)** | **Unspecified** | ⚠ Read paper, reimplement from scratch; do not vendor the repo |

---

## §9 Open questions / risks (honest accounting)

1. **scan-simulator license is unstated.** README does not declare. Mitigation: audit before any integration; if no license, reimplement the 5-6 transforms we need (chromatic aberration, lens defocus, motion blur, moire, scanner-lid shadow) in-house against OpenCV.
2. **SynthID-style inpainting depends on OCR + LLM correctness on seed invoice.** GLM-OCR fails ≈ 10-15 % of fields per `operator-findings-ie.md` §2.1; those failures propagate into synth labels. Mitigation: only use SynthID rewrite for the *content* axis, not the *layout* axis — keep the bbox skeleton from the seed (perfect by construction) and only swap inner text. Cross-validate every rewritten field with cross-field reconciliation (subtotal+VAT=total) before accepting.
3. **Layout-cluster diversity is bounded by the seeds.** 8-12 templates × 4× augmentation gives 32-48 layout-token combinations; real Limerick supplier mix has 50-200+ active layouts. Mitigation: synth gets you to 70-80 % F1 floor; the 50-100 real-correction fine-tune is where the long-tail layouts get learned. Don't over-invest in synth diversity past 12 base templates.
4. **VAT-code per-letter scheme on supermarket receipts is retailer-private.** Tesco's A/B/Z mapping isn't published; it's reverse-engineered from operator forums [30, 31]. Mitigation: synth template generates a *plausible* mapping consistent with footer-summary block; the model learns the *structure*, not the specific letter assignments. Real corrections from Adam's customers fix the letter mapping per chain.
5. **Ground-truth schema drift between synth and real.** If v0.4.0/v0.4.1 Pydantic schemas evolve, regenerate the synth corpus's `metadata.jsonl` (not the images — a script-only change). Plan for one regeneration cost per schema change. Lock the schema by v0.4.2 ship to minimize.
6. **Qwen2.5-VL fine-tune target_modules misconfiguration risk.** Qwen2.5-VL has `q_proj`/`v_proj`/etc. in *both* the language layers *and* the vision encoder. If LoRA targets unintentionally include vision-encoder modules while the encoder is "frozen," cross-modal alignment breaks (text fluent, vision fails) [39]. Mitigation: explicit target-module filter by name pattern (`"language_model.layers.*.q_proj"` only), and post-load assertion `[name for name, p in model.named_parameters() if 'vision' in name and p.requires_grad] == []`.

---

## §10 References

[1] Šimsa et al., **DocILE Benchmark for Document Information Localization and Extraction**, ICDAR 2023, arXiv:2302.05658 — https://arxiv.org/abs/2302.05658 ; repo https://github.com/rossumai/docile

[2] Mimesis (Apache-2.0 EU-locale data generator) — https://pypi.org/project/mimesis/

[3] Provectus, **Synthetic Invoice Dataset Generator (whitepaper)** — https://provectus.com/synthetic-invoice-dataset-generator-whitepaper/

[3a] Verma, **Generating Synthetic Invoices via Layout-Preserving Content Replacement (SynthID)**, arXiv:2508.03754 — https://arxiv.org/html/2508.03754v1

[3b] BevinV/Synthetic_Invoice_Generation (companion repo) — https://github.com/BevinV/Synthetic_Invoice_Generation

[4] SynthDoG (Donut Synthetic Document Generator) — https://github.com/clovaai/donut/tree/master/synthdog ; PyPI https://pypi.org/project/synthdog/

[5] Kim et al., **OCR-free Document Understanding Transformer (Donut)**, ECCV 2022, https://github.com/clovaai/donut

[6] synthtiger (PyPI) — https://pypi.org/project/synthtiger/

[7] aiviewz, **Generating Synthetic RTL OCR Data for Donut with SynthDoG-RTL**, dev.to 2025-09-23 — https://dev.to/aiviewz_team/generating-synthetic-rtl-ocr-data-for-donut-with-synthdog-rtl-3ghi

[8] Belval, **TextRecognitionDataGenerator (TRDG)**, https://github.com/Belval/TextRecognitionDataGenerator ; PyPI https://pypi.org/project/trdg/ ; docs https://textrecognitiondatagenerator.readthedocs.io/

[9] ritw.dev, **Synthetic Training Data When No Dataset Exists** — https://ritw.dev/blog/synthetic-training-data-when-no-dataset-exists/

[10] Hammoud et al., **Mind the Gap Between Synthetic and Real**, arXiv:2405.03243 — https://arxiv.org/html/2405.03243v1

[11] Bourne, **scrambledtext: training Language Models to correct OCR errors using synthetic data**, arXiv:2409.19735 — https://arxiv.org/html/2409.19735v1

[12] Northcutt et al., **cleanlab confident-learning** (cited in active-learning-flywheel.md, used for label-quality monitoring).

[13] Tatusch et al., **Data augmentation framework for OCR**, arXiv:2209.14970 — https://export.arxiv.org/pdf/2209.14970v1.pdf

[14] **Advancing Post-OCR Correction: A Comparative Study of Synthetic Data**, ACL Findings 2024, arXiv:2408.02253 — https://aclanthology.org/2024.findings-acl.361/

[15] **Effective Synthetic Data and Test-Time Adaptation for OCR Correction**, EMNLP 2024 — https://aclanthology.org/2024.emnlp-main.862/

[16] Provectus, op. cit. (see [3])

[17] Schmid, **Document AI: Fine-tuning Donut for document-parsing using Hugging Face Transformers**, philschmid.de — https://www.philschmid.de/fine-tuning-donut

[18] Desaraju, **OCR-free document understanding with Donut**, Towards Data Science — https://towardsdatascience.com/ocr-free-document-understanding-with-donut-1acfbdf099be

[19] sparkfish/augraphy (GitHub) — https://github.com/sparkfish/augraphy ; docs https://augraphy.readthedocs.io/

[20] Bensiali et al., **Augraphy: A Data Augmentation Library for Document Images**, arXiv:2208.14558 — https://arxiv.org/pdf/2208.14558

[21] s1mb1o/scan-simulator (GitHub, license TBD) — https://github.com/s1mb1o/scan-simulator

[22] Besnier et al. (cited via GaFi), arXiv:2305.10118 — https://export.arxiv.org/pdf/2305.10118v2.pdf

[23] **Synthetic Mixed Training**, arXiv:2603.23562 — https://arxiv.org/pdf/2603.23562v2

[24] Revenue.ie, **What information is required on a VAT invoice?** (2026-03-16) — https://www.revenue.ie/en/vat/vat-records-invoices-credit-notes/invoices/information-required-vat-invoice.aspx

[25] Invoice Data Extraction (vendor blog), **Ireland VAT Invoice Requirements: 2026 Compliance Guide** — https://invoicedataextraction.com/blog/ireland-vat-invoice-requirements

[26] Revenue.ie, **Part 18-02-04 Relevant Contracts Tax for Principal Contractors** — https://revenue.ie/en/tax-professionals/tdm/income-tax-capital-gains-tax-corporation-tax/part-18/18-02-04.pdf

[27] Chartered Accountants Ireland, **No 31 of 2010, Section 94 Revenue Information Note** (RCT VAT reverse-charge) — https://www.charteredaccountants.ie/taxsourcetotal/2010/en/act/pub/0031/notes/sec0094-1-notes.html

[28] Teagasc, **How to get your invoices paid (RCT supplier guidance)** — https://teagasc.ie/about/corporate-responsibility/information-for-suppliers/how-to-get-your-invoices-paid/

[29] Invoice Data Extraction, **Ireland RCT Invoice Requirements: eRCT Workflow Guide** — https://invoicedataextraction.com/blog/ireland-rct-invoice-requirements

[30] Brady & Associates, **Make sure to get a VAT receipt from big stores** — https://www.bradyassociates.ie/blog/2018/12/31/make-sure-to-get-a-vat-receipt-from-big-stores

[31] Tesco.com, **FAQ: Reclaiming VAT** — https://www.tesco.com/help/pages/in-store-faqs/payment-coupons-and-vouchers/reclaiming-vat

[32] AskAboutMoney IE forum, **Tesco Receipts** thread — https://www.askaboutmoney.com/threads/tesco-receipts.168165/

[33] ASDA Stores, **What information is on my till receipt?** — https://asda-stores.custhelp.com/app/answers/detail_grow/a_id/2140

[34] BankStatementLab, **Why Bank Statement PDF Parsing Fails Across Different Banks** (2026-02-14) — https://www.bankstatementlab.com/en/blog/en-bank-statement-pdf-format-varies-parsing-fails

[35] StatementSheet, **How to Convert AIB Bank Statements to Excel or CSV** — https://statementsheet.com/how-to-convert-aib-bank-statement-to-excel-csv/

[36] PDF2TEXT.AI, **Convert Revolut PDF to Excel** — https://pdf2text.ai/en/convert/revolut-statement-to-excel/

[37] Revenue.ie, **VAT records, invoices and credit notes index** — https://revenue.ie/en/vat/vat-records-invoices-credit-notes/index.aspx

[38] sandy1990418/Finetune-Qwen2.5-VL — https://github.com/sandy1990418/Finetune-Qwen2.5-VL

[39] Neural Base, **Qwen2-VL LoRA fine-tuning** (target_modules trap) — https://theneuralbase.com/lora-qlora/learn/advanced/qwen2-vl-lora-fine-tuning/

[40] zhangfaen/finetune-Qwen2.5-VL — https://github.com/zhangfaen/finetune-Qwen2.5-VL

[41] F22 Labs, **Complete Guide to Fine-tuning Qwen2.5 VL Model** — https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/

[42] NVIDIA Megatron Bridge, **Qwen2.5-VL** (PEFT options including freeze flags) — https://docs.nvidia.com/nemo/megatron-bridge/0.4.0/models/vlm/qwen2.5-vl.html

[43] AIB Business Help Centre, **Export Statements / Transactions** — https://aib.ie/business/help-centre/manage-my-accounts/how-do-i-export-my-statements-transactions

[44] aweher/fake-invoice-generator (Argentine-VAT example, Unlicense) — https://github.com/aweher/fake-invoice-generator

[45] khushaljethava/fakeinvoicegen (Faker + InvoiceGenerator wrapper) — https://github.com/khushaljethava/fakeinvoicegen

[46] hasff/python-invoice-pdf-generator (MIT, ReportLab production-style) — https://github.com/hasff/python-invoice-pdf-generator

[47] FakeInvoiceGen docs — https://fakeinvoicegen.readthedocs.io/

[48] ecmonline/invoice-generator (WeasyPrint + YAML) — https://github.com/ecmonline/invoice-generator

[49] initios/factura-pdf (Spanish-regs, ReportLab BSD-3) — https://github.com/initios/factura-pdf

[50] Anthropic Alignment, **Modifying LLM Beliefs with Synthetic Document Finetuning (SDF)** — https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/

[51] Document AI Transformers training notebook (philschmid) — https://github.com/philschmid/document-ai-transformers/blob/main/training/donut_sroie.ipynb

[52] Andyrasika/donut-base-sroie (HF model card) — https://huggingface.co/Andyrasika/donut-base-sroie

[53] DocILE landing page — https://docile.rossum.ai/

[54] DocILE 2023 lab teaser, arXiv:2301.12394 — https://export.arxiv.org/pdf/2301.12394v1.pdf

[55] DocSynth (Layout-Guided Synthesis paper, EmergentMind summary) — https://www.emergentmind.com/papers/2107.02638

[56] Conditional Data Synthesis Augmentation (CoDSA), arXiv:2504.07426 — https://arxiv.org/abs/2504.07426

[57] Graph-Augmented Document Layouts (GNN-based synthesis), arXiv:2412.03590 — https://arxiv.org/pdf/2412.03590

[58] Augraphy — list of augmentations, full reference — https://augraphy-doc.readthedocs.io/en/latest/doc/source/list_of_augmentations.html

[59] Augraphy — BleedThrough operator detail — https://augraphy-doc.readthedocs.io/en/latest/doc/source/augmentations/bleedthrough.html

[60] Augraphy "How Augraphy Works" — https://augraphy.readthedocs.io/en/latest/doc/source/how_augraphy_works.html

[61] Konfuzio, **Create data parsing tool with Python, SROIE dataset and machine learning** — https://konfuzio.com/en/create-data-parsing-tool-with-python-sroie-dataset-and-machine-learning/

---

## §11 Cross-references to companion files (do not duplicate)

- **Active learning loop, Label Studio integration, customer-correction capture UX** → `2026-05-04-active-learning-flywheel.md` §2-§4. Synth corpus's `metadata.jsonl` schema **must** match the Pydantic schema in `extract.py` to keep zero translation cost between synth pre-train and real-correction fine-tune.
- **OCR engine comparison, Tesseract / RapidOCR / GLM-OCR / PaddleOCR-VL choices** → `2026-05-04-local-ocr-deep-research.md`. Synth corpus is upstream of OCR — for the VLM track (Qwen2.5-VL) OCR isn't on the inference path; for the OCR-engine retrain track, TRDG-generated word-level images per §6.3 are the augmentation lever.
- **Vendor architectures (DocILE = Rossum origin; Nanonets two-stage curriculum)** → `2026-05-04-vendor-architectures.md` §2.2, §2.4. Our two-stage curriculum mirrors Nanonets-OCR2's published method, using DocILE's published synth-pretrain+real-finetune pattern.
- **Operator-real-world findings on supermarket-receipt patterns, AIB statement formats, RCT specifics** → `2026-05-04-operator-findings-ie.md` §2.4 (Aldi/Tesco summary block heuristic), §2.5 (bank statement page-stitch issue), §1 (PEPPOL Nov 2028 timeline).
- **Constrained-generation Pydantic round-trips, instructor lib choice** → `2026-05-04-hybrid-stack-libraries.md` §2. Synth labels feed into the same Pydantic schema instructor enforces at inference time — schema lock-in is the single most important durability decision for v0.4.2.
- **Touch gate / second-thing protocol** → `~/.claude/rules/common/touch-gate.md`. This research file is Claude-internal and has not yet been pressed back by a second thing. Recommended next step: Codex-peer review on §5 (corruption ratio) and §7 (implementation order), and Adam veto on the 8-12 seed templates list before annotation effort.

---
