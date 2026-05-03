# proof-fixtures — Public-domain hard documents for OCR/extraction benchmarking

A reproducible benchmark corpus for document-understanding pipelines, built from **public-domain primary sources** that are deliberately hard to extract: faded scans, handwritten amendments, table-spanning rows, mixed languages, multi-column layouts, century-old typography.

**License:** MIT (this repo's content + scripts). Source documents are public-domain in their respective jurisdictions; provenance + permissions per `fixtures/{set}/PROVENANCE.md`.

**Maintainer:** Callmeie Technologies (Limerick, Ireland). Used as the proof-of-work gallery for [callmeie.ie/docs/](https://callmeie.ie/docs/) Document Ops.

---

## Why this exists

OCR + LLM extraction benchmarks ([OmniDocBench](https://github.com/opendatalab/OmniDocBench), [DocVQA](https://www.docvqa.org/), etc.) measure happy-path performance on clean modern docs. They don't measure what happens when:

- Handwritten marginalia overlap printed text
- A 1900-era ledger has 32 columns and 200 rows on one A2 spread
- A document contains 3 languages and 4 alphabets
- A 1911 Census enumerator wrote in a slanted personal hand
- A 1990s Companies House annual return was photocopied 4 times before scanning

Document Ops claims its OSS ensemble + verifier-chain pipeline handles these failure modes correctly. **This repo is the receipt.**

---

## Hero set (7 sources)

| # | Set | Source | Rights | What it tests |
|---|---|---|---|---|
| 01 | 1900 US Census | [US National Archives](https://www.archives.gov/research/census/1900) | Public domain (US gov) | Ledger tables, faded enumerator handwriting, multi-column layout |
| 02 | NAI Calendars of Wills | [National Archives of Ireland](https://www.willcalendars.nationalarchives.ie/) | Permission requested 2026-05-04 | Cursive 19th-c. handwriting, IE place-names, deceased-effects schedules |
| 03 | UK Companies House pre-2000 returns | [Companies House](https://www.gov.uk/government/organisations/companies-house) | Crown Copyright (OGL) | Photocopy artifacts, form-based extraction, multi-page contiguous-row tables |
| 04 | Ellis Island manifests | [Statue of Liberty - Ellis Island Foundation](https://heritage.statueofliberty.org/) | Public domain (US gov, pre-1924) | Multi-language passenger names, transliteration, row-spanning entries |
| 05 | Project Gutenberg company filings (UK 1900s) | [Project Gutenberg](https://www.gutenberg.org/) | Public domain | Long historical contracts, antiquated typography, footnote/sidenote layout |
| 06 | SEC EDGAR 1990s 10-K filings | [SEC EDGAR](https://www.sec.gov/edgar.shtml) | Public domain (US gov filing) | Financial table extraction, multi-section continuity, dense numeric data |
| 07 | 1911 Ireland Census | [National Archives of Ireland (Census Search)](https://www.census.nationalarchives.ie/) | Permission requested 2026-05-04 | Gaelic-language headers, household-row aggregation, slanted enumerator hand |

---

## Repo structure

```
proof-fixtures/
├── README.md                # this file
├── LICENSE                  # MIT
├── .gitignore               # excludes raw scans (downloaded by script, not committed)
├── fixtures/
│   ├── 01-1900-us-census/
│   │   ├── PROVENANCE.md    # source URLs, dates accessed, rights
│   │   ├── samples/         # ~20 doc images (downloaded by build_corpus.py)
│   │   └── ground-truth/    # human-verified extraction targets in JSON
│   ├── 02-nai-calendars-of-wills/
│   ├── 03-uk-companies-house-pre2000/
│   ├── 04-ellis-island-manifests/
│   ├── 05-project-gutenberg-filings/
│   ├── 06-sec-edgar/
│   └── 07-1911-ireland-census/
├── scripts/
│   ├── build_corpus.py      # downloads + normalises hero set from public sources
│   ├── extract.py           # runs Document Ops OSS ensemble pipeline on a fixture set
│   └── score.py             # compares extraction output against ground-truth JSON
└── results/
    ├── 2026-05-XX-baseline.md   # first benchmark run, by set + by metric
    └── ...
```

---

## Reproducing the benchmark

```bash
# 1. Clone
git clone https://github.com/scruge1/proof-fixtures.git
cd proof-fixtures

# 2. Download hero set (~500MB, varies by set)
python scripts/build_corpus.py --all

# 3. Run Document Ops pipeline (or substitute your own pipeline)
python scripts/extract.py --set 01-1900-us-census --pipeline docops

# 4. Score against ground truth
python scripts/score.py --set 01-1900-us-census --output results/$(date +%Y-%m-%d)-mybench.md
```

---

## Metrics tracked per fixture set

- **Straight-through-rate (STR)** — % of documents where pipeline auto-exports without any field below confidence floor
- **Field-level accuracy when not flagged** — for documents that pass the gate, what % of critical fields match ground truth
- **Silent-failure rate** — % of auto-exported documents that contain a wrong critical field (the worst metric — measures verifier blind spots)
- **Bounce-back rate** — % of documents that route to human review
- **Time-per-doc** — wall-clock seconds, ZBook 8GB RTX baseline

---

## Contributing

This is a benchmark, not a leaderboard. PRs welcome for:

- Additional public-domain hard documents (with verifiable rights provenance)
- Ground-truth corrections (human review caught a wrong target value)
- Extraction pipeline implementations beyond Document Ops (lifts all boats)
- Better fixture metadata (failure-mode taxonomy, difficulty scoring)

PRs that add documents whose copyright status is unclear will be closed without merge. Provenance is the point.

---

## Versioning

- **v0.1 (2026-05-04)** — scaffold + hero-set definitions. Fixtures empty pending `build_corpus.py` first run. NAI permission emails outstanding for sets 02 + 07.
- v0.2 — first benchmark results (sets 01, 03, 04, 05, 06 — sets 02 + 07 gated on NAI permission).
- v1.0 — full hero set + Document Ops baseline + at least one third-party-pipeline submission.
