# Hybrid OCR + Structured-Extraction + Verification Stack — Library Research

**Date:** 2026-05-04
**Author:** Claude (Opus 4.7, 1M ctx) collaborator session
**Scope:** Libraries that compose into a v0.4 document-AI pipeline for sole-trader bookkeeping (invoices, receipts, bank statements, RCT, contracts, freight CMR).
**Hardware floor:** AMD Ryzen 5 3500U / 30 GB RAM / Vega 8 (no CUDA). Production target: Hetzner AX52 CPU. Possibly ZBook RTX (8 GB) later.
**License floor:** Apache-2.0 / MIT only. **GPL-3.0 disqualified up front** — Surya and Marker are out and not re-evaluated here.
**Already proven in v0.3 (not re-researched):** Tesseract 5, RapidOCR (ONNX), GLM-OCR Q8 (Ollama), Poppler.

---

## §1 Executive Summary — Top 3 Per Category

| Category | Pick #1 (chosen for v0.4) | Pick #2 | Pick #3 |
|---|---|---|---|
| Constrained generation | **`instructor`** | `outlines` | `pydantic-ai` |
| Layout-aware models (CPU) | **`LayoutParser` + Detectron2 ONNX** | `LayoutLMv3` (fine-tune later) | `Pix2Text` (math/table niche) |
| Table extraction | **`pdfplumber`** (native PDFs) + **`PaddleOCR PP-StructureV3`** (scans) | `docling` | `unstructured.io` |
| Embeddings | **`sentence-transformers` / `bge-small-en-v1.5`** | `all-MiniLM-L6-v2` | `jina-embeddings-v3` (multilingual fallback) |
| NER | **`spaCy` `en_core_web_md` + `EntityRuler`** | `gliner-base-v2.5` | `flair/ner-english` |
| Vector DB | **`sqlite-vec`** | `chromadb` | `lancedb` |
| PEPPOL UBL XML | **`lxml` + `python-en16931`** | `ubllib` | `pyschematron` (validation) |
| Doc classification | **few-shot LLM (GLM-OCR/Qwen2.5-VL via Ollama)** | TF-IDF + sklearn baseline | DistilBERT fine-tune (later) |
| Multi-page handling | **`pypdfium2` + per-page extract + reconcile pass** | `docling` (built-in) | `pdf2image` + custom reducer |

**Headline rationale:** v0.4 is CPU-only, sole-trader scale (low thousands of docs/month). The picks favor **embedded, single-process, deterministic** stacks over service-oriented "platform" tools. `instructor` over `outlines` is chosen because Adam's LLM path runs through Ollama (OpenAI-compat endpoint), where `instructor` hits the well-trodden path and `outlines` requires backend gymnastics. `sqlite-vec` over `chromadb` is chosen because Hetzner AX52 wants one process and one file, not a Rust daemon to babysit.

---

## §2 Constrained Generation Comparison

The job: turn GLM-OCR / Qwen2.5-VL output into Pydantic-validated `Invoice`, `LineItem`, `Receipt`, `RCTPayment` records, with retry on validation failure. Six libraries surveyed; three serious candidates.

### §2.1 Comparison table

| Library | License | Ollama path | llama.cpp path | Pydantic-native | Retry/validate loop | CPU friendly | Maintenance (2026) | First-use friction |
|---|---|---|---|---|---|---|---|---|
| **`instructor`** (567-labs/jxnl) | MIT | **First-class** via OpenAI-compat | Yes (via OpenAI-compat) | Yes | Built in (`max_retries`) | N/A — runs on host model | Active, 15+ provider integrations | Low — `from_openai(ollama_client)` |
| **`outlines`** (dottxt-ai) | Apache-2.0 | Partial — `format: json` mode lacks schema; OpenAI-compat works but bug #1221 reports `ValidationError` on Ollama+OpenAI | First-class — `models.llamacpp` | Yes (`output_type=PydanticModel`) | None native (you wrap) | Strong — FSM logits processing | Active, vLLM/SGLang adoption | Medium — backend choice matters; OOM reports on big schemas (#658) |
| **`pydantic-ai`** (Pydantic team) | MIT | OpenAI-compat works | Indirect | Yes (canonical) | Built in (validation retries) | Depends on host | Active, official Pydantic project | Low — but agent-first framing leaks into simple extract use |
| **`guidance`** (guidance-ai) | MIT | Limited | Yes | Indirect | Manual | Strong (with llguidance) | Active. Internal JSON format being phased out for Lark | High — DSL is its own language |
| **`lmformatenforcer`** | MIT | Indirect | Yes (via transformers) | Yes | None | Strong | Active | Medium — needs HF transformers integration |
| **`xgrammar`** (mlc-ai) | Apache-2.0 | Backend-only (vLLM/SGLang/TRT-LLM default) | Yes | Indirect | None | Strong (40 µs/token reported) | Active, becoming default backend | High — engine-level, not app-level |

### §2.2 Real-world signal

- **`instructor` + Ollama is the documented happy path.** Official integration page exists; `from_openai(OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))` and pass `response_model=PydanticModel`. `mode=instructor.Mode.JSON` is the working setting against most Ollama models per the integration notes.
- **`outlines` with Ollama is rougher.** Issue #1221 (`ValidationError when trying to generate JSON using Ollama with OpenAI API`) is the clearest user-visible report of the friction. The official llama.cpp path is solid (`models.llamacpp`), but going llama.cpp means dropping Ollama as the runner — a non-trivial regression for v0.4.
- **`xgrammar` won the speed crown** but not the app-layer crown. JSONSchemaBench (2025) showed Outlines at ~93% one-shot success vs XGrammar at 60-78%, but XGrammar dominated raw decoding speed (3.5x JSON, 10x CFG). Becoming the default in vLLM, SGLang, and TensorRT-LLM as of March 2026 means it shows up under the hood without being directly imported.
- **`guidance`** is being adopted in 60% of Fortune 500 AI teams per one report — but the DSL learning curve and the in-flight format migration (JSON → Lark) make it the wrong pick for an app that just wants Pydantic round-trips.

### §2.3 Pick — `instructor`

- Lowest friction with Ollama (already proven in v0.3 for GLM-OCR Q8).
- Built-in `max_retries` removes the need to hand-roll the validation loop the bookkeeping path needs (line-item count mismatch, total reconciliation failure, etc.).
- Pydantic-native, no DSL to learn, no separate inference engine to manage.
- Apache-2.0/MIT compliant.

### §2.4 Sources

- [GitHub — dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
- [Outlines docs — llama.cpp models](https://dottxt-ai.github.io/outlines/1.1.0/features/models/llamacpp/)
- [Outlines issue #1221 — Ollama+OpenAI ValidationError](https://github.com/dottxt-ai/outlines/issues/1221)
- [Outlines issue #658 — OOM with constrained JSON schema](https://github.com/dottxt-ai/outlines/issues/658)
- [Instructor — Ollama integration](https://python.useinstructor.com/integrations/ollama/)
- [Instructor — multi-language structured outputs](https://python.useinstructor.com/)
- [Pydantic AI docs](https://ai.pydantic.dev/)
- [Pydantic AI tutorial — DataCamp](https://www.datacamp.com/tutorial/pydantic-ai-guide)
- [Invoice extraction with Pydantic — invoicedataextraction.com](https://invoicedataextraction.com/blog/pydantic-invoice-extraction)
- [Microsoft Guidance — guidance-ai/guidance](https://github.com/guidance-ai/guidance)
- [llguidance — super-fast structured outputs](https://github.com/guidance-ai/llguidance)
- [XGrammar — mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
- [XGrammar paper — arXiv 2411.15100](https://arxiv.org/pdf/2411.15100)
- [MLC blog — XGrammar performance](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)
- [Guided Decoding paper — RAG context](https://arxiv.org/html/2509.06631v1)
- [JSONSchemaBench — arXiv 2501.10868](https://arxiv.org/html/2501.10868v1)

---

## §3 Layout-Aware Models (CPU-Runnable)

The job: turn a scanned-invoice page (after RapidOCR / Tesseract) into bbox + region-class output (header, vendor block, line-item table, total block, footer) so the line-item path is fed only the relevant pixels.

### §3.1 Comparison

| Model | License | Params | CPU latency (page) | RAM (peak) | Strength | Weakness for invoices |
|---|---|---|---|---|---|---|
| **LayoutParser** (Apache-2.0) | Apache-2.0 | varies (PubLayNet detectron2 ~50M) | 1.5-3 s/page (Vega 8 estimate) | 1.5-2 GB | Apache-2.0 model zoo, multi-backend (detectron2 / effdet / paddledetection) | Pre-trained models tuned for academic papers, not invoices — needs fine-tune |
| **LayoutLMv3** (CC-BY-NC) | **CC-BY-NC base weights** | 113M (base) / 368M (large) | 3-8 s/page | 2-4 GB | SOTA on FUNSD, CORD, RVL-CDIP — token + layout joint | **License gate — base weights non-commercial.** Architecture is MIT but the pretrained checkpoint is not. Self-pretrain or use derivative checkpoints with explicit Apache-2.0 declarations only. |
| **Donut** (MIT) | MIT | 200M | 5-15 s/page on CPU | 3-5 GB | OCR-free, end-to-end JSON output, fine-tunes well on CORD/SROIE | Slow on CPU; fine-tune-required for invoice variants |
| **DiT** (MIT) | MIT | 86M (base) / 304M (large) | 1-3 s/page (classification only) | 1-2 GB | Document image classifier, layout analysis, table detection | Classifier — not a structure extractor on its own |
| **Pix2Text** (MIT) | MIT | small (~100M aggregate) | 2-5 s/page | 1-2 GB | Apache-2.0; layouts + tables + math + 80 languages → Markdown | Math focus is overhead for invoices; table model less mature than PP-StructureV3 |
| **TrOCR** (MIT) | MIT | 334M (base) | 200-500 ms/line on CPU | 1.5 GB | SOTA handwriting recognition (>90% on cursive) | Text-recognition only, no layout |

### §3.2 LayoutLMv3 license trap

Multiple invoice-extraction tutorials use `microsoft/layoutlmv3-base` without flagging that the pretrained weights are **CC-BY-NC-4.0**. The architecture is MIT and a from-scratch retrain is fine, but the convenient HuggingFace checkpoint cannot ship in a commercial product without explicit licensing review. For v0.4 this means: **treat LayoutLMv3 as a fine-tune target only after we either (a) train from random init, (b) use a community Apache-2.0 derivative we can audit, or (c) negotiate Microsoft licensing.** Don't ship the base weights.

### §3.3 Pick — `LayoutParser` (with PaddleDetection backend) for layout, defer the deep model fine-tune

For v0.4 the pragmatic shape is:
1. **Bbox + region classification** → `LayoutParser` with a PubLayNet or DocLayNet model (Apache-2.0) as the layout zone detector.
2. **Inside each region** → existing v0.3 OCR (Tesseract / RapidOCR) for text, `pdfplumber` or PP-StructureV3 for tables.
3. **Defer LayoutLMv3 / Donut fine-tunes** until we have ≥200 labeled invoices and can justify the GPU rental for fine-tune (ZBook RTX path).

Pix2Text is held in reserve for any document with embedded math (technical specs / engineering invoices) — the math+table+text combo is unique among Apache-2.0 tools.

### §3.4 Sources

- [Layout-Parser/layout-parser](https://github.com/Layout-Parser/layout-parser)
- [LayoutParser — multi-backend support](https://layout-parser.readthedocs.io/en/latest/api_doc/models.html)
- [layoutparser-ort port — ONNX runtime](https://github.com/styrowolf/layoutparser-ort)
- [LayoutLMv3 HF docs](https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3)
- [Fine-tuning LayoutLMv3 for invoice processing — TDS](https://towardsdatascience.com/fine-tuning-layoutlm-v3-for-invoice-processing-e64f8d2c87cf/)
- [UbiAI — LayoutLMv3 vs v2.5 invoice comparison](https://ubiai.tools/fine-tuning-layoutlm-v3-for-invoice-processing-and-comparing-its-performance-to-layoutlm-v2-5-5/)
- [Donut — clovaai/donut](https://github.com/clovaai/donut)
- [Donut paper — arXiv 2111.15664](https://arxiv.org/abs/2111.15664)
- [Fine-tuning Donut for invoices — philschmid.de](https://www.philschmid.de/fine-tuning-donut)
- [Fine-tuning Donut for invoice recognition — TDS](https://towardsdatascience.com/fine-tuning-ocr-free-donut-model-for-invoice-recognition-46e22dc5cff1/)
- [scharnot/donut-invoices on HF](https://huggingface.co/scharnot/donut-invoices)
- [DiT — microsoft/unilm](https://github.com/microsoft/unilm/tree/master/dit)
- [DiT paper — arXiv 2203.02378](https://arxiv.org/abs/2203.02378)
- [microsoft/dit-large model card](https://huggingface.co/microsoft/dit-large)
- [Pix2Text — breezedeus/Pix2Text](https://github.com/breezedeus/Pix2Text)
- [TrOCR — microsoft/unilm/trocr](https://github.com/microsoft/unilm/blob/master/trocr/README.md)
- [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten)

---

## §4 Table Extraction Comparison

The job: extract line items from invoices. Reality split: native (digital-born) PDFs vs. scans.

### §4.1 Comparison table

| Tool | License | Native PDF | Scans (post-OCR) | CPU latency | Quality on invoice line items | Install pain |
|---|---|---|---|---|---|---|
| **`pdfplumber`** | MIT | **Best in class** — character-level positions | No (no OCR) | <1 s/page | Strong with `extract_tables()` + custom settings; pixel-level control | Pure Python, low |
| **`camelot`** | MIT | Good (lattice+stream modes) | No | 1-3 s/page | Good for bordered tables, weak on ambiguous columns | Needs Ghostscript binary; some Windows pain |
| **`tabula-py`** | MIT | Good | No | 2-5 s/page | Strong with `lattice=True` for grid-bordered ERP exports; weak otherwise | Needs Java runtime — acceptable but adds a dependency tier |
| **`docling`** (IBM) | MIT (Apache-2.0 for Granite-Docling-258M weights) | Yes | Yes (built-in) | 114 ms/page on L4 GPU; **3-8 s/page CPU estimate** | Reported 97.9% table extraction accuracy with Granite-Docling-258M | Heavy install (PyTorch + transformers + custom converters); fast-moving release cycle |
| **`markitdown`** (Microsoft) | MIT | Yes (basic) | Plugin (`markitdown-ocr` for OCR) | <1 s/page on native | Good for "doc → markdown for LLM" but less precise than pdfplumber on invoice tables | Low |
| **`PaddleOCR PP-StructureV3`** | Apache-2.0 | Yes (renders to image) | **Yes (designed for it)** | 2-6 s/page CPU | Strong: layout + table + cell coordinates → markdown/JSON | Medium — Paddle stack on Windows is the rough edge |
| **`unstructured.io`** | Apache-2.0 (community) | Yes | Yes (with `hi_res` strategy) | 5-15 s/page on `hi_res` | Trusted by ~1/3 of Fortune 500 per their docs; line-item extraction is reasonable but not the strongest | Heavy — pulls in detectron2 + many deps; CPU-core hogging reported (issue #3291) |

### §4.2 Real-world signal

- **For native PDFs, pdfplumber is the consensus winner.** Multiple 2026 reviews (BSWEN, Lido, invoicedataextraction.com) call it out for "complex invoice layouts where column boundaries are ambiguous." It respects the PDF's character positioning rather than guessing whitespace.
- **For scans, PP-StructureV3 is the strongest Apache-2.0 option.** PaddleOCR 3.0 ships layout + table + formula + reading-order recovery and emits markdown with cell coordinates. The official tutorial highlights it as "the most invoice-relevant feature."
- **Docling is the dark horse.** Granite-Docling-258M (Apache-2.0, January 2026) reports 97.9% table extraction accuracy at 258M params. Production deployment guides exist. The risk is the moving release cadence (v2.72.0 in February 2026 was the latest in our search).
- **Tabula-py's Java requirement is real.** For Hetzner AX52 deploy, JRE adds ~200 MB and a separate runtime to monitor. For a Python-first stack this is friction.

### §4.3 Pick — split path: `pdfplumber` (native) + `PP-StructureV3` (scans)

- **Native PDF path:** `pdfplumber` first. If `extract_tables()` returns nothing usable (heuristic: <80% of expected line-item rows), escalate.
- **Scan path:** `PP-StructureV3` after RapidOCR. Cell coordinates + reading order matter for line-item reconstruction.
- **Backup probe:** `docling` as the "throw the whole doc at one tool" fallback when pdfplumber and PP-StructureV3 both fail. Don't make it the front door — the dependency cost and release-velocity risk argue for fallback-only.

### §4.4 Sources

- [Python PDF table extraction comparison — invoicedataextraction.com](https://invoicedataextraction.com/blog/python-pdf-table-extraction-invoices)
- [How to extract tables from PDF (2026 guide) — Unstract](https://unstract.com/blog/extract-tables-from-pdf-python/)
- [Camelot wiki — comparison with other extractors](https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools)
- [pdfplumber vs PyMuPDF vs Tabula — BSWEN](https://docs.bswen.com/blog/2026-03-16-pdfplumber-vs-pymupdf/)
- [Best PDF data extraction tools 2026 — Lido](https://www.lido.app/blog/best-pdf-data-extraction-tools)
- [Docling production deployment guide — Iterathon](https://iterathon.tech/blog/docling-production-deployment-guide-2026)
- [Granite-Docling-258M release — InfoQ](https://www.infoq.com/news/2025/10/granite-docling-ibm/)
- [IBM Granite-Docling announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
- [microsoft/markitdown](https://github.com/microsoft/markitdown)
- [MarkItDown — InfoWorld review](https://www.infoworld.com/article/3963991/markitdown-microsofts-open-source-tool-for-markdown-conversion.html)
- [PaddleOCR — main repo](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR 3.0 technical report — arXiv 2507.05595](https://arxiv.org/html/2507.05595v1)
- [PP-StructureV3 introduction](https://paddlepaddle.github.io/PaddleOCR/v3.1.0/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html)
- [PaddleOCR vs Tesseract — Koncile](https://www.koncile.ai/en/ressources/paddleocr-analyse-avantages-alternatives-open-source)
- [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured)
- [Unstructured CPU-cores issue #3291](https://github.com/Unstructured-IO/unstructured/issues/3291)
- [Why I spent 2026 wrestling with 10 document parsers — Medium](https://medium.com/@ravi.retheesh/why-i-spent-2026-wrestling-with-these-10-document-parsers-unstructured-io-1e389ecf40db)

---

## §5 Embedding + NER Comparison

### §5.1 Embedding models for vendor disambiguation

The job: "Chadwicks" / "Chadwicks Ltd" / "CHADWICKS LIMITED" / "Chadwick's Builders Merchants" must resolve to the same supplier_id.

| Model | License | Dim | CPU latency | RAM | MTEB rank | Suitable for? |
|---|---|---|---|---|---|---|
| `all-MiniLM-L6-v2` | Apache-2.0 | 384 | **14.7 ms / 1k tokens, 68 ms end-to-end** | ~120 MB | Mid-tier (~58 avg) | Vendor-name dedup baseline; unbeatable speed/quality on short strings |
| `bge-small-en-v1.5` | MIT | 384 | ~70-100 ms (CPU short text) | ~130 MB | Higher than MiniLM (~62) | Best speed/accuracy for vendor strings |
| `bge-large-en-v1.5` | MIT | 1024 | 350 ms CPU (one report); 80 ms GPU | ~1.3 GB | Strong (~64) | Overkill for short strings; wait for ZBook GPU path |
| `multilingual-e5-large-instruct` | MIT | 1024 | Heavy on CPU | ~2 GB | Strong on multilingual | Overkill until we hit non-English Irish/Continental invoices |
| `jina-embeddings-v3` | Apache-2.0 | 1024 (configurable) | 570M params; CPU-feasible but slow | ~2 GB | SOTA-ish (8192-token context, beats `multilingual-e5-large-instruct` reportedly) | Worth piloting for multi-page contract clauses; not for vendor strings |

**Pick: `bge-small-en-v1.5` for vendor-name embeddings.** Reasons:
- 384-dim is plenty for short strings; cheap to store at sole-trader scale.
- Outperforms MiniLM by ~3-4 MTEB points without large RAM hit.
- MIT-licensed (BAAI publishes under MIT).
- Smaller index footprint than 1024-dim alternatives — matters for `sqlite-vec`.
- **Hybrid pattern:** vectorize names with character-n-gram TF-IDF, get top-k via cosine, then re-rank top-k with `rapidfuzz` Levenshtein distance. This pattern is the consensus 2026 recommendation across multiple guides (matchdatapro, dataladder, Practical Business Python).

### §5.2 NER for header fields

The job: extract ORG / DATE / MONEY / VAT-ID-like / INVOICE-NUMBER from the OCR text stream as a backstop / cross-check against the constrained-LLM extraction.

| Tool | License | Latency | Pre-trained quality on invoices | Trainable | Notes |
|---|---|---|---|---|---|
| **`spaCy en_core_web_md`** | MIT | <50 ms/page | OK (sm reports F1 ~0.85 on standard NER) — date/money formats often mis-tagged | Yes | `EntityRuler` is the right pattern: hand-write regex for VAT numbers + IE invoice formats, let statistical NER handle ORG. Multiple 2026 invoice-parsing guides recommend this hybrid. |
| **`gliner-base-v2.5`** | Apache-2.0 | ~200-400 ms/page | Strong zero-shot on out-of-domain NER (paper claims it beats ChatGPT on out-of-domain benchmarks) | Yes (zero-shot or fine-tune) | GLiNER2 (2025) unifies NER + classification + structured extraction in <500M params CPU-efficient. Worth piloting. |
| **`flair/ner-english`** | MIT | ~500 ms/page on CPU (PyTorch overhead) | Strong on PERSON/ORG/LOC/DATE | Yes | Slower than spaCy; bigger memory footprint; not worth the swap unless training a custom model |
| **`flair/ner-english-large`** | MIT | 1-3 s/page CPU | Best Flair tier | Yes | Too slow for batch CPU pipeline |
| **DistilBERT-NER fine-tune** | Apache-2.0 (model arch) | ~150 ms/page | Domain-specific; potentially best | Yes | Defer until labeled-data threshold reached |

**Pick: `spaCy` `en_core_web_md` + `EntityRuler` regex layer (VAT, invoice-number, IE-specific patterns).** Reasons:
- 50-100x faster than the Flair tier per page.
- Statistical NER handles ORG well; `EntityRuler` cleans up DATE/MONEY where the statistical layer over-fits to news-style training data.
- spaCy is trivially fine-tuneable later if recall on a specific entity type is poor.
- **Backup pilot:** GLiNER for the long tail (CMR-specific entities, freight broker names) — its zero-shot capability shortcuts the need to label.

### §5.3 Sources

- [Sentence Transformers — efficiency docs](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
- [Best open-source embedding models — Supermemory](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)
- [Pretrained models — Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- [sentence-transformers/all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [BGE / E5 / Jina comparison — Knightli (Apr 2026)](https://www.knightli.com/en/2026/04/23/compare-openai-bge-e5-gte-jina-embedding-models/)
- [Jina Embeddings v3 — arXiv 2409.10173](https://arxiv.org/abs/2409.10173)
- [Jina Embeddings v3 — official announcement](https://jina.ai/news/jina-embeddings-v3-a-frontier-multilingual-embedding-model/)
- [Best reranker models for RAG 2026 — BSWEN](https://docs.bswen.com/blog/2026-02-25-best-reranker-models/)
- [spaCy English models](https://spacy.io/models/en)
- [spacy/en_core_web_md model card](https://huggingface.co/spacy/en_core_web_md)
- [Invoice parsing with Python and spaCy — Vreamer](https://blogs.vreamer.space/invoice-parsing-with-python-and-spacy-a-software-engineers-guide-1cfee9d00022)
- [GLiNER — urchade/GLiNER](https://github.com/urchade/GLiNER)
- [GLiNER paper — arXiv 2311.08526](https://arxiv.org/abs/2311.08526)
- [GLiNER2 paper — arXiv 2507.18546](https://arxiv.org/html/2507.18546v1)
- [GLiNER zero-shot blog — Medium](https://netraneupane.medium.com/gliner-zero-shot-ner-outperforming-chatgpt-and-traditional-ner-models-1f4aae0f9eef)
- [flair/ner-english model card](https://huggingface.co/flair/ner-english)
- [flairNLP/flair](https://github.com/flairNLP/flair)
- [Beyond FuzzyWuzzy — matchdatapro](https://matchdatapro.com/beyond-fuzzywuzzy-a-better-way-to-match-and-clean-data/)
- [Top fuzzy matching tools 2026 — matchdatapro](https://matchdatapro.com/top-5-fuzzy-matching-tools-for-2026/)
- [Python record linking — Practical Business Python](https://pbpython.com/record-linking.html)
- [dedupeio/dedupe](https://github.com/dedupeio/dedupe)

---

## §6 Vector DB Pick — `sqlite-vec`

### §6.1 Comparison for sole-trader scale

Adam's scale: thousands of customer invoices over the bookkeeping lifetime, not millions. Embedding count ceiling for v0.4: ~50k vectors (line-items + vendor names + receipt entries combined). At this scale the choice is dominated by **operational simplicity**, not raw RPS.

| Tool | License | Embedded? | Storage | Query latency (50k @ 384-d) | Operational footprint | Best for |
|---|---|---|---|---|---|---|
| **`sqlite-vec`** | Apache-2.0 | **Yes — SQLite extension** | One `.db` file | <10 ms typical | Zero — just SQLite; no extra process | Sole-trader: one process, one file, no daemon |
| **`chromadb`** | Apache-2.0 | Yes (Python or Rust mode) | Directory of files | <50 ms | Low; 2025 Rust rewrite gave 4x writes; in-process or HTTP server modes | Prototyping; <10M vectors |
| **`lancedb`** | Apache-2.0 | Yes | Directory (Lance columnar format) | <50 ms; scales past memory | Low; disk-efficient zero-copy | Larger-than-RAM datasets, batch analytics |
| **`milvus-lite`** | Apache-2.0 | Yes (in-process Python) | Local file | <50 ms | Low for Lite; production path needs Standalone | Quickstart / future-scale path |
| **`qdrant`** | Apache-2.0 | No (server) — but single binary | Server data dir | <5 ms (best RPS in benchmarks) | Medium — own service to run | Production, >100k vectors, geo+filter queries |
| **`pgvector`** | PostgreSQL | Postgres extension | Postgres tables | <20 ms | Medium — needs Postgres | When Postgres is already in the stack |

### §6.2 Pick — `sqlite-vec`

- **One file, one process.** Hetzner AX52 wants this.
- Apache-2.0, mature SQLite under the hood (battle-tested as anything in software).
- Backup is `cp file.db backup.db` — no Qdrant snapshot ceremony.
- 50k 384-d vectors is well within sqlite-vec's comfort zone (it scales into millions).
- **Migration path:** if scale ever justifies, the schema fits Qdrant cleanly (vector + metadata payload).

`chromadb` is the runner-up — would be picked if the team was already used to it. `lancedb` wins only if we expect "larger-than-RAM" pressure, which sole-trader scale never produces.

### §6.3 Sources

- [Best vector databases 2026 — Encore](https://encore.dev/articles/best-vector-databases)
- [Best vector databases 2026 — Firecrawl blog](https://www.firecrawl.dev/blog/best-vector-databases)
- [Vector DB comparison 2026 — 4xxi](https://4xxi.com/articles/vector-database-comparison/)
- [Vector DB benchmarks 2026 — CallSphere](https://callsphere.ai/blog/vector-database-benchmarks-2026-pgvector-qdrant-weaviate-milvus-lancedb)
- [Qdrant benchmarks](https://qdrant.tech/benchmarks/)
- [VDBBench 1.0 — Milvus blog](https://milvus.io/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads.md)
- [Top 10 vector DBs 2026 — Second Talent](https://www.secondtalent.com/resources/top-vector-databases-for-llm-applications/)
- [Chroma vs LanceDB — Zilliz](https://zilliz.com/comparison/chroma-vs-lancedb)
- [Top open-source vector DBs 2025 — Medium](https://medium.com/@fendylike/top-5-open-source-vector-search-engines-a-comprehensive-comparison-guide-for-2025-e10110b47aa3)
- [Zvec — "SQLite of vector DBs" — Medium](https://medium.com/@AdithyaGiridharan/zvec-alibaba-just-open-sourced-the-sqlite-of-vector-databases-and-its-blazing-fast-15c31cbfebbf)

---

## §7 PEPPOL UBL XML in Python (state of the art)

### §7.1 The reality

PEPPOL invoices arrive as UBL 2.1 XML and bypass OCR entirely (per locked D18 — Adam's network). The Python tooling here is **smaller and rougher than the broader OCR ecosystem.** Most production PEPPOL implementations are Java (Mustang) or PHP (einvoicing/num-num). Python options exist but are sparser.

### §7.2 Python option survey

| Tool | Status | Use |
|---|---|---|
| **`lxml`** | Mature, ubiquitous | XML parsing + XSD validation. The base layer. |
| **`python-en16931`** (Invinet Sistemes) | Active; v0.1 docs (Feb 2024 latest) | Read/write/manage EN 16931. Generates valid PEPPOL BIS 3 UBL 2.1 XML. The most complete Python-native EN 16931 toolkit. |
| **`ubllib`** (glenfant) | Active | UBL 2.1 object marshalling — Python objects ↔ UBL XML. Lower-level than python-en16931. |
| **`pyschematron`** (robbert-harms) | Active; **Python 3.12 only** | Pure-Python ISO/IEC 19757-3:2020 Schematron validator using elementpath (XPath 1.0-3.1). Best Python-native option. |
| **`pyschematron`** (Ionite) | Alpha; moved to Codeberg | Alternative; not feature-complete |
| **`py-schematron-validator`** (SSRQ-SDS-FDS) | Active wrapper | Wraps SchXSLT under SaxonC HE — most production-grade if Java/SaxonC dependency is acceptable |
| **`saxonche`** | Active | Saxon C/Python bindings; XSLT 3.0 + Schematron. Used in EDocument app for ERPNext (`saxonche~=12.5.0`). |
| **`e-invoice-py`** (e-invoice.be) | SDK | Python SDK for e-invoice.be Peppol API — useful if outsourcing the access-point side; not what we need for parsing |

### §7.3 Validation pipeline (standard PEPPOL flow)

PEPPOL BIS validation requires three layered checks:
1. **UBL 2.1 XML Schema (XSD)** — structural validation. `lxml.etree.XMLSchema` does this directly.
2. **EN 16931 Schematron** — semantic rules from the EU directive (e.g., line totals must equal sum of line item totals).
3. **PEPPOL BIS Schematron** — additional rules from OpenPEPPOL.

Schematron files for both are at [OpenPEPPOL/peppol-bis](https://github.com/OpenPEPPOL/peppol-bis).

### §7.4 Pick — `lxml` + `python-en16931` + `pyschematron` (robbert-harms)

For v0.4 on the read-side (we're consuming PEPPOL invoices, not issuing them):

```
1. lxml — parse XML, run XSD schema validation
2. python-en16931 — map XML to typed Python Invoice/LineItem objects
3. pyschematron (robbert-harms) — run EN16931 + PEPPOL BIS Schematron rules
4. Map result into the same Pydantic Invoice schema used for OCR path
```

**Fallback:** if `pyschematron` proves too slow or hits XPath edge cases, escalate to `py-schematron-validator` (SaxonC). SaxonC adds a Java dependency but is the most battle-tested path. A single PEPPOL XML is small (typically <100 KB) so even Java-startup overhead per-doc is acceptable for batch nightly runs.

### §7.5 Sources

- [glenfant/ubllib](https://github.com/glenfant/ubllib)
- [Invinet python-en16931](https://invinet.github.io/python-en16931/build/html/invoice.html)
- [robbert-harms/pyschematron](https://github.com/robbert-harms/pyschematron)
- [pyschematron on PyPI](https://pypi.org/project/pyschematron/)
- [Ionite/pyschematron (alpha)](https://github.com/Ionite/pyschematron)
- [SSRQ-SDS-FDS/py-schematron-validator](https://github.com/SSRQ-SDS-FDS/py-schematron-validator)
- [Validating Peppol Documents — Ionite](https://ionite.net/news-articles/2023-08-17_validating_peppol_documents/)
- [PEPPOL document formats intro — Ionite](https://ionite.net/news-articles/2025-03-07_peppol_document_formats/)
- [OpenPEPPOL/peppol-bis Schematron files](https://github.com/OpenPEPPOL/peppol-bis/blob/master/rules/peppolbis-trdm010-2.0-invoice/Schematron/OPENPEPPOL/OPENPEPPOL-UBL-T10.sch)
- [Lino — generating Peppol XML files](https://dev.lino-framework.org/plugins/peppol/ubl.html)
- [Lino — Schematron validation guide](https://dev.lino-framework.org/topics/schematron.html)
- [Schematron with lxml — InterSystems DC](https://community.intersystems.com/post/schematron-xml-documents-validation-using-python)
- [e-invoice-be/e-invoice-py](https://github.com/e-invoice-be/e-invoice-py)
- [PEPPOL UBL format guide — Recommand](https://recommand.eu/en/docs/ubl-format-guide)

---

## §8 Document Classification

The job: route an incoming document to the right extractor (invoice / receipt / bank statement / RCT / contract / freight CMR) **before** running expensive extraction.

### §8.1 Approach comparison

| Approach | Setup cost | Inference latency | Accuracy on minimal data | Operates on |
|---|---|---|---|---|
| **Few-shot LLM prompt (GLM-OCR/Qwen2.5-VL)** | Zero — just write prompt | 2-5 s on Ollama Vega 8 | Strong with 1-3 examples per class | Image or text |
| **TF-IDF + sklearn LogisticRegression / SVM** | Need ~30-50 docs/class labeled | <10 ms | 85-90% on clear class boundaries | Text only |
| **DistilBERT fine-tune** | Need ~100 docs/class, GPU/CPU train day | 100-300 ms | 90-95% | Text |
| **Donut classifier fine-tune** | Need ~100-200 docs/class, GPU train | 5-15 s CPU | 95%+ | Image |
| **DiT classifier fine-tune** | ~50 docs/class | 1-3 s CPU | 90%+ | Image |

### §8.2 The 2026 nuance

Recent research (arXiv 2505.18215, "Do BERT-Like Bidirectional Models Still Perform Better on Text Classification in the Era of LLMs?") confirms BERT-likes still beat LLMs on **pattern-driven** tasks. But for **few-shot regimes with no labeled data**, LLMs win on time-to-first-result. For invoices vs receipts vs bank statements, the document boundaries are sufficiently pattern-rich that a tiny labeled set produces excellent classifiers.

### §8.3 Pick — staged

- **v0.4 Stage 1 (now):** few-shot prompt classifier through GLM-OCR (already loaded). Class set: invoice / receipt / bank_statement / RCT / contract / CMR / unknown. Confidence threshold: route to `unknown` for human review under threshold.
- **v0.4 Stage 2 (after ~50 docs/class collected):** TF-IDF + LogisticRegression baseline. Replaces the LLM call for the hot path. LLM falls back only when baseline confidence <0.7.
- **v0.5 (after ~500 docs/class):** DistilBERT fine-tune, replaces TF-IDF.
- **Never (probably):** Donut classifier — overkill for the speed-vs-accuracy tradeoff at this scale.

### §8.4 Sources

- [Long document classification benchmark 2025 — Procycons](https://procycons.com/en/blogs/long-document-classification-benchmark-2025/)
- [BERT vs LLMs on text classification — arXiv 2505.18215](https://arxiv.org/abs/2505.18215)
- [Invoice information extraction methods — arXiv 2510.15727](https://arxiv.org/html/2510.15727v1)
- [Document data extraction LLMs vs OCRs — Vellum](https://www.vellum.ai/blog/document-data-extraction-llms-vs-ocrs)
- [Best open-source LLM for document screening 2026 — SiliconFlow](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Document-screening)
- [Text classification pipeline — arXiv 2501.00174](https://arxiv.org/pdf/2501.00174)
- [Are we really making progress in text classification? — arXiv 2204.03954](https://arxiv.org/html/2204.03954v6)
- [Intent classification 2026 — Label Your Data](https://labelyourdata.com/articles/machine-learning/intent-classification)
- [Document intelligence with LLMs — Virtido](https://virtido.com/blog/document-intelligence-llm-extraction-guide)

---

## §9 Multi-Page Document Handling

### §9.1 The reality

Real invoices: 2-5 pages. Subscription/utility invoices: 1 page header + 5-30 pages of line items + 1-page total. Continuation totals appear at the foot of every page (`Continued on page 2…` then `Subtotal carried forward: $X`). Bank statements are 5-20 pages.

### §9.2 Patterns

**Pattern A — page-by-page extract, reduce at end:**
1. `pypdfium2` or `pdf2image` → list of page images
2. Per page: layout detect → extract structured fields + line-item rows
3. Reducer pass: concatenate line-item rows, sum, reconcile against the document `total` field discovered on the last page
4. Validation: if reducer total ≠ document total within tolerance → flag for human review

**Pattern B — docling whole-doc:**
- Docling handles multi-page natively, emits a single Markdown / JSON for the whole doc with reading-order preserved.
- Pro: less stitching code on our side.
- Con: less control over the per-page validation gate.

**Pattern C — hybrid:**
- Native PDF: pdfplumber per page, deterministic stitching (page numbers explicit in PDF).
- Scan PDF: pdf2image → per-page → reducer.
- Use Pattern A always; reach for Pattern B only as a fallback when Pattern A fails reconciliation.

### §9.3 Reconciliation rules (2026 best-practice consensus)

From the invoice extraction guides reviewed:
- **Cross-page total reconciliation:** `sum(line_item.line_total) == document.subtotal` and `document.subtotal + document.tax + document.shipping == document.total` within €0.05 tolerance.
- **Page count guard:** PDF metadata page count must match the page count seen by the extractor. Mismatch → corrupted upload or double-page scan failure.
- **Continuation marker recognition:** strings like "Continued on page", "Page X of Y", "Subtotal c/f", "Brought forward" — used to detect partial page breaks. Hand-write regex; no ML needed.
- **Line-item count guard:** if invoice claims `Line items: 47` in a header field, the extracted line-item count must match. This is an underused guard from invoice-extraction-domain experts (Nanonets / LlamaIndex services).

### §9.4 Pick — Pattern A with `pypdfium2`

`pypdfium2` (Apache-2.0/BSD) over `pdf2image` because:
- No Poppler bin requirement at runtime (we already have Poppler, but pypdfium2 also embeds PDFium directly).
- Faster page rendering than pdf2image's Poppler subprocess.
- Same Pillow-image output type so call-site code is identical.

Reducer pass implemented as plain Python over the per-page extract dicts. No framework needed.

### §9.5 Sources

- [invoice2data — invoice-x](https://github.com/invoice-x/invoice2data)
- [invoice2data on PyPI](https://pypi.org/project/invoice2data/)
- [How to extract data from invoices using Python — Nanonets](https://nanonets.com/blog/how-to-extract-data-from-invoices-using-python/)
- [Multi-page invoice OCR — LlamaIndex](https://www.llamaindex.ai/services/invoice-data-extraction-software)
- [AI OCR invoice extraction with Python — FlowHunt](https://www.flowhunt.io/blog/ai-ocr/)
- [Best Python PDF-to-text parser libraries 2026 — Unstract](https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/)
- [Python OCR — invoice data libraries tested — Python ML Daily](https://pythonmldaily.com/posts/python-ocr-invoice-data-pdf-image-libraries)

---

## §10 Concrete v0.4 Library Stack Picks

One pick per category, with rationale and the reject reasons.

### §10.1 The stack

```
┌──────────────────────────────────────────────────────────────────┐
│ INPUT                                                            │
│   PDF / image / PEPPOL UBL XML / EML attachment                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ ROUTING                                                          │
│   1. file-type detect (mime + magic)                             │
│   2. PEPPOL XML → §7 path                                        │
│   3. native PDF → pdfplumber primary                             │
│   4. scan / image → OCR path                                     │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ DOC CLASSIFICATION                                               │
│   few-shot LLM (GLM-OCR via Ollama)                              │
│   classes: invoice / receipt / bank_statement / RCT / contract / │
│            CMR / unknown                                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ LAYOUT + PAGE LOOP (per page)                                    │
│   pypdfium2 → page images                                        │
│   LayoutParser (PubLayNet/DocLayNet model) → bbox + zone class   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ TEXT + TABLE EXTRACTION (per zone)                               │
│   text zones:  Tesseract 5 / RapidOCR (v0.3 proven)              │
│   table zones: pdfplumber (native) | PP-StructureV3 (scans)      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ STRUCTURED FIELD EXTRACTION                                      │
│   instructor (Pydantic + Ollama OpenAI-compat)                   │
│   schemas: Invoice, Receipt, BankStatement, RCTPayment, Contract │
│   max_retries=3 with validation loop                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ NER BACKSTOP / CROSS-CHECK                                       │
│   spaCy en_core_web_md + EntityRuler                             │
│   confirms VAT IDs, dates, amounts, ORG (per Pydantic field)     │
│   conflict → flag for human review                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ VENDOR DEDUP                                                     │
│   bge-small-en-v1.5 embedding                                    │
│   sqlite-vec ANN top-k                                           │
│   rapidfuzz Levenshtein re-rank                                  │
│   threshold ≥0.92 → merge; 0.75-0.92 → human review              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ MULTI-PAGE RECONCILE (Pattern A)                                 │
│   per-page extracts merged → totals computed → match doc total   │
│   off-by-tolerance → flag                                        │
│   line-item count guard → flag                                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ PEPPOL XML PATH (separate branch from §7)                        │
│   lxml XSD validate → python-en16931 parse → pyschematron rules  │
│   → map into same Invoice Pydantic schema                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ OUTPUT                                                           │
│   Pydantic-validated record(s) + provenance trail                │
│   provenance: which engine produced each field, with confidence  │
└──────────────────────────────────────────────────────────────────┘
```

### §10.2 The picks (table form)

| Slot | Pick | License | Rationale | Rejected (and why) |
|---|---|---|---|---|
| Constrained generation | **`instructor`** | MIT | Lowest friction with Ollama (already in stack); Pydantic-native; built-in retry loop | `outlines` (Ollama+OpenAI bug #1221; llama.cpp swap regresses v0.3); `guidance` (DSL learning curve); `pydantic-ai` (agent-first leak); `xgrammar` (engine-level not app-level) |
| Layout detection | **`LayoutParser`** | Apache-2.0 | Apache-2.0 model zoo; multi-backend (detectron2/effdet/paddledetection); ONNX port available | `LayoutLMv3` (CC-BY-NC base weights — license trap); `Donut` (slow on CPU, requires fine-tune); `DiT` (classifier only) |
| Table extraction (native PDF) | **`pdfplumber`** | MIT | 2026 consensus winner for native-PDF tables; pixel-level control; pure Python | `camelot` (Ghostscript dep); `tabula-py` (Java dep); `markitdown` (less precise on invoice tables) |
| Table extraction (scans) | **`PaddleOCR PP-StructureV3`** | Apache-2.0 | Best Apache-2.0 layout+table+reading-order on scans; cell coordinates → markdown/JSON | `unstructured.io` (heavy deps; CPU-core hogging issue #3291); `docling` (release velocity risk — kept as fallback) |
| Embeddings | **`bge-small-en-v1.5`** via `sentence-transformers` | MIT (model) / Apache-2.0 (lib) | 384-d sweet spot; outperforms MiniLM by 3-4 MTEB points; small index footprint | `all-MiniLM-L6-v2` (good baseline but bge wins); `bge-large` (overkill at scale); `jina-v3` (heavy CPU; revisit when multilingual hits) |
| NER | **`spaCy en_core_web_md` + `EntityRuler`** | MIT | 50-100x faster than Flair on CPU; statistical NER + regex hybrid is the consensus pattern for invoices | `gliner-base-v2.5` (kept as long-tail pilot); `flair` (too slow); BERT fine-tune (defer until labeled data) |
| Vector DB | **`sqlite-vec`** | Apache-2.0 | One file, one process; SQLite reliability; backups are `cp` | `chromadb` (extra runtime); `lancedb` (no benefit at our scale); `qdrant` (separate service); `milvus-lite` (in-process Python but heavier) |
| PEPPOL XML | **`lxml` + `python-en16931` + `pyschematron`** | MIT/Apache | Most complete Python-native EN 16931 toolkit + ISO Schematron 19757-3:2020 native | `ubllib` (lower-level — used as backstop); SaxonC (Java dep — fallback only) |
| Doc classification | **few-shot LLM via GLM-OCR (Ollama)** | varies | Zero-cost setup; matures into TF-IDF baseline at 50 docs/class; DistilBERT at 500 docs/class | Donut classifier (overkill); BERT now (no labeled data yet) |
| Multi-page | **`pypdfium2` + per-page Pattern A reducer** | Apache-2.0/BSD | Faster than pdf2image; no extra Poppler subprocess; clean Python loop | `pdf2image` (works but slower); `docling` whole-doc (kept as fallback) |
| Vendor dedup re-rank | **`rapidfuzz`** | MIT | Fast token-based + Levenshtein; consensus 2026 pick over `fuzzywuzzy` | `fuzzywuzzy` (slower); `dedupe` (heavyweight for our scale) |
| PDF render | **`pypdfium2`** | Apache-2.0/BSD | Embedded PDFium; fast page rasterization | `pdf2image` (Poppler subprocess overhead) |

### §10.3 Hardware budget check

Worst-case invoice-day pipeline cost on Vega 8 (rough estimates from CPU-latency reports collected):
- LayoutParser per page: ~2 s
- RapidOCR per page: ~1 s
- PP-StructureV3 per page: ~4 s
- instructor + GLM-OCR Q8 per call: ~4-8 s
- spaCy NER: ~50 ms
- bge-small embed: ~80 ms
- sqlite-vec query: ~5 ms

Per-invoice (3-page average) ceiling: **~30-40 s end-to-end**. For 50 invoices/day batch: ~25-35 minutes nightly run. Comfortably within the Hetzner AX52 + Ryzen 5 3500U envelope.

### §10.4 What's deliberately NOT in the v0.4 stack

- **No GPU dependencies.** Anything that requires CUDA at install time is rejected (closes off Vega 8 + AX52).
- **No Java runtime in the hot path.** Tabula and SaxonC are fallback-only.
- **No "platform" SaaS-style tools.** Unstructured.io community version is technically Apache-2.0 but the dep weight + per-doc cost on CPU + community-vs-platform feature gap argues against it for sole-trader scale.
- **No GPL-3.0 anything.** Surya, Marker stay disqualified.
- **No vendor lock-in via cloud APIs.** Azure DI, AWS Textract, Google Document AI all out — they'd wipe the local-CPU thesis.
- **No agent frameworks for the extraction core.** `pydantic-ai` and `langchain` would add layers without solving the actual fields-from-pixels problem. Agents come later for the orchestration / human-review-loop layer.

### §10.5 Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `instructor` retry storm on bad OCR text | Medium | `max_retries=3` cap; surface failure as "human review" not "fail loop" |
| PaddleOCR install pain on Windows dev | High | Use Linux container for dev too; AX52 is Linux anyway |
| LayoutParser PubLayNet model under-trained on invoices | High | Plan a fine-tune sprint at v0.5 once 200+ labeled invoices collected |
| sqlite-vec scaling ceiling | Low at our scale | Schema migration path to Qdrant documented |
| `pyschematron` (robbert-harms) Python 3.12-only | Medium | Pin to 3.12 (already on 3.13 system; install side-by-side) — or fallback to py-schematron-validator |
| Docling release velocity churn | Medium | Use as fallback only; pin major version in requirements |
| `python-en16931` v0.1 maturity | High | Backstop with `ubllib` for raw object marshalling; fork if upstream stalls |

---

## §11 Citations Index (consolidated)

### Constrained generation (§2)
1. [GitHub — dottxt-ai/outlines](https://github.com/dottxt-ai/outlines)
2. [Outlines — llama.cpp models docs](https://dottxt-ai.github.io/outlines/1.1.0/features/models/llamacpp/)
3. [Outlines issue #1221 — Ollama+OpenAI ValidationError](https://github.com/dottxt-ai/outlines/issues/1221)
4. [Outlines issue #658 — OOM on constrained JSON schema](https://github.com/dottxt-ai/outlines/issues/658)
5. [Instructor — Ollama integration](https://python.useinstructor.com/integrations/ollama/)
6. [Instructor — multi-language structured outputs](https://python.useinstructor.com/)
7. [Pydantic AI docs](https://ai.pydantic.dev/)
8. [Pydantic AI tutorial — DataCamp](https://www.datacamp.com/tutorial/pydantic-ai-guide)
9. [Pydantic invoice extraction](https://invoicedataextraction.com/blog/pydantic-invoice-extraction)
10. [Microsoft Guidance — guidance-ai/guidance](https://github.com/guidance-ai/guidance)
11. [llguidance — super-fast structured outputs](https://github.com/guidance-ai/llguidance)
12. [XGrammar — mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar)
13. [XGrammar paper — arXiv 2411.15100](https://arxiv.org/pdf/2411.15100)
14. [MLC blog — XGrammar performance](https://blog.mlc.ai/2024/11/22/achieving-efficient-flexible-portable-structured-generation-with-xgrammar)
15. [Guided Decoding paper — arXiv 2509.06631](https://arxiv.org/html/2509.06631v1)
16. [JSONSchemaBench — arXiv 2501.10868](https://arxiv.org/html/2501.10868v1)
17. [Structured outputs in 2026 — DeepFounder](https://deepfounder.ai/structured-outputs-in-2026-how-to-make-llms-return-exactly-what-your-app-needs/)
18. [Pydantic for validating LLM outputs — Machine Learning Mastery](https://machinelearningmastery.com/the-complete-guide-to-using-pydantic-for-validating-llm-outputs/)

### Layout-aware models (§3)
19. [Layout-Parser/layout-parser](https://github.com/Layout-Parser/layout-parser)
20. [LayoutParser — multi-backend support](https://layout-parser.readthedocs.io/en/latest/api_doc/models.html)
21. [layoutparser-ort — ONNX port](https://github.com/styrowolf/layoutparser-ort)
22. [LayoutLMv3 HF docs](https://huggingface.co/docs/transformers/en/model_doc/layoutlmv3)
23. [Fine-tuning LayoutLMv3 for invoices — TDS](https://towardsdatascience.com/fine-tuning-layoutlm-v3-for-invoice-processing-e64f8d2c87cf/)
24. [Donut — clovaai/donut](https://github.com/clovaai/donut)
25. [Donut paper — arXiv 2111.15664](https://arxiv.org/abs/2111.15664)
26. [Fine-tuning Donut — philschmid.de](https://www.philschmid.de/fine-tuning-donut)
27. [DiT — microsoft/unilm/dit](https://github.com/microsoft/unilm/tree/master/dit)
28. [DiT paper — arXiv 2203.02378](https://arxiv.org/abs/2203.02378)
29. [Pix2Text — breezedeus/Pix2Text](https://github.com/breezedeus/Pix2Text)
30. [TrOCR — microsoft/unilm/trocr](https://github.com/microsoft/unilm/blob/master/trocr/README.md)

### Table extraction (§4)
31. [Python PDF table extraction comparison — invoicedataextraction.com](https://invoicedataextraction.com/blog/python-pdf-table-extraction-invoices)
32. [How to extract tables from PDF — Unstract (2026)](https://unstract.com/blog/extract-tables-from-pdf-python/)
33. [Camelot wiki — comparison](https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools)
34. [pdfplumber vs PyMuPDF vs Tabula — BSWEN](https://docs.bswen.com/blog/2026-03-16-pdfplumber-vs-pymupdf/)
35. [Granite-Docling-258M — InfoQ](https://www.infoq.com/news/2025/10/granite-docling-ibm/)
36. [Docling production deployment — Iterathon](https://iterathon.tech/blog/docling-production-deployment-guide-2026)
37. [microsoft/markitdown](https://github.com/microsoft/markitdown)
38. [PaddleOCR — main repo](https://github.com/PaddlePaddle/PaddleOCR)
39. [PaddleOCR 3.0 paper — arXiv 2507.05595](https://arxiv.org/html/2507.05595v1)
40. [PP-StructureV3 introduction](https://paddlepaddle.github.io/PaddleOCR/v3.1.0/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html)
41. [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured)
42. [Unstructured CPU-cores issue #3291](https://github.com/Unstructured-IO/unstructured/issues/3291)

### Embeddings + NER (§5)
43. [Sentence Transformers efficiency docs](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
44. [Best open-source embedding models — Supermemory](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)
45. [BGE / E5 / Jina comparison 2026 — Knightli](https://www.knightli.com/en/2026/04/23/compare-openai-bge-e5-gte-jina-embedding-models/)
46. [Jina Embeddings v3 — arXiv 2409.10173](https://arxiv.org/abs/2409.10173)
47. [GLiNER — urchade/GLiNER](https://github.com/urchade/GLiNER)
48. [GLiNER paper — arXiv 2311.08526](https://arxiv.org/abs/2311.08526)
49. [GLiNER2 paper — arXiv 2507.18546](https://arxiv.org/html/2507.18546v1)
50. [spaCy English models](https://spacy.io/models/en)
51. [Invoice parsing with Python and spaCy — Vreamer](https://blogs.vreamer.space/invoice-parsing-with-python-and-spacy-a-software-engineers-guide-1cfee9d00022)
52. [flairNLP/flair](https://github.com/flairNLP/flair)
53. [Beyond FuzzyWuzzy — matchdatapro](https://matchdatapro.com/beyond-fuzzywuzzy-a-better-way-to-match-and-clean-data/)
54. [Top fuzzy matching tools 2026 — matchdatapro](https://matchdatapro.com/top-5-fuzzy-matching-tools-for-2026/)
55. [dedupeio/dedupe](https://github.com/dedupeio/dedupe)
56. [Python record linking — Practical Business Python](https://pbpython.com/record-linking.html)

### Vector DBs (§6)
57. [Best vector databases 2026 — Encore](https://encore.dev/articles/best-vector-databases)
58. [Vector DB comparison 2026 — 4xxi](https://4xxi.com/articles/vector-database-comparison/)
59. [Vector DB benchmarks 2026 — CallSphere](https://callsphere.ai/blog/vector-database-benchmarks-2026-pgvector-qdrant-weaviate-milvus-lancedb)
60. [Qdrant benchmarks](https://qdrant.tech/benchmarks/)
61. [VDBBench 1.0 — Milvus blog](https://milvus.io/blog/vdbbench-1-0-benchmarking-with-your-real-world-production-workloads.md)
62. [Chroma vs LanceDB — Zilliz](https://zilliz.com/comparison/chroma-vs-lancedb)
63. [Zvec — "SQLite of vector DBs" — Medium](https://medium.com/@AdithyaGiridharan/zvec-alibaba-just-open-sourced-the-sqlite-of-vector-databases-and-its-blazing-fast-15c31cbfebbf)

### PEPPOL UBL (§7)
64. [glenfant/ubllib](https://github.com/glenfant/ubllib)
65. [Invinet python-en16931 docs](https://invinet.github.io/python-en16931/build/html/invoice.html)
66. [robbert-harms/pyschematron](https://github.com/robbert-harms/pyschematron)
67. [SSRQ-SDS-FDS/py-schematron-validator](https://github.com/SSRQ-SDS-FDS/py-schematron-validator)
68. [Validating Peppol Documents — Ionite](https://ionite.net/news-articles/2023-08-17_validating_peppol_documents/)
69. [PEPPOL document formats intro — Ionite](https://ionite.net/news-articles/2025-03-07_peppol_document_formats/)
70. [OpenPEPPOL/peppol-bis Schematron](https://github.com/OpenPEPPOL/peppol-bis/blob/master/rules/peppolbis-trdm010-2.0-invoice/Schematron/OPENPEPPOL/OPENPEPPOL-UBL-T10.sch)
71. [Lino — generating Peppol XML files](https://dev.lino-framework.org/plugins/peppol/ubl.html)
72. [PEPPOL UBL format guide — Recommand](https://recommand.eu/en/docs/ubl-format-guide)

### Document classification (§8)
73. [Long document classification benchmark — Procycons](https://procycons.com/en/blogs/long-document-classification-benchmark-2025/)
74. [BERT vs LLMs on text classification — arXiv 2505.18215](https://arxiv.org/abs/2505.18215)
75. [Invoice information extraction — arXiv 2510.15727](https://arxiv.org/html/2510.15727v1)
76. [Document data extraction LLMs vs OCRs — Vellum](https://www.vellum.ai/blog/document-data-extraction-llms-vs-ocrs)

### Multi-page (§9)
77. [invoice2data — invoice-x](https://github.com/invoice-x/invoice2data)
78. [How to extract data from invoices — Nanonets](https://nanonets.com/blog/how-to-extract-data-from-invoices-using-python/)
79. [Multi-page invoice OCR — LlamaIndex](https://www.llamaindex.ai/services/invoice-data-extraction-software)
80. [Best Python PDF-to-text libraries 2026 — Unstract](https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/)

---

## §12 Final Notes for Implementation

1. **Lock the picks before writing v0.4 code.** The danger zone is "while I'm here, let me also try docling" — that's where weeks disappear. The picks above are the picks. Docling, GLiNER, jina-v3, LayoutLMv3 are all on the **pilot** list, not the **build** list.
2. **Provenance is non-negotiable.** Every Pydantic field gets a sibling `_source` (which engine produced it: `pdfplumber` / `instructor+glm-ocr` / `peppol-xml` / `human`) and `_confidence` (float 0-1). Verification engine reads these.
3. **Build the verification engine before scaling extraction.** A single pipeline that produces 1000 unverified records is worse than a verified 50-record pipeline.
4. **Rejected option captured for memory:** `Surya`, `Marker`, `LayoutLMv3 base weights`, `unstructured.io heavy mode`, `tabula-py` (Java path), `Donut classifier as primary`, `xgrammar at app layer`, `pydantic-ai for the extract core`. Each rejected for license, dep weight, or scale-mismatch — see the per-section rationale above.
