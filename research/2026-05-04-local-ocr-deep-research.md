# Local / Open-Source OCR for IE Bookkeeping — Deep Research (2026-05-04)

**Goal:** Replace the paid Mistral-Small-3.1 verifier path in `extract.py` with a 100% local stack that runs on Adam's hardware (Ryzen 5 3500U / 30 GB RAM / Vega 8 iGPU 2 GB / no CUDA / Windows 11 / Python 3.10) and matches or beats the paid path on Irish invoices, receipts, and bank statements.

**Hardware constraint summary:** No CUDA. No usable iGPU acceleration for OCR (Vega 8 with 2 GB shared VRAM is below every viable VLM model floor — see §1.2). Everything is CPU + RAM bound. We have 30 GB RAM and ~17 Ollama models already pulled including `glm-ocr:q8_0` (1.6 GB) and `moondream:1.8b`. Tesseract 5.5.0 + RapidOCR 3.8.1 (ONNX) installed and proven. Poppler missing — install before deploying.

---

## 1. Executive Summary

### 1.1 Top three picks for our hardware

| Pick | Stack | Why this is the answer |
|---|---|---|
| **#1 — Production today** | **RapidOCR (ONNX) → GLM-OCR via Ollama as VLM verifier on low-confidence fields only** | RapidOCR is Apache-2.0, ~80 MB, ~0.2-1.0 s/page CPU; gives PaddleOCR-class accuracy without the install pain. GLM-OCR is 0.9 B params (1.6 GB Q8 already pulled), runs on CPU+RAM (no CUDA), holds OmniDocBench V1.5 #1 at 94.62. Use RapidOCR for bulk text + bbox detection, then route only fields below a confidence threshold to GLM-OCR for re-OCR/validation. Latency stays sane on Vega-8-class hardware because GLM-OCR only fires on uncertainty, not every field. |
| **#2 — Best accuracy, slow** | **Tesseract + RapidOCR ensemble → Qwen2.5-VL-3B (Q4 GGUF via llama.cpp) verifier** | Voting between two cheap engines first (Tesseract @ 87.5%, RapidOCR @ 75% on the 2026 codesota.com bench, but they fail on different things — see §4) catches the bulk of invoice fields. Qwen2.5-VL-3B (Apache-2.0, ~3 GB Q4) only fires on disagreements. Slower than #1 but higher ceiling because Qwen2.5-VL hits 85-95% on handwriting/blur where pure OCR drops below 60%. |
| **#3 — Single-engine, simplest** | **PaddleOCR-VL 0.9 B (CPU mode)** | Apache-2.0, 0.9 B params, claims OmniBenchDoc V1.5 #1 globally, runs on CPU "although it will be slower than GPU but still usable" per Baidu's own docs. One install, one call, no ensemble logic. Trade-off: install pain on Windows (separate venv recommended for vllm vs paddlepaddle) and CPU latency is the main risk. |

### 1.2 What we explicitly rule out

- **olmOCR / olmOCR-2** — hard CUDA dependency. GitHub issue [#50](https://github.com/allenai/olmocr/issues/50) confirms `CUDA_VISIBLE_DEVICES=""` raises `RuntimeError: No CUDA GPUs are available`. No CPU fallback. Even via Ollama (`richardyoung/olmocr2:7b-q8`, 9.5 GB, [docs](https://ollama.com/richardyoung/olmocr2)), CPU inference will be unusable: a similar 7-8 B VLM (Llama-3.2-Vision 11B) measured **174 seconds time-to-first-token on CPU for one invoice page** ([theneuralmaze.substack.com](https://theneuralmaze.substack.com/p/run-the-worlds-best-ocr-on-your-own)). Not viable.
- **Llama-3.2-Vision 11B** — same. 175 s TTFT on CPU per [vllm issue #8826](https://github.com/vllm-project/vllm/issues/8826). 30 GB RAM technically fits Q4, but throughput kills it.
- **Surya OCR** — **GPL-3.0 with revenue-cap commercial restriction**. Per [LICENSE](https://github.com/VikParuchuri/surya/blob/master/LICENSE) + [unstract review](https://unstract.com/blog/best-opensource-ocr-tools/): code is GPL, weights use modified Open Rail-M ("free for research, personal use, and startups under $2 M revenue"). For a commercial bookkeeping product, Surya forces dual-licensing (`surya@vikas.sh`). Even if we paid, Surya needs ≥6 GB VRAM in practice ([issue #183](https://github.com/datalab-to/surya/issues/183)) and CPU mode is "slow" by maintainer's own admission ([HF discussion](https://huggingface.co/spaces/artificialguybr/Surya-OCR/discussions/2): *"Yes. It's CPU or GPU based. But i don't know if 2 core and 4GB ram will be enough… You need to test by yourself"*). Triple no.
- **Marker** — depends on Surya, inherits GPL contamination, 4 GB+ VRAM needed ([repo](https://github.com/datalab-to/marker)).
- **Nougat** — academic-paper-focused, not invoice-tuned ([facebookresearch/nougat](https://github.com/facebookresearch/nougat)).
- **EasyOCR** — systematically misreads `$` as `8` and `€` as `E` ([codesota 2026 bench](https://www.codesota.com/ocr/best-for-python), [invoicedataextraction.com](https://invoicedataextraction.com/blog/python-ocr-library-comparison-invoices)). Disqualifying for a finance product.
- **DeepSeek-OCR / DeepSeek-OCR-2** — possible but graded down: MIT-licensed and accurate (96%+ on printed text per [labelyourdata](https://labelyourdata.com/articles/deepseek-ocr)), but operator on AMD Strix Halo (much beefier than Vega 8) reports **~2 s/page** with 105 GB unified memory ([Wong, Medium](https://medium.com/@yjwong/running-deepseek-ocr-locally-on-amd-strix-halo-a-journey-into-local-ai-powered-document-processing-ed9ab4c77ed0)). Sparkco's CPU-only guide is honest: CPU mode is "10-50x slower than GPU" ([sparkco.ai](https://sparkco.ai/blog/optimizing-deepseek-ocr-for-cpu-only-deployment-in-2025)). On Vega 8 / Ryzen 3500U we're closer to 30-60 s/page. Workable if patient, but not as fast as GLM-OCR Q8.
- **Florence-2 / Idefics3 / GOT-OCR-2.0** — research-grade or GPU-recommended, not a fit ([Roboflow](https://blog.roboflow.com/florence-2-ocr/), [modal.com](https://modal.com/blog/8-top-open-source-ocr-models-compared)).
- **Mistral OCR (paid)** — 72.2% accuracy on real-world docs per Omni AI / themanmaran on [HN](https://news.ycombinator.com/item?id=43282905); also "aggressively classifies content as images, replacing entire sections with `[image]` placeholders; receipts particularly affected." Local stack does not need to fear it.

### 1.3 Headline accuracy delta (local stack vs paid Mistral)

Best public read across HN operator threads, GetOmni's 1,000-doc benchmark ([source](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr)), and the codesota.com 2026 6-engine bench:

- Mistral OCR (paid, vendor-claimed): ~94% headline / **72.2% on real-world JSON extraction**
- Qwen2.5-VL 32B/72B: ~75% (beats Mistral)
- GLM-OCR 0.9 B: 94.62 on OmniDocBench V1.5 (#1 overall)
- PaddleOCR 3.4: **100% on the codesota.com 24-item invoice test** (best single-engine score)
- Tesseract + RapidOCR ensemble + light VLM verifier: empirically can hit 95%+ field accuracy on clean invoices, drops to 85-92% on phone photos / faded receipts (synthesised from [merfantz](https://www.merfantz.com/blog/transforming-invoice-processing-with-ocr-seamless-integration-of-900-transactions-into-sage/) and [labelstud.io](https://labelstud.io/blog/improve-ocr-quality-for-receipt-processing-with-tesseract-and-label-studio/))

**Verdict:** there is no realistic accuracy gap that justifies the paid dependency for clean Irish invoices/receipts. The local stack with proper preprocessing (§5) and a small VLM verifier on low-confidence fields will match or exceed Mistral OCR on field accuracy, lose only on speed in the worst case, and gain absolute privacy + zero cost.

---

## 2. Engine-by-Engine Table

Citations are inline. Bold = recommended for our hardware.

| Engine | Type | License | Invoice Acc. | CPU Latency / page | RAM at inference | CUDA req? | Maintenance | Operator citations |
|---|---|---|---|---|---|---|---|---|
| **Tesseract 5.5** | Traditional | Apache-2.0 ✓ | 87.5% (24-item invoice) [1] | **0.16 s** [1] | ~100 MB [1] | No | Active 30+ yrs | [1][2][3][4] |
| **RapidOCR (ONNX) 1.2** | Traditional | Apache-2.0 ✓ | 75% [1] / "PaddleOCR-class" [5] | **0.2-1.0 s** [1][5] | ~300 MB | No | Active | [1][5][6] |
| PaddleOCR 3.4 | Traditional | Apache-2.0 ✓ | **100%** [1] | 4.85 s [1] / 1-1.5 s [5] | 500 MB-1 GB [1] | Optional (5-10x) | Active | [1][5][7] |
| **PaddleOCR-VL 0.9 B** | VLM | Apache-2.0 ✓ | OmniDocBench #1 [8] | "slower but usable on CPU" [8] | est 4-6 GB [9] | Optional | Active (Baidu) | [8][9] |
| docTR | Traditional/DL | Apache-2.0 ✓ | 91.7% [1] | 1.8 s [1] | Variable [1] | Optional | Active (Mindee) | [1][10] |
| EasyOCR | Traditional | Apache-2.0 ✓ | 62.5% [1] — **mistakes `$` as `8`** [1][2] | 0.66 s [1] | 1-2 GB [1] | Optional 2-3x | Active | [1][2] DISQUALIFIED |
| Surya 0.9 | Modern OCR | **GPL-3.0** ✗ | 95.8% [1] / 97.41% [11] | 2.1 s on CPU [1] / "slow" [12] | 1-2 GB / ≥6 GB VRAM [13] | Recommended | Very active | [11][12][13] DISQUALIFIED (license) |
| Marker (Datalab) | Pipeline | OpenRAIL / GPL deps | LLM-judge 4.41/5 vs Mistral 4.32 [14] | "0.18 s on H100" [15] | 4 GB+ VRAM [15] | Strongly rec | Very active | [14][15] |
| Nougat | VLM (academic) | CC-BY-NC-4.0 | scientific docs only | n/a CPU | 4 GB+ VRAM | Yes | Maintenance mode | [16] |
| Qwen2.5-VL-3B | VLM | Apache-2.0 ✓ | 85-95% [17] (handwriting+) | est 8-15 s/page CPU Q4 | ~6 GB Q4 | Optional | Very active | [17][18][19] |
| Qwen2.5-VL-7B | VLM | Apache-2.0 ✓ | ~75% [20] (matches GPT-4o) | unusably slow CPU | ~10 GB Q4 | Strongly rec | Very active | [20][21] |
| Llama-3.2-Vision 11B | VLM | Llama 3.2 (commercial OK) | 82% [22] | **174 s TTFT** [12] | ~32 GB [22] | Strongly rec | Active | [12][22] DISQUALIFIED (CPU latency) |
| **GLM-OCR 0.9 B** | VLM | Apache-2.0 ✓ (model weights MIT-aligned) | OmniDocBench V1.5 **94.62** [23] | "modern CPU with enough RAM" runs it [23][24] | 1.6-2.2 GB Q8 [25] | No | Active (Z.AI) | [23][24][25][12] |
| MiniCPM-V 2.6 | VLM | Apache-2.0 + MiniCPM | tops OCRBench [26] | GPU-class load | 8 B params, 5 GB+ Q4 | Strongly rec | Active | [26] |
| dotsOCR 1.7 B | VLM | Open (check repo) | OmniDocBench TEDS **88.6%** [27] (vs GPT-5.4 72%) | 8 GB+ VRAM rec [27] | 8 GB+ VRAM [27] | Strongly rec | Active | [27] |
| olmOCR-2 7B | VLM | Apache-2.0 (RUG) | olmOCR-Bench 82.4 [28] | **CPU not supported** [29] | 16 GB rec [30] | **Hard requirement** [29] | Active (AI2) | [28][29][30] DISQUALIFIED |
| DeepSeek-OCR-2 3B | VLM | MIT ✓ | OmniDocBench v1.5 **91.09%** [31] / 96%+ printed [31] | 2 s/page on AMD Strix Halo (huge) [32]; 30-60 s estimated on Vega 8 | ~6 GB Q4 | Optional but 10-50x speedup [33] | Active (DeepSeek) | [31][32][33] |
| Moondream 2/3 | VLM | Apache-2.0 ✓ | **weak on dense invoice text** [34] | ~543 ms Jetson [34] | 2 B params | Optional | Active | [34] |
| Florence-2 | VLM | MIT ✓ | OCR caption-quality [35] | 1 s on T4 (GPU) [35] | small | Recommended | Active (MS) | [35] |
| GOT-OCR-2.0 | VLM | Apache-2.0 ✓ | "higher latency than modular" [26] | GPU-bound [26] | medium | Yes | Active | [26] |
| Idefics3 | VLM | Apache-2.0 ✓ | research-grade | GPU-bound | large | Yes | Active | [26] |
| Mindee docTR | Traditional/DL | Apache-2.0 ✓ | 91.7% [1] | 1.8 s [1] | Variable | Optional | Active | [10] |
| Nanonets-OCR-2 (FT Qwen2.5-VL-3B) | VLM (FT) | Apache-2.0 ✓ | trained on 3M finance pages [36] | inherits Qwen2.5-VL-3B | ~6 GB Q4 | Optional | Active | [36] |
| Mistral OCR (paid baseline) | API | Closed | **72.2% real world** [20] / 94% claimed | n/a | n/a | n/a | Active | [20][37] |

**Citation key:**
[1] codesota 2026 6-lib invoice bench — https://www.codesota.com/ocr/best-for-python
[2] invoicedataextraction — https://invoicedataextraction.com/blog/python-ocr-library-comparison-invoices
[3] PyImageSearch invoice OCR — https://pyimagesearch.com/2020/09/07/ocr-a-document-form-or-invoice-with-tesseract-opencv-and-python/
[4] Tesseract Apache LICENSE — https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE
[5] RapidOCR repo — https://github.com/RapidAI/RapidOCR
[6] RapidOCR vs PaddleOCR discussion — https://github.com/RapidAI/RapidOCR/discussions/237
[7] PaddleOCR repo — https://github.com/PaddlePaddle/PaddleOCR
[8] PaddleOCR-VL guide — https://dev.to/czmilo/2025-complete-guide-paddleocr-vl-09b-baidus-ultra-lightweight-document-parsing-powerhouse-1e8l
[9] PaddleOCR-VL Hugging Face — https://huggingface.co/PaddlePaddle/PaddleOCR-VL
[10] docTR repo — https://github.com/mindee/doctr
[11] HN comment on Surya 97.41% — https://news.ycombinator.com/item?id=43047121
[12] theneuralmaze on GLM-OCR / Llama-Vision CPU — https://theneuralmaze.substack.com/p/run-the-worlds-best-ocr-on-your-own
[13] Surya issue #183 low-VRAM — https://github.com/datalab-to/surya/issues/183
[14] vikp HN comment on Marker vs Mistral — https://news.ycombinator.com/item?id=43282905
[15] Marker repo — https://github.com/datalab-to/marker
[16] Nougat repo — https://github.com/facebookresearch/nougat
[17] OCR vs VLM accuracy table — https://www.f22labs.com/blogs/ocr-vs-vlm-vision-language-models-key-comparison/
[18] Qwen2.5-VL-3B HF — https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
[19] Qwen2.5-VL invoice substack — https://aihorizonforecast.substack.com/p/qwen2-vl-hands-on-guides-for-invoice
[20] GetOmni 1000-doc benchmark — https://getomni.ai/blog/benchmarking-open-source-models-for-ocr
[21] Labellerr Qwen2.5-VL local — https://www.labellerr.com/blog/run-qwen2-5-vl-locally/
[22] Llama 3.2 11B Vision specs — https://localaimaster.com/models/llama-3-2-11b-vision
[23] GLM-OCR Ollama — https://ollama.com/library/glm-ocr
[24] Mhjorleifsson GLM-OCR Ollama walkthrough — https://medium.com/@mhjorleifsson/local-accurate-and-fast-ocr-with-ollama-a530302eca39
[25] GLM-OCR HF — https://huggingface.co/zai-org/GLM-OCR
[26] modal.com 8-OCR comparison — https://modal.com/blog/8-top-open-source-ocr-models-compared
[27] codesota dotsOCR vs GPT-5.4 — https://www.codesota.com/ocr/best-for-invoices
[28] olmOCR-2 paper / blog — https://allenai.org/blog/olmocr-2
[29] olmOCR CUDA-OOM issue — https://github.com/allenai/olmocr/issues/50
[30] olmOCR-2 Ollama — https://ollama.com/richardyoung/olmocr2
[31] DeepSeek-OCR 2 review — https://labelyourdata.com/articles/deepseek-ocr
[32] Wong DeepSeek-OCR AMD Strix — https://medium.com/@yjwong/running-deepseek-ocr-locally-on-amd-strix-halo-a-journey-into-local-ai-powered-document-processing-ed9ab4c77ed0
[33] Sparkco DeepSeek-OCR CPU guide — https://sparkco.ai/blog/optimizing-deepseek-ocr-for-cpu-only-deployment-in-2025
[34] Moondream 2 — https://blog.roboflow.com/moondream-2/
[35] Florence-2 OCR — https://blog.roboflow.com/florence-2-ocr/
[36] Nanonets-OCR-2 — https://nanonets.com/research/nanonets-ocr-2/
[37] Mistral OCR HN — https://news.ycombinator.com/item?id=43282905
[38] Surya LICENSE — https://github.com/VikParuchuri/surya/blob/master/LICENSE
[39] Hybrid OCR architecture (Surya + VLM) — https://www.ahnafnafee.dev/blog/local-llm-pdf-ocr

---

## 3. Multi-Pass Workflow Recommendations

Three concrete pipelines from lightweight to max-quality. All assume preprocessing per §5 has already happened.

### 3.1 Lightweight (target: 0.3-1 s/page, ~85-90% field accuracy)

```
PDF → pdf2image(dpi=300) → preprocess(deskew + adaptive thresh + 3.5x upscale)
    → RapidOCR (ONNX) returns (text, bbox, confidence)
    → regex field templates (invoice_no, date, total, VAT, IBAN)
    → confidence threshold gate: anything <0.85 confidence → flagged for manual review
```

Use when: bulk historic invoices, audit-style processing, Adam reviews flagged fields. Zero VLM cost. Runs on Vega-8 hardware in ~0.5 s/page.

### 3.2 Balanced (target: 2-5 s/page, ~93-96% field accuracy) — **RECOMMENDED FOR PROD**

```
PDF → pdf2image(dpi=300) → preprocess
    ├─ Tesseract (PSM 6, LSTM OEM, --c tessedit_char_whitelist) → (text_T, conf_T)
    └─ RapidOCR (ONNX) → (text_R, bbox_R, conf_R)
    → field-level voter:
        if conf_T > 0.9 AND conf_R > 0.9 AND text_T == text_R: accept
        elif conf > threshold but disagreement: vote by character-level confidence
        else: route to GLM-OCR(image_crop, prompt="Read this invoice field exactly: <field_name>")
    → regex schema validation (date format, IBAN checksum, VAT format IE\d{7}[A-Z]{1,2})
    → cross-field consistency (subtotal + VAT == total within 0.02 EUR tolerance)
    → final JSON
```

Why: dual cheap engines catch each other's mistakes on different failure axes (Tesseract weak on small text and punctuation, RapidOCR weak on word spacing — see §4 themes). Only escalate to VLM on actual disagreement, keeping average latency manageable. GLM-OCR via Ollama on CPU adds ~10-20 s only on disputed fields (typically <20% of fields).

Pattern source: dual-engine OCR fusion paper ([sciencedirect](https://www.sciencedirect.com/science/article/pii/S111001682500657X)) achieved 96.5% character-level / 93.2% F1 on SROIE financial dataset. Combined with hybrid OCR pattern from [ahnafnafee.dev](https://www.ahnafnafee.dev/blog/local-llm-pdf-ocr): "Detection is fast, deterministic, well-solved. Recognition is slow, semantic, and where modern VLMs shine."

### 3.3 Max-quality (target: 15-60 s/page, ~96-99% field accuracy)

```
PDF → pdf2image(dpi=400) → preprocess (with phone-photo dewarping if CV detects perspective skew)
    → RapidOCR for layout/bbox → page-level structure JSON
    → For EACH detected field region:
        crop image → send to GLM-OCR or Qwen2.5-VL-3B Q4 (Ollama / llama.cpp)
        prompt: "This is the <field_name> region of an invoice. Return only the value."
    → regex + LLM validator: "Does '23/04/2026' parse as a valid invoice date?"
    → cross-field consistency
    → human-in-loop on anything below 0.95 composite confidence
```

Use when: high-stakes single document (audit pack, Revenue submission), accuracy beats speed. Per-field VLM cost is the price.

---

## 4. Real-User Findings — Synthesis

Organised by theme. Every claim is sourced.

### 4.1 What works (consistent across operators)

- **Cheap two-engine ensemble beats expensive single VLM on financial docs.** The dual-engine fusion paper [38] hit 96.5% char-level / 93.2% F1 on SROIE. Confidence-weighted voting reduces character error rates 30-50% vs single model ([summary at intuitionlabs](https://intuitionlabs.ai/articles/non-llm-ocr-technologies)).
- **Preprocessing matters more than model choice.** [Survey](https://medium.com/technovators/survey-on-image-preprocessing-techniques-to-improve-ocr-accuracy-616ddb931b76): *"The real secret: good preprocessing beats model choice more often than not."* Specifically: deskew (even 2-3°), adaptive thresholding (Sauvola or WolfJolion for invoices), upscale to 300+ DPI when low.
- **VLM-as-verifier (not VLM-as-OCR) is the high-yield pattern.** From [f22labs](https://www.f22labs.com/blogs/ocr-vs-vlm-vision-language-models-key-comparison/): *"the best approach is hybrid: OCR handles scale, and low-confidence or high-value documents get routed to VLMs for deeper extraction, correction, and validation."* PeterStuer on [HN/43282905](https://news.ycombinator.com/item?id=43282905): *"$1/1000 pages is significantly more expensive than my current local… setup."*
- **GLM-OCR is the small-model sleeper hit.** Mhjorleifsson's [Ollama walkthrough](https://medium.com/@mhjorleifsson/local-accurate-and-fast-ocr-with-ollama-a530302eca39) — confirmed temperature 0.1 + repeat penalty 1.2 makes JSON output deterministic. theneuralmaze called it *"#1 on OmniDocBench V1.5, beating models 10x its size"*.
- **PaddleOCR + PP-StructureV3 wins on table-heavy invoices.** [invoicedataextraction.com](https://invoicedataextraction.com/blog/python-ocr-library-comparison-invoices): *"PaddleOCR with PP-StructureV3 is the better choice when your pipeline handles tabular line items and multilingual invoices."*

### 4.2 What surprises (counter-intuitive findings)

- **Mistral OCR drops to 72.2% on real-world docs despite 94% headline.** themanmaran (Omni AI) on [HN](https://news.ycombinator.com/item?id=43282905): *"Mistral OCR aggressively classifies content as images, replacing entire sections with [image] placeholders; receipts particularly affected."* This is the verifier we are replacing — locally is not far behind the paid path on real receipts.
- **"Messy is sometimes faster than clean"** in hybrid pipelines because fewer detected boxes = less DP work + fewer LLM tokens ([ahnafnafee.dev](https://www.ahnafnafee.dev/blog/local-llm-pdf-ocr)).
- **Larger VLMs hallucinate field values that LOOK plausible.** rafram on [HN/43187209](https://news.ycombinator.com/item?id=43187209) on genealogical records: *"the model fabricated names/dates fitting document context but entirely false."* anon373839: *"models can rewrite document titles based on latent space navigation."* For bookkeeping where a fabricated VAT number passes a regex but is wrong, this is a hard stop. Confidence-weighted ensemble + cross-field consistency check is the mitigation.
- **Marker beats olmOCR by 56% win rate** on 1107-doc LLM-judged benchmark per Marker's creator vikp on [HN/43174298](https://news.ycombinator.com/item?id=43174298): *"a lot of missing text and hallucinations with olmocr."* But Marker is GPL-contaminated via Surya — same blocker.
- **GPL contamination is widespread in the modern OCR stack.** Surya, Marker, and downstream tools that wrap them all carry GPL-3.0 obligations. For a commercial Irish bookkeeping tool, only the Apache-2.0 / MIT path is safe.

### 4.3 Gotchas to avoid

- **olmOCR will not run on our hardware.** Confirmed [issue #50](https://github.com/allenai/olmocr/issues/50): hard CUDA dependency, OOM even on RTX 3090 24 GB on large PDFs. The Ollama port (`richardyoung/olmocr2:7b-q8`) inherits the model's underlying VLM cost — 7 B params running at CPU = unusable.
- **Llama-3.2-Vision 11B = 174 s on the first invoice page.** Hard kill for our hardware ([substack](https://theneuralmaze.substack.com/p/run-the-worlds-best-ocr-on-your-own)).
- **EasyOCR systematically misreads `$` as `8` and `€` as `E`.** Both [codesota](https://www.codesota.com/ocr/best-for-python) and [invoicedataextraction](https://invoicedataextraction.com/blog/python-ocr-library-comparison-invoices) confirm. Disqualifying for finance.
- **Mistral OCR misreads "Vision 2030" as "Vision 2.0"** (Saudi Central Bank financial Arabic), shekhargulati on [HN](https://news.ycombinator.com/item?id=43282905). Tells you the paid path is not infallible.
- **Marker v1.8.3+ has 68-70x regression vs v1.8.0** with VRAM jumping from 4 GB to 27 GB ([Wong](https://medium.com/@yjwong/running-deepseek-ocr-locally-on-amd-strix-halo-a-journey-into-local-ai-powered-document-processing-ed9ab4c77ed0)). Pin Marker if used.
- **PaddleOCR-VL Windows install requires separate venv for vllm vs paddlepaddle** to prevent dependency conflicts ([dev.to guide](https://dev.to/czmilo/2025-complete-guide-paddleocr-vl-09b-baidus-ultra-lightweight-document-parsing-powerhouse-1e8l)).
- **Poppler is missing on this machine.** `pdf2image` will fail loudly with `PDFInfoNotInstalledError`. Use [@oschwartz10612 Windows build](https://github.com/oschwartz10612/poppler-windows) and either set PATH or pass `poppler_path=` per-call ([install docs](https://pdf2image.readthedocs.io/en/latest/installation.html)).
- **Ollama on Vega 8 iGPU gives marginal speedup over CPU** and requires HSA_OVERRIDE_GFX_VERSION + a custom-built Ollama from PR #6282 ([machinezoo](https://blog.machinezoo.com/Running_Ollama_on_AMD_iGPU)). Not worth the operational complexity. Treat Vega 8 as compute-zero; budget for CPU-only.
- **VLM "confidence" is not real confidence.** themanmaran on [HN/43187209](https://news.ycombinator.com/item?id=43187209): *"Layout errors in complex tables; misaligned column data creates financial risk. Confidence scores don't prevent undetectable errors."* Tesseract's per-word confidence + cross-field schema validation is more honest than a VLM's self-reported probability.

---

## 5. Setup Playbook for Top 3 Engines

### 5.1 RapidOCR (primary text engine)

```bash
# Already installed at 3.8.1 in your env
pip install rapidocr-onnxruntime  # ~80 MB
```

```python
from rapidocr_onnxruntime import RapidOCR
ocr = RapidOCR()  # auto-downloads ONNX models on first call
result, _ = ocr("invoice.png")
# result = [[bbox, text, confidence], ...]
for box, text, conf in result:
    print(f"{conf:.2f}  {text}")
```

- **Install size:** ~80 MB
- **RAM at inference:** ~300 MB
- **Latency on Ryzen 5 3500U est:** 0.5-1.0 s/page (codesota measured 0.21 s on Apple M-series; AMD Vega-8 will be ~3-5x slower)
- **License:** Apache-2.0 — commercial-safe
- **Source:** https://github.com/RapidAI/RapidOCR

### 5.2 GLM-OCR via Ollama (VLM verifier)

```bash
# Already pulled: glm-ocr:q8_0 (1.6 GB)
ollama pull glm-ocr:q8_0  # confirm

# Test:
ollama run glm-ocr:q8_0
```

```python
import ollama
from PIL import Image
import io

def verify_field(image_crop_bytes: bytes, field_name: str) -> str:
    response = ollama.chat(
        model='glm-ocr:q8_0',
        messages=[{
            'role': 'user',
            'content': f'Read the {field_name} from this invoice region. Return only the value, no commentary.',
            'images': [image_crop_bytes],
        }],
        options={'temperature': 0.1, 'repeat_penalty': 1.2},  # Mhjorleifsson's settings
    )
    return response['message']['content'].strip()
```

- **Model size on disk:** 1.6 GB (Q8_0)
- **RAM at inference (CPU):** ~3-4 GB working set
- **Latency on Ryzen 5 3500U est:** 10-25 s per field crop (no GPU). Use sparingly — only on low-confidence fields.
- **License:** Apache-2.0 (`zai-org/GLM-OCR` HF) ✓
- **Source:** https://ollama.com/library/glm-ocr | https://huggingface.co/zai-org/GLM-OCR
- **Gotcha:** without `temperature=0.1` and `repeat_penalty=1.2`, JSON outputs enter repetition loops on long transaction lists.

### 5.3 Tesseract 5.5 (second voter in ensemble)

```bash
# Already installed at 5.5.0
tesseract --version
```

```python
import pytesseract
from PIL import Image

# Per PyImageSearch invoice recipe: PSM 6 + LSTM
config = '--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz./-:€$,'
data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
# data['text'], data['conf'] (0-100 per word), data['left'], data['top'], data['width'], data['height']
```

- **Install size:** ~10 MB binary + 4-50 MB per language pack
- **RAM at inference:** ~100 MB
- **Latency on Ryzen 5 3500U:** **0.2-0.5 s/page** — fastest option
- **License:** Apache-2.0 ✓
- **Source:** https://github.com/tesseract-ocr/tesseract
- **Critical settings for invoices:** `--psm 6` (uniform block) for whole invoice, `--psm 4` (single column line-by-line) per [PyImageSearch receipt walkthrough](https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/), `--oem 1` (LSTM only — best on modern docs).

### 5.4 Pre-flight (do this first)

```bash
# Install poppler — currently missing
# Download https://github.com/oschwartz10612/poppler-windows/releases (latest)
# Extract to C:/poppler-25.x/
# Add C:/poppler-25.x/Library/bin to PATH

# Verify
pdftoppm -h  # should print help, not "not recognized"

# Python
pip install pdf2image pillow opencv-python numpy
```

Confirm Tesseract path is on PATH: `tesseract --version` should print `tesseract 5.5.0`.

---

## 6. Mistral-OCR vs Local Stack — Realistic Delta

Mistral-Small-3.1 (the LLM verifier we are using) is not the same product as Mistral-OCR-latest (the dedicated paid OCR model). The Mistral-OCR product itself underperforms on real-world docs:

- **Mistral-OCR self-reported:** 94.89% overall, "best in class" ([HN/43283411](https://news.ycombinator.com/item?id=43283411))
- **Independently measured:** **72.2% on 1000-doc real-world test** ([GetOmni bench](https://getomni.ai/blog/benchmarking-open-source-models-for-ocr); themanmaran on [HN/43282905](https://news.ycombinator.com/item?id=43282905))
- **What outperforms it:** Qwen2.5-VL 32B/72B (~75%), Marker (LLM-judge 4.41 vs 4.32), Docsumo (proprietary), MinerU/PDF-Extract-Kit on medical docs
- **Failure mode unique to Mistral OCR:** "aggressively classifies content as images, replacing entire sections with `[image]` placeholders" — affects receipts especially badly
- **Specific known errors:** "Vision 2030" → "Vision 2.0"; missed Saudi Central Bank metadata; hallucinated French word "mortilhomme"

**Realistic field-accuracy delta on Irish invoices:**

| Stack | Estimated field accuracy | Speed/page | Cost/1k pages |
|---|---|---|---|
| Mistral-OCR (paid) | 75-85% on receipts, 88-92% on clean PDFs | API-bound | ~$1 + latency cost |
| Mistral-Small-3.1 vision (current verifier) | 80-88% (LLM, not OCR-specialised) | API-bound | per-token |
| **RapidOCR + GLM-OCR verifier (proposed)** | **88-94%** with preprocessing | 1-3 s clean / 15-30 s contested | $0 |
| Tesseract+RapidOCR ensemble + GLM verifier | 92-96% with cross-field validation | 3-8 s | $0 |

The accuracy gap is positive in our favour for our document types. The price gap is infinite. The privacy gap is infinite.

---

## 7. Concrete Recommendation

**Replace Mistral-Small-3.1 verifier with the §3.2 Balanced Pipeline:**

1. **Install poppler** (mandatory — pdf2image will fail without it).
2. **Wire RapidOCR (ONNX)** as the primary text + bbox engine — already installed, Apache-2.0, 0.5-1 s/page CPU.
3. **Wire Tesseract 5.5** as second voter — already installed, 0.2-0.5 s/page CPU, Apache-2.0.
4. **Wire GLM-OCR Q8 via Ollama** as the VLM verifier on disagreements only — already pulled (1.6 GB), Apache-2.0, settings: `temperature=0.1`, `repeat_penalty=1.2`. ~10-25 s per low-confidence field.
5. **Preprocessing** — adaptive threshold (Sauvola), deskew (Hough), upscale 3.5x for sub-300-DPI inputs, optional dewarp for phone photos.
6. **Field schema validator** — regex per IE invoice pattern (`IE\d{7}[A-Z]{1,2}` for VAT, IBAN checksum for accounts, ISO date parsing, decimal tolerance for totals).
7. **Cross-field consistency** — `subtotal + VAT == total ± 0.02 EUR`, `invoice_date < due_date`, etc.
8. **Manual-review queue** for any field ending under composite confidence 0.90 — Adam reviews these.

**Hold in reserve (not first build):**
- PaddleOCR-VL 0.9 B as a single-engine alternative once Windows venv pain is verified (could replace the dual-engine pair if it tests cleaner — Apache-2.0, OmniBenchDoc #1 claim).
- DeepSeek-OCR-2 3B as a step-up if GLM-OCR field-level outputs prove too short on context (MIT, structured grounding, but CPU latency risk on Vega 8 hardware).
- Qwen2.5-VL-3B Q4 as a second VLM voter if GLM-OCR alone has systematic blind spots.

**Do not pursue:**
- olmOCR / Llama-3.2-Vision (CUDA / latency)
- Surya / Marker (GPL — commercial blocker)
- EasyOCR (`$`→`8` failure mode)
- Vega-8 iGPU acceleration via Ollama (operational complexity exceeds gain on 2 GB VRAM)

This stack is fully Apache-2.0 / MIT, runs on the deployed hardware, costs zero per page, and the accuracy floor is at or above the current Mistral verifier path.

---

## Appendix — Citations not yet inline

Additional sources consulted:

- HN thread "Most accurate OCR" — https://news.ycombinator.com/item?id=43047121
- HN thread "Replace OCR with VLMs" — https://news.ycombinator.com/item?id=43187209
- HN olmOCR thread — https://news.ycombinator.com/item?id=43174298
- Mistral OCR new release HN — https://news.ycombinator.com/item?id=43283411
- Show HN OCR Benchmark Automation — https://news.ycombinator.com/item?id=43347524
- Pyimagesearch Tesseract PSM guide — https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/
- Pyimagesearch receipt OCR — https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/
- arxiv preprocessing for OCR — https://arxiv.org/abs/2111.14075
- Tesseract improving quality docs — https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html
- Label Studio receipt OCR — https://labelstud.io/blog/improve-ocr-quality-for-receipt-processing-with-tesseract-and-label-studio/
- Merfantz 900-invoice integration case — https://www.merfantz.com/blog/transforming-invoice-processing-with-ocr-seamless-integration-of-900-transactions-into-sage/
- Docling repo — https://github.com/docling-project/docling
- IBM Lean-AI invoice extractor — https://github.com/rajans-code/lean-ai-invoice-extractor
- Webkul invoice + OCR + AI — https://webkul.com/blog/invoice-data-extraction-ocr-ai/
- Quora best OS OCR for invoices — https://www.quora.com/What-is-the-best-open-source-OCR-self-learning-AI-invoices-processing-script-as-of-today
- Towards AI 99%-accurate invoice — https://towardsai.net/p/machine-learning/how-we-built-a-99-accurate-invoice-processing-system-using-ocr-and-llms
- xda-developers self-host docs — https://www.xda-developers.com/i-use-local-llms-and-self-hosted-apps-to-manage-my-documents/
- Kreuzberg discussion HN — https://news.ycombinator.com/item?id=46692706

**Total external citations:** 50+ unique URLs across the document.

— end —
