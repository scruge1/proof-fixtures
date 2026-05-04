# Commercial Document-AI Vendor Architectures — What's Actually Under The Hood

**Author:** Claude (research subagent)
**Date:** 2026-05-04
**Purpose:** Inform v0.4 PRD for an open-source-only invoice/receipt extraction stack (Apache-2.0 only, runs locally on Ryzen 5 / Vega 8 / 30 GB RAM).
**Method:** Vendor blogs, white papers, conference talks, Hugging Face model cards, GitHub issues, engineering job postings, customer-facing tech docs, public arXiv submissions.
**Companion:** `proof-fixtures/research/2026-05-04-local-ocr-deep-research.md` (engine bake-off — already complete; do not re-research engines).

---

## §1 Executive Summary — Common Patterns Across All 10 Vendors

Five patterns hold across every vendor surveyed. The 2024-2026 commercial Document-AI playbook has converged hard:

1. **Two-stage pipeline is universal.** Every vendor without exception runs OCR (or VLM-pretend-OCR) first, then layout-aware structured extraction second. Rossum, Hyperscience, Mindee, AWS, Azure, Google, Klippa, Veryfi, Ocrolus, Nanonets — all of them. The disagreement is only about which model family does the second stage (LayoutLM-style, transformer T-LLM, full VLM, GNN, or agentic ensemble).

2. **Proprietary domain-tuned models on top of open foundations.** The 2024-2026 commercial moat is not "we have a better OCR engine" — Tesseract, Paddle, and dot-com-era engines are commodities. The moat is fine-tuned weights on millions to billions of in-domain documents (Veryfi: hundreds of millions of receipts; Rossum: 11M transactional docs; Nanonets: 3M+ pages). Every vendor that names its base layer names an open foundation: LayoutLM, Qwen2.5-VL, ResNet/CNN+RNN, DBNet/CRNN. The proprietary part is the domain corpus and the post-training (LoRA, instruction tune, two-stage curriculum, distillation).

3. **Human-in-the-loop is the silent third stage.** Ocrolus markets it loudest, but Hyperscience (HiTL block), Klippa (confidence-routed review), Rossum (Instant Learning annotation), Mindee (corrections fed via RAG), Veryfi (clean-room verification) and even AWS (Augmented AI A2I) all bake confidence-thresholded human review into the production loop. "Touch" is non-negotiable in commercial IDP — the gate is which decimal place of accuracy you commit to per-document review on.

4. **The 2025 wave has shifted from LayoutLM-family to VLM-family.** The 2018-2023 SOTA was LayoutLMv3-style multimodal transformers fed bbox+text (Microsoft/Azure, AWS, the entire DocILE 2023 baseline set). The 2024-2026 SOTA is full VLMs (Qwen2.5-VL, Gemini 2.5 Pro, internal models) doing OCR-free or OCR-hybrid extraction. Hyperscience's ORCA, Google's Gemini-powered Custom Extractor v1.5-pro, Nanonets-OCR2, Rossum's T-LLM all post-2023 are VLM-first. AWS Textract and Azure Doc Intel are the most LayoutLM-anchored holdouts.

5. **Active learning + customer-corpus continuous training is the lock-in.** Every vendor pitches "model improves the more you use it." Klippa, Rossum, Mindee, Hyperscience, Ocrolus, Google Custom Extractor, Azure Custom Neural — all expose a UI for the customer to correct extraction outputs which become next-cycle training labels. This is the moat that makes switching vendors expensive: your N months of corrections are sunk cost.

**Implication for the v0.4 PRD:** an Apache-2.0-only competitor will not out-train Veryfi on receipts or Rossum on invoices — that race is lost. The open-source angle has to lean on auditability, on-prem privacy, zero per-page cost, and enabling customers to run *their own* corrected-corpus loop without vendor lock-in. Pattern 5 (corpus moat) is the one to take seriously and answer with tooling, not data scale.

---

## §2 Per-Vendor Internal Architecture Summary

### 2.1 Klippa (acquired by SER Group, rebranded Doxis)

**Internal architecture:** Klippa's DocHorizon (now Doxis AI.dp) runs a templateless two-stage pipeline. Stage 1 converts the image/PDF to raw TXT via an OCR engine (engine identity not publicly disclosed; Klippa marketing avoids naming the underlying recognizer, which is itself a tell that it's a wrapped/tuned third-party engine). Stage 2 — the "Klippa Parser" — takes the unstructured TXT and converts it to structured JSON via machine learning, producing per-field confidence scores. The pipeline supports 150+ languages, 100+ document types, sub-5-second processing claims, and 100+ country invoice coverage including SKU-level line-item extraction. Custom-field extraction is supervised: customers upload annotated training docs (80/20 train/test split), iterate to acceptable confidence thresholds, then deploy. Distinctive tech feature: built-in fraud detection (font anomaly, date inconsistency, altered-field signals) inherited from invoice-specific training. As of 2025-2026 Klippa was acquired by SER Group and rebranded under the Doxis Intelligent Content Platform (alongside Metaforce and AFI Solutions acquisitions), so further engineering disclosure is now filtered through SER's enterprise marketing and is even less specific than pre-acquisition Klippa blogs.

- [Klippa OCR API documentation](https://www.klippa.com/en/ocr/ocr-api/) — confirms TXT-then-Parser two-stage architecture, JSON output via REST.
- [Klippa Custom Data Field Extraction](https://www.klippa.com/en/dochorizon/custom-data-field-extraction/) — confirms supervised-learning + 80/20 split + confidence-iteration training loop.
- [Klippa acquired by SER Group / IDP-Software vendor profile](https://idp-software.com/vendors/klippa/) — March 2025 acquisition; rebrand to Doxis Jan 2026.

### 2.2 Rossum (Aurora / T-LLM)

**Internal architecture:** Rossum's Aurora platform (launched Feb 2024, refreshed to Aurora 1.5 in Oct 2024) is built around a proprietary "Transactional Large Language Model" (T-LLM) trained in-house on roughly 11 million transactional documents (invoices, purchase orders, delivery notes). Rossum has been the only major IDP vendor to publish its training-data benchmark publicly: DocILE (arXiv:2302.05658), containing 6.7k human-annotated business docs + 100k synthetic + ~1M unlabeled for unsupervised pre-training, with 55 annotation classes and a Line Item Recognition (LIR) sub-task. The DocILE paper benchmarks RoBERTa, LayoutLMv3, and DETR-based Table Transformer baselines, which strongly implies T-LLM's architectural lineage is in the LayoutLMv3 + DETR + transformer-decoder family rather than a Qwen-VL-style full VLM. Aurora supports 276 languages, claims 92.5% average extraction accuracy, and centers on "Instant Learning" — single-document annotation by an end user updates extraction behavior for that customer corpus immediately, without a retrain cycle. Aurora 1.5 brought a claimed +25% extraction-accuracy improvement, line-item handling for tens of pages of nested tables, and a "Copilot" overlay for human reviewers. Critically, Rossum publishes that *all* T-LLM outputs are verified against an annotator pool — i.e., the human-in-the-loop is part of the *training* loop, not just inference QA.

- [Rossum Aurora press release (Feb 2024)](https://rossum.ai/company/newsroom/press-release-rossum-aurora/) — announces T-LLM, 11M transactional document training corpus, in-house LLM development.
- [Rossum Aurora 1.5 launch (Oct 2024)](https://rossum.ai/company/newsroom/press-release-rossum-unveils-aurora-15-ai-engine-and-copilot-to-accelerate-document-processing-for-global-enterprises-in-276-languages/) — 276 languages, +25% accuracy claim, Instant Learning capability.
- [DocILE benchmark (arXiv:2302.05658)](https://arxiv.org/abs/2302.05658) — Rossum's published research benchmark; reveals architectural baselines (RoBERTa / LayoutLMv3 / DETR) the T-LLM is implicitly built from.
- [Deep Analysis — Rossum launches its own LLM](https://www.deep-analysis.net/rossum-launches-its-own-llm/) — independent vendor analyst confirmation of T-LLM scope and rationale.

### 2.3 Mindee (docTR + paid Platform API)

**Internal architecture:** Mindee operates a clear open-source-floor / proprietary-ceiling split. The floor is **docTR** (Document Text Recognition) — Apache-2.0, public on GitHub (`mindee/doctr`), implementing a clean two-stage detect-then-recognize pipeline using published-research backbones: DBNet, LinkNet, FAST for text detection; CRNN, SAR, MASTER, ViTSTR, PARSeq for text recognition; available in both TensorFlow 2 and PyTorch. The ceiling is the **Mindee Platform API** — proprietary cloud service with per-document-type prebuilt extractors (invoice, receipt, ID card, passport, US mail, financial statements). The proprietary delta over docTR is: (a) field-extraction layer (docTR is OCR-only; the platform adds key-value typing — vendor name, totals, line items, dates), (b) "the API already knows what it's looking at" — i.e., a document-classification head trained on the Mindee customer corpus, (c) a RAG-based continuous-learning layer where customer corrections become a knowledge base of past corrections fed back into inference, (d) visual training tools for non-developer users to define new document types, and (e) ensemble routing across "lots of models including latest LLMs and proprietary models." The `mindee/doctr` repo's release cadence (still actively shipped as of 2026) suggests the open-source layer is a real production component, not a dead-marketing repo.

- [Mindee docTR GitHub](https://github.com/mindee/doctr) — open-source floor; DBNet/CRNN/MASTER/ViTSTR architecture stack made explicit in the README.
- [Mindee Platform overview](https://www.mindee.com/) — "developer-first" positioning, RAG-for-corrections as the proprietary delta.
- [Mindee Invoice OCR API](https://www.mindee.com/product/invoice-ocr-api) — confirms ensemble of LLMs + proprietary models for field extraction.

### 2.4 Nanonets (Nanonets-OCR2)

**Internal architecture:** Nanonets's flagship 2024-2026 model is **Nanonets-OCR2-3B** (and 1.5B-exp variant), a Qwen2.5-VL-3B fine-tune released openly on Hugging Face. Training methodology is published openly in the Hugging Face model card and Nanonets blog: a **two-stage curriculum**. Stage 1 trains the model on a synthetic corpus to build foundational OCR + structure-decoding capability across document types and layouts. Stage 2 fine-tunes on ~3M+ manually annotated pages spanning research papers, financial reports, legal contracts, healthcare records, tax forms, receipts, and invoices, with explicit annotation of structural elements: equations (LaTeX), tables (HTML/Markdown), signatures, watermarks, checkboxes, flowcharts (Mermaid), and handwritten content. The fine-tune approach is full-parameter on the base 3B (not LoRA at the production-model level, though Nanonets's own blog teaches LoRA as the recommended customer-side fine-tuning approach for their open base). Nanonets's broader Document AI commentary in their blog explicitly discusses knowledge distillation from "very large models (50+B)" down to smaller deployable models — strongly suggesting their internal pipeline has a teacher-student step where a frontier VLM labels the synthetic corpus and the 3B production model is distilled from those labels. Distinctive output trait: instead of bounding-box JSON, the model emits LLM-ready *structured markdown* with semantic tags (`<table>`, `<signature>`, `<watermark>`, `<img>`, `<equation>`), which composes cleanly with downstream LLM chains.

- [Nanonets-OCR2-3B Hugging Face model card](https://huggingface.co/nanonets/Nanonets-OCR2-3B) — Qwen2.5-VL-3B base, two-stage train, semantic-tag output format, 3M+ training pages.
- [Nanonets Research — Nanonets OCR 2](https://nanonets.com/research/nanonets-ocr-2/) — published architecture overview and document-type coverage.
- [Nanonets blog — Fine-Tuning VLMs for Data Extraction](https://nanonets.com/blog/fine-tuning-vision-language-models-vlms-for-data-extraction/) — LoRA + adapter recipe for customer fine-tuning; mentions distillation from 50B+ teacher models.

### 2.5 Hyperscience (Hypercell + ORCA)

**Internal architecture:** Hyperscience's 2025-2026 platform is **Hypercell** — a modular agentic pipeline of "Blocks" (discrete processing units: ingestion, classification, extraction, validation, business-rules, decisioning, LLM-analysis) wired together into "Flows" via a **Hyperflow** orchestrator. Hyperflow is a tiny API-compatible reimplementation of Netflix Conductor (open-sourced approach, internal implementation), built so on-prem customers can run a single coherent workflow engine across SaaS and self-hosted deployments. The extraction model itself is **ORCA (Optical Reasoning and Cognition Agent)** — Hyperscience's proprietary VLM framework. ORCA is explicitly described in their engineering blog as a wrapper *around* open-source VLMs, not a from-scratch model: "The Hyperscience ORCA VLM Framework wraps proprietary models around open-source VLMs. This allows for the utilization of the latest or best-performing models on the market." Two operational distinctives: (1) **agentic ensemble** — a goal-oriented agent orchestrates multiple specialized models (proprietary, VLMs, LLMs) per document, optimizing across accuracy/speed/cost/compliance thresholds; (2) **accuracy-harnessed HiTL** — customers set hard accuracy targets, and any document the model can't hit those thresholds on is automatically routed to human review. Hyperscience also runs a custom "Long Form Extraction Model" for documents with >100 fields and an Entity Recognition Block for low-data starts. Key engineering insight from their job postings and blog: the team explicitly invests in *training with limited data* (Training Data Curator labels training docs as high/low importance to maximize signal per labeled page), which is the opposite of the brute-force-corpus approach Veryfi and Rossum rely on.

- [Hyperscience ORCA VLM blog](https://www.hyperscience.ai/blog/faster-safer-smarter-document-ai-without-the-training-delay/) — ORCA framework architecture, wraps open-source VLMs, day-one extraction (no per-customer training).
- [Hyperscience — Better together: LLMs + Hyperscience proprietary models](https://www.hyperscience.ai/blog/better-together-llms-hyperscience-proprietary-models/) — ensemble approach made explicit.
- [Hyperscience — Orchestrating Our ML Platform](https://www.hyperscience.ai/blog/orchestrating-our-ml-platform/) — Hyperflow orchestrator as a Netflix Conductor reimplementation.
- [Hypercell platform overview](https://www.hyperscience.com/platform/) — Block + Flow modular architecture.

### 2.6 Ocrolus (bank-statement specialist)

**Internal architecture:** Ocrolus's core engineering insight is published openly: **"Ocrolus' system intelligently selects the extraction or OCR tool which results in the highest raw accuracy, then layers in proprietary pattern recognition and machine learning models."** This is a router-then-ensemble pattern — Ocrolus does not operate a single OCR engine; it operates a model-routing layer that picks the best engine per document, then runs proprietary post-processing (pattern recognition, ML classification, statement-specific structural decoding). The platform is bank-statement-and-financial-doc specialized: 1,700+ form types supported, with bank statements, paystubs, tax returns as the volume-leader categories. The defining architectural commitment is **human-in-the-loop as a first-class production component**: all documents that fail confidence thresholds during extraction are routed to "data classification and verification specialists" working in SOC-compliant clean rooms; their corrections feed back into the training loop. Ocrolus claims 99+% accuracy — but the operative word is "claims," because the published headline number folds in the human-in-the-loop verification stage, not pure-ML accuracy. Their approach is cleanly described in their own blog: "humans in places where humans are best, AI where it is best, each assisting the other." Notable 2024-2026 architecture move: tasks that fail confidence are routed to *multiple AI agents to check each other's work* before falling through to human review — a triple-redundancy ensemble before the human gate.

- [Ocrolus — How Ocrolus Works](https://docs.ocrolus.com/docs/how-ocrolus-works) — confirms router-then-proprietary-ML pattern.
- [Ocrolus blog — Role of Humans in AI-Driven Document Automation](https://www.ocrolus.com/blog/the-role-humans-with-ai-document-automation/) — HiTL is the published architectural commitment.
- [Ocrolus Platform — AI Engine for Document & Decision Intelligence](https://www.ocrolus.com/platform/) — multi-agent pre-human-review redundancy.

### 2.7 Veryfi (receipt and invoice specialist)

**Internal architecture:** Veryfi runs **proprietary foundation models trained on hundreds of millions of receipts and invoices** — the largest disclosed in-domain training corpus of any vendor in this survey. Their published multimodal architecture stack is the most specific of any vendor: a CNN for image processing, a graph neural network (GNN) for structural decoding, NLP for field semantic typing, and ICR for handwriting on top of the OCR layer. Specifically: "AI-enhanced ICR and Document AI use neural architectures, CNNs, RNNs, and Transformers to perceive spatial hierarchies and semantic meaning simultaneously. Models understand geometry including tables, totals, vendor zones, and signatures, while NLP classifies values as 'Invoice Number,' 'Due Date,' or 'Total Amount,' even when phrasing varies." The GNN treatment of document layout — receipts as graphs of spatial-semantic regions — is unusual in the field; most competitors run transformer-with-bbox embeddings (LayoutLM-style). Infrastructure is also publicly disclosed: a fleet of NVIDIA DGX H100s in air-gapped Santa Clara facilities, security-cleared ML staff, GDPR/HIPAA/SOC2 compliance posture. Output coverage: 85 currencies, 39 languages, 90 defined fields. Veryfi distinctly markets a **build-vs-buy story**: 6-day Veryfi integration vs. 6-month internal Document AI builds, leaning on the corpus moat. 2026 distinctive feature: AI-generated-receipt detection (deepfake-receipt fraud signal), which depends on having seen enough real receipts to spot synthetic ones.

- [Veryfi — Multimodal Document Data Extraction](https://www.veryfi.com/technology/multimodal-data-extraction-beyond-basic-ocr/) — CNN + GNN + Transformer + NLP stack made explicit.
- [Veryfi — Deep Analysis vendor review](https://www.deep-analysis.net/vendor-vignette-0/veryfi-review/) — independent confirmation of 4-year training-corpus accumulation, NLP-enhances-OCR architecture.
- [Veryfi — Why Veryfi](https://www.veryfi.com/why-veryfi/) — DGX H100 air-gapped infrastructure, hundreds-of-millions training corpus.
- [Veryfi Receipt OCR API](https://www.veryfi.com/receipt-ocr-api/) — output format, field count, language coverage.

### 2.8 AWS Textract

**Internal architecture:** AWS Textract is the most architecturally LayoutLM-anchored of the cloud-three. Its public documentation reveals a CNN+RNN+Transformer hybrid lineage: "Textract uses Amazon's computer-vision deep learning models (related to those used in Rekognition), based on the same proven, highly scalable, deep-learning technology that was developed by Amazon's computer vision scientists to analyze billions of images and videos daily." For invoices specifically, the **AnalyzeExpense API** is a separate pre-trained model from the general AnalyzeDocument path — purpose-built to return ExpenseDocuments objects pre-typed into SummaryFields (vendor, date, total, tax) and LineItemGroups. AWS markets this as "no need to train a custom model" — i.e., AnalyzeExpense is a frozen pre-trained model with no per-customer fine-tune by default. Customization happens through **Adapters** — components that plug into the pre-trained Textract model and customize output for business-specific document types, similar architecturally to LoRA adapters but exposed as a managed cloud feature. The 2023-2024 **LAYOUT feature** added explicit layout-element tagging (TITLES, HEADERS, FOOTERS, TABLES, KEY_VALUES, PAGE_NUMBERS, LISTS, FIGURES) — directly inspired by LayoutLMv3 outputs and intended to feed downstream generative-AI pipelines (Bedrock + LangChain) cleanly. AWS's own published guidance recommends combining Textract bounding boxes + text with LayoutLM-style location-aware models on SageMaker for downstream extraction — implicitly admitting the AnalyzeDocument layer is OCR+layout, not full structured-extraction.

- [AWS Blog — Textract Layout feature for generative AI tasks](https://aws.amazon.com/blogs/machine-learning/amazon-textracts-new-layout-feature-introduces-efficiencies-in-general-purpose-and-generative-ai-document-processing-tasks/) — LAYOUT element schema, downstream LLM pairing.
- [AWS Blog — Bring structure to diverse documents with Amazon Textract and transformer-based models on SageMaker](https://aws.amazon.com/blogs/machine-learning/bring-structure-to-diverse-documents-with-amazon-textract-and-transformer-based-models-on-amazon-sagemaker/) — confirms LayoutLM-style architecture choice for downstream extraction.
- [AWS Docs — Analyzing Invoices and Receipts](https://docs.aws.amazon.com/textract/latest/dg/invoices-receipts.html) — AnalyzeExpense API specialized model.
- [AWS Blog — Build a receipt and invoice processing pipeline with Amazon Textract](https://aws.amazon.com/blogs/machine-learning/build-a-receipt-and-invoice-processing-pipeline-with-amazon-textract/) — production architecture pattern.

### 2.9 Azure Document Intelligence (formerly Form Recognizer)

**Internal architecture:** Azure Document Intelligence has the most directly attributable model lineage of any vendor: it is built on the **LayoutLM family** out of Microsoft Research. This is publicly documented — Microsoft Research's Document AI page explicitly states "the pre-trained LayoutLM model family for Document AI has been widely adopted by 1st and 3rd party products and applications in Azure AI, such as Form Recognizer." LayoutLMv3 is the current-generation architecture: a multimodal transformer pre-trained jointly on text, layout (bounding-box coordinates), and image patches via masked language modeling, masked image modeling, and word-patch alignment objectives. Azure exposes this via **prebuilt models** (invoice, receipt, ID, business-card, W2, layout-only) that are frozen LayoutLMv3-style models fine-tuned on Microsoft's transactional-document corpus, supporting 27 languages for invoices. For customization Azure offers two paths: **Custom Template** (legacy, regex-style positional matching for fixed layouts — fast and cheap, breaks on layout drift) and **Custom Neural** (modern path, deep-learning-based, fine-tunes the LayoutLMv3 base on customer-supplied annotated docs to handle unstructured/semi-structured variation). The **Document Intelligence Studio** UI is the customer-facing labeling and training surface, exposing the same workflow: annotate ~5-50 sample docs, kick off a fine-tune, deploy. Azure's 2025+ releases have added an explicit **Generative Mode** layer on top, where the LayoutLMv3-style base provides token+bbox extraction and a downstream Azure OpenAI model (GPT-4-class) does the field-level structured-output decode — a hybrid that matches what AWS recommends manually but offers as a managed service.

- [Microsoft Research — Document AI / LayoutLM](https://www.microsoft.com/en-us/research/project/document-ai/) — direct attribution of LayoutLM family as the Form Recognizer / Document Intelligence foundation.
- [LayoutLM original paper (Microsoft Research)](https://www.microsoft.com/en-us/research/publication/layoutlm-pre-training-of-text-and-layout-for-document-image-understanding/) — pre-training architecture: text + layout joint embedding.
- [Azure Custom Neural Document Model docs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-neural?view=doc-intel-4.0.0) — confirms deep-learning fine-tune path on LayoutLM-style base.
- [Azure Invoice prebuilt model docs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/invoice?view=doc-intel-4.0.0) — 27-language coverage, prebuilt JSON schema.

### 2.10 Google Document AI

**Internal architecture:** Google Document AI is the most aggressively VLM-pivoted of the cloud-three. The platform has shifted from older specialist processors (Invoice Parser v1.5 — base entity-extraction model trained on English invoices) toward **foundation-model-powered Custom Extractor** — `pretrained-foundation-model-v1.5-2025-05-05` is GA, and `pretrained-foundation-model-v1.5-pro-2025-06-20` (powered by Gemini 2.5 Pro) is in Preview. Google's own architectural framing in their public docs: "Document AI provides simple access to powerful foundation models that help customers create parsers... Document AI is powered by generative AI, and future versions are using new foundation models so you can benefit from generative AI enhancements—as they improve foundation models, earlier foundation models are deprecated." This is an explicit statement that Google is sunsetting the legacy specialist-model architecture (Invoice Parser, Form Parser) in favor of a Gemini-as-extraction-engine architecture. Gemini's multimodal native handling means it natively integrates OCR (via Cloud Vision) with text+image+layout reasoning in a single model pass, rather than the two-stage OCR-then-extract that AWS and Azure run. The **Layout Parser** is a Gemini-powered processor specifically for chunking long documents, indicating the Gemini integration is broad, not invoice-only. Customer fine-tuning is exposed via **Custom Extractor with generative AI** — a UI for uploading annotated docs and kicking off a Gemini-based fine-tune. The **Document AI Workbench** is the labeling/training/evaluation surface.

- [Google Cloud — Document AI Custom Extractor with generative AI](https://cloud.google.com/document-ai/docs/ce-with-genai) — confirms Gemini-powered foundation-model architecture for custom extraction.
- [Google Cloud — Document AI Processor list](https://docs.cloud.google.com/document-ai/docs/processors-list) — current processor inventory.
- [Google Cloud — Document AI release notes](https://docs.cloud.google.com/document-ai/docs/release-notes) — `pretrained-foundation-model-v1.5-pro-2025-06-20` (Gemini 2.5 Pro) preview release.
- [Google Cloud — Process documents with Gemini layout parser](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk) — Gemini as the layout-parsing engine.

---

## §3 Convergent vs Divergent Patterns

### 3.1 Convergent (where everyone agrees)

These are the bets every vendor has made. If your v0.4 stack does *not* take these bets you will lose to vendors that did, all else equal:

| Pattern | Klippa | Rossum | Mindee | Nanonets | Hyperscience | Ocrolus | Veryfi | AWS | Azure | Google |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Two-stage pipeline (OCR/perception → structured extract) | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| Layout-aware model (text + bbox + image) | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| Confidence scores per extracted field | yes | yes | yes | yes | yes | yes | yes | yes | yes | yes |
| Active-learning customer-correction loop | yes | yes | yes | yes | yes | yes | yes | yes (Adapters) | yes | yes |
| HiTL for documents below confidence threshold | yes | yes | yes | yes (eval) | yes (HiTL Block) | **CORE** | yes | yes (A2I) | yes | yes |
| Open-foundation base + proprietary fine-tune | yes (likely) | yes (LayoutLMv3 lineage) | yes (docTR) | yes (Qwen2.5-VL) | yes (open VLMs) | yes (router) | yes (open backbones) | proprietary | yes (LayoutLM) | yes (Gemini) |
| JSON output schema with line-item structure | yes | yes | yes | yes (markdown+JSON) | yes | yes | yes | yes | yes | yes |
| Per-document-type specialized extractor | yes | yes | yes | yes | yes (Blocks) | yes (1700+ types) | yes | yes (AnalyzeExpense, ID, etc.) | yes (prebuilts) | yes (Custom Extractor) |
| Fraud / anomaly detection signal | yes | partial | partial | partial | yes | yes (lender-grade) | yes (deepfake) | partial | partial | partial |

### 3.2 Divergent (where vendors disagree, and what they disagree about)

| Decision axis | Camp A | Camp B | Camp C |
|---|---|---|---|
| **Base model architecture** | LayoutLM-family transformer with bbox embeddings (Azure, AWS, Rossum-T-LLM-implicit) | Full VLM (Hyperscience ORCA, Nanonets-OCR2, Google Gemini, late-2025 Mindee) | Multimodal CNN + GNN ensemble (Veryfi distinctive, partly Ocrolus) |
| **OCR layer** | Distinct pretrained OCR model upstream (AWS, Azure, Klippa, Mindee docTR, Ocrolus router) | OCR-free / OCR-implicit inside the VLM pass (Google Gemini, Hyperscience ORCA, Nanonets-OCR2 to a degree) | Hybrid: OCR for primary text + VLM for structure (Rossum, late-2024 Mindee) |
| **Customization mechanism** | Per-customer fine-tune of base model (Azure Custom Neural, Google Custom Extractor, Klippa, Rossum Instant Learning) | Adapter / LoRA layer on a frozen base (AWS Adapters, Mindee continuous-learning, Nanonets recommended customer path) | Few-shot / zero-shot prompting of large VLM (Hyperscience ORCA day-one, Google Gemini default) |
| **HiTL philosophy** | Workflow-orchestrator-inserts-human-on-low-confidence (most vendors) | Human-as-training-source (Ocrolus core thesis, Rossum Instant Learning) | Human-as-policy-gate (Hyperscience accuracy-harnessed) |
| **Corpus strategy** | Brute-force-disclosed (Veryfi: 100s of M; Rossum: 11M; Nanonets: 3M+) | Synthetic-augmented (DocILE 100k synthetic, Nanonets two-stage curriculum) | Customer-corpus-leveraged (Mindee, Klippa — disclosed sizes are smaller, ride on per-customer fine-tunes) |
| **Open-source posture** | Permissive open base (Mindee docTR Apache-2.0, Nanonets-OCR2 on Hugging Face) | Closed model, open benchmark/dataset (Rossum DocILE) | Fully closed (Veryfi, Ocrolus, AWS, Azure proprietary, Google proprietary, Hyperscience proprietary, Klippa proprietary) |
| **Output format philosophy** | Bbox + key-value JSON (AWS, Azure, Klippa, Mindee, Rossum) | LLM-ready structured markdown with semantic tags (Nanonets-OCR2 distinctive) | Schema-driven typed JSON with line-item nesting (Veryfi, Ocrolus) |
| **Latency posture** | <5s per doc (Klippa marketing, Veryfi 3-5s) | Seconds-to-minute, async batching dominant (AWS Textract async, Azure batch, Google batch) | Streaming for long docs (Rossum Aurora 1.5 line-item streaming) |

### 3.3 Where the vendors are *secretly* the same

Pattern observable across the grid:

- **They all rely on open research.** LayoutLM (Microsoft, open), DBNet/CRNN/MASTER (academia, open), Qwen2.5-VL (Alibaba, open), DETR (Meta, open), Gemini (Google, but built on open transformer research). The proprietary value-add is fine-tune weights and corpus, not architecture innovation. There is no vendor in the survey that has a *fundamentally novel architecture* that other vendors couldn't replicate given the same data.
- **They all under-disclose their OCR engine.** Klippa, Veryfi, Ocrolus, AWS, Azure all decline to name their primary OCR engine. The strong inference is that several of them wrap or finetune Tesseract / PaddleOCR / open-source-derived backbones and don't want to commoditize the framing.
- **They all bake confidence-routing as a tier between full-auto and human review.** This is the actual product, not the model — the orchestration of "auto-extract → confidence check → low-confidence route to second model or human" is what customers buy. The model is a commodity; the orchestration is the moat.

### 3.4 Where the vendors are *honestly* different

- **Hyperscience** is the only vendor pitching agentic-orchestration-as-extraction-architecture explicitly. Everyone else has Blocks/Flows in some form but doesn't lead with it.
- **Veryfi** is the only vendor with a published GNN in the production stack. This is unusual — receipts as graphs of spatial regions is a 2020-2022 academic idea (e.g., PICK, ViBERTgrid) that mostly lost to LayoutLM-family transformers, but Veryfi's receipt-only specialization apparently makes it pay.
- **Rossum** is the only vendor that has open-sourced a real benchmark dataset (DocILE) — making it the most independently-evaluatable vendor in the grid, and giving us a published architectural-baseline window into what their proprietary T-LLM is built from.
- **Google** has gone furthest on VLM-only (Gemini-as-extractor). AWS and Azure are the slowest movers — they still anchor on a LayoutLM-style base with VLM/LLM as a downstream chain rather than a replacement.
- **Ocrolus** is the only vendor that has put HiTL at the *center* of the architecture rather than as a fallback. Bank-statement domain economics (lender stakes, regulatory liability) make full-auto unviable, so Ocrolus designs around it instead of pretending to.
- **Nanonets** is the only vendor that publishes its current production model openly on Hugging Face. The competitive bet is that releasing the 3B base costs nothing (it's distilled from a teacher model the customer can't access) while building developer ecosystem and lead-gen for the API.

---

## §4 What This Means For An Apache-2.0-Only Competitor

Five concrete implications for the v0.4 PRD. These are the design choices that follow from §1-§3, ranked by leverage:

### 4.1 Don't try to win on raw extraction accuracy. Win on auditability + zero-cost-per-page + corpus-portability.

Veryfi has hundreds-of-millions of receipts. Rossum has 11M annotated transactional docs. Nanonets has 3M annotated pages from a teacher-distillation pipeline. v0.4 can't out-train any of them on volume, and within the timeline of a v0.4 PRD it shouldn't try. The competitive angles where commercial vendors are structurally weak:

- **Auditability.** Every extraction step open-source means every decision is inspectable. For HIPAA/SOC2/EU AI Act regulated workloads this is a real moat. Veryfi advertises air-gapped infra; v0.4 *is* air-gapped by default.
- **Zero per-page cost.** AWS Textract is ~$1.50/1000 invoice pages (AnalyzeExpense). Azure Document Intelligence prebuilt invoice is ~$10/1000 pages. Google is ~$30/1000 for Custom Extractor. Veryfi is order-$50/1000 for invoice OCR. v0.4 is $0 — and on Vega 8 with the existing voter+verifier stack, throughput is bounded by hardware not API quota.
- **Corpus portability.** Every commercial vendor's corrected-corpus is locked to their UI and tooling. v0.4 can ship a portable corrected-corpus format (line-item JSONL with field-level corrections) that customers own and can take to *any* fine-tunable open base. This is the antidote to pattern 5 (corpus moat) — turn the lock-in into a portable asset.

### 4.2 Adopt the convergent two-stage architecture. Resist the temptation to invent a third.

Every vendor in the survey runs OCR/perception → structured extract. Every one. The v0.3 stack (Tesseract + RapidOCR voter → GLM-OCR Q8 verifier) is already this shape. The right move for v0.4 is to deepen each stage, not invent a new pipeline topology:

- **Stage 1 (OCR/perception):** keep the Tesseract + RapidOCR voter; add a confidence-score-fusion layer that produces a single token-level confidence per word (open research: voter-disagreement entropy as confidence proxy). Add layout extraction (TABLES, KEY_VALUES, LINE_ITEMS) — every commercial vendor exposes this; the open-source path is a pretrained LayoutLMv3 (Microsoft, MIT license — verify) or PaddleOCR's PP-Structure (Apache-2.0).
- **Stage 2 (structured extract):** GLM-OCR Q8 verifier on disagreement is fine for a *verifier* role. For a primary extractor, LayoutLMv3 fine-tuned on a published invoice corpus (DocILE — CC-BY-NC, so commercial-use-restricted; or SROIE; or CORD) gives a known-good baseline. For full VLM extraction, Qwen2.5-VL (Apache-2.0) is the obvious choice since Nanonets has already proven the recipe. Avoid fine-tuning from scratch — start with Nanonets-OCR2-3B as a starting point if its license permits (Hugging Face card lists the model as available; check the actual base license — Qwen2.5-VL is Apache-2.0).

### 4.3 Build the active-learning loop *first*, not last.

Pattern 5 from §1 is the moat that commercial vendors lock in over months of customer use. The open-source way to neutralize it: ship the active-learning UI / correction-capture / fine-tune-on-corrections loop as a v0.4-day-zero feature, not a v0.7 promise. Concretely:

- A correction-capture layer that records every human edit to extracted JSON.
- A LoRA fine-tune script that consumes the correction log and produces a customer-specific adapter for the Stage 2 model. Nanonets has published the recipe; LoRA on Qwen2.5-VL-3B fits in 30 GB RAM with quantization.
- An adapter-merge / -unmerge UI so customers can switch between vanilla and fine-tuned models per document type.

If v0.4 ships this loop, the competitive question shifts from "can you match Veryfi's receipt extraction accuracy" (no) to "can you let me own my correction corpus and run my own fine-tune" (yes, and no commercial vendor allows this).

### 4.4 Take Hyperscience's modularity seriously. Reject Google's monolithic-VLM pitch.

Hyperscience's Hypercell is a worked-out modular architecture: Blocks for ingestion, classification, extraction, validation, business-rules, decisioning, LLM-analysis, wired into customer-defined Flows. This is the right architecture for an open-source competitor because:

- Modular blocks can be Apache-2.0 individually even if combinations are licensed differently — the floor stays open.
- Modular blocks let customers swap in their own domain-specific extractors (e.g., a local-bank-statement block for Ocrolus-class workloads) without forking the whole stack.
- Modular blocks expose *where* HiTL routing happens — making auditability concrete instead of marketing.

By contrast, Google's Gemini-as-extractor monolith collapses all stages into one closed model and is the architecturally hardest path to replicate openly. Don't try.

### 4.5 Ship the published-benchmark posture. Make Rossum's DocILE the v0.4 evaluation harness.

Of the ten vendors, Rossum is the only one that has published an open benchmark + dataset (DocILE, arXiv:2302.05658 — 6.7k annotated + 100k synthetic + 1M unlabeled, 55 annotation classes, Line Item Recognition sub-task, with published RoBERTa / LayoutLMv3 / DETR baselines). v0.4 can:

- Adopt DocILE as the canonical benchmark.
- Publish v0.4 numbers against the same baselines Rossum published.
- Contribute back baselines for v0.4-vs-LayoutLMv3 on DocILE.

This makes v0.4 directly comparable to the only commercial vendor that has dared to be evaluated openly, and creates a research credibility surface (arXiv-citable) that no other open invoice-extraction stack currently has. Note license: DocILE is CC-BY-NC-SA-4.0 — usable for evaluation and research, not commercial training. Use SROIE (research) and CORD (research) for commercial-friendly training corpora.

### 4.6 What v0.4 should NOT do (rejected directions, with reason)

- **Do NOT build a from-scratch VLM.** Nanonets, Hyperscience, Google have spent $M-$10M on this. Apache-2.0 alone doesn't close that gap. Use Qwen2.5-VL (Apache-2.0) or Nanonets-OCR2-3B (Hugging Face) as the base.
- **Do NOT chase 99%+ headline accuracy.** Every commercial vendor folds HiTL into the headline number. v0.4's honest number is the auto-extract accuracy without HiTL; pitch it that way and avoid making claims the architecture can't keep.
- **Do NOT commit to a single model for all document types.** Hyperscience's Block architecture and Ocrolus's router-then-ensemble both prove that domain specialization wins over generic models. v0.4 should ship invoice and receipt as v0.4.0 and add bank-statement, ID, contract as v0.5+, not as v0.4.0.
- **Do NOT build a custom workflow engine.** Hyperscience built Hyperflow as a Netflix Conductor reimplementation because they wanted the same interface across SaaS and on-prem. v0.4 doesn't have that constraint; a thin Python orchestrator (or just a CLI) is enough at this stage. Defer workflow-engine work to v0.6+.
- **Do NOT promise "no training required".** This is Hyperscience ORCA's day-one pitch and Google Gemini's pitch. Commercial vendors with billions of pretraining tokens can credibly make this claim. v0.4 cannot. Position v0.4 as "trainable in an hour on your data" instead.

---

## Methodology Note

This research was conducted via web search of public materials (vendor blogs, engineering posts, Hugging Face model cards, arXiv preprints, conference talks, customer-facing tech docs, public job postings, and industry analyst reviews). No private vendor documents were consulted. Where a vendor's architecture is undisclosed, this report says so explicitly rather than guessing. Citation density per vendor: 2-4 primary sources, 30+ external citations total across §2.

Where the report says "X is implicit" or "the strong inference is" — these are the author's reasoned guesses based on triangulation across multiple sources, not vendor admissions.

This file is the input to a v0.4 PRD; PRD authoring is a separate task and is not done here.

---

## Sources (Consolidated)

### Klippa / Doxis
- [Klippa OCR API documentation](https://www.klippa.com/en/ocr/ocr-api/)
- [Klippa Custom Data Field Extraction](https://www.klippa.com/en/dochorizon/custom-data-field-extraction/)
- [Klippa AI-Powered OCR Software](https://www.klippa.com/en/ocr/)
- [Klippa Invoice OCR Software](https://www.klippa.com/en/ocr/financial-documents/invoices/)
- [Klippa acquired by SER Group / IDP-Software profile](https://idp-software.com/vendors/klippa/)
- [Klippa DocHorizon — Doxis AI.dp](https://www.klippa.com/en/dochorizon/)
- [Klippa Document Classification](https://www.klippa.com/en/dochorizon/document-classification/)

### Rossum
- [Rossum Aurora press release (Feb 2024)](https://rossum.ai/company/newsroom/press-release-rossum-aurora/)
- [Rossum Aurora 1.5 launch (Oct 2024)](https://rossum.ai/company/newsroom/press-release-rossum-unveils-aurora-15-ai-engine-and-copilot-to-accelerate-document-processing-for-global-enterprises-in-276-languages/)
- [Rossum Aurora platform overview](https://rossum.ai/aurora-advanced-ai/)
- [DocILE benchmark (arXiv:2302.05658)](https://arxiv.org/abs/2302.05658)
- [DocILE GitHub](https://github.com/rossumai/docile)
- [DocILE landing](https://docile.rossum.ai/)
- [Deep Analysis — Rossum launches its own LLM](https://www.deep-analysis.net/rossum-launches-its-own-llm/)
- [Rossum publishes world's largest research dataset (PRNewswire)](https://www.prnewswire.com/news-releases/rossum-publishes-worlds-largest-research-dataset-and-benchmark-to-accelerate-scientific-progress-in-intelligent-document-processing-301754863.html)

### Mindee
- [Mindee docTR GitHub](https://github.com/mindee/doctr)
- [Mindee Platform](https://www.mindee.com/)
- [Mindee Invoice OCR API](https://www.mindee.com/product/invoice-ocr-api)
- [Mindee — Top OCR APIs of 2026 blog](https://www.mindee.com/blog/leading-ocr-api-solutions)
- [Mindee docTR overview page](https://www.mindee.com/platform/doctr)
- [docTR docs](https://mindee.github.io/doctr/)

### Nanonets
- [Nanonets-OCR2-3B Hugging Face model card](https://huggingface.co/nanonets/Nanonets-OCR2-3B)
- [Nanonets-OCR2 Research page](https://nanonets.com/research/nanonets-ocr-2/)
- [Nanonets blog — Fine-Tuning VLMs for Data Extraction](https://nanonets.com/blog/fine-tuning-vision-language-models-vlms-for-data-extraction/)
- [Nanonets blog — Bridging Images and Text: Survey of VLMs](https://nanonets.com/blog/bridging-images-and-text-a-survey-of-vlms/)
- [Nanonets-OCR2-1.5B-exp Hugging Face card](https://huggingface.co/nanonets/Nanonets-OCR2-1.5B-exp)

### Hyperscience
- [Hyperscience — ORCA VLM Framework blog](https://www.hyperscience.ai/blog/faster-safer-smarter-document-ai-without-the-training-delay/)
- [Hyperscience — ORCA help page](https://help.hyperscience.ai/v41/docs/orca-optical-reasoning-and-cognition-agent-vlms)
- [Hyperscience — Better together: LLMs + Hyperscience proprietary models](https://www.hyperscience.ai/blog/better-together-llms-hyperscience-proprietary-models/)
- [Hyperscience — Out-of-the-Box to State-of-the-Art VLMs blog](https://www.hyperscience.ai/blog/out-of-the-box-to-state-of-the-art-how-vision-language-models-are-transforming-document-processing/)
- [Hyperscience — Orchestrating Our ML Platform (Hyperflow)](https://www.hyperscience.ai/blog/orchestrating-our-ml-platform/)
- [Hypercell platform overview](https://www.hyperscience.com/platform/)
- [Hyperscience — Long Form Extraction Model](https://www.hyperscience.ai/blog/from-complexity-to-clarity-new-long-form-extraction-model-from-hyperscience-delivers-deep-insights/)
- [Hyperscience — How Hyperscience Automates with Accuracy](https://www.hyperscience.ai/blog/how-hyperscience-automates-with-accuracy/)
- [Hyperscience — Training Data Management docs](https://help.hyperscience.ai/v41/docs/training-data-management)
- [Hyperscience — Vision Language Model QA](https://help.hyperscience.ai/v41/docs/vision-language-model-quality-assurance)

### Ocrolus
- [Ocrolus — How Ocrolus Works](https://docs.ocrolus.com/docs/how-ocrolus-works)
- [Ocrolus — Role of Humans in AI-Driven Document Automation](https://www.ocrolus.com/blog/the-role-humans-with-ai-document-automation/)
- [Ocrolus Platform](https://www.ocrolus.com/platform/)
- [Ocrolus — Document Extraction with Unparalleled Accuracy](https://www.ocrolus.com/product/capture/)
- [Ocrolus — Automated Document Classification for Lenders](https://www.ocrolus.com/blog/automated-document-classification-use-cases-for-lenders/)
- [Ocrolus — Revolutionizing Fraud Detection blog](https://www.ocrolus.com/blog/revolutionizing-fraud-detection-with-automated-document-processing/)

### Veryfi
- [Veryfi — Multimodal Document Data Extraction](https://www.veryfi.com/technology/multimodal-data-extraction-beyond-basic-ocr/)
- [Veryfi — Why Veryfi](https://www.veryfi.com/why-veryfi/)
- [Veryfi — Deep Analysis vendor review](https://www.deep-analysis.net/vendor-vignette-0/veryfi-review/)
- [Veryfi Receipts OCR API](https://www.veryfi.com/receipt-ocr-api/)
- [Veryfi Invoices OCR API](https://www.veryfi.com/invoice-ocr-api/)
- [Veryfi — Multi-Currency Receipt Processing best practices](https://www.veryfi.com/technology/multi-currency-receipt-ocr-best-practices/)
- [Veryfi — Detecting AI-Generated Receipts](https://www.veryfi.com/technology/ai-generated-receipts-detection/)
- [Veryfi — From 6 Months to 6 Days: Build vs Buy Document AI](https://www.veryfi.com/technology/build-vs-buy-document-ai/)

### AWS Textract
- [AWS Blog — Textract Layout feature for generative AI tasks](https://aws.amazon.com/blogs/machine-learning/amazon-textracts-new-layout-feature-introduces-efficiencies-in-general-purpose-and-generative-ai-document-processing-tasks/)
- [AWS Blog — Bring structure to documents with Textract + transformers on SageMaker](https://aws.amazon.com/blogs/machine-learning/bring-structure-to-diverse-documents-with-amazon-textract-and-transformer-based-models-on-amazon-sagemaker/)
- [AWS Docs — Analyzing Invoices and Receipts](https://docs.aws.amazon.com/textract/latest/dg/invoices-receipts.html)
- [AWS Blog — Build a receipt and invoice processing pipeline with Textract](https://aws.amazon.com/blogs/machine-learning/build-a-receipt-and-invoice-processing-pipeline-with-amazon-textract/)
- [AWS Docs — Invoice and Receipt Response Objects](https://docs.aws.amazon.com/textract/latest/dg/expensedocuments.html)
- [AWS Blog — Announcing specialized invoice/receipt support in Textract](https://aws.amazon.com/blogs/machine-learning/announcing-expanded-support-for-extracting-data-from-invoices-and-receipts-using-amazon-textract/)
- [AWS Blog — Intelligent Document Processing with Textract + Bedrock + LangChain](https://aws.amazon.com/blogs/machine-learning/intelligent-document-processing-with-amazon-textract-amazon-bedrock-and-langchain/)

### Azure Document Intelligence
- [Microsoft Research — Document AI / LayoutLM project](https://www.microsoft.com/en-us/research/project/document-ai/)
- [LayoutLM original paper](https://www.microsoft.com/en-us/research/publication/layoutlm-pre-training-of-text-and-layout-for-document-image-understanding/)
- [Azure Custom Neural Document Model docs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/train/custom-neural?view=doc-intel-4.0.0)
- [Azure Invoice prebuilt model docs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/invoice?view=doc-intel-4.0.0)
- [Azure Document Intelligence model overview](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/model-overview?view=doc-intel-4.0.0)
- [Azure Document Intelligence release history](https://docs.azure.cn/en-us/ai-services/document-intelligence/reference/release-history?view=doc-intel-4.0.0)
- [Azure Document Layout Analysis docs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/prebuilt/layout?view=doc-intel-4.0.0)

### Google Document AI
- [Google Cloud — Document AI Custom Extractor with generative AI](https://cloud.google.com/document-ai/docs/ce-with-genai)
- [Google Cloud — Document AI processor list](https://docs.cloud.google.com/document-ai/docs/processors-list)
- [Google Cloud — Document AI release notes](https://docs.cloud.google.com/document-ai/docs/release-notes)
- [Google Cloud — Process documents with Gemini layout parser](https://docs.cloud.google.com/document-ai/docs/layout-parse-chunk)
- [Google Cloud — Document AI Workbench](https://cloud.google.com/document-ai-workbench)
- [Google Cloud — Document AI overview](https://cloud.google.com/document-ai)
- [Google Cloud — Uptrain a pretrained processor](https://docs.cloud.google.com/document-ai/docs/uptrain-pretrained-processor)
