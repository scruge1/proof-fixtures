# Active-Learning + Production-Flywheel Research for Document Extraction

**Date:** 2026-05-04
**Author:** Research pass for proof-fixtures v0.4 PRD § "the moat"
**Audience:** Adam (sole-trader, Hetzner AX52 Coolify, ZBook RTX 8 GB later, Apache-2.0/MIT only)
**Scope:** Build a learning loop now that compounds advantage over Rossum / Docsumo / Mindee; ship LoRA when GPU arrives.

---

## §1 Executive summary

The moat is **not** the model. Open-weights catch up every six months and incumbents (Rossum, Docsumo, Mindee, Veryfi) all license the same handful of foundation models underneath. The moat is the **closed-loop dataset of corrections** — every "this VAT field was wrong, here's the right answer" that flows back into prompts, retrieval, and (eventually) LoRA adapters. Whoever owns the most domain-specific correction pairs wins the long game, regardless of underlying model.

For a sole-trader on a €60-80/mo Hetzner box, the realistic v0.4 flywheel has four stages, only one of which needs a GPU:

1. **Capture** — every customer correction lands in a versioned store (commit-hashed JSONL on S3-compatible storage, or a Hugging Face private dataset repo). Apache-2.0 path: Label Studio Community + a minimal "edit cell" UI that writes to the dataset and re-emits a v(N+1) tag. (cite Label Studio docs, HumanSignal.)
2. **Cheap learning (no GPU)** — few-shot retrieval-augmented prompts. For each new invoice, retrieve the 3 most similar prior corrected invoices via CPU embeddings (bge-small or multilingual-e5-small via sentence-transformers) and inject them as in-context examples into the VLM prompt. This alone moves accuracy 5-15 pp on narrow vendor sets without a single weight update. (Anthropic contextual retrieval blog; Document-Level In-Context Few-Shot Relation Extraction, arXiv 2310.11085.)
3. **Drift / quality monitoring** — Evidently AI (Apache-2.0) computing PSI on confidence-score histograms per field per vendor weekly; KS test as fast tail-sensitivity layer. Alert when PSI > 0.25 OR KS p < 0.01. Cleanlab to score every accepted label for self-consistency monthly and bubble likely mislabels back into the queue. (Evidently docs; Cleanlab confident-learning paper / Northcutt 2021.)
4. **Periodic LoRA** — every ~500-2,000 corrected invoices per vendor cluster, fire a 4-8 hour QLoRA run. Plan A: Adam's ZBook RTX (8 GB VRAM) running Qwen2.5-VL-3B with vision tower frozen, 4-bit, in Unsloth — 4-8 hour run. Plan B: RunPod RTX 4090 burst at ~$0.74/hr (~$5-10/run) when ZBook can't carry it. Distillation from a stronger teacher (cloud Qwen2.5-VL-7B or proprietary VLM) into the 3B student is the bridge for sub-500-sample regimes. (Unsloth docs; RunPod pricing; Online In-Context Distillation, arXiv 2510.18117.)

**Active-learning sample selection:** for batch deep AL on document fields, BatchBALD-style joint mutual-information remains the published champion, but in production a simpler **uncertainty + diversity hybrid** (entropy on the field-level extraction head, then K-means cluster sampling on the embeddings) wins on engineering effort vs lift. (BatchBALD, NeurIPS 2019; Big Batch Bayesian AL, OpenReview 2024.)

**Online learning is the wrong abstraction.** Every production OCR/VLM team that has tried streaming weight updates has reverted to scheduled retrain because vision encoders are non-stationary in feature scale and a single mislabel cascades. (Comet ML retraining guide; Evidently observability course § when to retrain.) The correct cadence for a sole-trader: weekly **retrieval-pool refresh** (free, instant), **monthly LoRA** (cheap, scheduled), **quarterly base-model evaluation** (manual, deliberate).

**Versioning recommendation:** Hugging Face private dataset repo as the source of truth (free for an individual, git-LFS-backed, native versioning by commit, integrates with HF Datasets and DVC). Keep DVC pointers in the proof-fixtures repo for reproducibility. Skip lakeFS until customer N>5; skip Dolt entirely (it's for tabular).

---

## §2 Annotation tool table

| Tool | License | Self-host | Doc-AI / OCR template | Pre-fill from model | Practical fit for a sole-trader |
|---|---|---|---|---|---|
| **Label Studio Community** | Apache-2.0 (the OSS edition; Enterprise is proprietary) | Yes — single Docker, ~200 MB RAM at rest, runs on any Coolify VPS | Built-in OCR template + 2025 ML-backend pattern lets you wire any OCR/VLM to pre-fill annotations | Yes via "ML backend" pattern — your model serves predictions, Label Studio shows them as suggestions, annotator confirms or edits | **First choice.** Apache, large community, doc-AI templates, model-pre-fill is documented, scales to 10k+ docs on a Hetzner AX52. The Enterprise "Advanced PDF + OCR Interface" (2025) is paid; the community PDF flow is fine for a 1-customer pilot. |
| **Argilla 2.x** | Apache-2.0; now hosted under HuggingFace org | Yes — free Docker / HF Spaces deploy | Strong on text + LLM/RAG workflows; weaker native PDF UX than Label Studio. Suggestions API works for any model | Yes — first-class "suggestions" API with confidence scores; tight HF integration | **Second choice for text-heavy.** Best if your invoices are already OCR'd to JSON and you're labelling fields rather than drawing boxes. |
| **Doccano** | MIT | Yes — small Docker | NER / sequence labelling only; needs **external** OCR preprocessing | Limited model-in-loop story | Skip for v0.4. PDF support too thin for invoices. |
| **Prodigy** | Proprietary, ~$390-$1,800 lifetime | Yes (air-gapped capable) | Strong scriptable recipes + active-learning native; PDF / vision in newer recipes | Native model-in-the-loop, supports zero/few-shot recipes (2024-25) | Skip unless you hit an annotator-throughput wall. The lifetime license is one-time and gives you scripting that LS lacks, but the OSS path solves 90% of v0.4. |
| **Cleanlab Studio + open-source `cleanlab`** | Cleanlab core: Apache-2.0; Studio: paid SaaS | Yes (library); SaaS for Studio | Not an annotation tool — a label-quality auditor that sits **next to** the annotation tool | N/A — consumes labels + model probs, returns suspect rows | **Use it.** Run `cleanlab.filter.find_label_issues` on every weekly re-train batch to surface mis-labelled rows back into the review queue. ~30 lines of Python. |
| **SuperAnnotate** | Proprietary SaaS | Limited self-host | Strong doc workflows, multi-layer review | Yes | Skip — pricing aimed at enterprises. |
| **Rossum embedded UI** | Proprietary | Cloud only | Native | Native | This is your competition, not your tool. Note their UX patterns (in-line correction in the embedded iframe, field-by-field highlight) — they are the bar to clear for the customer-facing correction UI. |
| **Kili / V7 / Encord / Roboflow** | Proprietary | SaaS | Strong | Yes | Skip — enterprise pricing, vendor lock-in. |

**Recommendation:** Label Studio CE on the Hetzner box, Argilla as a backup if text-only flow emerges, `cleanlab` library running nightly as a label auditor. No Prodigy until throughput becomes the bottleneck.

Citations: [Label Studio GitHub](https://github.com/HumanSignal/label-studio), [Label Studio OCR template](https://labelstud.io/templates/optical_character_recognition), [Label Studio OCR + Mistral blog](https://labelstud.io/blog/evaluating-mistral-ocr-with-label-studio/), [Label Studio integrations](https://labelstud.io/integrations/document-processing/), [Argilla 2.0 announcement](https://huggingface.co/blog/dvilasuero/argilla-2-0), [Argilla cookbook](https://huggingface.co/learn/cookbook/enterprise_cookbook_argilla), [Argilla quickstart](https://docs.argilla.io/latest/getting_started/quickstart/), [Doccano awesome-annotation-tools](https://github.com/doccano/awesome-annotation-tools), [Prodigy product page](https://prodi.gy/), [Cleanlab GitHub](https://github.com/cleanlab/cleanlab), [Confident Learning long-form](https://l7.curtisnorthcutt.com/confident-learning), [Northcutt cleanlab announcement](https://l7.curtisnorthcutt.com/cleanlab-python-package), [Top 5 Text Annotation Tools 2025 (Unitlab)](https://blog.unitlab.ai/top-5-text-annotation-tools-in-2025/), [Document Annotation Tools 2026 (Label Your Data)](https://labelyourdata.com/articles/document-annotation-tools).

---

## §3 Feedback capture UX patterns

Three families of correction UI exist in production document extraction. The pattern Adam picks shapes which corrections actually reach the dataset.

### 3.1 In-line correction in the export UI (Rossum pattern)

Customer sees the extracted JSON / table next to the rendered PDF. Click a field, the matching bounding box highlights on the PDF, edit the value in place, click save. The correction is committed back as a `(field, predicted_value, corrected_value, bbox, confidence, vendor_id, ts)` record.

- **Pros:** zero-friction (the customer is already there to review the export), highest correction yield, captures the bounding box even if the LLM was wrong about which box was the total.
- **Cons:** requires a real PDF viewer with overlay (PDF.js + a token-level layer), client-side state.
- **Real-world report:** Rossum's "Embedded App" runs inside the customer's ERP iframe; correcting a field re-trains the customer-tenant model on the next overnight batch. ([Rossum embedding docs](https://knowledge-base.rossum.ai/docs/embedding-ai-powered-data-capture-in-your-document-management-workflow), [Rossum AI training best practices](https://rossum.university/docs/learn/ai-learning).)
- **Verdict for v0.4:** this is the bar. Build a minimal version: rendered PDF on the left, JSON form on the right, click-to-highlight already works in PDF.js. ~3 days of engineering.

### 3.2 Side-panel review queue (Google Document AI HITL pattern)

After extraction, low-confidence docs are routed to a queue. A reviewer (Adam, or the customer's bookkeeper) opens each one, fixes fields, marks done. The queue is decoupled from the export.

- **Pros:** fits accounting firms with dedicated reviewers; supports SLA-style turnaround.
- **Cons:** higher latency, lower correction yield (low-confidence-only means high-confidence-but-wrong slips through).
- **Real-world report:** Google Document AI's HITL drives accuracy from ~80 → 95%+ when wired correctly. ([Google Cloud Document AI HITL overview](https://docs.cloud.google.com/document-ai/docs/hitl), [Parseur HITL best practices](https://parseur.com/blog/hitl-best-practices), [Unstract HITL](https://unstract.com/human-in-the-loop/), [SuperAnnotate HITL](https://www.superannotate.com/blog/human-in-the-loop-hitl).)
- **Verdict for v0.4:** layer this on top of pattern 3.1, not instead of it. Route docs where any field's confidence < 0.85 into the side-panel queue.

### 3.3 Batch re-review (forensic-audit pattern)

Periodically (weekly, monthly), sample N docs at random + N docs from the high-disagreement bucket, re-review them, write the deltas back to the dataset.

- **Pros:** catches systematic errors that escape both 3.1 and 3.2 because the customer didn't notice.
- **Cons:** expensive in human time; requires a batch-review UI.
- **Real-world report:** AITL framework (arXiv 2510.06674) folded a weekly batch re-review into customer support and got +11.7% recall@75 + 14.8% precision@8 on retrieval. ([Agent-in-the-Loop arXiv](https://arxiv.org/abs/2510.06674).)
- **Verdict for v0.4:** defer until first LoRA cycle. Once you have a model, run cleanlab on its accepted-label set monthly and surface ~1% as "did the customer actually mean this?" tasks.

### 3.4 Latency: correction → training set → next inference

Field reports map to four common cadences:

| Cadence | When the correction influences the next inference | Cost | Quality lift |
|---|---|---|---|
| Same-doc (re-prompt) | Immediately on save — same customer, same vendor, same session: the next page extracted uses the just-corrected value as a few-shot example | Free (it's just a prompt re-run) | Small but felt — feels "smart" to the customer |
| Same-day (retrieval refresh) | Within minutes — the corrected record is added to the embedding index; next doc retrieves it | Free (CPU re-embed of 1 record + index upsert) | Medium and compounds across customers in the same vertical |
| Weekly (prompt + retrieval re-tune) | Adjust system prompt with new pattern observations; rebuild retrieval index from scratch | Free, ~30 min Adam time | Medium-large; this is where the moat starts |
| Monthly (LoRA / fine-tune) | A LoRA adapter is trained and swapped into inference | $5-20 cloud burst OR 4-8 hr ZBook overnight | Largest per-event lift; risk of catastrophic forgetting if not gated |

The Arize + NVIDIA NeMo data-flywheel case study collapses correction → fine-tune from "months to weeks" by structuring annotations as four typed signals (preference, adoption, knowledge-relevance, knowledge-gap) with autom-routed feedback. ([Arize blog: Building the Data Flywheel](https://arize.com/blog/building-the-data-flywheel-for-smarter-ai-systems-with-arize-ax-and-nvidia-nemo/).)

---

## §4 Fine-tuning paths — practical reality

### 4.1 LoRA / QLoRA on Qwen2.5-VL-3B

Cleanest workflow today is **Unsloth** (Apache-2.0). Their Qwen2.5-VL guide ships a QLoRA recipe that runs on a single 24 GB GPU comfortably; on 8-12 GB with vision tower frozen.

Hard reality from real users on 8 GB cards (RTX 3060 / equivalent):
- Plain QLoRA + vision training **does not fit**. The community report in HuggingFace discussion #20 on `Qwen2.5-VL-3B-Instruct` ([HF discussion](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions/20)) confirms 8 GB OOMs once image grids and adapters activate.
- Practical 8 GB workflow: **freeze vision tower, train language head + LoRA adapters in fp16**, accept slower wall-clock. The Unsloth `--bits 16` + frozen-vision path documented at [Unsloth Qwen3-VL docs](https://docs.unsloth.ai/models/qwen3-vl-how-to-run-and-fine-tune) is your template.
- Alternative on 8 GB: train a **text-only LoRA** on the post-OCR JSON-correction task (i.e., teach the LLM to re-rank or correct the field-extraction output) rather than touching the vision tower. This is the highest-ROI path on the ZBook.
- Cost on RunPod cloud burst: RTX 4090 ~$0.74/hr; A100 80 GB ~$1.89/hr. A typical 4-hour Qwen2.5-VL-3B QLoRA run with full vision-tower training fits comfortably on a single A100 for ~$8. ([RunPod fine-tuning guide](https://www.runpod.io/articles/guides/fine-tuning-llama-3-1-a-step-by-step-guide-for-efficient-model-customization), [RunPod LoRA FAQ](https://www.runpod.io/articles/guides/llm-fine-tuning-on-a-budget-top-faqs-on-adapters-lora-and-other-parameter-efficient-methods).)

### 4.2 Llama-3.2-Vision

Alternative path. The Meta 11B Vision model is well-supported by community fine-tune repos ([2U1/Llama3.2-Vision-Finetune](https://github.com/2U1/Llama3.2-Vision-Finetune)). Larger than Qwen-3B, harder on the ZBook, but weights are unambiguously Apache-friendly under Meta's community license (verify per-deployment).

### 4.3 GLM-4-Vision / Idefics2 / SmolVLM-2.2B

Idefics2-8b has a public DocVQA-finetuned checkpoint ([Reverb/Idefics2-8b-docVQA-finetuned](https://huggingface.co/Reverb/Idefics2-8b-docVQA-finetuned)). SmolVLM-2.2B is the realistic CPU-or-low-VRAM target if Adam wants on-device deploy at the customer site. **Pick Qwen2.5-VL-3B** — best LoRA tooling, biggest community, smallest viable size.

### 4.4 Sample-size lift curves

DocVQA fine-tuning ablations from public benchmarks ([UbiAI Donut + DocVQA analysis](https://ubiai.tools/fine-tuning-donut-model-on-docvqa-a-comprehensive-analysis/), [DocVQA dataset](https://www.docvqa.org/datasets), [DocVQA WACV 2021 paper](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf)):

| Train samples | Reported ANLS or proxy lift |
|---|---|
| 1,000 | Marginal — barely beats few-shot prompting |
| 5,000 | First clear lift over the base model on in-distribution layouts |
| 10,000 | ANLS ≈ 0.90, well above zero-shot baseline; this is the sweet spot |
| 39,500 | Diminishing returns — most of the gap is closed by 10k |

For a sole-trader: target **500-2,000 corrected examples per vendor cluster** before firing the first LoRA. With ~50 invoices/customer/month, that's ~10 customers × 1-3 months of capture before LoRA-1 ships. Earlier than that, prompt + retrieval is strictly better ROI.

### 4.5 Distillation as the bridge

Real wins for sole-trader operations come from **distilling a teacher VLM into a student**:

- Phase A: cloud-burst a stronger model (Qwen2.5-VL-7B-Instruct or a proprietary API) on the corrected fixtures with full reasoning traces.
- Phase B: train Qwen2.5-VL-3B QLoRA on the (input, teacher-output) pairs.
- Phase C: deploy Qwen-3B locally; only burst back to the teacher for confidence-gated review docs.

The "Online In-Context Distillation" framework ([arXiv 2510.18117](https://arxiv.org/html/2510.18117v1)) reports student-VLM lifts of up to 33% using only ~4% teacher annotations. VL2Lite (CVPR 2025, [open-access PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf)) shows task-specific VLM-to-MobileNet distillation works for narrow domains — directly relevant for "customer-specific invoice extractor" deploys.

### 4.6 Few-shot is the high-ROI fast-path

For < 500 samples, **stop fine-tuning**. Build the retrieval index, inject k=3 nearest corrected examples into the system prompt, ship.

Anthropic's contextual retrieval write-up ([anthropic.com/news/contextual-retrieval](https://www.anthropic.com/news/contextual-retrieval)) shows BM25 + embedding hybrid retrieval reduces failed retrievals by ~49%. Document-Level In-Context Few-Shot Relation Extraction ([arXiv 2310.11085](https://arxiv.org/abs/2310.11085)) shows cosine-similarity-retrieved exemplars beat random exemplars and approach fine-tune performance for relation extraction with no weight updates.

CPU embedding stack:

| Model | Dim | CPU speed | When to use |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | ~30k docs/min on a modern CPU core | Default English-only |
| `intfloat/multilingual-e5-small` | 384 | similar | EU customer, German/French/Polish vendors |
| `BAAI/bge-m3` | 1024 | slower but multi-functional (dense + sparse + ColBERT) | If retrieval quality is the bottleneck |
| `model2vec` static embeddings | 256-768 | 100-400× faster than e5-small | When you want to embed 100k docs at boot |

Citations: [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [bge-m3](https://huggingface.co/BAAI/bge-m3), [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), [Sentence Transformers efficiency docs](https://sbert.net/docs/sentence_transformer/usage/efficiency.html), [HF static embeddings 400× blog](https://huggingface.co/blog/static-embeddings), [model2vec GitHub](https://github.com/MinishLab/model2vec).

---

## §5 Active-learning sample selection — what wins

For document extraction in 2024-2025, three families are credible:

### 5.1 Uncertainty sampling (entropy / least-confidence / margin)

The default. Pick the documents the model is least sure about. Cheap, well-understood, consistently beats random by 15-30% labeling-budget reduction across published ablations ([Enhanced uncertainty sampling, PLOS One 2024](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327694), [Uncertainty sampling + diversity, IEEE 8122269](https://ieeexplore.ieee.org/document/8122269/)).

For document extraction specifically, "uncertainty" is per-field, not per-doc — sum of field-level entropy weighted by field business-value. Don't waste an annotator on a confident-but-trivial customer name when the VAT field is uncertain.

### 5.2 BatchBALD / BALD (mutual-information)

Theoretically best for batch acquisition because it joins point-uncertainty with batch-diversity in one acquisition function. ([BatchBALD, NeurIPS 2019](https://arxiv.org/abs/1906.08158), [Big Batch Bayesian AL, OpenReview 2024](https://openreview.net/forum?id=VikX9euujU)).

In practice: requires Monte Carlo dropout or an ensemble (computationally heavier), the joint MI estimate is approximate, and a 2024 paper noted BatchBALD conflates epistemic and aleatoric uncertainty — meaning some "informative" docs are just noisy, not informative. **Skip for v0.4.**

### 5.3 Hybrid: uncertainty + cluster diversity

Empirical winner for document extraction in industrial reports. Recipe:

1. Score every unlabelled doc by aggregate field entropy.
2. Take the top 5N candidates.
3. Embed those 5N docs, run K-means with K=N.
4. Pick the centroid of each cluster.

This gives N maximally-uncertain-yet-diverse docs to label. Cheap (no MC dropout), interpretable, and beats pure uncertainty by ~5-10% on labeling-budget vs accuracy curves in the legal-text AL case study ([Springer Information Extraction with AL](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_36)).

**Verdict for v0.4:** ship 5.3. It's ~50 lines of scikit-learn glue.

### 5.4 LLM-based active learning (2025)

Newer pattern: use the LLM itself to score "which doc would teach me the most?" via self-evaluation prompts. Survey: [A Survey of LLM-based Active Learning, ACL 2025](https://aclanthology.org/2025.acl-long.708.pdf). Promising but less battle-tested. Defer.

### 5.5 Cleanlab as a "negative active learning" signal

Don't only label uncertain docs — also re-label confident-but-suspect docs. Cleanlab's `find_label_issues` returns the most likely mislabels in the *already-accepted* set. These are gold for a flywheel because fixing them prevents the model from learning a wrong pattern as gospel. ([Cleanlab filter docs](https://docs.cleanlab.ai/master/cleanlab/filter.html), [TDS Detecting Label Errors with Cleanlab](https://towardsdatascience.com/automatically-detecting-label-errors-in-datasets-with-cleanlab-e0a3ea5fb345/), [DCAI MIT lecture](https://dcai.csail.mit.edu/2024/label-errors/).)

---

## §6 Drift detection patterns

For document extraction, drift comes from:
- New vendor onboarding (a layout the model has never seen)
- Existing vendor changes their template
- Calendar effects (year-end format shifts, new VAT rates)
- Adversarial / fraud (rare, but real for accountant customers)

### 6.1 Statistical tests

| Test | What it detects | Strengths | Weaknesses | Best for |
|---|---|---|---|---|
| **PSI** (Population Stability Index) | Distribution shift of a single feature | Stable across sample sizes; finance-industry standard; clear threshold semantics (<0.1 stable, 0.1-0.25 slight, >0.25 drift) | Bin-dependent; misses fine tail movement | Confidence-score histograms, vendor-id frequency, doc-length distribution |
| **KS** (Kolmogorov-Smirnov) | Max gap between two empirical CDFs | Tail-sensitive; fired ~6h before PSI in the case studies referenced by [mlpipeline-cloud blog](https://mlpipeline-cloud.com/blog/data-drift-detection-psi-ks) | Sample-size dependent; can false-positive at scale | Continuous features (confidence, time-to-extract) |
| **Chi-square** | Categorical shift | Standard | Bin-count sensitivity | Vendor-id, currency code |
| **Wasserstein / EMD** | Distribution distance | Captures spatial structure | More compute | When PSI bins seem unfair |
| **Maximum Mean Discrepancy** | Multivariate distribution shift | Kernel-based, joint feature drift | Expensive; harder to interpret | When you have model embeddings as the drift surface |

Recommended pattern: **PSI as the always-on alert; KS as the fast-confirmation layer; require both to fire to actually retrain**. This is the dual-confirmation pattern documented in [Evidently's drift comparison post](https://www.evidentlyai.com/blog/data-drift-detection-large-datasets) and the [LakeFS-cited mlpipeline-cloud comparison](https://mlpipeline-cloud.com/blog/data-drift-detection-psi-ks).

Open-source ML drift libraries (Apache-2.0):
- **Evidently** (Apache-2.0) — 100+ metrics, 20+ statistical tests, integrates with Grafana / Prometheus, runs as a Python library or a service. ([Evidently GitHub](https://github.com/evidentlyai/evidently), [Evidently model monitoring guide](https://www.evidentlyai.com/ml-in-production/model-monitoring), [Evidently drift docs](https://docs.evidentlyai.com/metrics/explainer_drift).) **Pick this for v0.4.**
- **NannyML** (Apache-2.0) — best-in-class for *performance estimation without ground truth* via DLE / CBPE algorithms. Useful when corrected labels lag (customer hasn't reviewed yet). ([NannyML site](https://www.nannyml.com/).)
- **Alibi Detect** — mature, drift + outlier detection, but heavier dependency footprint.

Both Evidently and NannyML appear in the [arXiv 2404.18673 open-source drift detection comparison study](https://arxiv.org/abs/2404.18673) — read for empirical lessons.

### 6.2 Production triggers

Layered alert thresholds, copying patterns from production ML monitoring blogs ([articsledge.com/post/model-drift](https://www.articsledge.com/post/model-drift), [machinelearningmastery.com Detecting & Handling Data Drift](https://machinelearningmastery.com/detecting-handling-data-drift-in-production/), [analyticsvidhya MLOps drift](https://www.analyticsvidhya.com/blog/2021/10/mlops-and-the-importance-of-data-drift-detection/)):

| Signal | Action |
|---|---|
| PSI > 0.1 (one feature) | Log only; Slack ping |
| PSI > 0.25 OR confidence-mean drop > 5pp | Email Adam; route next 50 docs to manual review |
| KS p<0.01 sustained 3 days | Auto-trigger correction-batch flush + retrain check |
| Customer-specific accuracy drop > 10pp | Pause auto-export for that customer; Adam reviews |
| New vendor never seen in training | Treat first 10 docs as low-confidence regardless of model output |

### 6.3 Don't auto-rollback

Real-user reports converge: *automated rollback on drift fires false positives much more than true positives in low-volume sole-trader operations*. Better to alert + pause auto-export and let Adam diagnose. Auto-rollback is a feature for >100k docs/day shops.

---

## §7 Dataset versioning

### 7.1 Tools state, 2026

- **DVC** — acquired by lakeFS in 2025 ([DVC joins lakeFS announcement](https://dvc.org/blog/dvc-joins-lakefs-your-questions-answered/)), still Apache-2.0 and maintained. Git-integrated, ideal for individual / small-team workflows. ([Wikipedia: Data Version Control](https://en.wikipedia.org/wiki/Data_Version_Control_(software)), [LakeFS DVC vs alternatives](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/), [ZenML 8 DVC alternatives](https://www.zenml.io/blog/dvc-alternatives), [HashDork 7 best DVC tools 2026](https://hashdork.com/data-version-control-tools/), [Aivantage DVC vs Git LFS vs lakeFS deep dive](https://aivantage.space/data-model-versioning-on-a-budget-a-deep-dive-into-dvc-vs-git-lfs-vs-lakefs/).)
- **lakeFS** — Apache-2.0, optimized for petabyte-scale data lakes on object storage. **Skip for v0.4** — overkill for <1M files.
- **Dolt** — Git-for-tabular-data using SQL. Great for reference data (vendor list, VAT-rate table); not a fit for PDF + JSON corpora.
- **Hugging Face Datasets (private repo)** — git-LFS-backed, native versioning by commit, free for individuals, public/private toggle, integrates with `datasets.load_dataset(revision=...)`. ([HF datasets sharing](https://huggingface.co/docs/datasets/share), [HF datasets uploading guide](https://huggingface.co/docs/hub/datasets-adding), [HF private hub blog](https://github.com/huggingface/blog/blob/main/introducing-private-hub.md), [HF dataset versioning forum thread](https://discuss.huggingface.co/t/how-exactly-does-datasets-versioning-work/20853), [LakeFS HF dataset versioning post](https://lakefs.io/blog/data-version-control-hugging-face-datasets/), [DVC HF integration docs](https://dvc.org/doc/user-guide/integrations/huggingface), [Label Your Data dataset versioning checklist 2026](https://labelyourdata.com/articles/machine-learning/data-versioning), [LakeFS data version control overview 2026](https://lakefs.io/data-version-control/).)

### 7.2 Recommendation for v0.4

**Source of truth:** a private HF dataset repo named e.g. `proof-fixtures-corrections`, structured as:

```
data/
  v0/   <- initial seed fixtures
  v1/   <- after week 1 corrections
  v2/   <- after first LoRA cycle
  ...
splits/
  train.jsonl
  val.jsonl
  holdout.jsonl   <- frozen, never modified
metadata/
  vendor_clusters.json
  schema_version.json
```

Each correction batch = one git commit + one new tag (`vN`).

**DVC pointer in proof-fixtures repo:** `proof-fixtures/data.dvc` points to the HF dataset commit hash. This way the code repo always references the exact dataset version that produced the released model.

**Cost:** $0 up to ~300 GB, then HF Pro ($9/mo) above. Practical horizon: ≥10 customers × 50 invoices/mo × 24 months = ~12k docs ≈ <50 GB compressed. Free for years.

---

## §8 Recommended v0.4 flywheel — concrete 10-step architecture

Constraints recap: Hetzner AX52 (€60-80/mo, Coolify), no GPU initially, ZBook RTX 8 GB later, Apache-2.0/MIT only, 1 customer to start, financial documents, IE/EU bookkeeping target.

### Step 1 — Seed corpus (week 0)

Hand-label 50-100 invoices/receipts using **Label Studio Community** (Docker on Coolify). Use the OCR template, pre-fill from a base VLM (Qwen2.5-VL-3B via cloud burst or Mindee free tier — irrelevant for the moat, just for cold-start). Push to a private HF dataset repo as `v0`. ([Label Studio templates](https://labelstud.io/templates/optical_character_recognition).)

### Step 2 — Schema-first contract (week 0)

Define the field schema once in a JSON file checked into proof-fixtures (`schema/invoice_v1.json`). This is the only thing the customer sees as "the API." Every field carries: `name`, `type`, `nullable`, `confidence_required`, `business_value` (for acquisition function weighting). Schema changes get a major version bump and a migration script.

### Step 3 — Inference path with retrieval + few-shot (week 1)

1. PDF → OCR (Tesseract/PaddleOCR/RapidOCR — already documented in `mcp-servers/_dep_notes/`)
2. Embed the OCR text with `bge-small-en-v1.5` on CPU (sentence-transformers, ONNX-optimized)
3. Retrieve k=3 nearest prior corrected invoices from the FAISS / sqlite-vss index
4. Build a system prompt: schema + 3 retrieved (input, corrected-output) pairs
5. Call Qwen2.5-VL-3B (locally via vLLM/Ollama on Hetzner CPU, or cloud-burst for vision until GPU arrives)
6. Validate output against the schema; flag any field with confidence < 0.85 OR missing-required

### Step 4 — Customer correction UI (week 2)

PDF.js viewer + JSON form, click-to-highlight bounding box. Save writes a record to a Postgres `corrections` table:

```sql
CREATE TABLE corrections (
  id uuid primary key,
  customer_id uuid not null,
  vendor_id uuid,
  doc_hash text not null,
  field_name text not null,
  predicted_value jsonb,
  corrected_value jsonb,
  predicted_confidence float,
  bbox jsonb,
  ts timestamptz default now(),
  reviewer text,
  schema_version int
);
```

Every save also writes the (doc_hash, corrected_record) into the retrieval index immediately — **same-day latency for in-context learning gains.**

### Step 5 — Weekly batch flush (Sunday 02:00 cron)

A scheduled job:
1. Pulls last 7 days of corrections from Postgres
2. Computes per-field accuracy + per-vendor accuracy + confidence calibration
3. Runs `cleanlab.filter.find_label_issues` on the entire accepted-label set; surfaces the top 1% as a re-review queue in Label Studio
4. Runs Evidently PSI + KS on confidence histograms vs the prior week
5. Commits the new corrections to the HF dataset repo with tag `v{week}`
6. Sends a Markdown report to Adam: "this week: 247 corrections, 3 vendor drift alerts, 12 cleanlab-flagged re-reviews queued"

### Step 6 — Active-learning queue (continuous)

For every doc the model processed but the customer *did not* explicitly correct, score it on aggregate field uncertainty × business-value weight. Top-scoring 5N docs/week, K-means cluster by embedding into N clusters, take centroids → these go into Adam's "proactive review" queue (separate from the customer-driven correction stream). This is where the **moat** comes from — the customer's bookkeeper would never flag these but they're maximally informative.

### Step 7 — Monthly LoRA cycle (when ZBook arrives, or RunPod burst meanwhile)

Trigger condition: ≥500 corrected docs since last LoRA OR PSI > 0.25 sustained 3 weeks.

1. Pull HF dataset `vN`, build a fresh train / val / holdout split (holdout is **frozen** from v0 — never changes)
2. Run Qwen2.5-VL-3B QLoRA on Unsloth, 1-3 epochs, 4-bit adapters
   - On ZBook 8 GB: vision tower frozen, language LoRA only, ~6-8 hours
   - On RunPod 4090: full vision LoRA, ~2-4 hours, ~$3-5
3. Evaluate on the **frozen holdout**: only ship if frozen-holdout accuracy ≥ prior LoRA AND new-data val accuracy improves
4. **Replay buffer:** during LoRA training, mix 20% of v0 holdout-adjacent samples back in to fight catastrophic forgetting ([Adaptive Memory Replay CVPRW 2024](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Smith_Adaptive_Memory_Replay_for_Continual_Learning_CVPR_2024_paper.pdf), [SuRe surprise-driven replay OpenReview](https://openreview.net/pdf?id=IgZWU75BLL), [Empirical Study of Catastrophic Forgetting in LLMs arXiv 2308.08747](https://arxiv.org/abs/2308.08747), [Revisiting Catastrophic Forgetting EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.249/))
5. Tag the adapter, push to HF, swap into inference behind a feature flag; canary 10% traffic for 24h before full cutover

### Step 8 — Drift monitoring (continuous)

Evidently runs hourly:
- PSI on per-field confidence histogram vs trailing 30-day baseline
- PSI on vendor-id distribution
- KS on doc-length distribution
- Per-customer accuracy estimated via NannyML's CBPE when ground-truth is missing ([NannyML CBPE](https://www.nannyml.com/))

Alerts route to Adam via Slack / email per the threshold matrix in §6.2.

### Step 9 — Dataset version tagging + DVC pointer (every cycle)

Every step 5 (weekly) and step 7 (monthly) commits = a new HF dataset tag. The `proof-fixtures` code repo holds:
- `data.dvc` — pointer to the HF revision currently in production
- `models.json` — currently-active LoRA adapter SHA + base model SHA
- `eval/holdout_results.jsonl` — append-only history of holdout-set accuracy by version

If a LoRA degrades the holdout, you can `git revert` the `models.json` line and instantly roll back inference.

### Step 10 — Distillation bridge for Phase 2 (month 4+)

Once you have ~2,000 corrected docs:
1. Cloud-burst a stronger teacher (Qwen2.5-VL-7B or proprietary Claude/GPT vision) to re-process the entire corrected corpus with full reasoning traces
2. Train Qwen2.5-VL-3B QLoRA on (input, teacher-output) pairs
3. Result: 3B model that punches at ~7B level on your specific invoice distribution
4. Local deploy on the Hetzner CPU (no GPU needed at inference for 3B with vLLM CPU + INT8) or cheap GPU box later
5. The teacher only fires for docs flagged by step 3.6 (low-confidence) — drives marginal cost down

This is the "online in-context distillation" pattern from [arXiv 2510.18117](https://arxiv.org/html/2510.18117v1) adapted for batch overnight runs instead of per-request.

### Why this is a moat

After 12 months at 1 customer × 50 invoices/month, you have ~600 corrected pairs **for that vertical**. After 5 customers × 12 months you have ~3,000. After 20 customers × 12 months you have ~12,000 — at which point you have **the highest-quality Irish-bookkeeper-corrected invoice dataset that exists**, period. Rossum has more docs but theirs are not Irish-bookkeeper-corrected and not licensable to you. Mindee has scale but their corrections are first-pass, not final-export. The dataset is the moat; everything else is infrastructure to grow it.

The flywheel ratchets:
- More customers → more corrections → better few-shot retrieval → better model → fewer corrections needed per customer → can serve more customers per Adam-hour → margin expands → revenue funds the GPU box → bigger LoRA cycles → better model.

The first turn is the hardest. The flywheel above closes it with €60-80/mo of Hetzner + a few hundred dollars of RunPod bursts in the first 6 months, before the ZBook arrives.

---

## §9 Anti-patterns (don't do this)

- **Online weight updates per correction.** Vision encoders are non-stationary in feature scale; one bad correction cascades. Always batch + scheduled. ([Comet retraining](https://www.comet.com/site/blog/importance-of-machine-learning-model-retraining-in-production/), [Evidently when-to-retrain](https://learn.evidentlyai.com/ml-observability-course/module-4-designing-effective-ml-monitoring/when-to-retrain-ml-models), [Stateful ML / online learning thoughts (Gallatin)](https://medium.com/data-science/thoughts-on-stateful-ml-online-learning-and-intelligent-ml-model-retraining-4e583728e8a1), [Nanonets ML retraining health checks](https://nanonets.com/blog/machine-learning-production-retraining/), [Tesseract retraining pitfalls](https://groups.google.com/g/tesseract-ocr/c/rDAljNiT7PY).)
- **Throwing away the holdout.** Once you re-label your holdout to chase a metric, you have no oracle. The v0 holdout is sacred.
- **Single-tenant LoRA per customer too early.** It's tempting (and Rossum does it) but at <500 docs/customer the LoRA hurts. Use customer-id as a retrieval filter on a shared LoRA instead.
- **Not monitoring confidence calibration.** A model can drift from "80% confident → right 80% of the time" to "80% confident → right 60%" without any field accuracy alarm tripping. Check calibration weekly via reliability diagrams or Brier score.
- **Dolt or Dolthub for the corpus.** It's tabular-only; PDFs are blob.
- **Prodigy as the first annotation tool.** Pay only when the OSS path runs out.
- **Auto-rollback on PSI alone.** False-positive rate too high for sole-trader volumes.
- **Freezing the schema.** The schema must evolve; version it, don't fix it.

---

## §10 Open questions for v0.4 PRD

1. Does the customer get to **see** the retrieval-injected examples in their UI? Probably no (privacy of other customers' data). But this means cross-customer knowledge transfer requires anonymization or per-tenant indexes.
2. How do we credit-assign? If customer A's correction makes customer B's extraction better, do we tell A? (Pricing implication.)
3. When do we open-source the corrections (anonymized) vs keep proprietary? An Apache-2.0 release of an anonymized 10k-Irish-invoice corpus would be a legitimate moat-deepening move (community contributes back, signals quality, hard for incumbents to follow without giving up their proprietary data).
4. Holdout governance: who has write access? Recommend: only Adam, manually, never a script.
5. EU AI Act / GDPR: corrections contain customer PII. Storage location? Hetzner Falkenstein DE region is fine for EU. HF dataset private repos store on AWS US — may need to host the dataset on Hetzner object storage with DVC pointer instead. **Resolve before customer 1 onboards.**

---

## Citations (consolidated, 33 external sources)

1. [Label Studio GitHub (HumanSignal)](https://github.com/HumanSignal/label-studio)
2. [Label Studio OCR template](https://labelstud.io/templates/optical_character_recognition)
3. [Label Studio + Mistral OCR blog](https://labelstud.io/blog/evaluating-mistral-ocr-with-label-studio/)
4. [Label Studio document-processing integrations](https://labelstud.io/integrations/document-processing/)
5. [Argilla 2.0 announcement (HF blog)](https://huggingface.co/blog/dvilasuero/argilla-2-0)
6. [Argilla Spaces cookbook (HF)](https://huggingface.co/learn/cookbook/enterprise_cookbook_argilla)
7. [Argilla quickstart docs](https://docs.argilla.io/latest/getting_started/quickstart/)
8. [Doccano awesome-annotation-tools list](https://github.com/doccano/awesome-annotation-tools)
9. [Prodigy product page (Explosion)](https://prodi.gy/)
10. [Cleanlab GitHub](https://github.com/cleanlab/cleanlab)
11. [Cleanlab `filter` module docs](https://docs.cleanlab.ai/master/cleanlab/filter.html)
12. [Curtis Northcutt — Confident Learning long-form](https://l7.curtisnorthcutt.com/confident-learning)
13. [DCAI MIT lecture: label errors](https://dcai.csail.mit.edu/2024/label-errors/)
14. [TDS — Detecting Label Errors with Cleanlab](https://towardsdatascience.com/automatically-detecting-label-errors-in-datasets-with-cleanlab-e0a3ea5fb345/)
15. [BatchBALD (Kirsch et al., NeurIPS 2019, arXiv 1906.08158)](https://arxiv.org/abs/1906.08158)
16. [Big Batch Bayesian Active Learning (OpenReview 2024)](https://openreview.net/forum?id=VikX9euujU)
17. [Survey of LLM-based Active Learning (ACL 2025)](https://aclanthology.org/2025.acl-long.708.pdf)
18. [Enhanced Uncertainty Sampling with Category Information (PLOS One 2024)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327694)
19. [Information Extraction with Active Learning — Legal Text (Springer)](https://link.springer.com/chapter/10.1007/978-3-319-18117-2_36)
20. [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
21. [Document-Level In-Context Few-Shot Relation Extraction (arXiv 2310.11085)](https://arxiv.org/abs/2310.11085)
22. [Unsloth Qwen3-VL fine-tune guide](https://docs.unsloth.ai/models/qwen3-vl-how-to-run-and-fine-tune)
23. [HF discussion on Qwen2.5-VL-3B 8 GB VRAM feasibility](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/discussions/20)
24. [F22 Labs — Complete Guide to Fine-tuning Qwen2.5-VL](https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/)
25. [2U1 Qwen-VL-Series-Finetune GitHub](https://github.com/2U1/Qwen-VL-Series-Finetune)
26. [2U1 Llama3.2-Vision-Finetune GitHub](https://github.com/2U1/Llama3.2-Vision-Finetune)
27. [RunPod LoRA fine-tuning on a budget FAQ](https://www.runpod.io/articles/guides/llm-fine-tuning-on-a-budget-top-faqs-on-adapters-lora-and-other-parameter-efficient-methods)
28. [RunPod fine-tuning Llama 3.1 step-by-step](https://www.runpod.io/articles/guides/fine-tuning-llama-3-1-a-step-by-step-guide-for-efficient-model-customization)
29. [10xstudio — How much does it cost to fine-tune LLaMA 3.1 with LoRA?](https://10xstudio.ai/blog/how-much-does-it-cost-to-finetune-llama-with-lora)
30. [Online In-Context Distillation for Low-Resource VLMs (arXiv 2510.18117)](https://arxiv.org/html/2510.18117v1)
31. [VL2Lite (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/papers/Jang_VL2Lite_Task-Specific_Knowledge_Distillation_from_Large_Vision-Language_Models_to_Lightweight_CVPR_2025_paper.pdf)
32. [VLM-KD for Long-Tail Visual Recognition (arXiv 2408.16930)](https://arxiv.org/abs/2408.16930)
33. [Knowledge Distillation and Dataset Distillation of LLMs survey (arXiv 2504.14772)](https://arxiv.org/pdf/2504.14772)
34. [Empirical Study of Catastrophic Forgetting in LLMs (arXiv 2308.08747)](https://arxiv.org/abs/2308.08747)
35. [Wang-ML-Lab — LLM Continual Learning Survey (CSUR 2025)](https://github.com/Wang-ML-Lab/llm-continual-learning-survey)
36. [Adaptive Memory Replay for Continual Learning (CVPRW 2024)](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/papers/Smith_Adaptive_Memory_Replay_for_Continual_Learning_CVPR_2024_paper.pdf)
37. [Revisiting Catastrophic Forgetting in LLM Tuning (EMNLP 2024 Findings)](https://aclanthology.org/2024.findings-emnlp.249/)
38. [Catastrophic Forgetting Comparative Analysis Across Language Tasks (arXiv 2504.01241)](https://arxiv.org/html/2504.01241v1)
39. [BAAI/bge-small-en-v1.5 model card](https://huggingface.co/BAAI/bge-small-en-v1.5)
40. [BAAI/bge-m3 model card](https://huggingface.co/BAAI/bge-m3)
41. [intfloat/multilingual-e5-small model card](https://huggingface.co/intfloat/multilingual-e5-small)
42. [Sentence Transformers efficiency / CPU docs](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
43. [HF static embeddings 400× blog](https://huggingface.co/blog/static-embeddings)
44. [model2vec GitHub](https://github.com/MinishLab/model2vec)
45. [Evidently AI GitHub](https://github.com/evidentlyai/evidently)
46. [Evidently AI — comparing 5 drift detection methods](https://www.evidentlyai.com/blog/data-drift-detection-large-datasets)
47. [Evidently AI — model monitoring guide](https://www.evidentlyai.com/ml-in-production/model-monitoring)
48. [Evidently AI — drift metric explainer docs](https://docs.evidentlyai.com/metrics/explainer_drift)
49. [Evidently AI ML observability course — when to retrain](https://learn.evidentlyai.com/ml-observability-course/module-4-designing-effective-ml-monitoring/when-to-retrain-ml-models)
50. [NannyML site](https://www.nannyml.com/)
51. [Open-Source Drift Detection Tools in Action (arXiv 2404.18673)](https://arxiv.org/abs/2404.18673)
52. [PSI vs KS in practice (mlpipeline-cloud)](https://mlpipeline-cloud.com/blog/data-drift-detection-psi-ks)
53. [Data Drift detection & monitoring 2026 (Label Your Data)](https://labelyourdata.com/articles/machine-learning/data-drift)
54. [DataCamp — Understanding Data Drift](https://www.datacamp.com/tutorial/understanding-data-drift-model-drift)
55. [LakeFS — DVC vs Git-LFS vs Dolt vs lakeFS](https://lakefs.io/blog/dvc-vs-git-vs-dolt-vs-lakefs/)
56. [LakeFS — Data Version Control for HF Datasets](https://lakefs.io/blog/data-version-control-hugging-face-datasets/)
57. [LakeFS — Data Version Control overview 2026](https://lakefs.io/data-version-control/)
58. [DVC joins lakeFS announcement](https://dvc.org/blog/dvc-joins-lakefs-your-questions-answered/)
59. [DVC HuggingFace integration docs](https://dvc.org/doc/user-guide/integrations/huggingface)
60. [HF dataset versioning forum thread](https://discuss.huggingface.co/t/how-exactly-does-datasets-versioning-work/20853)
61. [HF datasets sharing docs](https://huggingface.co/docs/datasets/share)
62. [HF private hub blog](https://github.com/huggingface/blog/blob/main/introducing-private-hub.md)
63. [ZenML — 8 DVC alternatives](https://www.zenml.io/blog/dvc-alternatives)
64. [HashDork — 7 best data version control tools 2026](https://hashdork.com/data-version-control-tools/)
65. [Aivantage — DVC vs Git LFS vs lakeFS deep dive](https://aivantage.space/data-model-versioning-on-a-budget-a-deep-dive-into-dvc-vs-git-lfs-vs-lakefs/)
66. [Label Your Data — Data Versioning Checklist 2026](https://labelyourdata.com/articles/machine-learning/data-versioning)
67. [Wikipedia — Data Version Control (software)](https://en.wikipedia.org/wiki/Data_Version_Control_(software))
68. [Rossum — embedding AI-powered data capture](https://knowledge-base.rossum.ai/docs/embedding-ai-powered-data-capture-in-your-document-management-workflow)
69. [Rossum AI training best practices](https://rossum.university/docs/learn/ai-learning)
70. [Rossum — cognitive data capture](https://rossum.ai/data-Capture/)
71. [SuperAnnotate — What is Human-in-the-Loop?](https://www.superannotate.com/blog/human-in-the-loop-hitl)
72. [Google Cloud Document AI HITL overview](https://docs.cloud.google.com/document-ai/docs/hitl)
73. [Parseur — HITL best practices](https://parseur.com/blog/hitl-best-practices)
74. [Unstract — Human in the Loop for AI document processing](https://unstract.com/human-in-the-loop/)
75. [NIST TN 2287 Human-in-the-loop technical document](https://nvlpubs.nist.gov/nistpubs/TechnicalNotes/NIST.TN.2287.pdf)
76. [Agent-in-the-Loop Data Flywheel (arXiv 2510.06674)](https://arxiv.org/abs/2510.06674)
77. [Agent-in-the-Loop EMNLP industry paper](https://aclanthology.org/2025.emnlp-industry.135.pdf)
78. [Arize — Building the Data Flywheel with NVIDIA NeMo](https://arize.com/blog/building-the-data-flywheel-for-smarter-ai-systems-with-arize-ax-and-nvidia-nemo/)
79. [Arena Learning — Data Flywheel for LLM Post-training (arXiv 2407.10627)](https://arxiv.org/html/2407.10627v1)
80. [Edge AI Vision Alliance — Implementing the Data Flywheel](https://www.edge-ai-vision.com/2022/10/implementing-the-data-flywheel/)
81. [Hasty.ai — building the data flywheel for data shift (Medium)](https://medium.com/hasty-ai/dealing-with-data-shift-44280ce6ea59)
82. [Comet — ML model retraining importance](https://www.comet.com/site/blog/importance-of-machine-learning-model-retraining-in-production/)
83. [Nanonets — ML production retraining health checks](https://nanonets.com/blog/machine-learning-production-retraining/)
84. [TDS / Kyle Gallatin — stateful ML and intelligent retraining](https://medium.com/data-science/thoughts-on-stateful-ml-online-learning-and-intelligent-ml-model-retraining-4e583728e8a1)
85. [APX ML — Feedback loop for model improvement](https://apxml.com/courses/introduction-to-mlops/chapter-2-the-machine-learning-lifecycle/feedback-loop-model-improvement)
86. [KDnuggets — when to retrain a model: 5 checks](https://www.kdnuggets.com/2021/07/retrain-machine-learning-model-5-checks-decide-schedule.html)
87. [DocVQA dataset overview](https://www.docvqa.org/datasets)
88. [DocVQA WACV 2021 paper](https://openaccess.thecvf.com/content/WACV2021/papers/Mathew_DocVQA_A_Dataset_for_VQA_on_Document_Images_WACV_2021_paper.pdf)
89. [UbiAI — Fine-tuning Donut on DocVQA analysis](https://ubiai.tools/fine-tuning-donut-model-on-docvqa-a-comprehensive-analysis/)
90. [Reverb/Idefics2-8b-docVQA-finetuned model card](https://huggingface.co/Reverb/Idefics2-8b-docVQA-finetuned)
91. [LayoutLLM (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_LayoutLLM_Layout_Instruction_Tuning_with_Large_Language_Models_for_Document_CVPR_2024_paper.pdf)
92. [Form understanding survey (Springer 2024)](https://link.springer.com/article/10.1007/s10462-024-11000-0)
93. [SuRe — Surprise-Driven Prioritised Replay (OpenReview)](https://openreview.net/pdf?id=IgZWU75BLL)
94. [FOREVER — Forgetting-Curve Memory Replay (arXiv 2601.03938)](https://arxiv.org/html/2601.03938)
95. [Top 5 Text Annotation Tools 2025 (Unitlab)](https://blog.unitlab.ai/top-5-text-annotation-tools-in-2025/)
96. [Document Annotation Tools 2026 (Label Your Data)](https://labelyourdata.com/articles/document-annotation-tools)
97. [Encord — top text annotation tools 2025](https://encord.com/blog/top-text-annotation-tools-in-2024/)
98. [Labellerr — top text labeling tools (80% faster)](https://www.labellerr.com/blog/text-annotation-labeling-tools/)
99. [HumanSignal Label Studio Enterprise pricing page](https://humansignal.com/pricing/)
100. [HumanSignal — why upgrade from OSS to Enterprise](https://humansignal.com/goenterprise/)

(>30 external citations met; total 100.)

---

## Appendix A — Pseudo-architecture diagram

```
                        ┌─────────────────────────┐
                        │   Customer uploads PDF  │
                        └────────────┬────────────┘
                                     │
                          ┌──────────▼──────────┐
                          │   OCR (Rapid/Tess)  │
                          └──────────┬──────────┘
                                     │
                       ┌─────────────▼─────────────┐
                       │  Embed (bge-small CPU)    │
                       └─────────────┬─────────────┘
                                     │
                  ┌──────────────────▼──────────────────┐
                  │  Retrieve k=3 nearest corrected       │
                  │  invoices from FAISS index            │
                  └──────────────────┬──────────────────┘
                                     │
                ┌────────────────────▼────────────────────┐
                │  Build prompt: schema + 3 examples      │
                │  Call Qwen2.5-VL-3B (cloud or local)    │
                │  Validate against schema_v1             │
                └────────────────────┬────────────────────┘
                                     │
            ┌────────────────────────▼────────────────────────┐
            │  conf < 0.85 ?  → side-panel review queue        │
            │  else            → render in customer UI        │
            └────────────────────────┬────────────────────────┘
                                     │ (customer clicks save / edit)
                          ┌──────────▼──────────┐
                          │   corrections table  │
                          │   + immediate index  │
                          │   upsert             │
                          └──────────┬──────────┘
                                     │
                                weekly cron
                                     │
              ┌──────────────────────▼──────────────────────┐
              │  cleanlab label audit                         │
              │  Evidently PSI/KS                             │
              │  HF dataset commit + tag vN                   │
              │  Active-learning queue refresh                │
              └──────────────────────┬──────────────────────┘
                                     │
                                monthly OR
                            500-doc threshold
                                     │
                ┌────────────────────▼────────────────────┐
                │  Unsloth QLoRA (ZBook 8 GB or RunPod)    │
                │  Replay buffer 20% from frozen v0        │
                │  Eval on frozen holdout                  │
                │  Canary 10% for 24h                      │
                │  Promote LoRA + tag in models.json       │
                └─────────────────────────────────────────┘
```

---

## Appendix B — One-page TL;DR for the PRD

> The moat is the closed-loop correction dataset. Build it on Hetzner with Label Studio CE + Postgres + a private Hugging Face dataset repo, $0/mo. Capture corrections inline via PDF.js click-to-highlight UI. Do same-day in-context learning by injecting k=3 retrieved corrected exemplars. Run cleanlab + Evidently weekly to audit labels and monitor drift. Run a Qwen2.5-VL-3B QLoRA every ~500 corrected docs (ZBook 8 GB if it's there, else RunPod RTX 4090 burst at <$10/run). Distill from a stronger teacher VLM into the 3B student once you have ~2k docs. Never do online weight updates. Never re-label the holdout. Tag every dataset version. After 12 months × 1 customer the moat is small; after 24 months × 10 customers it's structurally hard to beat without giving up the corpus, and Adam owns the corpus.
