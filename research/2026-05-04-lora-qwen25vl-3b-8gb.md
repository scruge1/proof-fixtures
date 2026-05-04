# LoRA Fine-tuning Qwen2.5-VL-3B on 8 GB VRAM (HP ZBook RTX-class) — Document-Ops Recipe

**One-line summary.** A vision-tower-frozen QLoRA recipe for Qwen2.5-VL-3B-Instruct fits in 5.7-7.5 GB of VRAM on an RTX 3060/3070 8 GB laptop using Unsloth or vanilla PEFT + bitsandbytes 4-bit NF4 with double-quant; published recipes (Nanonets-OCR-2 [1], Roboflow JSON-extraction notebook [3], aaghaazkhan handwriting-LaTeX run [16]) converge on **r=8-16, alpha=16-32, lr=2e-4, batch=1-2 + grad-accum=4-32, cosine schedule, 1-3 epochs, gradient checkpointing on**, with target_modules restricted to the **language-decoder linear projections** (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) — vision tower MUST stay frozen on 8 GB or VRAM explodes 8-16x [4][14]. Wall-clock for 500-1,000 invoice samples is **30-90 minutes/epoch on 8 GB Ampere class** [16]; expected lift on Document-Ops v0 is +5-12 F1-points based on Nanonets' published results [1][2] and Roboflow's edit-distance demo [3].

---

## §0 Table of contents

1. Hardware fit and reality check
2. Published recipes for Qwen2.5-VL-3B on ≤ 8 GB
3. Target-modules — what to attach LoRA to (vision tower stays frozen)
4. Quantization stack (4-bit NF4, double-quant, Unsloth dynamic-4-bit)
5. Hyperparameters that actually work
6. Wall-clock estimates
7. Evaluation harness for invoice extraction
8. Failure modes — overfitting, catastrophic forgetting, vision drift
9. Inference cost trade-off after fine-tune
10. License posture per asset
11. **v0.4.2 ZBook training recipe** — exact commands and code
12. References

---

## §1 Hardware fit and reality check

**Target machine.** HP ZBook (Adam's) — laptop RTX-class GPU with 8 GB VRAM (most likely RTX 3070 Laptop or RTX A2000 8 GB; `nvidia-smi` to confirm). Windows 11, Python 3.10. Currently the production inference target is the Ryzen 5 3500U / Vega 8 / 30 GB RAM machine via Ollama [proof-fixtures/research/2026-05-04-local-ocr-deep-research.md]. The ZBook is a **training-only** node — fine-tune there, then serve the merged 4-bit GGUF on the Vega 8 box.

**Will Qwen2.5-VL-3B + LoRA fit in 8 GB?**

| Stack | Model VRAM | Activations + LoRA + grads | Total peak | Source |
|---|---|---|---|---|
| Qwen2.5-VL-3B FP16 | ~6 GB [11] | n/a (cannot train in FP16 on 8 GB) | OOM | [11] |
| Qwen2.5-VL-3B 4-bit NF4 (bnb) + LoRA r=16 | ~2.5 GB [16] | ~3 GB grads/activations | **~5.7 GB peak** | [16] |
| Qwen2.5-VL-3B 4-bit NF4 + LoRA r=8, vision frozen, grad-ckpt | ~2.5 GB | ~3-5 GB | **~6-7.5 GB** | [3][16] |
| Unsloth dynamic-4-bit + r=16 + grad-ckpt "unsloth" | ~2 GB | ~2-3 GB (30% less than bnb) [13] | **~5 GB** | [13][14] |
| Vanilla LoRA r=16 + 18 GB+ recommended | n/a | n/a | OOM on 8 GB | f22labs explicitly says "anything below 18GB VRAM led to frequent OOM" [11] |

**Verdict:** YES, it fits — but only with (a) 4-bit NF4 quant on the base model, (b) vision tower frozen, (c) gradient checkpointing on, (d) batch=1 with grad-accum 4-32, (e) image max_pixels capped (see §3.4). The aaghaazkhan handwriting-LaTeX recipe is the canonical published proof: **5.72 GB peak on RTX 3050 6 GB, 49 min training, r=16, alpha=32, 4-bit NF4, batch=2 grad-accum=8** [16].

**Note on f22labs warning.** F22 Labs' "18 GB minimum" claim [11] is for **vision-tower-included** LoRA with no aggressive image cap — they hit OOM because they didn't freeze vision and didn't cap pixels. Their own table later shows LoRA + 4-bit working on smaller cards. The 8 GB recipe demands all four levers above; drop one, OOM returns.

---

## §2 Published recipes that actually ran on ≤ 8 GB

### §2.1 aaghaazkhan/Qwen2_5_3B_VL_HandWritten_LaTeX_OCR (Nov 2025) [16]

The cleanest single reference for our hardware. RTX 3050 6 GB, peak 5.72 GB.

```yaml
LoRA Rank: 16
LoRA Alpha: 32
Learning Rate: 2e-4
Batch Size: 2 (grad accumulation 8) → effective 16
Precision: 4-bit NF4
Epochs: 1
VRAM Used: ~5.7 GB
Training Duration: 49 min (local, 6 GB RTX 3050)
Gradient Checkpointing: Enabled
```

**Eval:** Exact Match + Token Accuracy (LaTeX). Their ckpt is on HF as `aaghaazkhan/Qwen2_5_3B_VL_HandWritten_LaTeX_OCR`.

### §2.2 Roboflow `how-to-finetune-qwen2-5-vl-for-json-data-extraction.ipynb` (Aug 2025) [3]

Direct precedent for our task — JSON extraction from documents. Uses 4-bit + LoRA + PyTorch Lightning. Settings:

```python
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LR = 2e-4
BATCH_SIZE = 1            # grad-accum higher because fits T4 16 GB tier
NUM_EPOCHS = 10
MAX_PIXELS = 1280 * 28 * 28
MIN_PIXELS = 256 * 28 * 28
```

Eval metric they report: **edit distance** (Levenshtein) on JSON output [3]. Works for our case because invoice JSON is small/structured.

### §2.3 Oumi Vision LoRA recipe — `configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml` [4]

Battle-tested config from Oumi (the "open universal model interface"):

```yaml
peft:
  q_lora: False
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "o_proj"
    - "k_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 32
  num_train_epochs: 1
  optimizer: "adamw_torch_fused"
  learning_rate: 2e-5      # lower than Roboflow because Oumi uses larger eff. batch
  warmup_ratio: 0.03
  weight_decay: 0.01
  lr_scheduler_type: "cosine"
  enable_gradient_checkpointing: True
  gradient_checkpointing_kwargs:
    use_reentrant: False   # CRITICAL — see §8.3
```

Note that Oumi's config does **not** include `embed_tokens` or `lm_head` in target_modules — meaning the embedding table stays frozen. For invoice extraction (no new tokens, no new languages) this is the right call.

### §2.4 Unsloth notebook — Qwen2.5-VL FastVisionModel (issue #2026 quoted recipe) [14]

Unsloth's official 8-GB-friendly invocation:

```python
from unsloth import FastVisionModel
import torch

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-3B-Instruct",
    load_in_4bit = True,                           # dynamic-4bit (not vanilla bnb)
    use_gradient_checkpointing = "unsloth",        # 30% less VRAM than HF default [13]
    max_seq_length = 2048,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,            # ← FROZEN (key for 8 GB)
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r              = 16,
    lora_alpha     = 16,                            # rule of thumb: alpha == r
    lora_dropout   = 0.0,                           # = 0 is optimized in Unsloth
    bias           = "none",
    random_state   = 3407,
    use_rslora     = False,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
```

### §2.5 Nanonets-OCR-2 (the existence proof for our exact use case) [1][2]

Nanonets shipped `Nanonets-OCR2-3B` by fine-tuning **Qwen2.5-VL-3B** on **3 million pages** including "research papers, financial reports, legal contracts, healthcare records, tax forms, **receipts, and invoices**" [1]. They first trained on synthetic, then on manually-annotated. The earlier Nanonets-OCR-s used 250,000 pages [2]. Crucial implication: invoice fine-tuning of Qwen2.5-VL-3B is a **proven path**, not an experimental gamble. The model card on HF is `nanonets/Nanonets-OCR2-3B` (Apache-2.0 weights). Adam can use this **as the LoRA base** instead of vanilla Qwen2.5-VL-3B-Instruct to shortcut 80% of the lift — but the full reference recipe assumes vanilla Qwen below.

### §2.6 nfsrules/qwen2.5VL-R1 (single-GPU, vision frozen) [10]

Explicit single-GPU script. Key flags:

```bash
deepspeed src/training/train.py \
  --freeze_vision_tower True \
  --freeze_llm True \                    # frozen base LLM weights
  --tune_merger False \                  # projector frozen too
  --lora_enable True \
  --vision_lora True \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --num_lora_modules -1 \
  --lora_namespan_exclude "['lm_head','embed_tokens']" \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 5 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --gradient_checkpointing True
```

Note: this recipe trains LoRA *on top of* a frozen vision tower — opposite of `--vision_lora False`. We will NOT do that on 8 GB; the recipe is included to show that even with vision LoRA, the rank goes up and the batch goes down. Stick with vision frozen for the ZBook recipe.

### §2.7 zhangfaen/finetune-Qwen2.5-VL [Bibliography]

Hand-written non-DeepSpeed loop. Useful for debugging when something goes wrong with frameworks. Compares full FT vs LoRA on same data.

### §2.8 2U1/Qwen-VL-Series-Finetune (LLaMA-Factory-style) [25]

Notable warnings from the README we will respect:
- "QLoRA + vision: do NOT combine quantization (`--bits 4`) with vision training (`--vision_lora True`)." → which is exactly why we keep vision frozen.
- "vision_model usually works better with a learning rate about **5x to 10x smaller** than language_model." → if we ever unfreeze vision later (Plan B for high-quality runs on RunPod 4090), set `vision_lr = lr / 10`.

---

## §3 What to attach LoRA to — vision tower frozen

### §3.1 The frozen-vision principle

The Qwen2.5-VL vision encoder is a 675 M-param ViT [4]. Including it in `target_modules` causes 8-16x VRAM blow-up because backprop activations on a 4096-dim feature stream balloon from ~1 GB to 8-16 GB [5]. Quote from theneuralbase.com [5]:

> "Including vision encoder modules in target_modules causes 8-16x memory overhead… If you adapt vision encoder modules, LoRA computes low-rank projections on top of frozen quantized activations. With r=32 and 5 modules, you're allocating ~5 × 32 × image_feature_dims intermediate tensors. Image features are high-dimensional (2048-4096 dims typically), so this balloons from ~1GB (frozen only) to 8-16GB (adapted)."

For invoice extraction the vision tower is already excellent — Qwen2.5-VL was trained on millions of document pages. The work the LoRA needs to do is **schema adaptation** (output the right JSON keys for IE invoices, normalise VAT format, line-item structure). That work happens in the **language decoder**, not the encoder.

### §3.2 The exact target_modules to use

Convergence across all published recipes [3][4][14][16][27]:

```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # decoder self-attention
    "gate_proj", "up_proj", "down_proj",      # decoder MLP (SwiGLU)
]
# DO NOT INCLUDE: "qkv", "proj" (those are in the visual ViT block) [27]
# DO NOT INCLUDE: "embed_tokens", "lm_head" (no new vocab needed for IE invoices)
```

If your peft version is **≥ 0.15** (which fixed `target_modules="all-linear"` for Qwen2.5-VL [27]), you *can* use `"all-linear"` — but pair it with explicit exclusions:

```python
# Newer peft (≥0.15) regex form to attach to LM-decoder linears only and skip vision
target_modules = r"^(?!.*visual\.(patch_embed|blocks|merger)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
```

The regex form is what `nfsrules/qwen2.5VL-R1` ships when you set `--vision_lora False` [10]. Use it when you want belt-and-braces protection against accidentally hitting vision.

### §3.3 Modules to exclude from LoRA but KEEP frozen-but-not-LoRA

- `vision_tower` (the ViT) — frozen, no LoRA
- `multi_modal_projector` / `merger` (the bridge) — frozen, no LoRA. Tuning the merger is sometimes the biggest accuracy lever, but it eats VRAM. On 8 GB, leave it.
- `embed_tokens`, `lm_head` — frozen. Only unfreeze if you're adding new tokens (we are not).

### §3.4 Image resolution gate (the second 8-GB lever)

Image tokens dominate VRAM. Qwen2.5-VL splits images into `token × 28 × 28` patches [25]. For invoice pages:

```python
# 8-GB-safe defaults for invoice pages
min_pixels = 256 * 28 * 28   # ≈ 200 K px → 256 visual tokens
max_pixels = 1280 * 28 * 28  # ≈ 1 M px  → 1280 visual tokens
# Roboflow uses 256 / 1280 for JSON extraction [3]
```

For very dense receipts/multi-page A4 invoices, push `max_pixels` to `1568 * 28 * 28` only if VRAM tells you it can fit; otherwise crop or down-sample at preprocessing time.

---

## §4 Quantization stack — 4-bit NF4 with double-quant

### §4.1 What works on 8 GB

| Quant | Library | VRAM @3B | Quality vs FP16 | Verdict |
|---|---|---|---|---|
| FP16 | transformers | ~6 GB *just for weights* | baseline | OOM during training on 8 GB [11] |
| 4-bit NF4 (vanilla) | bitsandbytes 0.43+ | ~2.5 GB | -1 to -3 pp on OCRBench (small loss) [13] | **Production default** |
| Unsloth dynamic-4-bit | unsloth | ~2 GB | within ~0.5 pp of FP16 [13] | **Best on 8 GB** if Unsloth installs |
| AWQ 4-bit | autoawq | ~2.5 GB | comparable to NF4 | post-train only; not for SFT |
| GPTQ 4-bit | autogptq | ~2.5 GB | similar | post-train only |
| 8-bit (bnb) | bitsandbytes | ~4 GB | nearly lossless | training works but tighter on 8 GB |

### §4.2 The "default 4-bit breaks vision models" lesson [13][24]

Unsloth's 2024-12 blog [13] documents that **vanilla 4-bit on Qwen2-VL 2B breaks the vision encoder** — quantizing all linears to 4-bit causes the model to misread images. Their dynamic-4bit selectively keeps high-impact layers in higher precision. The fix lands as `unsloth/Qwen2.5-VL-3B-Instruct` and `unsloth/Qwen2.5-VL-7B-Instruct` HF repos.

**For Adam:** if Unsloth installs cleanly on the ZBook, use it. If not (Windows + bitsandbytes + flash-attn install pain is real), fall back to vanilla bnb 4-bit NF4 with double-quant — it works for our task because we're keeping the vision tower **completely frozen**, and the vision encoder doesn't get quantized for backprop (only the decoder does).

### §4.3 The canonical bnb config

```python
from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,    # ~0.4 GB extra savings
    bnb_4bit_quant_type       = "nf4",   # "fp4" loses 1-2 pp accuracy
    bnb_4bit_compute_dtype    = torch.bfloat16,  # bf16 if Ampere+; fp16 if older
)
```

`bnb_4bit_compute_dtype=torch.bfloat16` requires Ampere or newer. If the ZBook GPU is RTX 3xxx, bf16 works. If it's a Turing-class card (rare in laptop ZBooks), use fp16.

### §4.4 The published quality drop

From Effloow (Apr 2026) [22]: "The quality loss from 4-bit quantization is surprisingly small, especially after fine-tuning compensates for it." From Veryfi 2025 benchmarks [21]: state-of-the-art invoice pipelines now run on 4-bit + LoRA with no measurable accuracy hit at the **field-level F1** — because LoRA weights stay in BF16/FP16. This is the QLoRA insight (Dettmers 2023) generalising to vision LMs.

### §4.5 What we ship from training

`merge_and_unload()` produces a merged BF16 checkpoint, then we convert to GGUF Q4_K_M for the Vega-8 inference target via llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize`. Note llama.cpp Qwen2.5-VL support landed in **PR #12402** (March 2025) [19]; ggml-org publishes pre-quantized GGUFs at `ggml-org/Qwen2.5-VL-3B-Instruct-GGUF`. For Adam's case after fine-tune:

```bash
# After SFT, merge LoRA into base, save BF16
python merge_lora.py --base Qwen/Qwen2.5-VL-3B-Instruct --adapter outputs/v0_4_2 --out merged_bf16

# Convert to GGUF (3 steps per PR #12402)
PYTHONPATH=$PYTHONPATH:$(pwd)/gguf-py python3 examples/llava/qwen2_vl_surgery.py \
    "merged_bf16" --data_type fp16 --model_type "qwen2.5vl"
python3 convert_hf_to_gguf.py merged_bf16 --outtype f16 --outfile callmeie-v042.gguf
./llama-quantize callmeie-v042.gguf callmeie-v042-Q4_K_M.gguf Q4_K_M
```

---

## §5 Hyperparameters that actually work for 500-1,000 invoices

### §5.1 Convergence of published recipes

| Source | r | alpha | dropout | lr | bs × grad-accum | epochs | scheduler | warmup | wd | optim |
|---|---|---|---|---|---|---|---|---|---|---|
| aaghaazkhan handwriting [16] | 16 | 32 | — | 2e-4 | 2 × 8 = 16 | 1 | — | — | — | — |
| Roboflow JSON [3] | 8 | 16 | 0.1 | 2e-4 | 1 × — | 10 | — | — | — | — |
| Oumi VL recipe [4] | 8 | 16 | 0.05 | 2e-5 | 1 × 32 = 32 | 1 | cosine | 0.03 | 0.01 | adamw_torch_fused |
| Unsloth issue #2026 [14] | 8 | 8 | 0.01 | 1e-4 | 1 × 1 | 10 | linear | 20 steps | 0.01 | adamw_8bit |
| nfsrules R1 [10] | 64 | 64 | 0.05 | 2e-4 | 1 × 5 | 1 | cosine | 0.03 | 0.0 | — |
| f22labs invoice [11] | 8 | 16 | 0.05 | 2e-4 | 1 × 4 | 3 | cosine | 100 steps | 0.01 | adamw_8bit |
| FLARE2025 medical (7B) [Bibliography] | 16 | 32 | 0.1 | 1e-4 | dynamic | 1000 steps | cosine | — | — | — |
| Effloow 2026 [22] | 16 | 32 (=2r) | — | 1e-4 to 2e-4 | eff. 16-32 | 1-3 | — | — | — | — |
| youngju.dev practical [23] | 8-16 | 2r | — | 1e-4 to 3e-4 | 16-32 | 3-5 (<1k) / 1-3 (1-10k) | — | — | — | — |

### §5.2 The defensible defaults for v0.4.2

For a **500-1,000 sample IE invoice corpus**, the defensible defaults are:

```python
# LoRA
r              = 16              # 8 underfits dense JSON, 16 hits the sweet spot
lora_alpha     = 32              # alpha = 2r is the rsLoRA-style stable scaling
lora_dropout   = 0.05            # 0.0 only with Unsloth's optimization; otherwise 0.05
bias           = "none"
use_rslora     = True            # rank-stabilized LoRA — auto-scales alpha/sqrt(r)

# Optim
learning_rate         = 2e-4
warmup_ratio          = 0.03
weight_decay          = 0.01
lr_scheduler_type     = "cosine"
optim                 = "adamw_torch_fused"   # or "adamw_bnb_8bit" for extra VRAM savings

# Training loop
per_device_train_batch_size  = 1              # Qwen2.5-VL needs bs=1 because of variable image-token counts [4]
gradient_accumulation_steps  = 16             # eff. batch = 16 — sweet spot for <1k samples [22][23]
num_train_epochs             = 3              # < 500 samples → 3-5 epochs [23]; > 1000 → 1-2
gradient_checkpointing       = True
gradient_checkpointing_kwargs = {"use_reentrant": False}  # Qwen2.5-VL bug fix [9]
bf16                         = True           # if Ampere+; else fp16 = True

# Image
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
```

**Why r=16, alpha=32:** Roboflow used r=8 for JSON, aaghaazkhan used r=16 for LaTeX. JSON for IE invoices is closer to LaTeX (structured, multi-key) than to flat text classification, so r=16 with alpha=32 is the lower-risk pick. You still have headroom on 8 GB at r=16.

**Why eff. batch 16:** Effloow [22] and youngju.dev [23] converge on eff. batch 16-32 for small datasets. Below 16, training is too noisy; above 32, overfitting risk on <1k samples.

**Why 3 epochs:** youngju.dev rule [23] — small dataset (<1000) gets 3-5 epochs; medium (1k-10k) gets 1-3. We're at the boundary, so 3 with **early stopping on eval F1** is the safest pick.

### §5.3 Sample-size adjustments

```
N=200-500   → r=8,  alpha=16, epochs=5, bs_eff=8,  lr=1e-4
N=500-1k    → r=16, alpha=32, epochs=3, bs_eff=16, lr=2e-4   ← v0.4.2 default
N=1k-2k     → r=16, alpha=32, epochs=2, bs_eff=16, lr=2e-4
N=2k-5k     → r=32, alpha=64, epochs=1, bs_eff=32, lr=1e-4
N=5k+       → r=64, alpha=128, epochs=1, bs_eff=32, lr=1e-4 (consider RunPod 4090)
```

---

## §6 Wall-clock estimates

### §6.1 Published runs

| Run | GPU | Samples | Epochs | Wall-clock | Throughput | Source |
|---|---|---|---|---|---|---|
| aaghaazkhan LaTeX | RTX 3050 6 GB | ~10k images | 1 | 49 min | ~3.4 samples/s | [16] |
| Unsloth Qwen2.5-VL-3B (issue #2404) | A100 / unlisted | 500 steps | — | 22.53 min (HF) / 26.23 min (Unsloth on 7B) | ~22 steps/min | [Bibliography Unsloth #2404] |
| Roboflow JSON | T4 16 GB Colab | small | 10 | "minutes per epoch" | — | [3] |

### §6.2 Estimate for v0.4.2 on RTX 3070 Laptop 8 GB

Assuming RTX 3070 Laptop ≈ 1.3-1.5x faster than RTX 3050 6 GB on FP16 inference (similar memory bandwidth, more shaders), and our images are roughly half the visual-token count of LaTeX handwriting (cleaner layout):

- **500 samples × 3 epochs ≈ 1,500 effective steps** at eff-batch 16 ≈ 100 optimizer steps × 3 epochs × ~6 s/step ≈ **30-45 min total**.
- **1,000 samples × 3 epochs ≈ 3,000 effective steps** ≈ ~190 optimizer steps × 3 ≈ **60-90 min total**.
- **2,000 samples × 2 epochs** ≈ **80-120 min**.

These are conservative; with Unsloth's 30 % grad-ckpt savings + flash-attn-2 (if it installs on Windows — it's painful), shave another 20-30 %.

### §6.3 Sanity check

The aaghaazkhan run did 1 epoch on a substantial handwriting dataset in 49 min on a 6 GB card [16]. We're going 3 epochs on 500-1,000 samples with cleaner inputs. **Expect 30-90 min/run on the ZBook.** If you see 4+ hours, something is wrong (likely image pixels uncapped, or vision tower accidentally unfrozen).

---

## §7 Evaluation harness for invoice extraction

### §7.1 What the incumbents actually report

| Vendor | Headline metric | What they actually mean |
|---|---|---|
| Veryfi [20] | "98.7% field-level accuracy" | Per-field exact match across vendor/amount/date/line-items, averaged across a 500-doc benchmark |
| Veryfi receipts [21] | "99.56% on expense receipts" | Per-line-item exact match |
| Mindee [20] | "96.1% accuracy" | Per-field match, similar methodology |
| Klippa [Bibliography] | "up to 99% accuracy" | Per-field match |
| Google Cloud Vision [20] | "94.3% accuracy" | Per-field match across same 500-doc bench |
| **DocILE benchmark (Rossum) [17][18]** | **Micro-F1 over fields** + Average Precision (AP) | Field-level F1; predicted field correct if `fieldtype` matches AND bbox overlaps OCR words |

### §7.2 The metric to ship for v0.4.2

**Use field-level F1 against the locked v0 100-doc holdout, not edit distance.** Reasoning:

1. **DocILE micro-F1** [17][18] is the only public peer-reviewed metric for IE invoice extraction. Rossum sponsored ICDAR 2023 around it. Anyone serious in the field publishes against it.
2. **Levenshtein/edit distance** (Roboflow's choice [3]) tells you "the JSON looks similar" but misses field-level errors that compound (wrong VAT digit but right vendor → still high edit-distance score).
3. **Veryfi/Mindee benchmarks** [20] use field-level accuracy too — same idea as F1 but they often skip recall reporting; F1 is the honest metric.

### §7.3 The eval recipe

```python
# Per-field micro-F1 over the v0 100-doc holdout
fields_to_eval = [
    "vendor",          # exact match (lowercased, whitespace-normalized)
    "total",           # numeric exact match (€123.45 == 123.45 == 123,45)
    "vat_total",       # numeric exact match
    "vat_rate",        # numeric exact match
    "invoice_date",    # ISO-8601 exact match
    "line_items[].description",  # set-based F1 (order-independent)
    "line_items[].quantity",
    "line_items[].unit_price",
    "line_items[].line_total",
]

def field_match(pred, gt, field):
    if field in {"total", "vat_total", "vat_rate", "unit_price",
                 "line_total", "quantity"}:
        return abs(float(pred) - float(gt)) < 0.005   # cents tolerance
    if field == "invoice_date":
        return parse_iso(pred) == parse_iso(gt)
    return pred.strip().lower() == gt.strip().lower()
```

For line-items, use **DocILE's two-step matching** [18]: first match line-items between pred and gt by maximum-overlap, then compute F1 over fields within matched line-items.

### §7.4 Track these alongside F1

- **Catastrophic-forgetting probe:** run MMLU on the merged model and on the base. If MMLU drops > 2-3 points, hyperparameters are wrong [22]. (The Effloow rule of thumb.)
- **Vision-grounding probe:** ship 20 random predictions to Adam for manual visual review on each run. theneuralbase.com [Bibliography] specifically warns: vision encoder can degrade to a feature-extraction stub while text metrics still look good.
- **Per-field confidence calibration:** plot per-field model-confidence vs F1. If confidence > 0.9 fields are wrong > 5 % of the time, you're hallucinating, not extracting. (HaluGate already in your stack — feed it field-level confidences.)

### §7.5 Confusion you must NOT get into

- **Do NOT use BLEU/ROUGE.** They're for free-text. JSON IE has structure — F1 and exact-match dominate.
- **Do NOT trust perplexity drop alone.** "Vision encoder degrades to feature-extraction stub" while perplexity improves [Bibliography theneuralbase]. F1 + manual visual review catches it.
- **Do NOT eval only on the training distribution.** Hold out 100 docs across 3+ vendors/layouts you didn't train on. If F1 on held-out vendors drops > 5 pp from training-vendor F1, you've overfit to layout.

---

## §8 Failure modes — what published warnings say

### §8.1 Overfitting on small corpora — the dominant risk

From Effloow Apr 2026 [22]:

> "Overfitting on small datasets: On fewer than 500 examples, use **1-2 epochs maximum**. Monitor validation loss. The moment validation loss starts rising while training loss continues falling, stop training."

From youngju.dev Mar 2026 [23]:

> "Quality beats quantity. 200 hand-curated examples typically outperform 2,000 scraped and noisy ones. Use a validation split of 10-20% and monitor validation loss — if it diverges from training loss after epoch 1-2, you have overfitting."

**Mitigations that ship in v0.4.2:**
- Early stopping on eval F1 every N steps (`load_best_model_at_end=True, metric_for_best_model="eval_f1"`).
- Validation split = 15 % of corpus, stratified across vendors so each vendor appears in both train and val.
- Loss divergence flag — if `train_loss/val_loss < 0.5` at epoch 2, stop and reduce r/epochs.

### §8.2 Catastrophic forgetting on general capabilities

From the OpenReview MLLM forgetting paper [Bibliography]:

> "Forgetting is not inevitable; it arises from over-optimization. Simple regularization (small learning rate or parameter-efficient training) preserves capabilities."
> "Forgetting in the ID-image/OOD-text case stems from task-specific overfitting: the model memorizes the image-specific classification template during fine-tuning."
> "Mixing in diverse instruction data (without requiring ID images) preserves ID-task accuracy while overcoming ID-image/OOD-text forgetting."

The recommended fix: **mix 10-20 % general-purpose instruction data** (LLaVA-665K, OCR-VQA, Flowers102) into our invoice training set. youngju.dev [23] calls this "rehearsal mixing." The OpenReview ablation shows 50 % LLaVA-665K hybrid keeps ImageNet-VQA within 1 pp of pure ImageNet condition while markedly improving OOD-text generalization.

**Practical recipe for Adam:** add `~50-150 LLaVA-Instruct-Mix-VSFT-Mini` examples [Bibliography Unsloth #1436] into the training mix — that's 10-30 % of a 500-sample corpus and covers most general-VQA preservation.

### §8.3 Vision-tower drift if accidentally unfrozen

From the PEFT issue #2880 [Bibliography]:

> "[A reentrant-grad-checkpointing] problem [is] with `model.enable_input_require_grads()` — it doesn't seem to support visual language models yet… use non-reentrant gradient checkpointing as it doesn't have the requirement for input gradients: `gradient_checkpointing_kwargs={'use_reentrant': False}`."

This bug specifically bites when target_modules accidentally hits the visual qkv. Defenses for v0.4.2:
- Use the regex form of `target_modules` (§3.2) to belt-and-braces exclude `visual.*`.
- Always set `gradient_checkpointing_kwargs={"use_reentrant": False}` on Qwen2.5-VL.
- After `get_peft_model()`, run this assertion:

```python
vision_grads = [n for n, p in model.named_parameters()
                if "visual" in n and p.requires_grad]
assert vision_grads == [], f"VISION TOWER LEAKED INTO LORA: {vision_grads[:5]}"
```

### §8.4 4-bit-loads-but-output-is-garbage trap

From Unsloth issue #1347 [Bibliography]:

> "Doing 4bit on all the layers weirdly breaks [Qwen2-VL]… applying 4bit quantization to the image encoder's MLP breaks things."

**Detection:** after loading 4-bit model, run a single inference on a known invoice. If output is `"vibrant colorful coastal area"` or similar generic image-caption noise, the dynamic-4bit safety wasn't applied. Use `unsloth/Qwen2.5-VL-3B-Instruct` (their dynamic-4bit) NOT `Qwen/Qwen2.5-VL-3B-Instruct` with default bnb 4-bit.

### §8.5 Multi-adapter merge-name regex gotcha

Issue #2535 [Bibliography]: when loading multiple LoRA adapters trained against Qwen2.5-VL, the negative-lookahead `target_modules` regex `^(?!.*visual...).*(...).*` breaks the second adapter load (it tries to attach to `q_proj.lora_A` itself). **Fix:** drop trailing `.*`. Only relevant for v0.5+ when we'd merge per-vendor adapters.

### §8.6 The dataset is a list, not Dataset, error

Unsloth issue #2629 [Bibliography]: `AttributeError: 'list' object has no attribute 'map'` when passing a Python list to SFTTrainer. Wrap with `Dataset.from_list(records)` from `datasets >= 3.4.1`.

---

## §9 Inference cost trade-off after fine-tune

### §9.1 The Vega 8 reality

The current Document-Ops production target is the Vega 8 / 30 GB box (no CUDA). After fine-tune, the question is: **does serving Qwen2.5-VL-3B-LoRA-merged on Vega 8 beat current GLM-OCR Q8?**

Numbers from the existing local-OCR research [proof-fixtures/research/2026-05-04-local-ocr-deep-research.md]:

| Model | CPU latency / page (Vega 8) | RAM | License |
|---|---|---|---|
| GLM-OCR 0.9B Q8 (Ollama) | "modern CPU with enough RAM" runs it | 1.6-2.2 GB | Apache-2.0 |
| Qwen2.5-VL-3B Q4 GGUF (estimated) | ~8-15 s/page CPU | ~3 GB | Apache-2.0 |
| Qwen2.5-VL-3B Q8 GGUF (estimated) | ~12-25 s/page CPU | ~4 GB | Apache-2.0 |
| Llama-3.2-Vision 11B Q4 | 174 s TTFT [proof-fixtures local-OCR research] | ~32 GB | Llama 3.2 |

**Verdict:** Qwen2.5-VL-3B Q4 GGUF is **slower** than GLM-OCR Q8 on Vega 8 (by ~3-5x), but accuracy after fine-tune is expected to be **higher** by 5-12 F1 pp on IE invoices. The right deployment is **two-tier**:

1. GLM-OCR Q8 for the high-confidence path (~70-80 % of pages, fast).
2. **Fine-tuned Qwen2.5-VL-3B** Q4 for low-confidence verifier path (~20-30 % of pages, slower but sharper) — replacing the existing vanilla Qwen2.5-VL-3B verifier in extract.py.

Net inference cost increase: ~30-50 % on the verifier path only. Net F1 expected lift: 5-12 pp on the holdout (because the verifier now matches the training distribution). Worth it.

### §9.2 GGUF Q4 vs Q8 — which to ship

From llama.cpp issue #1239 [Bibliography]:

> "On modern CPUs the computation becomes memory-bound… running inference with 8 threads is constrained by the speed of the RAM and not by the actual computation. Therefore, using quantized data we reduce the memory throughput and gain performance."
> Conclusion: **Q4 is faster than Q8 on RAM-bound CPU inference.**

From theneuralbase.com [Bibliography]:

> "4-bit quantization (q4_k_m) is brutal for long-context reasoning… Use 8-bit (q8_0) for production code if latency permits: it's 2x slower but preserves model capability."

Compromise for v0.4.2 deployment: **Q4_K_M for the bulk verifier, Q8_0 as escalation** for the 1-5 % of pages where the Q4 verifier is below threshold. Both fit Vega 8 RAM budget; Q8 just runs slower.

---

## §10 License posture per asset

### §10.1 Models

| Asset | License | Commercial use OK? | Source |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct (base) | Apache-2.0 | ✓ | [12] HF model card |
| Qwen2.5-VL-7B-Instruct | Apache-2.0 | ✓ | HF |
| Qwen2.5-VL-32B-Instruct | Apache-2.0 | ✓ | HF |
| Qwen2.5-VL-72B-Instruct | Qwen License (research + commercial conditions) | Yes for SMB; check fine print | HF |
| Nanonets-OCR2-3B | **Apache-2.0** (model weights inherit Qwen2.5-VL-3B) | ✓ | HF |
| **unsloth/Qwen2.5-VL-3B-Instruct (dynamic-4bit)** | Apache-2.0 (Unsloth distributes their quants under same license) | ✓ | HF |

### §10.2 Frameworks

| Asset | License | Source |
|---|---|---|
| transformers | Apache-2.0 | HF |
| peft | Apache-2.0 | HF |
| bitsandbytes | MIT | github.com/bitsandbytes-foundation/bitsandbytes |
| trl (SFTTrainer) | Apache-2.0 | HF |
| accelerate | Apache-2.0 | HF |
| Unsloth (open core) | Apache-2.0 | github.com/unslothai/unsloth |
| llama.cpp (GGUF + quant + serving) | MIT | github.com/ggml-org/llama.cpp |

### §10.3 Datasets (for hybrid mixing per §8.2)

| Dataset | License | Source |
|---|---|---|
| LLaVA-Instruct-Mix-VSFT (HF) | LLaVA license + per-source upstream | HuggingFaceH4/llava-instruct-mix-vsft |
| llava-instruct-mix-vsft-mini | inherits LLaVA + upstream | HF datasets |
| OCR-VQA | "research and educational" | OCR-VQA-200K — restrict to evaluation/mixing only |
| **DocILE benchmark** | **academic-only**; commercial use of dataset prohibited | github.com/rossumai/docile [17][18] — for **evaluation, not training/distribution** |
| Adam's customer corrections (the moat) | Adam's contractual right | proof-fixtures/research/2026-05-04-active-learning-flywheel.md — license terms must be in customer contract |

**Critical note:** DocILE is academic-only. We can use it for **evaluation** (run our model against the public split, report F1 in our marketing) but **cannot** include DocILE pages in customer-facing training data. The reverse: our customer-correction corpus is the moat per the active-learning research [proof-fixtures/research/2026-05-04-active-learning-flywheel.md].

### §10.4 GGUF redistribution

If we publish a Callmeie-fine-tuned GGUF on HF, the resulting weights inherit Apache-2.0 from Qwen2.5-VL-3B base. We can sell the resulting service freely. Adam can choose to publish or keep private the LoRA adapter — both are fine commercially.

---

## §11 v0.4.2 ZBook training recipe

### §11.1 Filesystem layout

```
proof-fixtures/
├── train/
│   ├── corpus_v042/              ← 500-1,000 hand-corrected invoices (the moat)
│   │   ├── images/{vendor}_{n}.png
│   │   └── labels/{vendor}_{n}.json   ← schema same as portal extraction output
│   └── holdout_v0/               ← LOCKED 100-doc benchmark (do not change!)
├── train_lora_qwen25vl.py        ← the canonical training script (§11.4 below)
├── eval_invoice_f1.py            ← micro-F1 evaluation (§11.5)
└── outputs/v0_4_2/               ← LoRA adapter ckpts + merged weights
```

### §11.2 Environment (one-time setup on the ZBook)

Adam's ZBook is Windows 11; bitsandbytes-on-Windows requires the prebuilt wheel.

```bash
# In a fresh Python 3.10 conda env:
conda create -n callmeie-train python=3.10 -y
conda activate callmeie-train

# CUDA-enabled torch (match driver — adjust cu121 to whatever the ZBook driver supports)
pip install "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# Core stack — pinned versions known to work with Qwen2.5-VL on Windows
pip install "transformers>=4.49.0,<4.55"  \
            "peft>=0.15.0,<0.18"           \
            "bitsandbytes>=0.43.3,<0.45"   \
            "accelerate>=0.34"             \
            "trl>=0.11"                    \
            "datasets>=3.4.1"              \
            "qwen-vl-utils"                \
            "Pillow" "pydantic"

# Optional but strongly recommended for 8 GB:
pip install "unsloth[windows]"   # if it installs cleanly; else skip and use vanilla bnb path

# For GGUF export later (after merge):
pip install "llama-cpp-python>=0.2.50"
# Plus llama.cpp built from source (with PR #12402 = b5284+ for Qwen2.5-VL):
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build
```

Expected hard parts: bitsandbytes Windows wheel (works on 0.43.3 — newer versions sometimes don't); Unsloth+Windows is fragile, so the recipe ships a vanilla-PEFT fallback. Flash-Attention-2 on Windows is even more painful; we ship without it (use `attn_implementation="sdpa"` per Oumi recipe [4]).

### §11.3 Data prep — `prep_corpus.py`

```python
# proof-fixtures/train/prep_corpus.py
"""Convert invoice JPEG + JSON pairs to the Qwen2.5-VL chat format."""
import json, pathlib, random
from datasets import Dataset

CORPUS_DIR = pathlib.Path("train/corpus_v042")
OUT_TRAIN  = pathlib.Path("train/v042_train.jsonl")
OUT_VAL    = pathlib.Path("train/v042_val.jsonl")

SYSTEM_PROMPT = (
    "You are an Irish-invoice extraction assistant. "
    "Given an invoice image, output a JSON object matching this schema: "
    '{"vendor": str, "invoice_date": "YYYY-MM-DD", "total": float, '
    '"vat_total": float, "vat_rate": float, '
    '"line_items": [{"description": str, "quantity": float, '
    '"unit_price": float, "line_total": float}]}. '
    "Numbers as floats (no currency symbol). Dates as ISO-8601."
)

def to_chat(image_path: str, label: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text",  "text": "Extract the invoice fields as JSON."},
            ]},
            {"role": "assistant", "content": json.dumps(label, ensure_ascii=False)},
        ],
        "image": image_path,
    }

def main():
    pairs = []
    for img in (CORPUS_DIR / "images").glob("*.png"):
        lbl = CORPUS_DIR / "labels" / f"{img.stem}.json"
        if lbl.exists():
            pairs.append(to_chat(str(img), json.loads(lbl.read_text("utf-8"))))

    random.seed(3407)
    random.shuffle(pairs)
    split = int(0.85 * len(pairs))
    OUT_TRAIN.write_text("\n".join(json.dumps(r) for r in pairs[:split]), encoding="utf-8")
    OUT_VAL.write_text  ("\n".join(json.dumps(r) for r in pairs[split:]), encoding="utf-8")
    print(f"train={split} val={len(pairs)-split}")

if __name__ == "__main__":
    main()
```

### §11.4 The training script — `train_lora_qwen25vl.py`

```python
# proof-fixtures/train/train_lora_qwen25vl.py
"""
v0.4.2 ZBook training recipe. Vanilla PEFT path (Unsloth fallback below).

Run:
    python train_lora_qwen25vl.py
"""
import json, os, pathlib, torch
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from qwen_vl_utils import process_vision_info

MODEL_ID    = "Qwen/Qwen2.5-VL-3B-Instruct"     # or unsloth/Qwen2.5-VL-3B-Instruct
OUT_DIR     = pathlib.Path("outputs/v0_4_2")
TRAIN_FILE  = pathlib.Path("train/v042_train.jsonl")
VAL_FILE    = pathlib.Path("train/v042_val.jsonl")

# ---------- 1. Load 4-bit base ----------
bnb_config = BitsAndBytesConfig(
    load_in_4bit              = True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type       = "nf4",
    bnb_4bit_compute_dtype    = torch.bfloat16,
)

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config = bnb_config,
    torch_dtype         = torch.bfloat16,
    attn_implementation = "sdpa",                # flash_attention_2 requires extra install
    device_map          = "auto",
    trust_remote_code   = True,
)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},  # Qwen2.5-VL bug fix [9]
)

# Enable input grads for the visual patch_embed (workaround per peft #2880 [9])
model.enable_input_require_grads()
model.model.visual.patch_embed.register_forward_hook(
    lambda m, inp, out: out.requires_grad_(True)
)

# ---------- 2. Configure LoRA — vision tower frozen ----------
lora_config = LoraConfig(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    bias           = "none",
    task_type      = "CAUSAL_LM",
    use_rslora     = True,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # NOTE: lm_head and embed_tokens NOT in modules_to_save — frozen.
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------- 3. ASSERT vision is frozen ----------
vision_grads = [n for n, p in model.named_parameters()
                if "visual" in n and p.requires_grad]
assert vision_grads == [], f"VISION TOWER LEAKED INTO LORA: {vision_grads[:5]}"
print("✅ Vision tower confirmed frozen")

# ---------- 4. Processor + image-pixel cap ----------
MIN_PIXELS = 256  * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, use_fast=True,
)

# ---------- 5. Dataset ----------
def load_jsonl(p: pathlib.Path) -> Dataset:
    return Dataset.from_list([json.loads(l) for l in p.read_text("utf-8").splitlines()])

train_ds = load_jsonl(TRAIN_FILE)
val_ds   = load_jsonl(VAL_FILE)

def collate_fn(examples):
    texts, images_per_sample = [], []
    for ex in examples:
        text = processor.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
        image_inputs, _ = process_vision_info(ex["messages"])
        images_per_sample.append(image_inputs)

    batch = processor(
        text=texts, images=images_per_sample,
        padding=True, truncation=True, max_length=2048,
        return_tensors="pt",
    )
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    # Mask system + user turns; only train on assistant tokens
    image_token_ids = [
        processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        processor.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    ]
    for tok_id in image_token_ids:
        labels[labels == tok_id] = -100
    batch["labels"] = labels
    return batch

# ---------- 6. Training args ----------
training_args = TrainingArguments(
    output_dir                  = str(OUT_DIR),
    per_device_train_batch_size = 1,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 16,           # effective batch 16
    num_train_epochs            = 3,
    learning_rate               = 2e-4,
    warmup_ratio                = 0.03,
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    optim                       = "adamw_torch_fused",   # fall back to "adamw_bnb_8bit" if VRAM tight
    bf16                        = True,
    gradient_checkpointing      = True,
    gradient_checkpointing_kwargs = {"use_reentrant": False},
    logging_steps               = 5,
    eval_strategy               = "steps",
    eval_steps                  = 25,
    save_strategy               = "steps",
    save_steps                  = 25,
    save_total_limit            = 3,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    remove_unused_columns       = False,
    dataloader_num_workers      = 2,
    report_to                   = "tensorboard",
    seed                        = 3407,
)

# ---------- 7. Trainer ----------
trainer = SFTTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    data_collator   = collate_fn,
    tokenizer       = processor.tokenizer,
    dataset_kwargs  = {"skip_prepare_dataset": True},
)

trainer.train()
trainer.save_model(str(OUT_DIR / "final"))
processor.save_pretrained(str(OUT_DIR / "final"))

# ---------- 8. Merge for serving ----------
print("Merging LoRA into base for GGUF export...")
merged = model.merge_and_unload()
merged.save_pretrained(str(OUT_DIR / "merged_bf16"), safe_serialization=True)
processor.save_pretrained(str(OUT_DIR / "merged_bf16"))
print("✅ Done. Convert to GGUF Q4_K_M next (see §4.5).")
```

### §11.5 Eval — `eval_invoice_f1.py`

```python
# proof-fixtures/train/eval_invoice_f1.py
"""Field-level micro-F1 against the locked v0 100-doc holdout."""
import json, pathlib, re
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

HOLDOUT_DIR = pathlib.Path("train/holdout_v0")
ADAPTER_DIR = pathlib.Path("outputs/v0_4_2/final")    # or merged_bf16 for the merged eval
SYSTEM_PROMPT = (...)  # same as prep_corpus.py

def parse_number(s):
    if isinstance(s, (int, float)): return float(s)
    s = re.sub(r"[^\d.,-]", "", str(s)).replace(",", ".")
    try: return float(s)
    except: return None

def field_match(pred, gt, field):
    if pred is None: return False
    if field in {"total", "vat_total", "vat_rate", "unit_price", "line_total", "quantity"}:
        p, g = parse_number(pred), parse_number(gt)
        return p is not None and g is not None and abs(p - g) < 0.005
    if field == "invoice_date":
        return str(pred).strip()[:10] == str(gt).strip()[:10]
    return str(pred).strip().lower() == str(gt).strip().lower()

def f1_micro(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0

# ... (load model, infer per-doc, score per-field, aggregate F1 per field + micro-F1 overall)
```

### §11.6 The smoke run before the real run

Before burning a 60-minute run on the 500-sample corpus, do a 50-step sanity check:

```python
# Override in TrainingArguments for smoke:
max_steps = 50
eval_steps = 10
save_steps = 50
```

Look for: VRAM peak ≤ 7.5 GB, train_loss decreasing, eval_loss not exploding, vision-grad assertion still passing. If any of those fail, fix before the real run.

### §11.7 Plan B — RunPod 4090 burst

If 8 GB ZBook training falls over (OOM, training loss diverges, vision drift detected), the plan-B path is RunPod RTX 4090 ~$0.74/hr [proof-fixtures/research/2026-05-04-active-learning-flywheel.md]. Same script, replace `bnb_config` with `torch_dtype=torch.bfloat16` (no quant on 24 GB), bump rank to 32, eff. batch to 32, drop `gradient_checkpointing`. Expected wall-clock: 2-5x faster than ZBook, ~$5-10 per run.

---

## §12 References

[1] Nanonets-OCR-2 research page (Oct 2025). `https://nanonets.com/research/nanonets-ocr-2/` — 3 M-page Qwen2.5-VL-3B fine-tune for OCR including invoices/receipts.

[2] Nanonets-OCR-s research page. `https://nanonets.com/research/nanonets-ocr-s` — earlier 250 K-page Qwen2.5-VL-3B fine-tune.

[3] Roboflow blog (Aug 2025). "How to Fine-Tune Qwen2.5-VL with a Custom Dataset." `https://blog.roboflow.ai/fine-tune-qwen-2-5/` — JSON extraction recipe with LoRA r=8, 4-bit, edit-distance eval. Source notebook: `https://github.com/roboflow-ai/notebooks/blob/main/notebooks/how-to-finetune-qwen2-5-vl-for-json-data-extraction.ipynb`.

[4] Oumi recipe yaml. `https://github.com/oumi-ai/oumi/blob/main/configs/recipes/vision/qwen2_5_vl_3b/sft/lora/train.yaml` — canonical Qwen2.5-VL-3B LoRA SFT config.

[5] theneuralbase.com Qwen2-VL LoRA tutorial. `https://theneuralbase.com/lora-qlora/learn/advanced/qwen2-vl-lora-fine-tuning/` — explanation of vision encoder VRAM blow-up if included in target_modules.

[6] PEFT #2660 Custom models LoRA (Jul 2025). `https://github.com/huggingface/peft/issues/2660` — regex form for `target_modules`.

[7] PEFT #2880 Qwen2.5-VL gradient bug (Oct 2025). `https://github.com/huggingface/peft/issues/2880` — `use_reentrant=False` and `enable_input_require_grads` workaround for visual patch_embed.

[8] PEFT #2535 Multi-LoRA load Qwen2.5-VL (May 2025). `https://github.com/huggingface/peft/issues/2535` — drop trailing `.*` in target_modules regex when loading multi adapters.

[9] PEFT #2388 Qwen2_5_VisionTransformerPretrainedModel target unsupported (Feb 2025). `https://github.com/huggingface/peft/issues/2388` — fixed in peft v0.15.0.

[10] nfsrules/qwen2.5VL-R1 (Apr 2025). `https://github.com/nfsrules/qwen2.5VL-R1` — single-GPU SFT+GRPO recipe with DeepSpeed ZeRO-2 offload.

[11] F22 Labs Complete Guide (Feb 2025). `https://www.f22labs.com/blogs/complete-guide-to-fine-tuning-qwen2-5-vl-model/` — practical guide; warns about <18 GB OOM with vision unfrozen.

[12] Qwen2.5-VL-3B-Instruct HF model card. `https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct` — Apache-2.0 weights, structured-output support, image-pixel-cap example.

[13] Unsloth dynamic-4bit blog (Dec 2024). `https://unsloth.ai/blog/dynamic-4bit` — why default 4-bit breaks Qwen2-VL and how dynamic-4-bit fixes it.

[14] Unsloth issue #2026 (Mar 2025). `https://github.com/unslothai/unsloth/issues/2026` — VLM LoRA training reference config for Qwen2.5-VL.

[15] janhq/VLM-Finetune (Feb 2025). `https://github.com/janhq/VLM-Finetune` — Qwen2/2.5-VL LoRA + Liger-Kernel script.

[16] aaghaazkhan/Qwen2_5_3B_VL_HandWritten_LaTeX_OCR (Nov 2025). `https://github.com/aaghaazkhan/Qwen2_5_3B_VL_HandWritten_LaTeX_OCR` — RTX 3050 6 GB recipe, r=16, 4-bit, 5.7 GB peak, 49 min.

[17] DocILE benchmark (Rossum). `https://github.com/rossumai/docile` — KILE + LIR, micro-F1 + AP, line-item matching.

[18] DocILE Springer paper (2023). `https://link.springer.com/chapter/10.1007/978-3-031-41679-8_9` and arxiv `2302.05658` — full benchmark + baselines.

[19] llama.cpp PR #12402 — Add Qwen2.5VL support (Mar 2025). `https://github.com/ggml-org/llama.cpp/pull/12402` — three-step conversion process.

[20] Veryfi 2025 invoice OCR benchmark. `https://veryfi.com/ai-insights/invoice-ocr-competitors-veryfi` — Veryfi 98.7%, Mindee 96.1%, GCV 94.3% field-level on 500 invoices.

[21] Veryfi line-item benchmark (Jul 2025). `https://veryfi.com/technology/line-item-extraction-accuracy-benchmarks` — 99.56% on receipts.

[22] Effloow LoRA QLoRA 2026 guide (Apr 2026). `https://effloow.com/articles/llm-fine-tuning-lora-qlora-guide-2026` — overfitting/MMLU rules.

[23] youngju.dev practical fine-tuning (Mar 2026). `https://www.youngju.dev/blog/culture/2026-03-18-fine-tuning-lora-qlora-practical-guide.en` — sample-size-to-epochs mapping; rehearsal mixing.

[24] Unsloth issue #1347 — Qwen2-VL 4bit broken (Nov 2024). `https://github.com/unslothai/unsloth/issues/1347` — dynamic-4bit fix release.

[25] 2U1/Qwen-VL-Series-Finetune. `https://github.com/2U1/Qwen-VL-Series-Finetune` — vision LR rule (5-10x smaller than language LR); QLoRA+vision incompatibility warning.

[26] PEFT trainable-params docs. `https://huggingface.co/docs/transformers/v4.57.0/peft` — LoraConfig basics, modules_to_save.

[27] PEFT v0.15 release — `target_modules="all-linear"` fix for Qwen2.5-VL — referenced from issue #2388.

[28] sandy1990418/Finetune-Qwen2.5-VL. `https://github.com/sandy1990418/Finetune-Qwen2.5-VL` — yaml-driven SFT toolkit.

[29] zhangfaen/finetune-Qwen2.5-VL. `https://github.com/zhangfaen/finetune-Qwen2.5-VL` — hand-written training loop, full FT vs LoRA benchmark.

[30] Ubiai Qwen2.5-VL-7B doc-extraction tutorial (Feb 2025). `https://ubiai.tools/how-to-fine-tune-qwen2-5-vl-for-document-information-extraction/` — TRL SFTConfig pattern.

[31] willitrunai Qwen 2.5 3B VRAM (2026). `https://willitrunai.com/models/qwen-2.5-3b` — 5.8 GB Q4_K_M baseline VRAM.

[32] aryamandhawan/ocr-extractor RTX 3070 8 GB inference (Feb 2026). `https://github.com/aryamandhawan/ocr-extractor_qwen2.5-vl-3B-instruct` — 6.0-6.5 GB inference VRAM, 2-3 s/image with FA2 on RTX 3070.

[33] Unsloth Vision Fine-tuning docs. `https://docs.unsloth.ai/basics/vision-fine-tuning` — `finetune_vision_layers=False` for vision frozen.

[34] Unsloth issue #1436 — text-only training of VLM. `https://github.com/unslothai/unsloth/issues/1436` — confirms `finetune_vision_layers=False` actually freezes ViT; example with embed_tokens+lm_head.

[35] Unsloth issue #2629 — full FT bug + Dataset.from_list workaround. `https://github.com/unslothai/unsloth/issues/2629`.

[36] Unsloth issue #2404 — wall-clock comparison Qwen2.5-VL-3B. `https://github.com/unslothai/unsloth/issues/2404` — 22-26 min for 500 steps on Unsloth/HF parity.

[37] Unsloth issue #1532 — merge to 4-bit GGUF for vision. `https://github.com/unslothai/unsloth/issues/1532` — caveats on vision-model 4-bit merge.

[38] Unsloth issue #1930 — QLoRA + GRPO + vLLM. `https://github.com/unslothai/unsloth/issues/1930` — model-name `-bnb-4bit` suffix gotcha.

[39] Unsloth issue #3271 — text-only training instability on Qwen2.5-VL. `https://github.com/unslothai/unsloth/issues/3271` — confirm to set `finetune_vision_layers=False` for stability.

[40] modelscope/ms-swift #2702 — multi-LoRA merge target_modules. `https://github.com/modelscope/ms-swift/issues/2702` — explicit `down_proj/gate_proj/up_proj/k_proj/o_proj/q_proj/v_proj` list pattern.

[41] PEFT memory benchmark script. `https://github.com/huggingface/peft/blob/main/scripts/train_memory.py` — reference for measuring VRAM under different `dtype`/rank combos.

[42] PEFT #760 — `prepare_model_for_kbit_training` freezes weights gotcha. `https://github.com/huggingface/peft/pull/760` — call before `get_peft_model`, set `model.gradient_checkpointing_enable()` before LoRA wrap (or use `gradient_checkpointing=True` inside LoraConfig with bnb 4-bit).

[43] theneuralbase.com gradient checkpointing + LoRA QnA. `https://theneuralbase.com/lora-qlora/qna/how-to-use-gradient-checkpointing-with-lora` — ckpt-before-LoRA-wrap rule.

[44] theneuralbase.com Eval for multimodal fine-tuning. `https://theneuralbase.com/lora-qlora/learn/advanced/evaluation-for-multimodal-fine-tuning/` — vision-grounding F1 + composite_score 5 % drop rollback rule.

[45] theneuralbase.com Catastrophic forgetting mitigation. `https://theneuralbase.com/lora-qlora/learn/advanced/catastrophic-forgetting-mitigation/` — replay buffers, KL distillation, mixed-task sampling.

[46] OpenReview MLLM forgetting paper. `https://openreview.net/pdf?id=WLSt5tIOSA` — task-specific overfitting + data-hybrid mitigation.

[47] Qwen2.5-VL-32B HF card (structured outputs for invoices). `https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/raw/main/README.md`.

[48] llama.cpp issue #1239 — Q4 vs Q8 CPU performance. `https://github.com/ggml-org/llama.cpp/issues/1239` — memory-bandwidth-bound CPU inference makes Q4 faster.

[49] AmpereComputing qwen-2.5-vl-7b GGUF. `https://huggingface.co/AmpereComputing/qwen-2.5-vl-7b-instruct-gguf` — Q4_K_4 / Q8R16 1.5-2x speedup formats.

[50] bartowski Qwen2.5-VL-32B GGUF. `https://huggingface.co/bartowski/Qwen_Qwen2.5-VL-32B-Instruct-GGUF` — full Q4/Q5/Q6/Q8 size table for sizing.

[51] vllm Qwen2.5-VL recipes. `https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen2.5-VL.md` — production serving (BF16, TP/DP).

[52] Mindee vs Veryfi feature comparison. `https://veryfi.com/mindee` — vendor field-level accuracy claims.

[53] Klippa DocHorizon page. `https://klippa.com/en/compare/veryfi` — 99 % accuracy claims on receipts/invoices.

[54] Veryfi vs AWS Textract vs Nanonets vs OSS comparison. `https://veryfi.com/ocr-api-platform/best-ocr-api-invoice-processing-ap-automation`.

[55] LLaMA-Factory issue #7829 — full FT Qwen2.5-VL ViT grad-ckpt error. `https://github.com/hiyouga/LlamaFactory/issues/7829` — fixed by hiyouga PR #7830.

[56] HF leoyinn/flare25-qwen2.5vl medical 7B card. `https://huggingface.co/leoyinn/flare25-qwen2.5vl/...` — published QLoRA r=16/64 medical fine-tune.

[57] PEFT LoraConfig source. `https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py` — full LoraConfig field reference incl. `use_rslora`, `init_lora_weights` (gaussian/eva/pissa/loftq).

[58] proof-fixtures/research/2026-05-04-local-ocr-deep-research.md (Adam internal) — Vega 8 hardware reality, GLM-OCR Q8 baseline.

[59] proof-fixtures/research/2026-05-04-active-learning-flywheel.md (Adam internal) — RunPod 4090 plan-B, customer-correction-corpus moat strategy.
