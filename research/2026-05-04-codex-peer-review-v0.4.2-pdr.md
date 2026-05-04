# Codex peer review — v0.4.2 PDR v0.1 → v0.2

**Date:** 2026-05-04
**Reviewer:** Codex (GPT-5.4) via mcp__codex-peer__ask_codex
**Review target:** `document-ops-portal/CALLMEIE-DOCAI-V0.4.2-CORPUS-BOOTSTRAP-PDR.md` v0.1
**Verdict:** **BLOCK** (5 P0 fixes + §A-G recommendations all applied to v0.2)

## P0 fixes applied to v0.2

| # | Issue | Fix in v0.2 |
|---|---|---|
| 1 | FUNSD listed as TRAIN but eval research says non-commercial / EVAL-only | FUNSD moved to EVAL ONLY in §3 license table; removed from §4 pretraining substrate |
| 2 | Eval N canonicalization (§2 says 160, §8 says 180, §9 says ≥7-of-10 vs ≥4-of-10 acceptance criteria) | Locked: N=180, ≥7-of-10 metrics, no >2pp regression, ≥5pp lift on real/TSTR shard with paired bootstrap significance |
| 3 | "Corpus-as-moat thesis VALIDATED before customers" overclaims | Downgraded to "credible-bench improvement validates synth-pretrain pipeline." Real moat thesis requires customer corrections (PRD §8 Open Q2). |
| 4 | DocILE-derived metrics in public/marketing headline = NC license risk | Locked: DocILE results private-only by default. Public marketing requires legal permission. ShareAlike doesn't infect aggregate factual metrics; NC restricts commercial promotion. |
| 5 | Two-tier deployment claim (GLM-OCR primary + fine-tuned 3B verifier) overstates current extract.py architecture — verifier is text-only Ollama JSON | Reworded: vision-crop verifier requires extract.py refactor (image crop path + model registry + GGUF serving). Captured as D-V0.4.2-18 deferred-or-in-scope decision. |

## §A-G area fixes applied

**§A Corruption ratio**: was 30/50/20 (3-bucket). New: **30 clean / 45 light / 20 heavy / 5 very-heavy** (4-bucket). Very-heavy capped to "orientation/occlusion robustness" labeled separately. >50% occlusion → red-team EVAL only (not training, unless obscured fields marked unavailable). 90/180/270 rotation → separate EXIF test shard.

**§B Build order**: Step 0 INSERTED before existing Step 1 — schema/contracts (JSON schema for output, placeholder names, VAT constants, layout-slot taxonomy, DVC/license manifest). Step 7 public-corpus-fetch can run parallel only AFTER Step 0 conventions exist.

**§C Holdout**: 6-axis disjointness rule extended. Add: (a) "no shared CSS/includes/template partials except VAT constants" between train + eval generators, (b) commit-hash pin both generators, (c) MinHash near-duplicate leakage check (per eval-without-customers.md lines 93-122). Adam's 10 invoices relabeled "personal-real-IE probe" NOT "gold production distribution." CORD eval relabeled "cross-locale benchmark" NOT "distribution-shift proof."

**§D Metrics**: tighter thresholds + significance gate. ANLS 0.93 → **0.90** (eval research §4.1 anchor). KIEval group-F1 0.75 → **0.80** (eval research target). Pass rule = absolute gates per metric + ≥5pp micro-F1 lift on real/TSTR shard with paired bootstrap or McNemar significance + no real-shard regression >2pp. Run baseline first on full N=180 eval with current GLM-OCR Q8 BEFORE training (not 1-sample v0.3).

**§E License posture**: DocILE remains EVAL-ONLY private R&D; never train, never redistribute, never marketing-public. NC commercial-promo restriction explicitly captured.

**§F Missing risks**: 4 added to §11:
1. Synth-pretty-good failure (model good on synth+public, fails real IE customer docs). Mitigation: only real/TSTR shards in headline metrics; synth-holdout diagnostic only.
2. Catastrophic forgetting gate not in metric battery. Mitigation: MMLU-style probe added to §9 (metric #11) + visual review on 20 random preds + calibration ECE.
3. Vision-tower leak risk during LoRA. Mitigation: regex target_modules + runtime assertion `assert all(p.requires_grad is False for p in model.visual.parameters())`.
4. Verifier refactor scope. Current extract.py text-only Ollama JSON. Fine-tuned 3B vision verifier needs image-crop path + model registry naming + GGUF/Ollama serving validation. Documented as D-V0.4.2-18.

**§G Decisions added**: D-V0.4.2-12 through D-V0.4.2-19 (8 new). See PDR §12 v0.2.

## Codex full review text

(Verbatim from `mcp__codex-peer__ask_codex` 2026-05-04T14:23 — reproduced for audit.)

```
**VERDICT: Block**

The corpus-bootstrap direction is technically plausible, but this PDR should not lock as v1.0 until these are fixed:

1. Remove FUNSD from TRAIN. The PDR lists FUNSD as CC-BY-4.0 TRAIN in §3/§2, but the eval research says FUNSD is non-commercial only and eval-only (`eval-without-customers.md` §2.1 lines 59-60, §7.3 lines 261/278).
2. Canonicalize eval N and pass criteria. PDR §2 says 160 docs and ≥4-of-10 metrics; §8 totals 180; §9 says ≥7-of-10 (`PDR` lines 27, 31, 203-221, 252-253).
3. Downgrade the thesis claim. v0.4.2 can validate "synthetic corpus improves this pipeline on a credible bench," not the full "corpus-as-moat" thesis before customers.
4. Keep DocILE out of public/marketing headline metrics unless legal permission is obtained. CC BY-NC-SA permits sharing/adaptation only under NC/SA terms; commercial promotion is the risky part, not just training contamination. Source: Creative Commons CC BY-NC-SA 4.0 terms and DocILE GitHub/benchmark docs.
5. Reword the two-tier deployment claim. `extract.py` has a verifier registry and Qwen fallback, but the actual verifier call is text-only via Ollama, not a vision-crop fine-tuned 3B verifier (`extract.py` lines 1109-1246, 1421-1435).

[full §A-G text continues — see codex_review_final.md in temp]
```

## Cross-refs

- `document-ops-portal/CALLMEIE-DOCAI-V0.4.2-CORPUS-BOOTSTRAP-PDR.md` v0.2 (this review applied)
- `proof-fixtures/research/2026-05-04-{synth-invoice-gen,lora-qwen25vl-3b-8gb,eval-without-customers}.md` (anchors)
- `~/.claude/rules/common/touch-gate.md` (Codex peer review = valid second-thing)
