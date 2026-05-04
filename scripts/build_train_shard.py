#!/usr/bin/env python3
"""v0.4.1 Stage 9 — build training shard from corrections + extractions.

Skeleton: dedupes corrections by (extraction_id, field_id, latest submitted_at);
joins with the original extraction payload to produce one training pair per
corrected field; emits metrics.json with corpus stats.

Real LoRA / distillation prompt-formatting is v0.5 work (per PRD §5).
This stage exists so that DVC can track the curation step's inputs/outputs and
the metrics pipeline can chart corpus growth over time.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("build_train_shard")


def _read_jsonl(path: Path):
    if not path.exists():
        log.warning("%s missing — treating as empty", path)
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                log.warning("Skipping malformed line: %s", e)
    return out


def _dedupe_corrections(corrections: list[dict]) -> list[dict]:
    """Keep latest correction per (extraction_id, field_id) by submitted_at.

    Append-only stream may contain multiple corrections for the same field
    over time (reviewer corrected twice). The training shard takes the latest.
    """
    latest: dict[tuple[str, str], dict] = {}
    for c in corrections:
        key = (str(c.get("extraction_id")), str(c.get("field_id")))
        prev = latest.get(key)
        if prev is None or str(c.get("submitted_at", "")) > str(prev.get("submitted_at", "")):
            latest[key] = c
    return list(latest.values())


def _build_training_pairs(corrections: list[dict],
                          extractions_by_id: dict[str, dict]) -> list[dict]:
    """For each correction, emit a {input, target} training pair.

    `input` is the original extraction context (raw OCR text + the model's
    prediction). `target` is the corrected value. v0.5 will format these into
    actual LoRA training prompts; for now we emit the raw structure.
    """
    pairs = []
    for c in corrections:
        eid = str(c.get("extraction_id"))
        ext = extractions_by_id.get(eid)
        if ext is None:
            log.debug("correction %s -> unknown extraction; skip", c.get("id"))
            continue
        pairs.append({
            "extraction_id": eid,
            "doc_id": ext.get("doc_id"),
            "tenant_id": str(c.get("tenant_id")),
            "field_id": c.get("field_id"),
            "original_value": c.get("original_value"),
            "corrected_value": c.get("corrected_value"),
            "reason": c.get("reason"),
            "evidence_bbox": c.get("evidence_bbox"),
            "original_form": ext.get("original_form"),
            "source_engine": ext.get("source_engine"),
            "submitted_at": c.get("submitted_at"),
        })
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Build training shard from corrections + extractions")
    ap.add_argument("--corrections", type=Path, default=Path("corpus/corrections.jsonl"))
    ap.add_argument("--extractions", type=Path, default=Path("corpus/extractions.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("corpus/train_shard.jsonl"))
    ap.add_argument("--metrics", type=Path, default=Path("corpus/metrics.json"))
    args = ap.parse_args()

    corrections = _read_jsonl(args.corrections)
    extractions = _read_jsonl(args.extractions)
    log.info("Loaded: corrections=%d extractions=%d", len(corrections), len(extractions))

    extractions_by_id = {str(e.get("id", e.get("extraction_id", ""))): e for e in extractions}
    deduped = _dedupe_corrections(corrections)
    pairs = _build_training_pairs(deduped, extractions_by_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Per-field correction counts feed v0.4 PRD Open Question 2 (corpus
    # accumulation pace verification: ~100 corrections/customer/month).
    by_field = defaultdict(int)
    by_tenant = defaultdict(int)
    for p in pairs:
        by_field[p["field_id"]] += 1
        by_tenant[p["tenant_id"]] += 1
    metrics = {
        "training_pairs": len(pairs),
        "raw_corrections": len(corrections),
        "deduped_corrections": len(deduped),
        "per_field_counts": dict(by_field),
        "per_tenant_counts": dict(by_tenant),
        "extractions_total": len(extractions),
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Wrote %d training pairs -> %s", len(pairs), args.out)
    log.info("Metrics -> %s", args.metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
