#!/usr/bin/env python3
"""
Document Ops extraction pipeline (v0.1)

Runs the OSS OCR ensemble + verifier chain on a directory of documents:
  Tesseract 5 + PaddleOCR-VL + PP-StructureV3 (OCR layer)
  -> Local Qwen2.5-VL via Ollama (primary verifier)
  -> Mistral Small 3.1 via API (paid fallback)
  -> Gemini 2.5 Flash via API (optional Auto Plus cross-check)

Outputs per-document JSON with extracted fields, per-field confidence scores,
verifier verdicts, and a final pass/bounce decision against the 0.98 gate.

Usage:
  python scripts/extract.py --set 01-1900-us-census
  python scripts/extract.py --doc fixtures/03-uk-companies-house-pre2000/samples/example.pdf
  python scripts/extract.py --set 06-sec-edgar --pipeline tesseract-only --no-verifier
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ============================================================================
# Configuration
# ============================================================================

CRITICAL_FIELDS = ("vendor", "total", "vat", "date")
CONFIDENCE_GATE = 0.98
DEFAULT_VERIFIER_CHAIN = ("ollama-qwen", "mistral", "gemini")
REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
RESULTS_DIR = REPO_ROOT / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract")


# ============================================================================
# Output schema (versioned — bump on breaking change)
# ============================================================================

SCHEMA_VERSION = "0.1"


@dataclass
class FieldExtraction:
    field_id: str
    raw_text: str
    parsed_value: Any
    ocr_confidence: float
    verifier_verdict: Optional[str] = None       # confirm | flag | reject
    verifier_confidence: Optional[float] = None
    verifier_reason: Optional[str] = None
    evidence_bbox: Optional[list[float]] = None  # [x0, y0, x1, y1] page-relative
    final_confidence: float = 0.0


@dataclass
class DocumentExtraction:
    doc_id: str
    source_path: str
    schema_version: str = SCHEMA_VERSION
    pipeline: str = "docops"
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    ocr_engines_used: list[str] = field(default_factory=list)
    verifiers_used: list[str] = field(default_factory=list)
    fields: list[FieldExtraction] = field(default_factory=list)
    gate_passed: bool = False
    bounce_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)


# ============================================================================
# OCR layer — adapter pattern, graceful fallback per engine
# ============================================================================


class OCREngine:
    name = "abstract"

    def available(self) -> bool:
        return False

    def extract(self, doc_path: Path) -> dict[str, Any]:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    name = "tesseract-5"

    def __init__(self):
        self._loaded = False
        try:
            import pytesseract  # noqa: F401
            self._loaded = True
        except ImportError:
            log.debug("pytesseract not installed; Tesseract engine disabled")

    def available(self) -> bool:
        return self._loaded

    def extract(self, doc_path: Path) -> dict[str, Any]:
        import pytesseract
        from PIL import Image

        # Convert PDF page-1 to image if needed (real impl: pdf2image for all pages)
        if doc_path.suffix.lower() == ".pdf":
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(str(doc_path), dpi=300, first_page=1, last_page=1)
                img = images[0]
            except ImportError:
                raise RuntimeError("pdf2image required for PDF — pip install pdf2image")
        else:
            img = Image.open(doc_path)

        # Get word-level confidence via image_to_data
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = []
        for i, word in enumerate(data["text"]):
            if word.strip() and int(data["conf"][i]) > 0:
                words.append({
                    "text": word,
                    "confidence": int(data["conf"][i]) / 100.0,
                    "bbox": [
                        data["left"][i], data["top"][i],
                        data["left"][i] + data["width"][i],
                        data["top"][i] + data["height"][i],
                    ],
                })
        return {
            "engine": self.name,
            "raw_text": pytesseract.image_to_string(img),
            "words": words,
        }


class PaddleOCRVLEngine(OCREngine):
    """PaddleOCR-VL 1.5 (Apache 2.0, 0.9B vision model). Higher accuracy than
    Tesseract on rotated / multi-column / faded scans."""

    name = "paddleocr-vl-1.5"

    def __init__(self):
        self._ocr = None
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except ImportError:
            log.debug("paddleocr not installed; PaddleOCR-VL engine disabled")
        except Exception as e:
            log.warning("PaddleOCR-VL init failed: %s", e)

    def available(self) -> bool:
        return self._ocr is not None

    def extract(self, doc_path: Path) -> dict[str, Any]:
        result = self._ocr.ocr(str(doc_path), cls=True)
        words = []
        full_text = []
        for page_result in (result or []):
            for line in (page_result or []):
                bbox, (text, conf) = line
                words.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": [float(c) for pt in bbox for c in pt][:4],
                })
                full_text.append(text)
        return {
            "engine": self.name,
            "raw_text": "\n".join(full_text),
            "words": words,
        }


class PPStructureV3Engine(OCREngine):
    """PaddleOCR PP-StructureV3 — table + form extraction."""

    name = "pp-structure-v3"

    def __init__(self):
        self._engine = None
        try:
            from paddleocr import PPStructure
            self._engine = PPStructure(show_log=False)
        except ImportError:
            log.debug("PPStructure not installed; PP-StructureV3 engine disabled")
        except Exception as e:
            log.warning("PP-StructureV3 init failed: %s", e)

    def available(self) -> bool:
        return self._engine is not None

    def extract(self, doc_path: Path) -> dict[str, Any]:
        # PPStructure operates on images; convert PDF page-1 first
        if doc_path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path
            images = convert_from_path(str(doc_path), dpi=300, first_page=1, last_page=1)
            import numpy as np
            img = np.array(images[0])
        else:
            import cv2
            img = cv2.imread(str(doc_path))
        result = self._engine(img)
        # PPStructure returns list of regions (table, text, figure, ...)
        return {
            "engine": self.name,
            "regions": [
                {"type": r.get("type"), "bbox": r.get("bbox"), "text": str(r.get("res", ""))[:500]}
                for r in result
            ],
        }


class OCREnsemble:
    """Runs available OCR engines + merges results. The verifier layer
    operates on the merged ensemble output, not on a single engine."""

    def __init__(self, engine_names: tuple[str, ...] = ("tesseract", "paddleocr-vl", "pp-structure")):
        registry: list[OCREngine] = []
        if "tesseract" in engine_names:
            registry.append(TesseractEngine())
        if "paddleocr-vl" in engine_names:
            registry.append(PaddleOCRVLEngine())
        if "pp-structure" in engine_names:
            registry.append(PPStructureV3Engine())
        self.engines = [e for e in registry if e.available()]
        if not self.engines:
            raise RuntimeError(
                "No OCR engine available. Install at least pytesseract:\n"
                "  pip install pytesseract pillow pdf2image\n"
                "  + system Tesseract binary (https://github.com/UB-Mannheim/tesseract/wiki on Windows)"
            )

    def run(self, doc_path: Path) -> dict[str, Any]:
        results = {}
        for eng in self.engines:
            try:
                results[eng.name] = eng.extract(doc_path)
            except Exception as e:
                log.warning("OCR engine %s failed on %s: %s", eng.name, doc_path.name, e)
                results[eng.name] = {"engine": eng.name, "error": str(e)}
        return results


# ============================================================================
# Verifier layer — chain pattern, cheapest tier first
# ============================================================================


class Verifier:
    name = "abstract"
    cost_per_doc_eur = 0.0

    def available(self) -> bool:
        return False

    def verify(self, ocr_output: dict[str, Any], fields: list[FieldExtraction]) -> dict[str, Any]:
        """Returns {field_id: {verdict, confidence, reason}} per critical field."""
        raise NotImplementedError


VERIFIER_PROMPT = """\
You are a verifier for an automated invoice/receipt extraction pipeline.
You are given OCR output and a candidate field extraction.

Your job: score the candidate field against the OCR text. Score from 0.0 to 1.0
where 1.0 means "this field is confidently correct based on the visible OCR text".

You must NOT propose a replacement value. You score only.

Return strict JSON:
  {"verdict": "confirm" | "flag" | "reject", "confidence": <float>, "reason": "<short>"}

Field to verify:
  field_id: %FIELD_ID%
  candidate_value: %CANDIDATE_VALUE%

OCR raw text (truncated):
%OCR_TEXT%
"""


class OllamaQwenVerifier(Verifier):
    """Local Qwen2.5-VL-7B via Ollama. Free, runs on ZBook 8GB RTX."""

    name = "ollama-qwen2.5-vl-7b"
    cost_per_doc_eur = 0.0

    def __init__(self, model: str = "qwen2.5vl:7b", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import requests
            r = requests.get(f"{self.endpoint}/api/tags", timeout=2)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def verify(self, ocr_output, fields):
        import requests
        results = {}
        ocr_text = self._merge_ocr_text(ocr_output)[:4000]
        for f in fields:
            prompt = (VERIFIER_PROMPT
                      .replace("%FIELD_ID%", f.field_id)
                      .replace("%CANDIDATE_VALUE%", str(f.parsed_value))
                      .replace("%OCR_TEXT%", ocr_text))
            try:
                r = requests.post(
                    f"{self.endpoint}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
                    timeout=60,
                )
                response_text = r.json().get("response", "{}")
                results[f.field_id] = json.loads(response_text)
            except Exception as e:
                results[f.field_id] = {"verdict": "flag", "confidence": 0.0,
                                       "reason": f"verifier-error: {e}"}
        return results

    def _merge_ocr_text(self, ocr_output: dict) -> str:
        return "\n".join(
            r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict)
        )


class MistralVerifier(Verifier):
    """Mistral Small 3.1 via paid API (Scale tier — no training on customer
    content per Mistral commercial DPA)."""

    name = "mistral-small-3.1"
    cost_per_doc_eur = 0.001

    def __init__(self, model: str = "mistral-small-latest"):
        self.model = model
        self._client = None
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            log.debug("MISTRAL_API_KEY not set; Mistral verifier disabled")
            return
        try:
            from mistralai import Mistral
            self._client = Mistral(api_key=api_key)
        except ImportError:
            log.debug("mistralai package not installed (pip install mistralai)")

    def available(self) -> bool:
        return self._client is not None

    def verify(self, ocr_output, fields):
        results = {}
        ocr_text = "\n".join(r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict))[:4000]
        for f in fields:
            prompt = (VERIFIER_PROMPT
                      .replace("%FIELD_ID%", f.field_id)
                      .replace("%CANDIDATE_VALUE%", str(f.parsed_value))
                      .replace("%OCR_TEXT%", ocr_text))
            try:
                resp = self._client.chat.complete(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                results[f.field_id] = json.loads(resp.choices[0].message.content)
            except Exception as e:
                results[f.field_id] = {"verdict": "flag", "confidence": 0.0,
                                       "reason": f"mistral-error: {e}"}
        return results


class GeminiVerifier(Verifier):
    """Gemini 2.5 Flash via paid API @ europe-west4 (Auto Plus cross-check)."""

    name = "gemini-2.5-flash"
    cost_per_doc_eur = 0.0005

    def __init__(self):
        self._client = None
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            log.debug("GEMINI_API_KEY/GOOGLE_API_KEY not set; Gemini verifier disabled")
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            log.debug("google-genai package not installed")

    def available(self) -> bool:
        return self._client is not None

    def verify(self, ocr_output, fields):
        results = {}
        ocr_text = "\n".join(r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict))[:4000]
        for f in fields:
            prompt = (VERIFIER_PROMPT
                      .replace("%FIELD_ID%", f.field_id)
                      .replace("%CANDIDATE_VALUE%", str(f.parsed_value))
                      .replace("%OCR_TEXT%", ocr_text))
            try:
                resp = self._client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"response_mime_type": "application/json"},
                )
                results[f.field_id] = json.loads(resp.text)
            except Exception as e:
                results[f.field_id] = {"verdict": "flag", "confidence": 0.0,
                                       "reason": f"gemini-error: {e}"}
        return results


VERIFIER_REGISTRY = {
    "ollama-qwen": OllamaQwenVerifier,
    "mistral": MistralVerifier,
    "gemini": GeminiVerifier,
}


def select_verifier(chain: tuple[str, ...]) -> Optional[Verifier]:
    """Walks the verifier chain; returns the first available verifier."""
    for name in chain:
        cls = VERIFIER_REGISTRY.get(name)
        if not cls:
            continue
        v = cls()
        if v.available():
            log.info("Verifier selected: %s (%s eur/doc)", v.name, v.cost_per_doc_eur)
            return v
    log.warning("No verifier available — running OCR-only mode (no confidence verification)")
    return None


# ============================================================================
# Field extraction — placeholder rules-layer (real impl uses prompt extraction)
# ============================================================================


def extract_fields_from_ocr(ocr_output: dict) -> list[FieldExtraction]:
    """Stub: pulls candidate values for critical fields from OCR text via
    simple regex. Real impl uses a structured-extraction prompt OR a
    fine-tuned NER model. v0.1 ships the stub so the pipeline runs end-to-end."""
    import re
    text = "\n".join(r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict))
    avg_conf = _avg_word_confidence(ocr_output)
    fields = []
    for fid in CRITICAL_FIELDS:
        candidate, parsed = _placeholder_extract(text, fid)
        fields.append(FieldExtraction(
            field_id=fid,
            raw_text=candidate,
            parsed_value=parsed,
            ocr_confidence=avg_conf,
        ))
    return fields


def _placeholder_extract(text: str, field_id: str) -> tuple[str, Any]:
    import re
    if field_id == "vendor":
        m = re.search(r"^([A-Z][A-Za-z &.,'-]{3,60})$", text, re.MULTILINE)
        return (m.group(0), m.group(0)) if m else ("", None)
    if field_id == "total":
        m = re.search(r"(?:total|amount due)[:\s]*€?\s*([\d,]+\.\d{2})", text, re.IGNORECASE)
        return (m.group(0), float(m.group(1).replace(",", ""))) if m else ("", None)
    if field_id == "vat":
        m = re.search(r"vat[:\s]*€?\s*([\d,]+\.\d{2})", text, re.IGNORECASE)
        return (m.group(0), float(m.group(1).replace(",", ""))) if m else ("", None)
    if field_id == "date":
        m = re.search(r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})", text)
        return (m.group(0), m.group(0)) if m else ("", None)
    return ("", None)


def _avg_word_confidence(ocr_output: dict) -> float:
    confs = []
    for r in ocr_output.values():
        if isinstance(r, dict) and "words" in r:
            confs.extend(w["confidence"] for w in r["words"] if "confidence" in w)
    return sum(confs) / len(confs) if confs else 0.0


# ============================================================================
# Gate — applies 0.98 critical-field threshold
# ============================================================================


def apply_gate(extraction: DocumentExtraction, threshold: float = CONFIDENCE_GATE) -> None:
    failing = []
    for f in extraction.fields:
        if f.field_id in CRITICAL_FIELDS:
            f.final_confidence = min(
                f.ocr_confidence,
                f.verifier_confidence if f.verifier_confidence is not None else f.ocr_confidence,
            )
            if f.final_confidence < threshold:
                failing.append(f"{f.field_id} ({f.final_confidence:.2f})")
    if failing:
        extraction.gate_passed = False
        extraction.bounce_reason = "below-threshold: " + ", ".join(failing)
    else:
        extraction.gate_passed = True


# ============================================================================
# Main pipeline
# ============================================================================


def extract_one(doc_path: Path, ensemble: OCREnsemble, verifier: Optional[Verifier]) -> DocumentExtraction:
    extraction = DocumentExtraction(
        doc_id=doc_path.stem,
        source_path=str(doc_path.resolve()),
    )

    # OCR layer
    log.info("OCR: %s", doc_path.name)
    ocr_output = ensemble.run(doc_path)
    extraction.ocr_engines_used = [e.name for e in ensemble.engines]

    # Field extraction
    extraction.fields = extract_fields_from_ocr(ocr_output)

    # Verifier layer
    if verifier is not None:
        log.info("Verify: %s via %s", doc_path.name, verifier.name)
        verdicts = verifier.verify(ocr_output, extraction.fields)
        for f in extraction.fields:
            v = verdicts.get(f.field_id, {})
            f.verifier_verdict = v.get("verdict")
            f.verifier_confidence = v.get("confidence")
            f.verifier_reason = v.get("reason")
        extraction.verifiers_used = [verifier.name]

    # Gate
    apply_gate(extraction)

    extraction.completed_at = time.time()
    log.info(
        "Result %s: gate=%s%s",
        doc_path.name,
        "PASS" if extraction.gate_passed else "BOUNCE",
        f" ({extraction.bounce_reason})" if extraction.bounce_reason else "",
    )
    return extraction


def discover_docs(set_slug: Optional[str], single_doc: Optional[Path]) -> list[Path]:
    if single_doc:
        return [single_doc]
    if not set_slug:
        raise ValueError("Provide --set or --doc")
    set_dir = FIXTURES_DIR / set_slug / "samples"
    if not set_dir.exists():
        raise FileNotFoundError(f"No fixtures at {set_dir}. Run scripts/build_corpus.py first.")
    docs = [p for p in set_dir.iterdir() if p.suffix.lower() in (".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff")]
    if not docs:
        log.warning("Fixture set %s is empty (no docs in samples/) — placeholder only.", set_slug)
    return sorted(docs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Document Ops extraction pipeline (v0.1)")
    ap.add_argument("--set", dest="set_slug", help="Fixture set slug, e.g. 01-1900-us-census")
    ap.add_argument("--doc", type=Path, help="Single document path")
    ap.add_argument("--pipeline", default="docops",
                    help="Pipeline preset: docops (default) | tesseract-only | paddle-only")
    ap.add_argument("--no-verifier", action="store_true", help="Skip verifier layer (OCR only)")
    ap.add_argument("--verifier-chain", default=",".join(DEFAULT_VERIFIER_CHAIN),
                    help="Comma-separated verifier preference order")
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                    help="Where to write per-doc JSON results")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    # Pick OCR engines per pipeline preset
    if args.pipeline == "tesseract-only":
        engines = ("tesseract",)
    elif args.pipeline == "paddle-only":
        engines = ("paddleocr-vl", "pp-structure")
    else:
        engines = ("tesseract", "paddleocr-vl", "pp-structure")

    ensemble = OCREnsemble(engines)
    log.info("OCR ensemble: %s", [e.name for e in ensemble.engines])

    verifier = None if args.no_verifier else select_verifier(tuple(args.verifier_chain.split(",")))

    docs = discover_docs(args.set_slug, args.doc)
    if not docs:
        log.error("No documents to process.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.set_slug:
        out_dir = args.output_dir / args.set_slug
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = args.output_dir

    pass_count, bounce_count, error_count = 0, 0, 0
    for doc in docs:
        try:
            ext = extract_one(doc, ensemble, verifier)
            (out_dir / f"{ext.doc_id}.json").write_text(
                json.dumps(asdict(ext), indent=2, default=str),
                encoding="utf-8",
            )
            if ext.gate_passed:
                pass_count += 1
            else:
                bounce_count += 1
        except Exception as e:
            log.error("Doc %s crashed pipeline: %s", doc.name, e)
            error_count += 1

    log.info("=== Done: %d passed gate, %d bounced, %d errors ===",
             pass_count, bounce_count, error_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
