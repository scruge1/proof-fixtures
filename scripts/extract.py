#!/usr/bin/env python3
"""
Document Ops extraction pipeline (v0.3 — local-only, no paid API)

Replaces v0.2's Mistral + Gemini paid-API verifier chain with a fully
Apache-2.0 / MIT local stack per
`proof-fixtures/research/2026-05-04-local-ocr-deep-research.md`.

OCR layer (per `mcp-servers/_dep_notes/{pytesseract,rapidocr}.md`):
  Tesseract 5.5.0 (UB-Mannheim Windows build at C:/Program Files/Tesseract-OCR)
  + RapidOCR 3.8.1 (ONNX runtime, models bundled, AMD-CPU-friendly).

Voter layer (NEW — replaces blind ensemble):
  TwoEngineVoter — Tesseract + RapidOCR field-level vote.
    accept if both engines agree above threshold.
    flag for verifier if engines disagree OR confidence < 0.85.

Verifier layer (NEW — local-only, no paid API):
  -> GLMOCRVerifier (Ollama glm-ocr:q8_0, 1.6GB, OmniDocBench V1.5 #1).
       temperature=0.1, repeat_penalty=1.2 (prevents JSON loops).
  -> OllamaQwenVerifier (qwen2.5vl:7b fallback, slower CPU).
  No third-country flow. Zero per-doc cost. All Apache-2.0.

Outputs per-document JSON with extracted fields, per-field confidence,
voter agreement signal, verifier verdicts, schema-validation results,
cross-field consistency, and final pass/bounce decision against 0.98 gate.

Usage:
  python scripts/extract.py --set 00-test
  python scripts/extract.py --doc fixtures/00-test/samples/test-invoice-001.png
  python scripts/extract.py --set 06-sec-edgar --pipeline tesseract-only --no-verifier
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
VOTER_AGREEMENT_THRESHOLD = 0.85          # below = route to verifier
VOTER_TEXT_FUZZ_DISTANCE = 2              # Levenshtein tolerance on agreement
DEFAULT_VERIFIER_CHAIN = ("ollama-glm-ocr", "ollama-qwen")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"
RESULTS_DIR = REPO_ROOT / "results"

# Auto-detect Poppler binary directory (pdf2image dep). If not found, PDF
# input crashes with PDFInfoNotInstalledError.
def _detect_poppler() -> Optional[str]:
    if env := os.environ.get("POPPLER_PATH"):
        return env
    candidates = [
        r"C:\Users\a33_s\AppData\Local\Microsoft\WinGet\Packages\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe\poppler-25.07.0\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\poppler\bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
    ]
    for p in candidates:
        if Path(p).exists() and (Path(p) / ("pdftoppm.exe" if os.name == "nt" else "pdftoppm")).exists():
            return p
    return None

POPPLER_PATH = _detect_poppler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("extract")


# ============================================================================
# Output schema (versioned — bump on breaking change)
# ============================================================================

SCHEMA_VERSION = "0.3"


@dataclass
class FieldExtraction:
    field_id: str
    raw_text: str
    parsed_value: Any
    ocr_confidence: float
    voter_agreement: Optional[bool] = None       # both engines saw same value
    voter_engines_agreed: list[str] = field(default_factory=list)
    verifier_verdict: Optional[str] = None       # confirm | flag | reject
    verifier_confidence: Optional[float] = None
    verifier_reason: Optional[str] = None
    schema_valid: Optional[bool] = None          # regex / format check pass
    schema_reason: Optional[str] = None
    evidence_bbox: Optional[list[float]] = None
    final_confidence: float = 0.0


@dataclass
class DocumentExtraction:
    doc_id: str
    source_path: str
    schema_version: str = SCHEMA_VERSION
    pipeline: str = "docops-local-only"
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    ocr_engines_used: list[str] = field(default_factory=list)
    verifiers_used: list[str] = field(default_factory=list)
    fields: list[FieldExtraction] = field(default_factory=list)
    cross_field_checks: dict[str, Any] = field(default_factory=dict)
    gate_passed: bool = False
    bounce_reason: Optional[str] = None
    errors: list[str] = field(default_factory=list)


# ============================================================================
# Image preprocessing — single highest-yield lever per research §4.1
# ============================================================================


def preprocess_image(img_array):
    """Returns a binarized, upscaled, deskewed numpy uint8 image suitable for
    Tesseract + RapidOCR. Implements the §3.2 Balanced Pipeline preprocessing.
    """
    import cv2
    import numpy as np

    if img_array is None:
        raise ValueError("preprocess_image received None")

    # Grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_array.copy()

    # 3.5x upscale (Lanczos) — research consensus, biggest accuracy lever on
    # sub-300-DPI inputs
    h, w = gray.shape
    if max(h, w) < 2000:
        upscaled = cv2.resize(gray, (int(w * 3.5), int(h * 3.5)), interpolation=cv2.INTER_LANCZOS4)
    else:
        upscaled = gray

    # Deskew (Hough-based via minAreaRect of dark pixels)
    inverted = cv2.bitwise_not(upscaled)
    coords = np.column_stack(np.where(inverted > 64))
    if len(coords) > 200:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5 and abs(angle) < 15:
            (uh, uw) = upscaled.shape
            M = cv2.getRotationMatrix2D((uw // 2, uh // 2), angle, 1.0)
            upscaled = cv2.warpAffine(upscaled, M, (uw, uh), borderValue=255, flags=cv2.INTER_CUBIC)

    # Adaptive Gaussian threshold — better than Otsu on uneven invoice scans
    binarized = cv2.adaptiveThreshold(
        upscaled, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )
    return binarized


def _load_doc_as_array(doc_path: Path):
    """Load a doc (PDF or image) as a numpy array. Uses Poppler for PDFs."""
    import cv2
    import numpy as np

    if doc_path.suffix.lower() == ".pdf":
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise RuntimeError("pdf2image not installed. pip install pdf2image. See _dep_notes/pdf2image.md")
        kwargs = {"dpi": 300, "first_page": 1, "last_page": 1}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH
        try:
            images = convert_from_path(str(doc_path), **kwargs)
        except Exception as e:
            raise RuntimeError(f"PDF -> image failed (Poppler? path={POPPLER_PATH}): {e}")
        return np.array(images[0])
    else:
        img = cv2.imread(str(doc_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {doc_path}")
        return img


# ============================================================================
# OCR layer
# ============================================================================


class OCREngine:
    name = "abstract"

    def available(self) -> bool:
        return False

    def extract(self, doc_path: Path, preprocessed) -> dict[str, Any]:
        raise NotImplementedError


class TesseractEngine(OCREngine):
    """Tesseract 5.5 via pytesseract. Per `_dep_notes/pytesseract.md`."""

    name = "tesseract-5"

    def __init__(self):
        self._loaded = False
        try:
            import pytesseract
            tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(tesseract_path):
                pytesseract.pytesseract.tesseract_cmd = tesseract_path
            self._loaded = True
        except ImportError:
            log.debug("pytesseract not installed; Tesseract engine disabled")

    def available(self) -> bool:
        return self._loaded

    def extract(self, doc_path: Path, preprocessed) -> dict[str, Any]:
        import pytesseract
        from PIL import Image

        # PSM 6 = uniform block, OEM 1 = LSTM only (best on modern docs)
        config = "--oem 1 --psm 6"
        pil_img = Image.fromarray(preprocessed)
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
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
            "raw_text": pytesseract.image_to_string(pil_img, config=config),
            "words": words,
        }


class RapidOCREngine(OCREngine):
    """RapidOCR 3.8.1 via ONNX runtime. Per `_dep_notes/rapidocr.md`."""

    name = "rapidocr-3.8.1"

    def __init__(self):
        self._engine = None
        try:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        except ImportError:
            log.debug("rapidocr not installed; RapidOCR engine disabled")
        except Exception as e:
            log.warning("RapidOCR init failed: %s", e)

    def available(self) -> bool:
        return self._engine is not None

    def extract(self, doc_path: Path, preprocessed) -> dict[str, Any]:
        result = self._engine(preprocessed, use_det=True, use_cls=False, use_rec=True)
        _b = getattr(result, "boxes", None)
        boxes = list(_b) if _b is not None else []
        txts = getattr(result, "txts", []) or []
        scores = getattr(result, "scores", []) or []

        words = []
        for i, txt in enumerate(txts):
            if i < len(boxes) and i < len(scores):
                box = boxes[i]
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                words.append({
                    "text": txt,
                    "confidence": float(scores[i]),
                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                })
        return {
            "engine": self.name,
            "raw_text": "\n".join(txts),
            "words": words,
        }


class OCREnsemble:
    def __init__(self, engine_names: tuple[str, ...] = ("tesseract", "rapidocr")):
        registry: list[OCREngine] = []
        if "tesseract" in engine_names:
            registry.append(TesseractEngine())
        if "rapidocr" in engine_names:
            registry.append(RapidOCREngine())
        self.engines = [e for e in registry if e.available()]
        if not self.engines:
            raise RuntimeError(
                "No OCR engine available. See _dep_notes/pytesseract.md or _dep_notes/rapidocr.md.\n"
                "  Tesseract binary: C:/Program Files/Tesseract-OCR/tesseract.exe (UB-Mannheim Windows)\n"
                "  RapidOCR: pip install 'rapidocr==3.8.1'"
            )

    def run(self, doc_path: Path, preprocessed) -> dict[str, Any]:
        results = {}
        for eng in self.engines:
            try:
                results[eng.name] = eng.extract(doc_path, preprocessed)
            except Exception as e:
                log.warning("OCR engine %s failed on %s: %s", eng.name, doc_path.name, e)
                results[eng.name] = {"engine": eng.name, "error": str(e)}
        return results


# ============================================================================
# Field extraction — regex stubs (v0.4 will use structured-prompt extraction)
# ============================================================================

IE_VAT_REGEX = re.compile(r"\b(?:VAT(?:\s*No\.?)?[:\s]*)?(IE\d{7}[A-Z]{1,2})\b", re.IGNORECASE)
IBAN_REGEX = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16})\b")
DATE_REGEX = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
# Match either €/EUR before or after the number
MONEY_PATTERN = r"(?:€|EUR)\s*([\d,]+\.\d{2})|([\d,]+\.\d{2})\s*(?:€|EUR)"
MONEY_REGEX = re.compile(MONEY_PATTERN, re.IGNORECASE)


def _extract_candidate(text: str, field_id: str) -> tuple[str, Any]:
    if field_id == "vendor":
        # Prefer first all-caps line of >=4 chars (typical IE invoice header).
        # Then fall back to mixed-case heading line. v0.4 will use NER.
        excluded_kws = ("invoice", "total", "vat", "page", "date", "subtotal",
                        "amount", "balance", "tax", "due", "received")
        for line in text.splitlines():
            stripped = line.strip()
            if (4 <= len(stripped) <= 60
                    and re.match(r"^[A-Z][A-Z0-9 &.,'-]{3,60}$", stripped)
                    and not any(kw in stripped.lower() for kw in excluded_kws)):
                return stripped, stripped
        for line in text.splitlines():
            stripped = line.strip()
            if (4 <= len(stripped) <= 60
                    and re.match(r"^[A-Z][A-Za-z0-9 &.,'-]{3,60}$", stripped)
                    and not any(kw in stripped.lower() for kw in excluded_kws)):
                return stripped, stripped
        return ("", None)

    if field_id == "total":
        # Walk lines; "total" line that is NOT a subtotal line gets priority.
        # If money is on the next line (RapidOCR splits "Total:\nEUR 283.52"),
        # combine the lines.
        lines = text.splitlines()
        money_re = re.compile(r"(?:€|EUR)\s*([\d,]+\.\d{2})|([\d,]+\.\d{2})\s*(?:€|EUR)", re.IGNORECASE)
        for i, line in enumerate(lines):
            ll = line.lower()
            if re.search(r"\btotal\b", ll) and "subtotal" not in ll:
                # try same line first
                m = money_re.search(line)
                if not m and i + 1 < len(lines):
                    m = money_re.search(lines[i + 1])
                if m:
                    val = m.group(1) or m.group(2)
                    return line.strip(), float(val.replace(",", ""))
        # Fallback: "Amount Due" or "Grand Total" + largest money on page
        m = re.search(
            r"(?:amount\s+due|grand\s+total)[:\s]*\n?\s*(?:€|EUR)?\s*([\d,]+\.\d{2})",
            text, re.IGNORECASE,
        )
        if m:
            return m.group(0), float(m.group(1).replace(",", ""))
        money = MONEY_REGEX.findall(text)
        nums = [float((a or b).replace(",", "")) for a, b in money if (a or b)]
        if nums:
            return f"EUR {max(nums):.2f}", max(nums)
        return "", None

    if field_id == "vat":
        # IE invoices write VAT lines as "VAT 23%: EUR 53.02" or "VAT IE 6543210C ..."
        # Skip the IE VAT-number form; we want the monetary value.
        # Strategy: find "VAT" not followed by " IE" or " No." then capture money.
        for line in text.splitlines():
            if re.search(r"\bvat\b", line, re.IGNORECASE) and not re.search(r"\bvat\s+(no\.?|IE\d)", line, re.IGNORECASE):
                m = re.search(r"(?:€|EUR)\s*([\d,]+\.\d{2})|([\d,]+\.\d{2})\s*(?:€|EUR)", line, re.IGNORECASE)
                if m:
                    val = m.group(1) or m.group(2)
                    return line.strip(), float(val.replace(",", ""))
        return "", None

    if field_id == "date":
        m = DATE_REGEX.search(text)
        return (m.group(0), m.group(0)) if m else ("", None)

    return ("", None)


def _avg_word_confidence(ocr_output: dict, engine_name: str) -> float:
    r = ocr_output.get(engine_name)
    if not isinstance(r, dict):
        return 0.0
    confs = [w["confidence"] for w in r.get("words", []) if "confidence" in w]
    return sum(confs) / len(confs) if confs else 0.0


# ============================================================================
# TwoEngineVoter — field-level voting between Tesseract and RapidOCR
# ============================================================================


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()) if s else ""


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[-1]


def _values_match(p1, p2, fid: str) -> bool:
    """Field-aware semantic equality. Tolerates float drift on money fields
    and raw-text differences when parsed values are equivalent."""
    if p1 is None or p2 is None:
        return p1 == p2
    if fid in ("total", "vat"):
        try:
            return abs(float(p1) - float(p2)) < 0.01
        except (TypeError, ValueError):
            return str(p1).strip() == str(p2).strip()
    return _normalize_text(str(p1)) == _normalize_text(str(p2))


def two_engine_vote(ocr_output: dict) -> list[FieldExtraction]:
    """Per-engine field extraction, then vote field-by-field. Engines agree
    when `parsed_value` matches semantically (numeric tolerance for money,
    normalized string for text). Falls back to Levenshtein on raw_text only
    when parsed values are both None but raw_texts exist."""
    per_engine: dict[str, dict[str, tuple[str, Any]]] = {}
    for eng_name, res in ocr_output.items():
        if not isinstance(res, dict) or "raw_text" not in res:
            continue
        per_engine[eng_name] = {
            fid: _extract_candidate(res["raw_text"], fid)
            for fid in CRITICAL_FIELDS
        }

    fields: list[FieldExtraction] = []
    for fid in CRITICAL_FIELDS:
        candidates = []  # [(engine_name, raw, parsed)]
        for eng_name, ext in per_engine.items():
            raw, parsed = ext.get(fid, ("", None))
            if raw or parsed is not None:
                candidates.append((eng_name, raw, parsed))

        if not candidates:
            fields.append(FieldExtraction(
                field_id=fid, raw_text="", parsed_value=None,
                ocr_confidence=0.0, voter_agreement=False,
                voter_engines_agreed=[],
            ))
            continue

        # Pick the candidate with the most informative raw_text as the reference
        ref_engine, ref_raw, ref_parsed = max(candidates, key=lambda c: len(c[1] or ""))
        agreed = [ref_engine]
        for eng_name, raw, parsed in candidates:
            if eng_name == ref_engine:
                continue
            if _values_match(ref_parsed, parsed, fid):
                agreed.append(eng_name)
            elif ref_parsed is None and parsed is None:
                # Fall back to raw_text Levenshtein when both engines failed
                if _levenshtein(_normalize_text(ref_raw), _normalize_text(raw)) <= VOTER_TEXT_FUZZ_DISTANCE:
                    agreed.append(eng_name)

        avg_confs = [_avg_word_confidence({e: ocr_output[e]}, e) for e, _, _ in candidates if e in ocr_output]
        ocr_conf = sum(avg_confs) / len(avg_confs) if avg_confs else 0.0
        full_agreement = len(agreed) == len(candidates) and len(candidates) >= 2

        fields.append(FieldExtraction(
            field_id=fid,
            raw_text=ref_raw,
            parsed_value=ref_parsed,
            ocr_confidence=ocr_conf,
            voter_agreement=full_agreement,
            voter_engines_agreed=agreed,
        ))
    return fields


# ============================================================================
# Schema validation — regex per IE invoice patterns
# ============================================================================


def validate_field_schema(f: FieldExtraction) -> tuple[bool, str]:
    if f.parsed_value is None:
        return False, "no value extracted"

    if f.field_id == "vendor":
        s = str(f.parsed_value)
        if 3 <= len(s) <= 80 and any(c.isalpha() for c in s):
            return True, "ok"
        return False, "vendor length or alpha check failed"

    if f.field_id in ("total", "vat"):
        try:
            v = float(f.parsed_value)
            if 0 <= v <= 1_000_000:
                return True, "ok"
            return False, f"value out of plausible range: {v}"
        except (TypeError, ValueError):
            return False, f"not a number: {f.parsed_value!r}"

    if f.field_id == "date":
        s = str(f.parsed_value)
        m = DATE_REGEX.match(s)
        if not m:
            return False, "date regex no match"
        try:
            parts = re.split(r"[/-]", m.group(1))
            d, mo, y = (int(p) for p in parts)
            if y < 100:
                y += 2000
            if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2100:
                return True, "ok"
            return False, f"date parts invalid: d={d} m={mo} y={y}"
        except Exception as e:
            return False, f"date parse error: {e}"

    return True, "ok"


def cross_field_consistency(fields: list[FieldExtraction]) -> dict[str, Any]:
    """`subtotal + vat ≈ total` within ±0.02 EUR tolerance. We don't have
    subtotal as a critical field yet; check `vat <= total`."""
    by_id = {f.field_id: f for f in fields}
    checks = {}

    total = by_id.get("total")
    vat = by_id.get("vat")
    if total and vat and isinstance(total.parsed_value, (int, float)) and isinstance(vat.parsed_value, (int, float)):
        ratio = vat.parsed_value / total.parsed_value if total.parsed_value > 0 else 0
        # IE std rate 23%, reduced 13.5%, super-reduced 9% / 4.8% / 0%
        plausible_rates = (0.0, 0.048, 0.09, 0.135, 0.21, 0.23, 0.30)  # ratios off the gross
        # For VAT-on-net invoices, vat/total ≈ rate / (1 + rate). Convert.
        net_ratios = [r / (1 + r) for r in plausible_rates]
        plausible = any(abs(ratio - pr) < 0.005 for pr in net_ratios)
        checks["vat_total_ratio"] = round(ratio, 4)
        checks["vat_rate_plausible"] = plausible
    else:
        checks["vat_total_ratio"] = None
        checks["vat_rate_plausible"] = None

    return checks


# ============================================================================
# Verifier layer — local-only Ollama
# ============================================================================


class Verifier:
    name = "abstract"
    cost_per_doc_eur = 0.0

    def available(self) -> bool:
        return False

    def verify(self, ocr_output, fields):
        raise NotImplementedError


VERIFIER_PROMPT = """\
You are a verifier for an automated invoice/receipt extraction pipeline.
You score a candidate field against the OCR text. Score 0.0 to 1.0,
where 1.0 means "this field is confidently correct based on the visible OCR text".

You must NOT propose a replacement value. You score only.

Return strict JSON: {"verdict": "confirm" | "flag" | "reject", "confidence": <float>, "reason": "<short>"}

Field to verify:
  field_id: %FIELD_ID%
  candidate_value: %CANDIDATE_VALUE%

OCR raw text (truncated):
%OCR_TEXT%
"""


def _check_ollama_model(model_substring: str, endpoint: str) -> bool:
    try:
        import requests
        r = requests.get(f"{endpoint}/api/tags", timeout=2)
        if r.status_code != 200:
            return False
        tags = r.json().get("models", [])
        return any(model_substring in t.get("name", "") for t in tags)
    except Exception:
        return False


def _ollama_verify_text(model: str, endpoint: str, ocr_text: str,
                        fields: list[FieldExtraction], options: dict) -> dict[str, dict]:
    """Generic text-only verification call to Ollama JSON endpoint."""
    import requests
    results = {}
    for f in fields:
        prompt = (VERIFIER_PROMPT
                  .replace("%FIELD_ID%", f.field_id)
                  .replace("%CANDIDATE_VALUE%", str(f.parsed_value))
                  .replace("%OCR_TEXT%", ocr_text))
        try:
            r = requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": options,
                },
                timeout=180,
            )
            response_text = r.json().get("response", "{}")
            results[f.field_id] = json.loads(response_text)
        except Exception as e:
            results[f.field_id] = {"verdict": "flag", "confidence": 0.0,
                                   "reason": f"verifier-error: {e}"}
    return results


class GLMOCRVerifier(Verifier):
    """GLM-OCR Q8 via Ollama. Per research §1.1 + §5.2:
    OmniDocBench V1.5 #1 at 94.62, 1.6GB on disk, ~3-4GB RAM at inference,
    Apache-2.0. Settings temperature=0.1 + repeat_penalty=1.2 prevent
    JSON repetition loops on transaction lists (Mhjorleifsson)."""

    name = "ollama-glm-ocr-q8"
    cost_per_doc_eur = 0.0

    def __init__(self, model: str = "glm-ocr:q8_0", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is None:
            self._available = _check_ollama_model("glm-ocr", self.endpoint)
        return self._available

    def verify(self, ocr_output, fields):
        ocr_text = "\n".join(
            r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict)
        )[:4000]
        return _ollama_verify_text(
            self.model, self.endpoint, ocr_text, fields,
            options={"temperature": 0.1, "repeat_penalty": 1.2},
        )


class OllamaQwenVerifier(Verifier):
    """Qwen2.5-VL fallback. Slower than GLM-OCR on CPU per research §2 table
    (~8-15s/page Q4 vs glm-ocr at 1.6GB model). Apache-2.0."""

    name = "ollama-qwen2.5-vl"
    cost_per_doc_eur = 0.0

    def __init__(self, model: str = "qwen2.5vl:7b", endpoint: str = "http://localhost:11434"):
        self.model = model
        self.endpoint = endpoint
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is None:
            self._available = _check_ollama_model("qwen2.5vl", self.endpoint)
        return self._available

    def verify(self, ocr_output, fields):
        ocr_text = "\n".join(
            r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict)
        )[:4000]
        return _ollama_verify_text(
            self.model, self.endpoint, ocr_text, fields,
            options={"temperature": 0.1},
        )


VERIFIER_REGISTRY = {
    "ollama-glm-ocr": GLMOCRVerifier,
    "ollama-qwen": OllamaQwenVerifier,
}


def select_verifier(chain: tuple[str, ...]) -> Optional[Verifier]:
    for name in chain:
        cls = VERIFIER_REGISTRY.get(name)
        if not cls:
            continue
        v = cls()
        if v.available():
            log.info("Verifier selected: %s (%s eur/doc)", v.name, v.cost_per_doc_eur)
            return v
    log.warning("No verifier available — running OCR + voter only (no LLM verification)")
    return None


# ============================================================================
# Gate — applies 0.98 critical-field threshold
# ============================================================================


def apply_gate(extraction: DocumentExtraction, threshold: float = CONFIDENCE_GATE) -> None:
    """Composite confidence per field:

    - voter agreement + schema valid: trust the consensus (verifier was skipped
      by design per research §3.2). final = min(1.0, ocr_conf * 1.05).
    - voter disagreement: verifier weighs in. final = ocr*0.3 + verifier*0.55 +
      schema*0.15.
    - voter disagreement + no verifier: low trust. final = ocr * 0.5.
    """
    failing = []
    for f in extraction.fields:
        if f.field_id not in CRITICAL_FIELDS:
            continue
        if f.parsed_value is None:
            f.final_confidence = 0.0
        elif f.voter_agreement and f.schema_valid:
            f.final_confidence = min(1.0, f.ocr_confidence * 1.05)
        elif f.verifier_confidence is not None:
            schema_pass = 1.0 if f.schema_valid else 0.0
            f.final_confidence = min(
                1.0,
                f.ocr_confidence * 0.30
                + f.verifier_confidence * 0.55
                + schema_pass * 0.15,
            )
        else:
            f.final_confidence = f.ocr_confidence * 0.5
        if f.final_confidence < threshold:
            failing.append(f"{f.field_id}={f.final_confidence:.2f}")
    extraction.gate_passed = not failing
    extraction.bounce_reason = "below-threshold: " + ", ".join(failing) if failing else None


# ============================================================================
# Main pipeline
# ============================================================================


def extract_one(doc_path: Path, ensemble: OCREnsemble, verifier: Optional[Verifier]) -> DocumentExtraction:
    extraction = DocumentExtraction(
        doc_id=doc_path.stem,
        source_path=str(doc_path.resolve()),
    )

    # Load + preprocess
    log.info("Load+preprocess: %s", doc_path.name)
    img_array = _load_doc_as_array(doc_path)
    preprocessed = preprocess_image(img_array)

    # OCR layer
    log.info("OCR: %s", doc_path.name)
    ocr_output = ensemble.run(doc_path, preprocessed)
    extraction.ocr_engines_used = [e.name for e in ensemble.engines]

    # Voter — field-level extraction + agreement check
    extraction.fields = two_engine_vote(ocr_output)

    # Schema validation per-field
    for f in extraction.fields:
        ok, reason = validate_field_schema(f)
        f.schema_valid = ok
        f.schema_reason = reason

    # Verifier layer — only on disagreement OR low confidence per research §3.2
    if verifier is not None:
        fields_to_verify = [
            f for f in extraction.fields
            if (not f.voter_agreement) or f.ocr_confidence < VOTER_AGREEMENT_THRESHOLD
        ]
        if fields_to_verify:
            log.info("Verify %d/%d fields via %s", len(fields_to_verify), len(extraction.fields), verifier.name)
            verdicts = verifier.verify(ocr_output, fields_to_verify)
            for f in fields_to_verify:
                v = verdicts.get(f.field_id, {})
                f.verifier_verdict = v.get("verdict")
                f.verifier_confidence = v.get("confidence")
                f.verifier_reason = v.get("reason")
            extraction.verifiers_used = [verifier.name]
        else:
            log.info("Verifier skipped: voter unanimous + ocr_conf >= %s", VOTER_AGREEMENT_THRESHOLD)

    # Cross-field consistency
    extraction.cross_field_checks = cross_field_consistency(extraction.fields)

    # Gate
    apply_gate(extraction)

    extraction.completed_at = time.time()
    log.info(
        "Result %s: gate=%s%s (%.1fs)",
        doc_path.name,
        "PASS" if extraction.gate_passed else "BOUNCE",
        f" ({extraction.bounce_reason})" if extraction.bounce_reason else "",
        extraction.completed_at - extraction.started_at,
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
        log.warning("Fixture set %s is empty (no docs in samples/)", set_slug)
    return sorted(docs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Document Ops extraction pipeline (v0.3 — local-only)")
    ap.add_argument("--set", dest="set_slug", help="Fixture set slug, e.g. 00-test")
    ap.add_argument("--doc", type=Path, help="Single document path")
    ap.add_argument("--pipeline", default="docops",
                    help="Pipeline preset: docops (default) | tesseract-only | rapidocr-only")
    ap.add_argument("--no-verifier", action="store_true", help="Skip verifier layer (OCR + voter only)")
    ap.add_argument("--verifier-chain", default=",".join(DEFAULT_VERIFIER_CHAIN),
                    help="Comma-separated verifier preference order")
    ap.add_argument("--output-dir", type=Path, default=RESULTS_DIR,
                    help="Where to write per-doc JSON results")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if args.pipeline == "tesseract-only":
        engines = ("tesseract",)
    elif args.pipeline == "rapidocr-only":
        engines = ("rapidocr",)
    else:
        engines = ("tesseract", "rapidocr")

    ensemble = OCREnsemble(engines)
    log.info("OCR ensemble: %s", [e.name for e in ensemble.engines])
    log.info("Poppler path: %s", POPPLER_PATH or "NOT FOUND — PDF input will fail")

    verifier = None if args.no_verifier else select_verifier(tuple(args.verifier_chain.split(",")))

    docs = discover_docs(args.set_slug, args.doc)
    if not docs:
        log.error("No documents to process.")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = args.output_dir / args.set_slug if args.set_slug else args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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
            log.error("Doc %s crashed pipeline: %s", doc.name, e, exc_info=True)
            error_count += 1

    log.info("=== Done: %d passed gate, %d bounced, %d errors ===",
             pass_count, bounce_count, error_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
