#!/usr/bin/env python3
"""
Document Ops extraction pipeline (v0.4 — local-only, Hypercell modular).

Stage layout per CALLMEIE-DOCAI-V0.4-PRD.md §4:
  Stage 1  ingest+provenance — pypdfium2 render or pdfplumber native text;
           VATCA s.84(3) original-form attestation; sha256 hash; ingest ts.
  Stage 2  classification (deferred — v0.4.1)
  Stage 3  preprocess (Lanczos 3.5x + Hough deskew + adaptive threshold)
  Stage 4  OCR ensemble (Tesseract + RapidOCR) [+PaddleOCR PP-StructureV3 v0.4.x]
  Stage 5  voter + extract (regex stubs in v0.4.0; instructor+Pydantic v0.4.x)
  Stage 6  vision verifier (GLM-OCR Q8 / Qwen2.5-VL) — local Ollama, zero cost
  Stage 7  HaluGate confidence gate (0.98 critical) + cross-field validation
  Stage 8  Label Studio CE corrections (v0.4.1)
  Stage 9  Hetzner Object Storage + DVC + drift dashboard (v0.4.2)

OCR layer (per `mcp-servers/_dep_notes/{pytesseract,rapidocr}.md`):
  Tesseract 5.5.0 + RapidOCR 3.8.1 (ONNX, AMD-CPU-friendly).

Ingest layer NEW v0.4 (per `_dep_notes/{pypdfium2,pdfplumber}.md`):
  pypdfium2 5.7.1 — embedded PDFium, no Poppler subprocess. Replaces pdf2image.
  pdfplumber 0.11.9 — native-PDF text extractor; bypasses OCR on digital PDFs.

Verifier layer (local-only, no paid API):
  GLM-OCR Q8 via Ollama (OmniDocBench V1.5 #1 at 94.62, 1.6GB, Apache-2.0)
  Qwen2.5-VL fallback. No third-country flow. Zero per-doc cost.

Outputs per-document JSON with provenance block, extracted fields, per-field
confidence, voter agreement, verifier verdicts, schema-validation results,
cross-field consistency, and final pass/bounce decision against 0.98 gate.

Usage:
  python scripts/extract.py --set 00-test
  python scripts/extract.py --doc fixtures/00-test/samples/test-invoice-001.png
  python scripts/extract.py --set 06-sec-edgar --pipeline tesseract-only --no-verifier
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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

SCHEMA_VERSION = "0.4"


@dataclass
class IngestProvenance:
    """Stage 1 — VATCA s.84(3) compliant provenance block.

    `original_form` flags whether the source was digitally born or scanned —
    Section 84(3) of the VAT Consolidation Act 2010 requires Customers to retain
    originals on Customer-controlled storage for 6 years; this block lets us
    attest at extraction time which form we received.
    """
    original_form: str                              # digital_native | image_only_pdf | paper_scan | photo
    source_engine: str                              # pdfplumber-0.11.9 | pypdfium2-5.7.1+ocr | cv2.imread+ocr
    file_sha256: str
    file_size_bytes: int
    ingest_timestamp_utc: str                       # RFC3339
    page_count: int
    pdf_metadata: Optional[dict[str, Any]] = None   # Title/Author/CreationDate/...
    pdf_version: Optional[str] = None
    is_tagged: Optional[bool] = None
    vatca_attestation: str = "original-form-preserved"


@dataclass
class IngestResult:
    """Stage 1 output — passed to downstream stages without re-reading the file."""
    provenance: IngestProvenance
    image_array: Any = None                          # numpy array for OCR path
    native_text: Optional[dict[str, Any]] = None     # pdfplumber engine result
    source_path: str = ""


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
class LineItem:
    """v0.4.1 Stage 4b — invoice line item from PaddleOCR PP-StructureV3 / RapidTable.
    Fields all optional because the table parser may yield partial rows."""
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    amount: Optional[float] = None
    raw_cells: list[str] = field(default_factory=list)
    row_idx: int = 0


@dataclass
class DocumentExtraction:
    doc_id: str
    source_path: str
    schema_version: str = SCHEMA_VERSION
    pipeline: str = "docops-local-only-v0.4"
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    provenance: Optional[IngestProvenance] = None
    ocr_engines_used: list[str] = field(default_factory=list)
    verifiers_used: list[str] = field(default_factory=list)
    fields: list[FieldExtraction] = field(default_factory=list)
    line_items: list[LineItem] = field(default_factory=list)
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


# ============================================================================
# Stage 1 — Ingest with provenance (VATCA s.84(3) compliant)
# ============================================================================


def _file_sha256(path: Path, chunk: int = 65_536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ingester:
    """Stage 1 — VATCA s.84(3) compliant doc ingest.

    Native PDFs (digitally born) → pdfplumber path. No OCR run; native text and
    bbox positions are exact. Confidence locked to 1.0.

    Image-only PDFs / scans / photos → pypdfium2 (PDFs) or cv2.imread (images)
    → numpy array → Stage 3 preprocessing → Stage 4 OCR ensemble.

    Provenance is attached to the DocumentExtraction so downstream stages and
    audit logs can prove (1) what form the original was in, (2) when we ingested,
    (3) the unmutated SHA-256 of the bytes we received.
    """

    NATIVE_TEXT_THRESHOLD = 50    # min chars to call a PDF "native"; below → OCR
    DEFAULT_DPI = 300              # render DPI for image_only_pdf path

    def __init__(self, dpi: int = DEFAULT_DPI, native_threshold: int = NATIVE_TEXT_THRESHOLD):
        self.dpi = dpi
        self.native_threshold = native_threshold

    def ingest(self, doc_path: Path) -> IngestResult:
        sha = _file_sha256(doc_path)
        size = doc_path.stat().st_size
        ts = _utc_now_rfc3339()
        suffix = doc_path.suffix.lower()

        if suffix == ".pdf":
            return self._ingest_pdf(doc_path, sha, size, ts)
        if suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"):
            return self._ingest_image(doc_path, sha, size, ts)
        raise ValueError(f"Unsupported file type: {suffix} ({doc_path.name})")

    def _ingest_pdf(self, doc_path: Path, sha: str, size: int, ts: str) -> IngestResult:
        try:
            import pdfplumber
        except ImportError:
            raise RuntimeError("pdfplumber not installed. pip install pdfplumber. See _dep_notes/pdfplumber.md")

        with pdfplumber.open(str(doc_path)) as pdf:
            n_pages = len(pdf.pages)
            metadata = dict(pdf.metadata) if pdf.metadata else {}
            first = pdf.pages[0] if pdf.pages else None
            native_text = (first.extract_text() or "") if first else ""

            if native_text and len(native_text.strip()) >= self.native_threshold:
                # Native-PDF fast path — no OCR
                words = first.extract_words(extra_attrs=["fontname"]) if first else []
                tables = first.extract_tables() if first else []
                native = {
                    "engine": "pdfplumber-0.11.9",
                    "raw_text": native_text,
                    "words": [
                        {
                            "text": w["text"],
                            "confidence": 1.0,                   # native = exact
                            "bbox": [float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])],
                        }
                        for w in words
                    ],
                    "tables": tables,
                }
                prov = IngestProvenance(
                    original_form="digital_native",
                    source_engine="pdfplumber-0.11.9",
                    file_sha256=sha, file_size_bytes=size, ingest_timestamp_utc=ts,
                    page_count=n_pages, pdf_metadata=metadata,
                )
                return IngestResult(
                    provenance=prov, image_array=None,
                    native_text=native, source_path=str(doc_path.resolve()),
                )

        # Native text too short — fall through to render path (image_only_pdf)
        arr = self._render_pdf_first_page(doc_path)
        prov = IngestProvenance(
            original_form="image_only_pdf",
            source_engine="pypdfium2-5.7.1+ocr",
            file_sha256=sha, file_size_bytes=size, ingest_timestamp_utc=ts,
            page_count=n_pages, pdf_metadata=metadata,
        )
        return IngestResult(
            provenance=prov, image_array=arr,
            native_text=None, source_path=str(doc_path.resolve()),
        )

    def _render_pdf_first_page(self, doc_path: Path):
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise RuntimeError("pypdfium2 not installed. pip install pypdfium2. See _dep_notes/pypdfium2.md")
        import cv2
        scale = self.dpi / 72
        with pdfium.PdfDocument(str(doc_path)) as doc:
            if not len(doc):
                raise RuntimeError(f"PDF has zero pages: {doc_path}")
            page = doc.get_page(0)
            try:
                bitmap = page.render(scale=scale, rev_byteorder=True)
                arr = bitmap.to_numpy()      # (H, W, 4) BGRA when rev_byteorder=True
                bitmap.close()
            finally:
                page.close()
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        return arr

    def _ingest_image(self, doc_path: Path, sha: str, size: int, ts: str) -> IngestResult:
        import cv2
        img = cv2.imread(str(doc_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {doc_path}")
        # photo vs scan distinction is heuristic on EXIF / aspect ratio (deferred);
        # default to paper_scan for non-PDF rasters
        prov = IngestProvenance(
            original_form="paper_scan",
            source_engine="cv2.imread+ocr",
            file_sha256=sha, file_size_bytes=size, ingest_timestamp_utc=ts,
            page_count=1,
        )
        return IngestResult(
            provenance=prov, image_array=img,
            native_text=None, source_path=str(doc_path.resolve()),
        )


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
# Stage 4b — Line-item table extraction via RapidTable (v0.4.1)
# ============================================================================
# Per CALLMEIE-DOCAI-V0.4-PRD.md Stage 4 — table layer for invoice line items.
# rapid-table 3.0.2 wraps PaddleOCR PP-Structure ONNX (Apache-2.0) without the
# paddlepaddle 3.x runtime that fails on Windows (PIR onednn ConvertPirAttribute2RuntimeAttribute).
# Per `_dep_notes/rapid_table.md`. ~2.5-3s/page on Vega 8.
#
# Skipped on digital_native PDFs (pdfplumber.extract_tables() is exact and free).


def _flatten_native_tables(tables: list[list[list[str]]]) -> list[list[str]]:
    """pdfplumber.extract_tables() returns list[table] where each table is list[row]
    where each row is list[cell-or-None]. Flatten into list[row] across all tables.
    """
    flat: list[list[str]] = []
    for tbl in tables or []:
        for row in tbl or []:
            flat.append([("" if c is None else str(c)) for c in row])
    return flat


_LINE_ITEM_HEADER_HINTS = (
    "description", "item", "qty", "quantity", "rate", "unit", "value",
)
# Rows containing these keywords are summary rows (subtotal/vat/total) — drop.
_LINE_ITEM_SUMMARY_KEYWORDS = (
    "subtotal", "sub-total", "vat ", "vat:", "total:", "amount due",
    "balance due", "grand total", "tax:", "tax ",
)
_MONEY_REGEX_TABLE = re.compile(
    r"(?:€|EUR)?\s*([\d,]+\.\d{2})|([\d,]+\.\d{2})\s*(?:€|EUR)?", re.IGNORECASE,
)


def _parse_money_cell(s: str) -> Optional[float]:
    if not s:
        return None
    m = _MONEY_REGEX_TABLE.search(s)
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or "").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_qty_cell(s: str) -> Optional[float]:
    if not s:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


class LineItemsExtractor:
    """Stage 4b — invoice line items via RapidTable (PP-Structure ONNX).

    Skipped on digital_native PDFs — pdfplumber.extract_tables() output is
    already wired into the native_text path (see Ingester._build_native_result).
    """

    name = "rapid-table-3.0.2-ppstructure-en"

    def __init__(self):
        self._engine = None
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            from rapid_table import RapidTable, RapidTableInput, ModelType, EngineType
            input_args = RapidTableInput(
                model_type=ModelType.PPSTRUCTURE_EN,
                engine_type=EngineType.ONNXRUNTIME,
                use_ocr=True,
            )
            self._engine = RapidTable(input_args)
            self._available = True
        except Exception as e:
            log.debug("rapid-table init failed: %s", e)
            self._available = False
        return self._available

    def extract(self, image_array) -> list[LineItem]:
        if not self.available():
            return []
        try:
            out = self._engine(image_array)
        except Exception as e:
            log.warning("LineItemsExtractor predict failed: %s", e)
            return []

        if not out.pred_htmls:
            return []
        return self._parse_html_table(out.pred_htmls[0])

    def _parse_html_table(self, html: str) -> list[LineItem]:
        try:
            import bs4
        except ImportError:
            log.warning("bs4 not installed; skipping HTML table parse")
            return []
        soup = bs4.BeautifulSoup(html, "lxml")
        items: list[LineItem] = []
        # Find header-row index via hint keywords; rows after that are line-items.
        rows = soup.find_all("tr")
        header_idx = -1
        for i, tr in enumerate(rows):
            cells = [td.get_text(strip=True).lower() for td in tr.find_all("td")]
            joined = " ".join(cells)
            if sum(1 for hint in _LINE_ITEM_HEADER_HINTS if hint in joined) >= 2:
                header_idx = i
                break

        for i, tr in enumerate(rows):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            cells = [c for c in cells if c]  # drop empty
            if not cells:
                continue
            if i <= header_idx:
                continue
            # Drop summary rows (subtotal / vat / total)
            joined_lower = " ".join(cells).lower()
            if any(kw in joined_lower for kw in _LINE_ITEM_SUMMARY_KEYWORDS):
                continue
            # Drop rows with no money cell — likely header / vendor / metadata.
            money_cells = [(idx, _parse_money_cell(c)) for idx, c in enumerate(cells)]
            money_cells = [(idx, v) for idx, v in money_cells if v is not None]
            if not money_cells:
                continue
            amount = money_cells[-1][1]
            description = max((c for c in cells if not _parse_money_cell(c)),
                              key=len, default="") or None
            qty = _parse_qty_cell(description or "") if description else None
            items.append(LineItem(
                description=description,
                quantity=qty,
                amount=amount,
                raw_cells=cells,
                row_idx=i,
            ))
        return items


# ============================================================================
# Stage 5 — LLM extraction via instructor + Pydantic + Ollama JSON-mode
# ============================================================================
# Per CALLMEIE-DOCAI-V0.4-PRD.md D-V0.4-03 — instructor over regex stubs.
# instructor 1.15.1 + Ollama 0.6.0 (OpenAI-compatible API at /v1).
# Pydantic 2.11.5 schema-validates the output before we trust it.
# Falls back to regex stubs if Ollama unavailable or model unloaded.
# Acts as a third voter candidate alongside Tesseract + RapidOCR regex.


try:
    from pydantic import BaseModel as _PydanticBaseModel, Field as _PydanticField
    _PYDANTIC_OK = True
except ImportError:
    _PYDANTIC_OK = False


if _PYDANTIC_OK:
    class IrishInvoiceFields(_PydanticBaseModel):
        """Pydantic schema for IE invoice / receipt critical-field extraction.

        instructor enforces JSON-mode return matching this schema. Optional
        fields default to None when the field is genuinely missing or unreadable;
        confidence is the model's self-rating 0.0-1.0.
        """
        vendor: Optional[str] = _PydanticField(
            None,
            description="Vendor / supplier / merchant company name (e.g. 'Tesco Ireland Limited').",
        )
        total: Optional[float] = _PydanticField(
            None,
            description="Total amount paid INCLUDING VAT, in EUR. Numeric only, no currency symbol.",
        )
        vat: Optional[float] = _PydanticField(
            None,
            description="VAT amount in EUR (the tax line, NOT the rate%). Numeric only.",
        )
        date: Optional[str] = _PydanticField(
            None,
            description="Invoice / receipt date in DD/MM/YYYY (Irish convention) or YYYY-MM-DD.",
        )
        vendor_confidence: float = _PydanticField(0.0, ge=0.0, le=1.0)
        total_confidence: float = _PydanticField(0.0, ge=0.0, le=1.0)
        vat_confidence: float = _PydanticField(0.0, ge=0.0, le=1.0)
        date_confidence: float = _PydanticField(0.0, ge=0.0, le=1.0)


_LLM_EXTRACT_SYSTEM = """\
You extract critical fields from an Irish invoice or receipt OCR text.
Be strict: return None for any field not clearly present.
Numbers must be numeric (no €/EUR symbol). Dates as DD/MM/YYYY.
Return JSON matching the schema. No prose, no markdown, JSON only.
Common IE patterns:
  - Vendor: typically the all-caps company name in the header (e.g. CHADWICKS LIMITED, Tesco Ireland Limited).
  - Total: "Total", "Amount Due", "Grand Total" — NOT "Subtotal".
  - VAT: line labeled "VAT 23%:", "VAT", "Tax". The MONEY value, not the percentage.
  - Date: any DD/MM/YYYY, DD-MM-YYYY, DD/MM/YY near "Date", "Invoice Date".
"""


class LLMExtractor:
    """Stage 5 — instructor + Pydantic + Ollama JSON-mode field extractor.

    Per `_dep_notes/`:
      ollama 0.6.0 + instructor 1.15.1 + openai 2.0.0 + pydantic 2.11.5
      Ollama exposes OpenAI-compatible API at http://localhost:11434/v1
      Model: glm-ocr:q8_0 (default) — same model as Stage 6 verifier; reuse cuts cold start.

    No third-country flow. Zero per-doc cost. Apache-2.0.
    """

    name = "ollama-instructor-glm-ocr"
    cost_per_doc_eur = 0.0

    def __init__(self, model: str = "glm-ocr:q8_0", endpoint: str = "http://localhost:11434/v1"):
        self.model = model
        self.endpoint = endpoint
        self._client = None
        self._available: Optional[bool] = None

    def available(self) -> bool:
        if self._available is not None:
            return self._available
        if not _PYDANTIC_OK:
            self._available = False
            return False
        try:
            import instructor  # noqa: F401
            from openai import OpenAI  # noqa: F401
        except ImportError:
            log.debug("instructor / openai missing; LLMExtractor disabled")
            self._available = False
            return False
        # Don't probe Ollama here — let the first .extract() call surface the error.
        # We probe Ollama tags using the existing _check_ollama_model helper.
        bare_endpoint = self.endpoint.replace("/v1", "")
        model_substr = self.model.split(":")[0]
        self._available = _check_ollama_model(model_substr, bare_endpoint)
        return self._available

    def _get_client(self):
        if self._client is None:
            import instructor
            from openai import OpenAI
            base = OpenAI(base_url=self.endpoint, api_key="ollama")
            self._client = instructor.from_openai(base, mode=instructor.Mode.JSON)
        return self._client

    def extract(self, ocr_text: str) -> Optional["IrishInvoiceFields"]:
        if not self.available():
            return None
        try:
            client = self._get_client()
            return client.chat.completions.create(
                model=self.model,
                response_model=IrishInvoiceFields,
                messages=[
                    {"role": "system", "content": _LLM_EXTRACT_SYSTEM},
                    {"role": "user", "content": f"OCR text:\n{ocr_text[:6000]}"},
                ],
                temperature=0.1,
                max_retries=2,
                timeout=180,
            )
        except Exception as e:
            log.warning("LLMExtractor failed (fall back to regex): %s", e)
            return None


def llm_to_engine_output(llm_result: "IrishInvoiceFields", engine_name: str) -> dict[str, Any]:
    """Pack LLM-extracted fields into an `OCREngine.extract`-shaped dict so the
    voter and downstream code paths treat it as another OCR engine candidate.
    `confidence` is the model's self-rating per field; `bbox` is None (LLM has
    no spatial grounding without the vision crop step in Stage 6)."""
    raw_text = "\n".join(filter(None, [
        llm_result.vendor,
        f"Total: {llm_result.total}" if llm_result.total is not None else None,
        f"VAT: {llm_result.vat}" if llm_result.vat is not None else None,
        f"Date: {llm_result.date}" if llm_result.date else None,
    ]))
    words = []
    for fid, value, conf in [
        ("vendor", llm_result.vendor, llm_result.vendor_confidence),
        ("total", llm_result.total, llm_result.total_confidence),
        ("vat", llm_result.vat, llm_result.vat_confidence),
        ("date", llm_result.date, llm_result.date_confidence),
    ]:
        if value is not None:
            words.append({
                "text": str(value),
                "confidence": float(conf),
                "bbox": None,
                "field_id": fid,        # marker for _extract_candidate to short-circuit
            })
    return {
        "engine": engine_name,
        "raw_text": raw_text,
        "words": words,
        "_llm_fields": {
            "vendor": (llm_result.vendor, llm_result.vendor_confidence),
            "total": (llm_result.total, llm_result.total_confidence),
            "vat": (llm_result.vat, llm_result.vat_confidence),
            "date": (llm_result.date, llm_result.date_confidence),
        },
    }


# ============================================================================
# Field extraction — regex stubs + LLM short-circuit (v0.4)
# ============================================================================

IE_VAT_REGEX = re.compile(r"\b(?:VAT(?:\s*No\.?)?[:\s]*)?(IE\d{7}[A-Z]{1,2})\b", re.IGNORECASE)
IBAN_REGEX = re.compile(r"\b([A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16})\b")
DATE_REGEX = re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
# Match either €/EUR before or after the number
MONEY_PATTERN = r"(?:€|EUR)\s*([\d,]+\.\d{2})|([\d,]+\.\d{2})\s*(?:€|EUR)"
MONEY_REGEX = re.compile(MONEY_PATTERN, re.IGNORECASE)


def _extract_candidate(text: str, field_id: str) -> tuple[str, Any]:
    if field_id == "vendor":
        # Vendor name lives in the document header region (first ~8 lines on
        # IE invoice / receipt / LoE templates). Earlier rev searched the whole
        # document and picked all-caps body section headers ("SERVICES RENDERED",
        # "ITEMS", "PAYMENT DETAILS") over the actual vendor when the vendor
        # name was mixed-case. v0.4 will replace this with NER.
        # Strip trailing all-caps doc-type words FIRST so "Dr. Smith Clinic
        # RECEIPT" becomes "Dr. Smith Clinic" before keyword filtering.
        doctype_suffix = re.compile(
            r"\s+(RECEIPT|INVOICE|STATEMENT|CREDIT\s+NOTE|LETTER\s+OF\s+ENGAGEMENT|"
            r"VAT\s+INVOICE|TAX\s+INVOICE|FEE\s+NOTE|UTILITY\s+BILL|DELIVERY\s+NOTE|"
            r"BILL|REPORT)\s*$"
        )
        # Body-section / accounting-noise lines that are NEVER the vendor.
        excluded_kws = ("invoice no", "invoice number", "total", "vat reg",
                        "vat exempt", "page", "date", "due date", "subtotal",
                        "amount due", "balance", "tax point", "received",
                        "services rendered", "items", "description", "qty",
                        "unit price", "payment details", "billing period",
                        "transaction", "delivery note ref", "fees")
        head = [ln.strip() for ln in text.splitlines()[:8]]

        def _ok(stripped: str) -> Optional[str]:
            cleaned = doctype_suffix.sub("", stripped).strip()
            if not (4 <= len(cleaned) <= 80):
                return None
            if any(kw in cleaned.lower() for kw in excluded_kws):
                return None
            return cleaned

        # Pass 1 — all-caps header (high-signal vendor brand line).
        for stripped in head:
            if not re.match(r"^[A-Z][A-Z0-9 &.,'-]{3,60}$", doctype_suffix.sub("", stripped).strip()):
                continue
            cleaned = _ok(stripped)
            if cleaned:
                return cleaned, cleaned
        # Pass 2 — mixed-case header.
        for stripped in head:
            if not re.match(r"^[A-Z][A-Za-z0-9 &.,'-]{3,80}$", doctype_suffix.sub("", stripped).strip()):
                continue
            cleaned = _ok(stripped)
            if cleaned:
                return cleaned, cleaned
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
        # LLM extractor short-circuit: if engine has `_llm_fields`, take those directly.
        llm_fields = res.get("_llm_fields")
        if llm_fields:
            per_engine[eng_name] = {
                fid: (str(llm_fields[fid][0]) if llm_fields[fid][0] is not None else "",
                      llm_fields[fid][0])
                for fid in CRITICAL_FIELDS if fid in llm_fields
            }
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
        # Native pdfplumber path = exact extraction, treat as trivially agreed.
        is_native = len(candidates) == 1 and candidates[0][0].startswith("pdfplumber-")
        if is_native:
            full_agreement = True
            ocr_conf = 1.0
        else:
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
    """Stage 7 — HaluGate cross-field validation.

    - vat / total ratio against IE statutory rates (0%, 4.8%, 9%, 13.5%, 23%).
    - vat <= total (sanity invariant; vat > total is always a bounce).
    - date plausibility (1900-2100, regex-validated).
    - bounce_conditions list — if non-empty, apply_gate will fail the doc
      regardless of per-field confidence.
    """
    by_id = {f.field_id: f for f in fields}
    checks: dict[str, Any] = {}
    bounce: list[str] = []

    total = by_id.get("total")
    vat = by_id.get("vat")
    if total and vat and isinstance(total.parsed_value, (int, float)) and isinstance(vat.parsed_value, (int, float)):
        ratio = vat.parsed_value / total.parsed_value if total.parsed_value > 0 else 0
        # IE rates per Revenue: 23% (std), 13.5% (reduced), 9% (tourism/printed), 4.8% (livestock), 0% (zero-rated)
        plausible_rates = (0.0, 0.048, 0.09, 0.135, 0.23)
        net_ratios = [r / (1 + r) for r in plausible_rates]
        plausible = any(abs(ratio - pr) < 0.005 for pr in net_ratios)
        checks["vat_total_ratio"] = round(ratio, 4)
        checks["vat_rate_plausible"] = plausible
        if not plausible and total.parsed_value > 0:
            bounce.append(
                f"vat/total ratio {ratio:.4f} not within ±0.005 of IE rates "
                f"(0%, 4.8%, 9%, 13.5%, 23%)"
            )
        if vat.parsed_value > total.parsed_value:
            bounce.append(f"vat ({vat.parsed_value}) exceeds total ({total.parsed_value})")
    else:
        checks["vat_total_ratio"] = None
        checks["vat_rate_plausible"] = None

    # Date plausibility (already validate_field_schema covers regex/range — surface as cross-field anyway)
    date = by_id.get("date")
    if date and not date.schema_valid and date.parsed_value is not None:
        bounce.append(f"date schema invalid: {date.schema_reason}")

    checks["bounce_conditions"] = bounce
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


HALUGATE_FLOOR = 0.80   # Stage 7 — token-confidence floor; below = warn always


def apply_gate(extraction: DocumentExtraction, threshold: float = CONFIDENCE_GATE) -> None:
    """Stage 7 HaluGate composite gate.

    Per-field composite confidence:
    - voter agreement + schema valid: trust the consensus (verifier was skipped
      by design per research §3.2). final = min(1.0, ocr_conf * 1.05).
    - voter disagreement: verifier weighs in. final = ocr*0.3 + verifier*0.55 +
      schema*0.15.
    - voter disagreement + no verifier: low trust. final = ocr * 0.5.

    Bounce conditions (any one fails the doc):
    - any critical field final_confidence < `threshold` (0.98 default)
    - cross_field_checks.bounce_conditions non-empty (vat ratio implausible,
      vat > total, date invalid)

    Floor (warning, not bounce):
    - any field final_confidence < HALUGATE_FLOOR (0.8) — recorded in
      `bounce_reason` as `[WARN]` even when the gate otherwise passes.
    """
    failing = []
    below_floor = []
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
        if f.final_confidence < HALUGATE_FLOOR:
            below_floor.append(f"{f.field_id}={f.final_confidence:.2f}")

    cross_bounces = list(extraction.cross_field_checks.get("bounce_conditions", []) or [])

    extraction.gate_passed = (not failing) and (not cross_bounces)

    parts = []
    if failing:
        parts.append("below-threshold: " + ", ".join(failing))
    if cross_bounces:
        parts.append("cross-field: " + "; ".join(cross_bounces))
    if below_floor and not failing:
        parts.append(f"halugate-floor<{HALUGATE_FLOOR}: " + ", ".join(below_floor) + " [WARN]")
    extraction.bounce_reason = " | ".join(parts) if parts else None


# ============================================================================
# Main pipeline
# ============================================================================


def extract_one(
    doc_path: Path,
    ensemble: OCREnsemble,
    verifier: Optional[Verifier],
    ingester: Ingester,
    llm_extractor: Optional[LLMExtractor] = None,
    line_items_extractor: Optional[LineItemsExtractor] = None,
) -> DocumentExtraction:
    extraction = DocumentExtraction(
        doc_id=doc_path.stem,
        source_path=str(doc_path.resolve()),
    )

    # Stage 1 — ingest with provenance
    log.info("Stage 1 ingest: %s", doc_path.name)
    ingest = ingester.ingest(doc_path)
    extraction.provenance = ingest.provenance
    log.info(
        "  original_form=%s source=%s sha256=%s pages=%d",
        ingest.provenance.original_form,
        ingest.provenance.source_engine,
        ingest.provenance.file_sha256[:12] + "…",
        ingest.provenance.page_count,
    )

    if ingest.native_text is not None:
        # Native PDF fast path — pdfplumber output IS the OCR output
        log.info("Stage 4 SKIP — native PDF, using pdfplumber as single engine")
        ocr_output = {ingest.native_text["engine"]: ingest.native_text}
        extraction.ocr_engines_used = [ingest.native_text["engine"]]
        # Native tables come from pdfplumber.extract_tables() at ingest time.
        native_tables = ingest.native_text.get("tables") or []
        for row_idx, row in enumerate(_flatten_native_tables(native_tables)):
            cells = [c.strip() if c else "" for c in row]
            if not any(cells):
                continue
            money_cells = [_parse_money_cell(c) for c in cells]
            amount = next((m for m in reversed(money_cells) if m is not None), None)
            description = max((c for c in cells if not _parse_money_cell(c)),
                              key=len, default="") or None
            extraction.line_items.append(LineItem(
                description=description,
                amount=amount,
                raw_cells=cells,
                row_idx=row_idx,
            ))
    else:
        # Stage 3 + 4 — preprocess + OCR ensemble
        log.info("Stage 3 preprocess: %s", doc_path.name)
        preprocessed = preprocess_image(ingest.image_array)
        log.info("Stage 4 OCR: %s", doc_path.name)
        ocr_output = ensemble.run(doc_path, preprocessed)
        extraction.ocr_engines_used = [e.name for e in ensemble.engines]
        # Stage 4b — line items via RapidTable / PP-Structure (image path only)
        if line_items_extractor is not None and line_items_extractor.available():
            log.info("Stage 4b line items: %s (%s)", doc_path.name, line_items_extractor.name)
            try:
                extraction.line_items = line_items_extractor.extract(ingest.image_array)
                log.info("  found %d line items", len(extraction.line_items))
            except Exception as e:
                log.warning("Line-items extraction failed: %s", e)
                extraction.errors.append(f"line_items: {e}")

    # Stage 5a — instructor + Pydantic + Ollama JSON-mode (LLM-based extract).
    # Joins ocr_output as a third voter candidate alongside Tesseract + RapidOCR.
    # SKIP on digital_native PDFs — pdfplumber output is exact (conf 1.0), and
    # LLM call would add ~30-78s on Vega 8 without accuracy lift (v0.4.1).
    is_digital_native = (
        ingest.provenance is not None
        and ingest.provenance.original_form == "digital_native"
    )
    if llm_extractor is not None and llm_extractor.available() and not is_digital_native:
        combined_text = "\n".join(
            r.get("raw_text", "") for r in ocr_output.values() if isinstance(r, dict)
        )
        log.info("Stage 5a LLM extract: %s (model=%s)", doc_path.name, llm_extractor.model)
        llm_result = llm_extractor.extract(combined_text)
        if llm_result is not None:
            ocr_output[llm_extractor.name] = llm_to_engine_output(llm_result, llm_extractor.name)
            extraction.ocr_engines_used.append(llm_extractor.name)
    elif is_digital_native and llm_extractor is not None:
        log.info("Stage 5a SKIP — digital_native PDF, pdfplumber already exact (conf 1.0)")

    # Stage 5b — voter + regex fallback (LLM short-circuits regex when present)
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
    ap.add_argument("--ingest-dpi", type=int, default=300,
                    help="Render DPI for image_only PDFs (Stage 1 fallback path)")
    ap.add_argument("--llm-extract", dest="llm_extract", action="store_true",
                    default=True, help="Stage 5 LLM extract via instructor + Ollama (default ON)")
    ap.add_argument("--no-llm-extract", dest="llm_extract", action="store_false",
                    help="Disable Stage 5 LLM extract (regex stubs only)")
    ap.add_argument("--llm-extract-model", default="glm-ocr:q8_0",
                    help="Ollama model for LLM extract (default: glm-ocr:q8_0)")
    ap.add_argument("--line-items", dest="line_items", action="store_true",
                    default=True, help="Stage 4b line-item extract via RapidTable (default ON)")
    ap.add_argument("--no-line-items", dest="line_items", action="store_false",
                    help="Disable Stage 4b line-item extract")
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
    log.info("Poppler path: %s (legacy fallback only — Stage 1 uses pypdfium2)", POPPLER_PATH or "—")

    ingester = Ingester(dpi=args.ingest_dpi)
    log.info("Ingester: pypdfium2 + pdfplumber (DPI=%d, native_threshold=%d)",
             ingester.dpi, ingester.native_threshold)

    llm_extractor: Optional[LLMExtractor] = None
    if args.llm_extract:
        candidate = LLMExtractor(model=args.llm_extract_model)
        if candidate.available():
            log.info("LLMExtractor: %s (instructor + Ollama JSON-mode)", candidate.name)
            llm_extractor = candidate
        else:
            log.warning("LLMExtractor model unavailable (%s); falling back to regex stubs",
                        args.llm_extract_model)

    line_items_extractor: Optional[LineItemsExtractor] = None
    if args.line_items:
        candidate = LineItemsExtractor()
        if candidate.available():
            log.info("LineItemsExtractor: %s", candidate.name)
            line_items_extractor = candidate
        else:
            log.warning("LineItemsExtractor unavailable (rapid-table import failed)")

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
            ext = extract_one(doc, ensemble, verifier, ingester, llm_extractor, line_items_extractor)
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
