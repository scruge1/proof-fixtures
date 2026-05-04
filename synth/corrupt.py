"""Step 4 corruption pipeline — Augraphy 30/45/20/5 split.

Per PDR v0.2 §5 + Codex §A: synthetic invoice corruption distribution
            clean       30%
            light       45%
            heavy       20%
            very-heavy   5%

Bucket → Augraphy stack mapping locked here. Each call to `corrupt()` takes
a clean PDF (rendered by `synth/render.py`) and a target bucket, rasterises
the PDF page(s) to BGR np.ndarray, applies the bucket's Augraphy pipeline,
and writes the corrupted output as a PDF (single-page, image-embedded).

CONTRACT (downstream):
  - Caller passes (in_pdf_path, bucket, out_pdf_path, seed).
  - Caller is responsible for choosing the bucket per `bucket_for_seed(seed)`
    weighted draw.
  - Output PDF is image-only (NOT digital_native); flag in ground-truth
    `original_form` accordingly: `image_only_pdf` for light/heavy/very-heavy.

Design notes:
  - Augraphy expects np.ndarray BGR uint8.
  - PIL → np: `np.array(pil_img)` returns RGB; `cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)`.
  - Reverse for write-back: `cv2.cvtColor(out, cv2.COLOR_BGR2RGB)` → `Image.fromarray(rgb)`.
  - `numba` JIT warmup is ~5s on first pipeline call; cache pipelines per bucket.
  - Random seed via `np.random.seed(seed)` BEFORE pipeline call (Augraphy
    uses numpy global RNG).
"""
from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pypdfium2 as pdfium
from PIL import Image
import cv2

# Lazy-import augraphy so unit tests / non-corrupt callers don't pay the
# numba JIT warmup unless they actually need corruption.
_AUGRAPHY_LOADED = False
_PIPELINE_CACHE: dict[str, "AugraphyPipeline"] = {}


Bucket = Literal["clean", "light", "heavy", "very-heavy"]


# ─── Bucket weights (per PDR v0.2 §5) ────────────────────────────────────────

BUCKET_WEIGHTS: dict[Bucket, float] = {
    "clean":       0.30,
    "light":       0.45,
    "heavy":       0.20,
    "very-heavy":  0.05,
}


def bucket_for_seed(seed: int) -> Bucket:
    """Deterministic bucket selection per seed. Uses weighted-cumulative draw
    so the same seed always maps to the same bucket."""
    rng = random.Random(seed)
    r = rng.random()
    cum = 0.0
    for bucket, w in BUCKET_WEIGHTS.items():
        cum += w
        if r < cum:
            return bucket
    return "very-heavy"  # rounding tail


# ─── Augraphy pipeline cache ─────────────────────────────────────────────────


def _build_pipeline(bucket: Bucket):
    """Build a pipeline for the given bucket. Cached per bucket — numba JIT
    warmup runs once per pipeline (not per doc).

    Augraphy 8.2.6 API: augmentations applied sequentially in their phase.
    `Jpeg` class (NOT `JpegCompression`). `OneOf` removed in 8.x — apply
    augmentations unconditionally and rely on phase-ordering for variety.

    Imports are at function-top because Python treats `from X import Y`
    INSIDE a conditional as a local binding for the whole function scope —
    second-call lookup fails with UnboundLocalError if the import is gated.
    Module cache makes subsequent calls cheap regardless.
    """
    from augraphy import AugraphyPipeline
    from augraphy.augmentations import (
        BadPhotoCopy, BleedThrough, DirtyDrum, Folding, Geometric,
        Jpeg, NoiseTexturize,
    )

    if bucket == "clean":
        return None  # caller short-circuits (no pipeline needed)

    if bucket == "light":
        # Mild scan-quality augmentation: slight skew + light JPEG.
        return AugraphyPipeline(
            ink_phase=[],
            paper_phase=[],
            post_phase=[
                Geometric(rotate_range=(-2, 2)),
                Jpeg(quality_range=(80, 92)),
            ],
        )

    if bucket == "heavy":
        # Aggressive: heavy skew, low JPEG, paper texture, ink bleed-through.
        return AugraphyPipeline(
            ink_phase=[BleedThrough()],
            paper_phase=[NoiseTexturize()],
            post_phase=[
                Geometric(rotate_range=(-8, 8)),
                Folding(fold_count=2),
                Jpeg(quality_range=(35, 60)),
                DirtyDrum(),
            ],
        )

    if bucket == "very-heavy":
        # Orientation/occlusion robustness ONLY (Codex §A).
        return AugraphyPipeline(
            ink_phase=[BleedThrough()],
            paper_phase=[NoiseTexturize()],
            post_phase=[
                Geometric(rotate_range=(-90, 90)),
                BadPhotoCopy(),
                Jpeg(quality_range=(30, 50)),
            ],
        )

    raise ValueError(f"Unknown bucket {bucket!r}")


def _get_pipeline(bucket: Bucket):
    if bucket == "clean":
        return None
    if bucket not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[bucket] = _build_pipeline(bucket)
    return _PIPELINE_CACHE[bucket]


# ─── PDF → image → corrupt → PDF ─────────────────────────────────────────────


@dataclass(frozen=True)
class CorruptionResult:
    bucket: Bucket
    in_path: Path
    out_path: Path
    page_count: int
    file_size_bytes: int


def render_page_to_bgr(in_pdf: Path, page_idx: int = 0, dpi: int = 200) -> np.ndarray:
    """Rasterise one page of a PDF to a BGR uint8 ndarray (Augraphy input format)."""
    pdf = pdfium.PdfDocument(str(in_pdf))
    page = pdf[page_idx]
    bitmap = page.render(scale=dpi / 72)
    pil_img = bitmap.to_pil().convert("RGB")
    rgb = np.array(pil_img)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def bgr_to_pdf(bgr: np.ndarray, out_pdf: Path, dpi: int = 200) -> int:
    """Write a BGR image as a single-page PDF. Returns file size in bytes.

    Uses PIL's PDF writer (no GTK / WeasyPrint needed — pure-PIL). Output is
    image-only, NOT digital_native. Caller must update ground-truth
    `original_form` to `image_only_pdf` accordingly.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(str(out_pdf), "PDF", resolution=dpi)
    return out_pdf.stat().st_size


def corrupt(
    in_pdf: Path,
    bucket: Bucket,
    out_pdf: Path,
    seed: int,
    dpi: int = 200,
) -> CorruptionResult:
    """Apply bucket's corruption pipeline to a PDF. Writes corrupted PDF to
    out_pdf. Returns CorruptionResult metadata."""
    np.random.seed(seed)  # Augraphy uses numpy global RNG

    if bucket == "clean":
        # Just copy the input (no corruption); maintain the bucket flag in
        # ground-truth for downstream tracking.
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        out_pdf.write_bytes(in_pdf.read_bytes())
        return CorruptionResult(
            bucket=bucket, in_path=in_pdf, out_path=out_pdf,
            page_count=_count_pages(in_pdf),
            file_size_bytes=out_pdf.stat().st_size,
        )

    pipeline = _get_pipeline(bucket)
    if pipeline is None:
        raise RuntimeError(f"No pipeline for bucket {bucket!r}")

    # For multi-page PDFs (e.g. bank_statement), corrupt each page then
    # re-stitch. v0.4.2-step2 prototype: only first page corrupted; multi-page
    # support deferred to v0.4.3 once Step 9 baseline establishes which
    # multi-page templates need it.
    bgr_in = render_page_to_bgr(in_pdf, page_idx=0, dpi=dpi)
    bgr_out = pipeline(bgr_in)
    file_size = bgr_to_pdf(bgr_out, out_pdf, dpi=dpi)

    return CorruptionResult(
        bucket=bucket, in_path=in_pdf, out_path=out_pdf,
        page_count=1, file_size_bytes=file_size,
    )


def _count_pages(pdf_path: Path) -> int:
    pdf = pdfium.PdfDocument(str(pdf_path))
    return len(pdf)


# ─── Smoke entry point ──────────────────────────────────────────────────────


def smoke_test() -> int:
    """Run a 4-bucket smoke test on the tradesman_rct prototype output.

    Reads `corpus/synth-prototype/tradesman_rct/synth-03043cc28092.pdf` and
    emits 4 corrupted variants under `corpus/synth-corrupted-smoke/`."""
    repo_root = Path(__file__).resolve().parent.parent
    in_pdf = repo_root / "corpus" / "synth-prototype" / "tradesman_rct" / "synth-03043cc28092.pdf"
    if not in_pdf.exists():
        print(f"SMOKE FAIL: source PDF not found at {in_pdf}", flush=True)
        return 1
    out_dir = repo_root / "corpus" / "synth-corrupted-smoke"
    for i, bucket in enumerate(("clean", "light", "heavy", "very-heavy"), start=1):
        out_pdf = out_dir / f"smoke-{bucket}.pdf"
        result = corrupt(in_pdf, bucket, out_pdf, seed=42 + i)
        print(f"  OK {bucket:12s} -> {out_pdf.name} {result.file_size_bytes//1024}K", flush=True)
    print(f"\nWrote 4 smoke-corrupted PDFs to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(smoke_test())
