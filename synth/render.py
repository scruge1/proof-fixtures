"""HTML+CSS → PDF renderer using WeasyPrint 68 + GTK3 runtime.

Per PDR v0.2 §7 Step 3. Critical Windows wiring:

    GTK3 runtime (libgobject-2.0-0.dll, libpango-1.0-0.dll, libcairo-2.dll, ...)
    is installed at C:\\Program Files\\GTK3-Runtime Win64\\bin via tschoonj
    winget package. We MUST call os.add_dll_directory() BEFORE importing
    weasyprint, else the cffi loader picks up Tesseract-OCR's stale GTK
    DLLs and fails with OSError 0x7e (libgobject-2.0-0.dll).

Render flow:
    template_family → load .html + .css from templates/ → Jinja2 fill →
    WeasyPrint render → PDF bytes → write to disk → return sha256 + size.
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ─── GTK3 DLL discovery + fontconfig wiring ──────────────────────────────────

GTK3_DIR = Path(r"C:\Program Files\GTK3-Runtime Win64")
GTK3_BIN_DIR = GTK3_DIR / "bin"
GTK3_FONTCONFIG_FILE = GTK3_DIR / "etc" / "fonts" / "fonts.conf"


def _ensure_gtk3_dlls() -> None:
    """Prepend GTK3 bin to DLL search path AND set FONTCONFIG_FILE BEFORE
    WeasyPrint loads its cffi bindings. Idempotent — safe to call repeatedly.

    Raises RuntimeError if GTK3 isn't installed (rather than silently failing
    later inside cffi with the cryptic 0x7e error)."""
    if sys.platform != "win32":
        return
    if not GTK3_BIN_DIR.exists():
        raise RuntimeError(
            f"GTK3 runtime not found at {GTK3_BIN_DIR}. "
            f"Install via: winget install --id tschoonj.GTKForWindows "
            f"--silent --accept-source-agreements --accept-package-agreements "
            f"(needs admin)."
        )
    if not (GTK3_BIN_DIR / "libgobject-2.0-0.dll").exists():
        raise RuntimeError(
            f"GTK3 dir exists but libgobject-2.0-0.dll missing — install corrupt? {GTK3_BIN_DIR}"
        )
    os.add_dll_directory(str(GTK3_BIN_DIR))
    # Point fontconfig at GTK3's bundled fonts.conf so font discovery works.
    if GTK3_FONTCONFIG_FILE.exists() and "FONTCONFIG_FILE" not in os.environ:
        os.environ["FONTCONFIG_FILE"] = str(GTK3_FONTCONFIG_FILE)
        os.environ["FONTCONFIG_PATH"] = str(GTK3_FONTCONFIG_FILE.parent)


_ensure_gtk3_dlls()

# Now safe to import
from jinja2 import Environment, FileSystemLoader  # noqa: E402
from weasyprint import HTML, CSS                  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent  # proof-fixtures/
TEMPLATES_DIR = REPO_ROOT / "synth" / "templates"


def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_to_pdf(
    template_family: str,
    content: dict[str, Any],
    out_path: Path,
) -> dict[str, Any]:
    """Render a single document to PDF. Returns provenance metadata dict
    suitable for `ground_truth_provenance` per output_schema.json.

    Args:
        template_family: e.g. "tradesman_rct" — must have matching
            templates/{family}.html + templates/{family}.css files.
        content: dict from content_engine.build(template_family, seed).
        out_path: absolute path where PDF will be written.
    """
    html_template_path = TEMPLATES_DIR / f"{template_family}.html"
    css_template_path = TEMPLATES_DIR / f"{template_family}.css"
    if not html_template_path.exists():
        raise FileNotFoundError(f"Template HTML missing: {html_template_path}")
    if not css_template_path.exists():
        raise FileNotFoundError(f"Template CSS missing: {css_template_path}")

    env = _jinja_env()
    template = env.get_template(f"{template_family}.html")
    rendered_html = template.render(**content)

    css = CSS(filename=str(css_template_path))
    pdf_bytes = HTML(
        string=rendered_html,
        base_url=str(TEMPLATES_DIR),
    ).write_pdf(stylesheets=[css])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)

    return {
        "original_form": "digital_native",
        "source_engine": f"weasyprint-{_weasyprint_version()}+jinja2",
        "file_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
        "file_size_bytes": len(pdf_bytes),
        "ingest_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_count": _count_pdf_pages(pdf_bytes),
        "vatca_attestation": "original-form-preserved",
    }


def _weasyprint_version() -> str:
    import weasyprint
    return getattr(weasyprint, "__version__", "unknown")


def _count_pdf_pages(pdf_bytes: bytes) -> int:
    """Count pages in PDF via pypdfium2 (already installed for extract.py)."""
    try:
        import pypdfium2 as pdfium
        from io import BytesIO
        pdf = pdfium.PdfDocument(BytesIO(pdf_bytes))
        return len(pdf)
    except Exception:
        # Fallback: count "/Type /Page" markers in PDF stream (approximate).
        return max(1, pdf_bytes.count(b"/Type /Page") - pdf_bytes.count(b"/Type /Pages"))
