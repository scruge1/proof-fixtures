#!/usr/bin/env python3
"""
build_corpus.py — populate hero-set fixtures from public-domain sources.

Downloads representative documents (~5-20 per set) into
fixtures/{set}/samples/. Files are gitignored — corpus rebuilds reproducibly
from this script.

Each set has its own fetcher because the source APIs/structures differ
(NAI uses search forms, EDGAR has REST, Companies House has filings API,
Project Gutenberg has direct download URLs).

This is v0.1 — fetchers are skeletons that pull the FIRST-PAGE-friendly
public sample URLs documented per fixture set. Full crawls + auth-gated
sources (NAI Calendars 02, 1911 Census 07) are gated on permission and
deferred to v0.2.

Usage:
  python scripts/build_corpus.py --set 01-1900-us-census
  python scripts/build_corpus.py --all
  python scripts/build_corpus.py --set 06-sec-edgar --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_corpus")

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "fixtures"

# ----------------------------------------------------------------------------
# Per-set source definitions
# ----------------------------------------------------------------------------

# 01 — 1900 US Census
# https://www.archives.gov/research/census/1900 — sample population schedules
# (open public-domain; pre-1924). Direct image URLs from NARA's online catalog.
US_CENSUS_1900_SAMPLES = [
    # NARA scanned population schedules — these are illustrative URLs.
    # Real implementation should query the NARA Catalog API
    # (https://catalog.archives.gov/api/v1) for population-schedule scans.
    # Placeholder list documents the sourcing pattern; v0.2 wires the API.
    # "https://catalog.archives.gov/OpaAPI/media/.../1900-census-page-001.jpg",
]

# 03 — UK Companies House pre-2000 returns
# https://www.gov.uk/government/organisations/companies-house — Crown Copyright
# OGL — historical filings via the Find-a-Company API.
COMPANIES_HOUSE_SAMPLES = [
    # Pulled via Companies House API: GET /company/{number}/filing-history
    # Filter: type IN (AR, AA, AP01) AND date < 2000-01-01
    # Real impl: build_companies_house_corpus(api_key)
]

# 04 — Ellis Island manifests
# https://heritage.statueofliberty.org/ — public domain, pre-1924
ELLIS_ISLAND_SAMPLES = [
    # Direct image URLs from Ellis Island Foundation public archive
    # https://www.archives.gov/research/immigration/passenger-arrival
]

# 05 — Project Gutenberg historical filings
# Free download by ebook ID. Format: https://www.gutenberg.org/files/{id}/{id}-pdf.pdf
GUTENBERG_SAMPLES = [
    # Pick texts containing tabular financial data, articles of association, etc.
    # Example seed: "The Articles of Association" (id varies by edition)
    # ("https://www.gutenberg.org/files/{id}/{id}-pdf.pdf", "{slug}.pdf"),
]

# 06 — SEC EDGAR 1990s 10-K filings
# https://www.sec.gov/cgi-bin/browse-edgar — public domain US gov filings
# Real fetcher uses: GET https://data.sec.gov/submissions/CIK{cik}.json
EDGAR_SAMPLES = [
    # Sample CIKs from 1990s-active public companies; their 10-K HTML/PDF
    # fetched via https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}
]


# ----------------------------------------------------------------------------
# Generic fetcher
# ----------------------------------------------------------------------------


def download(url: str, dest: Path, timeout: int = 30) -> bool:
    """Polite single-file download. Skips if dest already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("skip (cached): %s", dest.name)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import requests
    except ImportError:
        log.error("requests not installed — pip install requests")
        return False
    headers = {
        "User-Agent": "Callmeie-Document-Ops-Benchmark/0.1 (https://callmeie.ie/docs/; hello@callmeie.ie)",
        "Accept": "application/pdf, image/*, text/html",
    }
    try:
        log.info("GET %s", url)
        with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        time.sleep(0.5)  # politeness — don't hammer source servers
        log.info("Saved %s (%d bytes)", dest.name, dest.stat().st_size)
        return True
    except Exception as e:
        log.warning("Download failed %s: %s", url, e)
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


# ----------------------------------------------------------------------------
# Per-set builders (v0.1 stubs — wire real sources in v0.2)
# ----------------------------------------------------------------------------


def build_set_01_census(limit: int) -> int:
    """1900 US Census — NARA Catalog API (https://catalog.archives.gov/api/v1).
    v0.1: documents the sourcing pattern; live API integration deferred."""
    set_dir = FIXTURES_DIR / "01-1900-us-census" / "samples"
    set_dir.mkdir(parents=True, exist_ok=True)
    log.info("Set 01: NARA Catalog API integration is v0.2 — see PROVENANCE.md.")
    return 0


def build_set_02_nai_wills(limit: int) -> int:
    """NAI Calendars of Wills — gated on NAI permission email (D22 from §12).
    v0.1: skip until permission@nationalarchives.ie response received."""
    log.info("Set 02: gated on NAI permission email — skipping (re-run v0.2 after permission).")
    return 0


def build_set_03_companies_house(limit: int) -> int:
    """UK Companies House filing-history API. Free with API key.
    v0.1: documents sourcing; live integration deferred to v0.2."""
    log.info("Set 03: Companies House filing-history API integration is v0.2.")
    log.info("Source: https://developer.company-information.service.gov.uk/")
    return 0


def build_set_04_ellis_island(limit: int) -> int:
    """Ellis Island Foundation passenger manifests — public domain pre-1924."""
    log.info("Set 04: Ellis Island archive integration is v0.2.")
    log.info("Source: https://heritage.statueofliberty.org/")
    return 0


def build_set_05_gutenberg(limit: int) -> int:
    """Project Gutenberg UK 1900s company filings / articles of association."""
    log.info("Set 05: Project Gutenberg corpus selection deferred to v0.2.")
    log.info("Source: https://www.gutenberg.org/  (search: 'company' OR 'articles of association' filtered to <1924).")
    return 0


def build_set_06_edgar(limit: int) -> int:
    """SEC EDGAR 10-K filings 1993-2000. Polite fetch via official API."""
    set_dir = FIXTURES_DIR / "06-sec-edgar" / "samples"
    set_dir.mkdir(parents=True, exist_ok=True)
    seed_ciks = [320193, 789019, 1018724, 1067983, 1018230]  # Apple, Microsoft, Amazon, Berkshire, eBay
    saved = 0
    for cik in seed_ciks[:limit]:
        # SEC EDGAR submissions JSON — public, no auth, polite UA required
        url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
        dest = set_dir / f"cik-{cik:010d}-submissions.json"
        if download(url, dest):
            saved += 1
    log.info("Set 06: %d submission JSONs fetched. v0.2 will pull individual 10-Ks from index files.", saved)
    return saved


def build_set_07_ireland_census(limit: int) -> int:
    """1911 Ireland Census — gated on NAI permission (D22)."""
    log.info("Set 07: gated on NAI permission email — skipping.")
    return 0


SET_BUILDERS = {
    "01-1900-us-census": build_set_01_census,
    "02-nai-calendars-of-wills": build_set_02_nai_wills,
    "03-uk-companies-house-pre2000": build_set_03_companies_house,
    "04-ellis-island-manifests": build_set_04_ellis_island,
    "05-project-gutenberg-filings": build_set_05_gutenberg,
    "06-sec-edgar": build_set_06_edgar,
    "07-1911-ireland-census": build_set_07_ireland_census,
}


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="set_slug", help="One of: " + ", ".join(SET_BUILDERS))
    ap.add_argument("--all", action="store_true", help="Build all 7 sets")
    ap.add_argument("--limit", type=int, default=20, help="Max docs per set (default: 20)")
    args = ap.parse_args()

    if not args.set_slug and not args.all:
        ap.error("Pass --set <slug> or --all")

    target_sets = list(SET_BUILDERS) if args.all else [args.set_slug]
    total = 0
    for s in target_sets:
        builder = SET_BUILDERS.get(s)
        if not builder:
            log.error("Unknown set: %s", s)
            continue
        log.info("=== Building set %s ===", s)
        n = builder(args.limit)
        total += n
        log.info("=== Set %s: %d files saved ===", s, n)

    log.info("Total files fetched: %d", total)
    if total == 0:
        log.warning(
            "v0.1 ships sourcing patterns + permission-gated stubs. Live API "
            "integrations land in v0.2 after Adam confirms NAI permissions + "
            "Companies House API key."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
