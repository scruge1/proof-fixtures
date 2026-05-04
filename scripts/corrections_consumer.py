#!/usr/bin/env python3
"""Corrections consumer — drains Postgres NOTIFY -> DVC-tracked JSONL.

v0.4.1 Stage 8 daemon per CALLMEIE-DOCAI-V0.4-PRD.md D-V0.4-06 / D-V0.4-07.

Subscribes to the `corrections_channel` PostgreSQL channel populated by the
`notify_correction()` trigger added in alembic migration 0002_corrections.
For each NOTIFY, fetches the new correction row and appends it to the
DVC-tracked corpus JSONL. The corpus is what the LoRA / distillation pipeline
trains on — the moat per D-V0.4-04.

Append-only. One JSONL row per correction. Never overwrites.
DVC push runs separately (e.g. nightly cron) to push to Hetzner Object Storage.

Run as systemd unit on AX52 — see infra/label-studio/README-DEPLOY.md.

Env vars:
    DATABASE_URL       postgres://user:pass@host:5432/document_ops_portal
    CORPUS_PATH         absolute path to corrections.jsonl (default: ./corpus/corrections.jsonl)
    POLL_TIMEOUT_SEC   integer, fallback poll interval if NOTIFY misses (default 300)

CLI:
    python corrections_consumer.py            # daemon mode (LISTEN forever)
    python corrections_consumer.py --once     # process pending then exit (for tests)
    python corrections_consumer.py --backfill # replay all corrections (rebuild JSONL)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import select
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

# psycopg 3 (preferred) — falls back to psycopg2 if installed
try:
    import psycopg                          # type: ignore
    from psycopg.rows import dict_row        # type: ignore
    _PSYCOPG_VERSION = 3
except ImportError:
    try:
        import psycopg2                                    # type: ignore
        from psycopg2.extras import RealDictCursor          # type: ignore
        _PSYCOPG_VERSION = 2
    except ImportError:
        sys.stderr.write("psycopg or psycopg2 required: pip install 'psycopg[binary]'\n")
        sys.exit(2)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("corrections_consumer")


CHANNEL = "corrections_channel"


def _connect(dsn: str):
    if _PSYCOPG_VERSION == 3:
        # autocommit required for LISTEN/NOTIFY
        return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)
    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    conn.set_isolation_level(0)  # ISOLATION_LEVEL_AUTOCOMMIT
    return conn


def _fetch_correction(conn, correction_id: str) -> Optional[dict[str, Any]]:
    sql = """
        SELECT
            c.id, c.tenant_id, c.extraction_id, c.user_id, c.field_id,
            c.original_value, c.corrected_value, c.reason, c.evidence_bbox,
            c.label_studio_task_id, c.submitted_at,
            e.doc_id, e.file_sha256, e.original_form, e.source_engine,
            e.schema_version, e.pipeline
        FROM corrections c
        JOIN extractions e ON e.id = c.extraction_id
        WHERE c.id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (correction_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def _stream_pending(conn, since_id: Optional[str] = None) -> Iterator[dict[str, Any]]:
    """Replay all corrections (or those after `since_id`) in submission order."""
    sql = """
        SELECT
            c.id, c.tenant_id, c.extraction_id, c.user_id, c.field_id,
            c.original_value, c.corrected_value, c.reason, c.evidence_bbox,
            c.label_studio_task_id, c.submitted_at,
            e.doc_id, e.file_sha256, e.original_form, e.source_engine,
            e.schema_version, e.pipeline
        FROM corrections c
        JOIN extractions e ON e.id = c.extraction_id
        ORDER BY c.submitted_at ASC, c.id ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        for row in cur:
            yield dict(row)


def _append_jsonl(corpus_path: Path, record: dict[str, Any]) -> None:
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = {
        # Cast UUID/datetime to string for stable JSON
        k: (str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) else v)
        for k, v in record.items()
    }
    with corpus_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(serialized, ensure_ascii=False) + "\n")


def run_daemon(dsn: str, corpus_path: Path, poll_timeout: int) -> None:
    log.info("Connecting to Postgres (%s)...", dsn.split("@")[-1])
    conn = _connect(dsn)
    log.info("LISTEN %s", CHANNEL)
    with conn.cursor() as cur:
        cur.execute(f"LISTEN {CHANNEL}")

    stop = {"flag": False}

    def _on_signal(sig, frame):
        log.info("Signal %s — exiting", sig)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    log.info("Daemon ready — waiting for corrections (corpus=%s)", corpus_path)
    while not stop["flag"]:
        if _PSYCOPG_VERSION == 3:
            # psycopg3 yields Notify objects directly via .notifies() generator
            try:
                gen = conn.notifies(timeout=poll_timeout)
                notify = next(gen, None)
                if notify is None:
                    log.debug("Poll timeout (%ds) — fallback resync", poll_timeout)
                    continue
                payload = notify.payload
            except StopIteration:
                continue
        else:
            r, _, _ = select.select([conn], [], [], poll_timeout)
            if not r:
                log.debug("Poll timeout (%ds) — fallback resync", poll_timeout)
                continue
            conn.poll()
            while conn.notifies:
                payload = conn.notifies.pop(0).payload
                _process_one(conn, payload, corpus_path)
            continue

        # psycopg3 path falls through to here
        _process_one(conn, payload, corpus_path)

    log.info("Closing connection")
    conn.close()


def _process_one(conn, correction_id: str, corpus_path: Path) -> None:
    log.info("NOTIFY correction id=%s", correction_id)
    record = _fetch_correction(conn, correction_id)
    if record is None:
        log.warning("Correction %s not found (deleted?)", correction_id)
        return
    _append_jsonl(corpus_path, record)
    log.info("Appended -> %s (field=%s)", corpus_path.name, record.get("field_id"))


def run_backfill(dsn: str, corpus_path: Path) -> None:
    log.info("Backfill mode — replaying all corrections to %s", corpus_path)
    if corpus_path.exists():
        backup = corpus_path.with_suffix(corpus_path.suffix + f".bak.{int(time.time())}")
        corpus_path.rename(backup)
        log.info("Existing corpus backed up to %s", backup)
    conn = _connect(dsn)
    n = 0
    for record in _stream_pending(conn):
        _append_jsonl(corpus_path, record)
        n += 1
    log.info("Backfill done — %d corrections appended", n)
    conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Corrections consumer (Postgres NOTIFY -> JSONL)")
    ap.add_argument("--once", action="store_true",
                    help="Process current pending notifications then exit (for tests)")
    ap.add_argument("--backfill", action="store_true",
                    help="Replay ALL corrections from DB into a fresh JSONL")
    args = ap.parse_args()

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log.error("DATABASE_URL not set")
        return 2

    corpus_path = Path(os.environ.get("CORPUS_PATH", "./corpus/corrections.jsonl"))
    poll_timeout = int(os.environ.get("POLL_TIMEOUT_SEC", "300"))

    if args.backfill:
        run_backfill(dsn, corpus_path)
        return 0

    if args.once:
        # Run daemon for a short window then exit (smoke test)
        os.environ["POLL_TIMEOUT_SEC"] = "5"
        run_daemon(dsn, corpus_path, 5)
        return 0

    run_daemon(dsn, corpus_path, poll_timeout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
