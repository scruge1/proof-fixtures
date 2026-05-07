#!/usr/bin/env python3
"""Push gate-failed extractions to Label Studio review queue (Rescue tier).

v0.4.1 Stage 8 second-half daemon, paired with `corrections_consumer.py`.

Doc Ops Rescue + Export tier (€499/mo) promises 10 human rescues per month
with a 3-business-day SLA. The mechanism: any extraction where
`gate_passed = FALSE` AND `tenant.plan = 'rescue_export'` should appear as
a task in that tenant's Label Studio project. Adam clears the queue from
the UI; on save, the existing `/api/corrections` webhook fires + the
existing corrections_consumer appends to the DVC-tracked corpus JSONL.

This script is the missing piece between extraction and Label Studio: it
lists pending rescues + creates Label Studio tasks for each. Idempotent —
checks each extraction has not already been pushed (records its own
push-marker AuditLog row).

Plus a "rescue SLA monitor" mode: counts rescues older than 2 business
days that have not been resolved, and emails Adam if any. This is the
load-bearing alert for the 3-day promise.

Run modes:
    python push_rescues_to_label_studio.py            # one-shot push pending
    python push_rescues_to_label_studio.py --watch    # poll loop, 60s interval
    python push_rescues_to_label_studio.py --sla-check # email Adam on overdue

Env vars:
    DATABASE_URL          (Doc Ops Postgres)
    LABEL_STUDIO_URL      (e.g. https://review.callmeie.ie)
    LABEL_STUDIO_TOKEN    (Token-prefixed bearer for the Adam-super account)
    SLA_BUSINESS_DAYS     (default 3 — Rescue tier SLA)
    SLA_ALERT_EMAIL       (default adam@callmeie.ie)

Schedule:
- Push loop: systemd unit on AX52, restart=always
- SLA monitor: cron 09:00 IST daily, fires only on real overdue rescues
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PORTAL_ROOT = REPO_ROOT.parent / "document-ops-portal"
if str(PORTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(PORTAL_ROOT))

log = logging.getLogger("push_rescues")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def _import_app():
    """Lazy import — avoids needing the portal repo present at module load."""
    from sqlmodel import Session, select  # noqa: F401
    from app.db import engine  # noqa: F401
    from app.models import AuditLog, Extraction, Tenant  # noqa: F401
    return locals()


def _label_studio_token() -> tuple[str, str]:
    base = os.environ.get("LABEL_STUDIO_URL", "").rstrip("/")
    token = os.environ.get("LABEL_STUDIO_TOKEN", "")
    if not (base and token):
        sys.exit("LABEL_STUDIO_URL + LABEL_STUDIO_TOKEN required (see infra/label-studio/README)")
    return base, token


def _project_id_for_tenant(tenant_slug: str, base: str, token: str) -> int | None:
    """Look up Label Studio project_id by name=docops-{slug}."""
    import requests
    name = f"docops-{tenant_slug}"
    r = requests.get(
        f"{base}/api/projects?name={name}",
        headers={"Authorization": f"Token {token}"}, timeout=15,
    )
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        log.warning("no Label Studio project for tenant %s (expected name=%s) — run scripts/provision_tenant.py first", tenant_slug, name)
        return None
    return int(results[0]["id"])


def _push_one_extraction(project_id: int, extraction, base: str, token: str) -> int | None:
    """Create one Label Studio task for the rescue. Returns task_id on success."""
    import requests
    payload = {
        "data": {
            "extraction_id": str(extraction.id),
            "tenant_id": str(extraction.tenant_id),
            "doc_id": getattr(extraction, "doc_id", "") or "",
            "source_path": getattr(extraction, "source_path", "") or "",
            "bounce_reason": extraction.bounce_reason or "",
            "fields": extraction.payload or {},
            "created_at": extraction.created_at.isoformat() if getattr(extraction, "created_at", None) else "",
        },
    }
    r = requests.post(
        f"{base}/api/projects/{project_id}/import",
        headers={"Authorization": f"Token {token}"},
        json=[payload], timeout=30,
    )
    if r.status_code >= 400:
        log.error("LS import failed for ext=%s: %s %s", extraction.id, r.status_code, r.text[:300])
        return None
    body = r.json()
    # /import returns {"task_count": 1, "annotation_count": ...}; for the new
    # task id we need the most recent task in the project, but for our
    # idempotency marker we just record "pushed" with the extraction_id.
    return int(body.get("task_count", 0))


def push_pending(dry_run: bool = False) -> dict[str, int]:
    """Find rescue extractions that have not been pushed yet, push each."""
    g = _import_app()
    Session, select = g["Session"], g["select"]
    engine, AuditLog, Extraction, Tenant = g["engine"], g["AuditLog"], g["Extraction"], g["Tenant"]
    base, token = _label_studio_token()

    with Session(engine) as session:
        rescue_tenants = list(session.exec(
            select(Tenant)
            .where(Tenant.plan == "rescue_export")
            .where(Tenant.suppressed == False)  # noqa: E712
        ).all())
        if not rescue_tenants:
            log.info("no rescue-tier tenants; nothing to push")
            return {"pushed": 0, "skipped": 0, "tenants": 0}

        already_pushed = {
            row.target_id for row in session.exec(
                select(AuditLog).where(AuditLog.action == "rescue_pushed_to_label_studio")
            ).all()
        }

        pushed = skipped = 0
        for tenant in rescue_tenants:
            project_id = _project_id_for_tenant(tenant.slug, base, token)
            if project_id is None:
                continue
            pending = list(session.exec(
                select(Extraction)
                .where(Extraction.tenant_id == tenant.id)
                .where(Extraction.gate_passed == False)  # noqa: E712
                .order_by(Extraction.created_at)
            ).all())
            for ext in pending:
                if str(ext.id) in already_pushed:
                    skipped += 1
                    continue
                if dry_run:
                    log.info("[dry-run] would push extraction %s to project %s", ext.id, project_id)
                    pushed += 1
                    continue
                ok = _push_one_extraction(project_id, ext, base, token)
                if ok is not None:
                    session.add(AuditLog(
                        tenant_id=tenant.id, action="rescue_pushed_to_label_studio",
                        target_type="extraction", target_id=str(ext.id),
                        payload_json={"project_id": project_id},
                    ))
                    session.commit()
                    pushed += 1
                else:
                    skipped += 1
        return {"pushed": pushed, "skipped": skipped, "tenants": len(rescue_tenants)}


def _is_business_day(d: datetime) -> bool:
    return d.weekday() < 5


def _business_days_between(start: datetime, end: datetime) -> int:
    """Count business days strictly between start and end (exclusive on both
    ends — close enough for SLA arithmetic at this scale)."""
    if end <= start:
        return 0
    days = 0
    cursor = start.date()
    while cursor < end.date():
        cursor = cursor + timedelta(days=1)
        if cursor.weekday() < 5:
            days += 1
    return days


def sla_check() -> dict:
    """Email Adam if any rescue is >SLA_BUSINESS_DAYS old without resolution."""
    g = _import_app()
    Session, select = g["Session"], g["select"]
    engine, AuditLog, Extraction, Tenant = g["engine"], g["AuditLog"], g["Extraction"], g["Tenant"]
    sla_days = int(os.environ.get("SLA_BUSINESS_DAYS", "3"))
    alert_to = os.environ.get("SLA_ALERT_EMAIL", "adam@callmeie.ie")

    with Session(engine) as session:
        rescue_tenants = list(session.exec(
            select(Tenant).where(Tenant.plan == "rescue_export").where(Tenant.suppressed == False)  # noqa: E712
        ).all())
        if not rescue_tenants:
            return {"ok": True, "overdue": 0}

        # "Resolved" = the extraction has been re-extracted post-correction OR
        # has gate_passed flipped True. For now: gate_passed flipped True is the
        # only signal we have on a flat schema. (When corrections_consumer
        # fully closes the loop, an `extraction.resolved_at` column will be the
        # canonical signal.)
        now = datetime.now(UTC)
        overdue: list[dict] = []
        for tenant in rescue_tenants:
            pending = list(session.exec(
                select(Extraction)
                .where(Extraction.tenant_id == tenant.id)
                .where(Extraction.gate_passed == False)  # noqa: E712
            ).all())
            for ext in pending:
                bd = _business_days_between(ext.created_at, now)
                if bd > sla_days:
                    overdue.append({
                        "tenant": tenant.slug, "extraction_id": str(ext.id),
                        "bounce_reason": ext.bounce_reason or "",
                        "business_days_late": bd,
                    })

    if overdue:
        # Send via Resend SMTP (already in stack)
        try:
            from app.auth.smtp import send_email  # type: ignore
            body = (
                f"Rescue SLA breach — {len(overdue)} extraction(s) past {sla_days} business days:\n\n"
                + "\n".join(
                    f"  • [{o['tenant']}] {o['extraction_id']} — {o['business_days_late']}bd late — {o['bounce_reason']}"
                    for o in overdue[:30]
                )
                + (f"\n\n…and {len(overdue) - 30} more." if len(overdue) > 30 else "")
                + f"\n\nClear queue: review.callmeie.ie\n"
            )
            send_email(alert_to, f"[RESCUE SLA] {len(overdue)} overdue", body)
        except Exception as e:
            log.exception("could not send SLA alert email: %s", e)
    return {"ok": not overdue, "overdue": len(overdue), "items": overdue[:10]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--watch", action="store_true",
                    help="Poll loop, 60s interval (systemd-friendly)")
    ap.add_argument("--sla-check", action="store_true",
                    help="Email Adam if any rescue is past the business-day SLA")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report without pushing")
    args = ap.parse_args()

    if args.sla_check:
        result = sla_check()
        log.info("sla_check: %s", result)
        return 0 if result["ok"] else 1

    if args.watch:
        log.info("watching for rescue-tier extractions (60s poll)")
        while True:
            try:
                summary = push_pending(args.dry_run)
                if summary["pushed"]:
                    log.info("pushed %d (skipped %d) across %d tenants",
                             summary["pushed"], summary["skipped"], summary["tenants"])
            except KeyboardInterrupt:
                log.info("watcher interrupted")
                return 0
            except Exception:
                log.exception("push loop failed; sleeping then retrying")
            time.sleep(60)

    summary = push_pending(args.dry_run)
    log.info("pushed %d (skipped %d) across %d tenants",
             summary["pushed"], summary["skipped"], summary["tenants"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
