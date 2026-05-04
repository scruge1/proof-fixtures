"""DVC + license manifest builders for synth corpus + public datasets.

Every doc in the corpus carries a manifest entry: source, license, role
(TRAIN / EVAL / EVAL-ONLY-PRIVATE), commit-hash of generator (synth) or
upstream version (public), MinHash signature for leakage gate.

Per PDR v0.2 §3 + §7 Step 0 + Codex P0-4 (DocILE PRIVATE-ONLY R&D).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


Role = Literal["TRAIN", "EVAL", "EVAL-ONLY-PRIVATE", "RED-TEAM-EVAL", "BACKFILL"]
# BACKFILL = retro-imported pre-v0.4.2 corpus entries; admitted to manifest for
# provenance only. Never used in TRAIN, EVAL, or RED-TEAM splits until promoted
# via explicit role-flip + commit-hash pin. Per D-V0.4.2-21 (Adam 2026-05-04).
LicenseTag = Literal[
    "Apache-2.0",
    "MIT",
    "CC-BY-4.0",
    "CC-BY-NC-SA-4.0",          # eval-only or eval-only-private
    "CDLA-Permissive-1.0",
    "OGL-v3",
    "Public-record",
    "Adam-personal-consent",
    "BSD-3-Clause",
    "CC0-1.0",
]


@dataclass(frozen=True)
class CorpusEntry:
    doc_id: str
    source: str                  # e.g. "synth-tradesman_rct" / "CORD-train" / "DocILE-eval"
    license_tag: LicenseTag
    role: Role
    file_path: str               # relative to repo root
    file_sha256: str
    file_size_bytes: int
    template_family: str | None  # synth only
    corruption_bucket: str | None  # synth only: clean | light | heavy | very-heavy
    commit_hash: str | None      # synth only — generator git rev
    upstream_version: str | None  # public only — dataset version tag
    minhash_hex: str | None      # 16-byte MinHash digest for leakage gate
    created_at: str              # RFC3339 UTC
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _file_sha256(path: Path, chunk: int = 65_536) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _utc_now_rfc3339() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_dvc_manifest(entries: list[CorpusEntry], out_path: Path) -> None:
    """Writes a JSONL manifest of corpus entries. DVC tracks the manifest
    file (the corpus content itself is content-tracked separately by DVC)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")


# Roles that may carry CC-BY-NC-SA-4.0 (or other non-commercial) content.
# EVAL-ONLY-PRIVATE: explicit Codex P0-4 carve-out for DocILE-class private R&D.
# BACKFILL:          provenance-only retro-import (never used in TRAIN/EVAL/RED-TEAM
#                    until role-flipped via explicit promotion, which MUST re-run
#                    validate_license_posture under the new role per D-V0.4.2-21).
_NC_PERMITTED_ROLES: frozenset[str] = frozenset({"EVAL-ONLY-PRIVATE", "BACKFILL"})


def build_license_manifest(entries: list[CorpusEntry], out_path: Path) -> None:
    """Writes a per-source license summary for legal audit. Group entries
    by source + license_tag, count documents, mark role.
    """
    summary: dict[tuple[str, str, str], int] = {}
    for e in entries:
        key = (e.source, e.license_tag, e.role)
        summary[key] = summary.get(key, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Corpus license manifest\n")
        f.write(f"_Generated {_utc_now_rfc3339()}_\n\n")
        f.write("| Source | License | Role | Count |\n")
        f.write("|---|---|---|---|\n")
        for (source, lic, role), count in sorted(summary.items()):
            warn = ""
            if lic == "CC-BY-NC-SA-4.0" and role not in _NC_PERMITTED_ROLES:
                warn = " ⚠️ NC license — must be EVAL-ONLY-PRIVATE or BACKFILL per Codex P0-4 + D-V0.4.2-21"
            f.write(f"| {source} | {lic} | {role} | {count} |{warn}\n")
        f.write("\n")
        f.write("**Codex P0-4 + D-V0.4.2-21 enforcement:** any row with license "
                "`CC-BY-NC-SA-4.0` MUST have role in {`EVAL-ONLY-PRIVATE`, "
                "`BACKFILL`}. Any other role is a license-posture violation. "
                "Promoting a `BACKFILL` row to `TRAIN`/`EVAL`/`RED-TEAM-EVAL` "
                "MUST re-run `validate_license_posture` under the new role.\n")


def validate_license_posture(entries: list[CorpusEntry]) -> list[str]:
    """Returns list of violation strings. Empty list = clean.

    BACKFILL is exempt because retro-import entries are provenance-only and
    never used for TRAIN/EVAL/RED-TEAM. Any role-flip BACKFILL→active must
    re-run this validator under the new role (caller responsibility).
    """
    violations = []
    for e in entries:
        if e.license_tag == "CC-BY-NC-SA-4.0" and e.role not in _NC_PERMITTED_ROLES:
            violations.append(
                f"VIOLATION: {e.doc_id} ({e.source}) license={e.license_tag} "
                f"but role={e.role} — must be EVAL-ONLY-PRIVATE or BACKFILL per Codex P0-4 + D-V0.4.2-21"
            )
    return violations
