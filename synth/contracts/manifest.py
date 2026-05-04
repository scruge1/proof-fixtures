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


Role = Literal["TRAIN", "EVAL", "EVAL-ONLY-PRIVATE", "RED-TEAM-EVAL"]
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
            if lic == "CC-BY-NC-SA-4.0" and role != "EVAL-ONLY-PRIVATE":
                warn = " ⚠️ NC license — must be EVAL-ONLY-PRIVATE per Codex P0-4"
            f.write(f"| {source} | {lic} | {role} | {count} |{warn}\n")
        f.write("\n")
        f.write("**Codex P0-4 enforcement:** any row with license `CC-BY-NC-SA-4.0` "
                "MUST have role `EVAL-ONLY-PRIVATE`. Any other role is a "
                "license-posture violation.\n")


def validate_license_posture(entries: list[CorpusEntry]) -> list[str]:
    """Returns list of violation strings. Empty list = clean."""
    violations = []
    for e in entries:
        if e.license_tag == "CC-BY-NC-SA-4.0" and e.role != "EVAL-ONLY-PRIVATE":
            violations.append(
                f"VIOLATION: {e.doc_id} ({e.source}) license={e.license_tag} "
                f"but role={e.role} — must be EVAL-ONLY-PRIVATE per Codex P0-4"
            )
    return violations
