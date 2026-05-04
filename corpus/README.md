# corpus/

DVC-tracked. Hetzner Object Storage remote. The CORE asset of v0.4 — the
correction-driven training corpus that becomes the moat per D-V0.4-04 once
real customer extractions start landing.

## What lives here

| File | Source | Tracked by | Notes |
|---|---|---|---|
| `corrections.jsonl` | `corrections_consumer.py` LISTEN drain | DVC | Append-only. One row per Label Studio correction insert. |
| `extractions.jsonl` | `extract.py --emit-corpus` (v0.4.2) | DVC | Append-only. One row per gate-pass extraction (training-positive). |
| `train_shard.jsonl` | `dvc repro shard_train` | DVC | Curated subset for LoRA training (v0.5). |
| `holdout/` | Adam-curated 100 docs | DVC | Frozen v0 holdout set (D-V0.4-10). |

`.dvc/.gitignore` keeps the actual bytes out of git. `git add corpus/.dvc/*.dvc`
adds the content-hash pointers; `dvc push` ships the bytes to Hetzner.

## DVC remote

`hetzner` remote configured in `.dvc/config`:
- URL: `s3://callmeie-corpus`
- Endpoint: `https://fsn1.your-objectstorage.com` (Nuremberg)
- Region: `fsn1`
- Auth: `.dvc/config.local` (gitignored) OR env vars
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (DVC reuses S3 SDK).

Why Hetzner Object Storage and not HuggingFace private datasets?
HF private dataset stores data in AWS US — hard GDPR/Schrems II blocker for
IE customer data. Hetzner Object Storage stays intra-EEA on Nuremberg.
See `proof-fixtures/research/2026-05-04-vendor-architectures.md` §License Traps.

## Operational

### Adam-keyboard one-time setup (Hetzner Cloud Console)

1. Hetzner Cloud Console -> Object Storage -> Create Bucket
   - Name: `callmeie-corpus`
   - Region: `fsn1` (Nuremberg) — must match DPA v0.4 §8.1 residency
   - Access: Private (default)
2. Generate credentials — save access key + secret to vault.
3. Copy `.dvc/config.local.example` -> `.dvc/config.local`. Paste real keys.
4. Test: `dvc remote test hetzner` (verifies bucket reachable + creds valid).

### First push

```bash
cd proof-fixtures
mkdir -p corpus
# After the first correction lands via corrections_consumer.py:
dvc add corpus/corrections.jsonl
git add corpus/corrections.jsonl.dvc corpus/.gitignore
git commit -m "data(corpus): seed corrections.jsonl"
dvc push
```

### Cron schedule (AX52)

Append + push nightly:
```cron
# /etc/cron.d/callmeie-corpus-push
30 02 * * *  callmeie  cd /opt/callmeie/proof-fixtures && dvc add corpus/corrections.jsonl && dvc push
```

Or wire as a Coolify "Scheduled Task" pointing to `scripts/dvc_push.sh`
(skeleton at `proof-fixtures/scripts/dvc_push.sh`).

## License posture

- DVC: Apache-2.0
- s3fs / aiobotocore: Apache-2.0
- Hetzner Object Storage: not a sub-processor under DPA v0.4 §7 — Hetzner
  itself IS already the listed sub-processor for compute, this is a different
  service surface from the same vendor. No new sub-processor entry needed.

## Cross-refs

- `.dvc/config` — remote config
- `.dvc/config.local.example` — credential template
- `scripts/corrections_consumer.py` — appends to `corrections.jsonl`
- `scripts/dvc_push.sh` — cron-style push helper
- `document-ops-portal/CALLMEIE-DOCAI-V0.4-PRD.md` D-V0.4-07 — Hetzner Object Storage decision
- `proof-fixtures/research/2026-05-04-active-learning-flywheel.md` — flywheel design
