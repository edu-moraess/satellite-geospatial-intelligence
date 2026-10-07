# VYRA GEOCORE — Recovery Rebuild v1

This branch contains the reproducible recovery pipeline for the GEOCORE dataset rebuild.

Storage policy:
- GitHub: code, manifests, checkpoints, audits, hashes and metadata.
- Google Drive: large source archives, raster data, patch shards and other heavy artifacts.

Execution policy:
1. Validate before processing.
2. Persist every completed gate.
3. Compute SHA-256 for every persisted artifact.
4. Never overwrite historical checkpoints.
5. Never execute split or training before all required gates PASS.

Branch: `rebuild/geocore-v1`
