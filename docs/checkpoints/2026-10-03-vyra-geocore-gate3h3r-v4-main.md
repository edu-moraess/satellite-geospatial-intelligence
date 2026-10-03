# VYRA GEOCORE — Gate 3H.3R v4 Main Checkpoint

Date: 2026-10-03

## Frozen state

This checkpoint records the validated computational state after the CRS-aware Gate 3H.3R v4 rerun and before the targeted diagnostic of the remaining low-validity sample.

### Gate status

- Gate: 3H.3R
- Version: v4
- Decision: `NOT_EVALUATED`
- ML training: `BLOCKED`
- Gate 3H.4: `BLOCKED`

### Results

- Canonical samples processed: 29/29
- Status: 29/29 `AUDITED`
- Patch geometry: 224x224 for all samples
- Minimum geometry margin: 257 px
- Median geometry margin: 1396 px
- Zero-margin samples: 0
- Negative-margin samples: 0
- Samples with CRS: 29/29
- Unique raster CRS: 22
- Spectral usable fraction — minimum: 0.9289
- Spectral usable fraction — median: 1.0000
- Spectral usable fraction — mean: 0.9975
- Negative reflectance samples: 0

### Radiometric contract

`reflectance = DN_EarthSearch * 0.0001`

Additional offset: `0.0`.

### Outstanding diagnostic

Class 05 — `SrublandClose` has a valid fraction of 0.9289 and requires targeted diagnosis before the gate can be formally evaluated.

Class 20 — `WetlandMarshl` has a valid fraction of 0.9999 and remains documented as a near-complete-validity sample.

### Spatial correction validated

The v4 rerun uses CRS-aware coordinate transformation:

WGS84 (EPSG:4326) → raster native CRS → `ds.index(x, y)` → 224x224 patch.

This resolves the prior failure mode where WGS84 coordinates were passed directly to projected rasters.

### Scope protection

The frozen v4 evidence must not be overwritten by the next diagnostic. Any follow-up analysis must be recorded in a new checkpoint.

No ML training or Gate 3H.4 progression should occur until Gate 3H.3R is formally evaluated.
