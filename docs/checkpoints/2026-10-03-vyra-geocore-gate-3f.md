# VYRA / GEOCORE — Dataset Audit Checkpoint

**Checkpoint:** 2026-10-03  
**Branch:** `feat/vyra-geocore-dataset-manifest`  
**Purpose:** preserve the exact state of the Sentinel2GlobalLULC v2.1 audit and STAC patch-validation work before continuing.

## Repository baseline

- Repository: `edu-moraess/satellite-geospatial-intelligence`
- Phase 1 baseline: `c72647417ec678000b7e0983039437cff3c05086`
- Phase 1 full suite previously validated: **100 passed**
- Dataset manifest branch previously validated:
  - `tests/test_ml_dataset_manifest.py`: **6 passed**
  - full suite: **125 passed**
- ML feature branch state previously validated: **119 passed**
- Current manifest infrastructure commit: `17c15e8811b1ff18b8c6a7c81549f3b868055a00`

## Sentinel2GlobalLULC v2.1 audit state

Dataset index recovered locally from the CSV archive:

- Total samples: **194,877**
- Classes: **29**
- Countries: **210**
- Coordinates unique: **194,877**
- Repeated coordinates: **0**
- Coordinates assigned to multiple classes: **0**
- Brazil samples: **11,079**
- Brazil represented classes: **10**

Image ID audit:

- dtype: `int64`
- unique Image IDs: **93,465**
- duplicate records by Image ID: **101,412**
- Image ID is not treated as a Sentinel-2/STAC scene identifier.
- Geographic coordinates are the primary spatial bridge.

Spatial-grid audit:

- Multiple tested classes showed median latitude/longitude grid spacing of approximately **0.0201222624 degrees**.
- This is consistent with the documented 224 x 224 pixel / 10 m tile geometry, while physical longitude distance varies with latitude.

Dataset class counts:

| Class | Count |
|---|---:|
| 1 Barren lands | 14,000 |
| 2 Moss and Lichen | 4,656 |
| 3 Grasslands | 8,869 |
| 4 Open Shrublands | 14,000 |
| 5 Close Shrublands | 11,937 |
| 6 Open Deciduous Broadleaf Forests | 4,437 |
| 7 Close Deciduous Broadleaf Forests | 1,348 |
| 8 Dense Deciduous Broadleaf Forests | 14,000 |
| 9 Open Deciduous Needleleaf Forests | 10,438 |
| 10 Close Deciduous Needleleaf Forests | 6,380 |
| 11 Dense Deciduous Needleleaf Forests | 2,880 |
| 12 Open Evergreen Broadleaf Forests | 567 |
| 13 Close Evergreen Broadleaf Forests | 1,258 |
| 14 Dense Evergreen Broadleaf Forests | 14,000 |
| 15 Open Evergreen Needleleaf Forests | 3,914 |
| 16 Close Evergreen Needleleaf Forests | 3,872 |
| 17 Dense Evergreen Needleleaf Forests | 13,991 |
| 18 Mangrove Wetlands | 416 |
| 19 Swamp Wetlands | 487 |
| 20 Marshland Wetlands | 4,205 |
| 21 Marine Water Bodies | 14,000 |
| 22 Continental Water Bodies | 14,000 |
| 23 Permanent Snow | 14,000 |
| 24 Croplands Flooded with Seasonal Water | 2,004 |
| 25 Cereal Irrigated Cropland | 842 |
| 26 Cereal Rainfed Cropland | 1,020 |
| 27 Irrigated Broadleaf Cropland | 353 |
| 28 Rainfed Broadleaf Cropland | 413 |
| 29 Urban and Built-up Areas | 12,590 |

## STAC spatial/temporal bridge

Selected validation point:

- Dataset class: **29 Urban and Built-up Areas**
- Latitude: **-23.532986**
- Longitude: **-46.693710**
- Country: Brazil
- Pixel purity: **100**
- Number of S2 images: **574**

Earth Search item selected:

- Item ID: `S2B_23KLP_20201028_1_L2A`
- Datetime: **2020-10-28 13:18:59.838Z**
- Platform: **Sentinel-2B**
- Processing baseline: **05.00**
- Cloud cover: **0.017233**
- CRS: **EPSG:32723**

Earth Search asset mapping:

- B02 -> `blue`
- B03 -> `green`
- B04 -> `red`
- B08 -> `nir`
- B11 -> `swir16`

Important methodological constraint:

> Sentinel2GlobalLULC is a composite dataset. A single STAC scene must not be treated as an exact reconstruction of the original dataset tile. The current STAC work establishes spatial/temporal availability and enables a future reconstruction experiment.

## Gate 3F — 10 m patch

B04 reference grid:

- CRS: **EPSG:32723**
- Resolution: **10 x 10 m**
- Scene size: **10980 x 10980**
- Bounds: left **300000**, bottom **7290220**, right **409800**, top **7400020**
- Point projected coordinates:
  - X: **327097.7077359791**
  - Y: **7396456.263231723**
- 224 x 224 window:
  - col_off **2597**
  - row_off **244**
  - width **224**
  - height **224**

10 m patch statistics:

| Band | Shape | Dtype | Min | Max | Mean | Resolution |
|---|---|---|---:|---:|---:|---|
| B02 | 224 x 224 | uint16 | 182 | 8792 | 1091.841119 | 10 m |
| B03 | 224 x 224 | uint16 | 195 | 11288 | 1310.113341 | 10 m |
| B04 | 224 x 224 | uint16 | 184 | 10192 | 1537.820751 | 10 m |
| B08 | 224 x 224 | uint16 | 353 | 11049 | 2353.082390 | 10 m |

## B11 correction

Initial extraction incorrectly used a 224 x 224 window on the 20 m grid. That would represent a 4.48 km x 4.48 km footprint and was rejected.

Corrected B11 extraction:

- Asset:
  `https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/23/K/LP/2020/10/S2B_23KLP_20201028_1_L2A/B11.tif`
- CRS: **EPSG:32723**
- Resolution: **20 x 20 m**
- Scene size: **5490 x 5490**
- Same projected center:
  - X: **327097.7077359791**
  - Y: **7396456.263231723**
- Pixel center:
  - col: **1354.8853867989565**
  - row: **178.18683841382153**
- Correct window:
  - col_off **1298**
  - row_off **122**
  - width **112**
  - height **112**
- Shape: **112 x 112**
- dtype: **uint16**
- min: **986**
- max: **10323**
- mean: **2730.915656887755**
- nodata: **0**

This B11 window represents the same nominal **2.24 km x 2.24 km** physical footprint as the 224 x 224 10 m patch.

## Current gate status

**PASS / validated:**
- Dataset CSV recovery
- 29-class inventory
- coordinate uniqueness
- spatial-grid consistency
- Earth Search spatial/temporal retrieval
- STAC asset availability
- 10 m patch extraction
- corrected 20 m B11 extraction

**NOT YET COMPLETE:**
- current-item radiometric scale/offset verification for all five assets
- explicit nodata/validity-mask validation on the patch
- physically aligned resampling of B11 to 10 m, if required by the ML feature contract
- NDVI / NDWI / NDBI reconstruction
- RGB/false-color visual inspection
- validation of temporal-composite reconstruction against Sentinel2GlobalLULC methodology
- decision on direct 29-class supervision versus reconstructed multispectral supervision
- training

## Next execution point

Resume from:

**Gate 3G — radiometric and feature-contract validation of the real STAC patch.**

Do not start model training before Gate 3G and the composite/reconstruction decision are complete.

