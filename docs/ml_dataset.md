# GEOCORE / VYRA ML Dataset

## Purpose

The GEOCORE land-cover dataset is an auditable geospatial perception dataset
intended to become a training and evaluation input for VYRA.

The dataset is designed around four independent dimensions:

- **Land cover:** the original ESA WorldCover 2021 v200 class code.
- **Geographic domain:** biome, country, continent and coordinates.
- **Climate domain:** a controlled metadata category.
- **Acquisition context:** Sentinel-2 scene and acquisition date.

The manifest does not turn biome or climate into land-cover labels. They are
domain metadata used to measure geographic and environmental diversity and
generalization.

## Label contract

The source label registry preserves all 11 WorldCover classes:

| Code | Class |
| ---: | --- |
| 10 | Tree cover |
| 20 | Shrubland |
| 30 | Grassland |
| 40 | Cropland |
| 50 | Built-up |
| 60 | Bare/sparse vegetation |
| 70 | Snow/ice |
| 80 | Permanent water |
| 90 | Herbaceous wetland |
| 95 | Mangroves |
| 100 | Moss/lichen |

The existing GEOCORE five-class mapping remains a separate experimental
training contract. The manifest never overwrites the original WorldCover code.

## Spatial split

Train/validation/test separation must be based on spatial blocks rather than
random individual pixels. This reduces leakage from neighboring, highly
correlated pixels.

## Dataset construction sequence

1. Discover candidate Sentinel-2 scenes and WorldCover coverage.
2. Build and audit the manifest.
3. Measure class, biome, climate and geographic distributions.
4. Select samples with explicit coverage targets.
5. Download only the selected raster windows.
6. Extract spectral features.
7. Create spatial train/validation/test splits.
8. Train the offline LightGBM baseline.
9. Evaluate by class and by geographic/environmental domain.
10. Register the resulting experimental model for VYRA.

No training metrics are implied by this infrastructure alone.
