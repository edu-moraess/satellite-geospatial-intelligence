# GEOCORE

Experimental **Geospatial Computing & Intelligence** platform for **Sentinel-2** Earth Observation analysis: catalog search, AOI-based download, spectral indices, land-cover classification, change detection, and a Geospatial AI inference interface.

Repository: [edu-moraess/satellite-geospatial-intelligence](https://github.com/edu-moraess/satellite-geospatial-intelligence)

---

## Overview

The project implements a modular pipeline for remote-sensing analysis over user-defined areas of interest (AOI):

1. Search the **Sentinel-2 L2A** catalog (Microsoft Planetary Computer STAC).
2. Optionally constrain the search with a **polygon drawn on the map** (or a point + area size).
3. Download required spectral bands (B02, B03, B04, B08, B11) for a selected scene.
4. Validate rasters, build RGB / false-color composites, and compute **NDVI**, **NDWI**, and **NDBI**.
5. Derive **land-cover** classes from those indices.
6. Compare two scenes for **change detection**.
7. Run **object-detection style inference** when a model checkpoint is available, with optional **GeoJSON** export of georeferenced detections.
8. Prepare a gated **Earth Observation deep-learning layer** for multispectral foundation models, with explicit input-band and checkpoint validation.

The application is experimental. Results depend on scene quality, cloud cover, AOI geometry, algorithm parameters, and data availability.

---

## Architecture

The application is organized as a thin Streamlit shell over testable scientific workflows:

\`\`\`
app.py
  ├── Mission Control (ui/mission_control.py)
  ├── Geospatial Operations (src/map_view.py)
  ├── Satellite Archive (ui/catalog.py)
  ├── Active Observation
  ├── Spectral Intelligence
  ├── Land Cover
  ├── Change Intelligence
  ├── Geospatial AI
  └── Processing Pipeline

src/
  ├── domain modules       # catalog, AOI, raster, spectral, detection
  └── workflows/           # orchestration of multi-step scientific analysis

ui/
  ├── mission_control.py   # mission inputs + summary
  ├── catalog.py           # archive presentation + selection
  ├── components.py        # reusable cards and metrics
  ├── layout.py            # structural UI primitives
  ├── status.py            # pipeline state
  └── theme.py             # visual system

tests/
  └── unit + architecture guardrails
\`\`\`

**Design principles**

- \`app.py\` coordinates; it does not implement scientific algorithms.
- \`src/\` remains independent from Streamlit wherever possible.
- One satellite archive is the source of truth for scene selection.
- Active imagery is explicit: select → download → process → analyze.
- AI is checkpoint-gated. A missing checkpoint produces \`MODEL UNAVAILABLE\`; it never fabricates detections.
- UI state and scientific state are kept separate and reset deliberately when the catalog changes.
- The interface uses one dark visual system with consistent surfaces, controls and semantic status colors.

---

## Features

Implemented in the current codebase:

| Feature | Notes |
|---------|--------|
| Sentinel-2 L2A search | STAC via Planetary Computer |
| AOI | Point + area size, or polygon drawn on the map |
| Band download | B02, B03, B04, B08, B11 → GeoTIFF under `data/raw/<SCENE_ID>/` |
| Raster validation | Shape / validity checks before analysis |
| RGB & false-color | Natural color and NIR-based false color |
| NDVI / NDWI / NDBI | Spectral indices with summary statistics |
| Land-cover analysis | Rule-based classification from indices |
| Change detection | Index difference between two scenes |
| Geospatial AI interface | Tiling + detector API; requires a real checkpoint |
| GeoJSON export | Georeferenced detections when transform/CRS are available |
| Pipeline status | Catalog → Imagery → Spectral → Change → AI |

**Not implemented as production models:** model checkpoints are not committed. The registry defines `remote_sensing_baseline` with `checkpoint=None`; inference is designed to fail safely when no weights are present (no fabricated detections).

---

## User Interface

The interface is structured as a single mission workspace rather than duplicated catalog/analysis surfaces:

1. **Mission Control** — sensor, AOI, time window and cloud filter.
2. **Geospatial Operations** — interactive AOI map and telemetry.
3. **Satellite Archive** — one authoritative scene list with explicit selection.
4. **Active Observation** — download state and natural/false-color imagery.
5. **Spectral Intelligence** — NDVI, NDWI and NDBI.
6. **Land Cover** — rule-based spectral classification.
7. **Change Intelligence** — explicit Before/After comparison.
8. **Geospatial AI** — real-checkpoint inference only.
9. **Processing Pipeline** — operational state of the current mission.

The UI is designed to remain readable on desktop and mobile widths without relying on light Streamlit surfaces.

---

## Data Sources

| Source | Usage |
|--------|--------|
| [Microsoft Planetary Computer STAC](https://planetarycomputer.microsoft.com/api/stac/v1) | Catalog search (`pystac-client` + `planetary-computer` signing) |
| Collection `sentinel-2-l2a` | Sentinel-2 Level-2A scenes |
| Local `data/raw/` | Downloaded band GeoTIFFs per scene ID |

Bands used: **B02** (Blue), **B03** (Green), **B04** (Red), **B08** (NIR), **B11** (SWIR).

---

## Project Structure

```
satellite-geospatial-intelligence/
├── app.py                 # Thin Streamlit application shell
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── src/
│   ├── aoi.py             # AOI / bbox helpers from map drawings
│   ├── catalog.py         # STAC search
│   ├── config.py          # Paths, STAC URL, band map
│   ├── downloader.py      # Band download + directory handling
│   ├── geospatial.py      # Band read / alignment
│   ├── raster_validation.py
│   ├── spectral.py        # NDVI / NDWI / NDBI
│   ├── visualization.py   # RGB / false color
│   ├── index_visualization.py
│   ├── classification.py
│   ├── land_cover.py
│   ├── change_detection.py
│   ├── change_visualization.py
│   ├── object_detection.py
│   ├── tiling.py
│   ├── detector_model.py
│   ├── model_registry.py
│   ├── model_inference.py
│   ├── geospatial_detections.py
│   ├── map_view.py        # Folium map panel
│   └── workflows/
│       ├── __init__.py
│       ├── imagery.py      # Scene processing orchestration
│       └── change.py       # Before/after orchestration
├── ui/
│   ├── theme.py
│   ├── layout.py
│   ├── components.py
│   ├── status.py
│   └── navigation.py
└── tests/                 # pytest suite
```

Runtime directories such as `data/` and `outputs/` are created as needed and are not required in the repository tree.

---

## Installation

```bash
git clone https://github.com/edu-moraess/satellite-geospatial-intelligence.git
cd satellite-geospatial-intelligence

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

Dependencies (from `requirements.txt`): `streamlit`, `numpy`, `rasterio`, `geopandas`, `shapely`, `pystac-client`, `planetary-computer`, `folium`, `streamlit-folium`, `matplotlib`.

Heavy ML stacks (e.g. torch / TerraTorch) remain intentionally omitted until a real compatible checkpoint and its input pipeline are validated.

---

## Run

```bash
streamlit run app.py
```

Default browser opens the Streamlit app. Configure AOI and temporal filters in **Mission Control**, search the catalog, download a scene, then use the analysis workspace.

---

## Testing

```bash
python -m compileall .
pytest -v
```

The current suite covers AOI/bbox, downloader helpers, spectral indices, raster validation, classification, change detection, tiling, object-detection utilities, and map-view helpers. **72 tests** are expected to pass in a clean environment with `requirements.txt` installed.

---

## Deployment

Suitable for [Streamlit Community Cloud](https://streamlit.io/cloud):

| Setting | Value |
|---------|--------|
| Repository | `edu-moraess/satellite-geospatial-intelligence` |
| Branch | `main` |
| Main file | `app.py` |
| Python packages | `requirements.txt` |

No application secrets are required for public Planetary Computer STAC access as used by this code. Theme is controlled via `.streamlit/config.toml` (primary color only; light/dark follows Streamlit).

---

## Scientific Scope

This is a **research / experimental** tool, not an operational certification system.

Outputs depend on:

- image quality and cloud contamination;
- spatial resolution and AOI extent;
- choice of index and change threshold;
- availability of STAC assets;
- presence of a valid AI checkpoint (if using Geospatial AI).

Spectral values should be interpreted in the context of sensor characteristics and preprocessing.

---

## Limitations

- **AI models:** no checkpoint is shipped; the registry lists a baseline definition with `checkpoint=None`. Inference is not operational without external weights.
- **Area metrics:** land-cover and change areas use a fixed 10 m pixel size assumption for Sentinel-2 10 m bands; this is an approximation, not a full geodetic area computation for every edge case.
- **Network dependency:** catalog search and downloads require access to Planetary Computer / asset URLs.
- **Scale:** AOI size is intentionally constrained in the UI for interactive use; very large regions are not the target workflow.

---

## Roadmap

### Implemented

- Sentinel-2 L2A search and AOI-constrained download
- Raster validation and spectral indices
- Land-cover and change-detection workflows
- Map-based AOI drawing and GeoJSON export path for detections
- Automated unit tests for core modules

### Planned

- Connect the deep-learning layer to a real compatible multispectral checkpoint
- Add Sentinel-2 B8A/B12 ingestion and a 6-band resampling path for compatible Earth Observation foundation models
- Optional real model checkpoints and documented inference dependencies
- Stronger geodetic area statistics
- CI (e.g. GitHub Actions) running `pytest` on each push
- Temporal stacks / multi-date dashboards beyond pairwise change
- Benchmarking harness for detection quality when models are available

---

## Engineering Principles

- **Separation of concerns:** UI (`ui/`) does not implement scientific algorithms; `src/` does not depend on Streamlit layout.
- **Validation before analysis:** rasters are checked before spectral and detection pipelines.
- **Reproducibility:** dependencies pinned by name in `requirements.txt`; tests exercise pure functions without live STAC when possible.
- **Honest experimental scope:** missing checkpoints do not invent detections.

---

## Author

[edu-moraess](https://github.com/edu-moraess)

---

*GEOCORE — experimental Earth Observation / remote sensing analysis on Sentinel-2.*


### Scientific processing safeguards

Sentinel-2 L2A downloads now resolve Earth Search semantic asset keys for B8A, B12 and SCL, decode the STAC radiometric scale/offset contract without double-applying the BOA offset, and preserve an AOI-level Scene Classification quality report. Cloud, cloud-shadow, snow and invalid pixels are excluded from quantitative spectral processing.

The deep-learning layer registers a Prithvi-EO-2.0 Burn Scars checkpoint contract but deliberately blocks inference for raw Sentinel-2 L2A until an explicit Sentinel-2/HLS harmonization path is validated. The six-band model contract is B02, B03, B04, B8A, B11 and B12; the Burn Scars checkpoint was fine-tuned on HLS data.


### Sentinel-2 → Prithvi adapter

The deep-learning layer now contains a dedicated adapter that orders B02/B03/B04/B8A/B11/B12, resamples the native 10 m/20 m Sentinel-2 inputs to a 30 m target grid using area averaging, and applies the published Prithvi Burn Scars normalization constants. The adapter reports its provenance explicitly as `spatially_harmonized`.

It does **not** claim to be an HLS S30 product. NASA HLS additionally applies spectral bandpass adjustment and nadir/BRDF normalization. Consequently the Burn Scars checkpoint remains disabled until a validated HLS-equivalent harmonization path is established.


### Real Deep Learning runtime

GEOCORE now registers the official **Prithvi-EO-2.0-300M-BurnScars** checkpoint and includes a lazy TerraTorch/PyTorch inference runtime. The model consumes six bands at 512×512 and predicts two classes: Not burned and Burn scar. The official repository reports inference on HLS imagery and publishes the model checkpoint at approximately 1.3 GB. The optional runtime is isolated from the base Streamlit dependency set to avoid making the core application depend on a very large ML stack.

Sentinel-2 inference remains gated until the GEOCORE Sentinel-2 → HLS S30 harmonization is scientifically validated. Spatial resampling to 30 m and applying model normalization alone does not establish equivalence to HLS.
