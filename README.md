# Satellite Geospatial Intelligence Models

This directory contains model checkpoints used by the
Geospatial AI pipeline.

## Important

Model files are intentionally NOT committed to GitHub.

Large checkpoints should be downloaded separately or
stored through an appropriate model registry.

Expected structure:

models/
    model_name/
        checkpoint.pt

The application should verify the checkpoint before
running inference.

No detections are fabricated when a checkpoint is absent.