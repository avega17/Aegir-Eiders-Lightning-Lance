# Architecture

- Operational datastore: LanceDB at ./data/lancedb
- System of record: DuckLake (DuckDB + Postgres catalog)
- Pipelines: Apache Hamilton for composable dataflows
- Training: PyTorch Lightning with W&B logging
- Indexing: Emphasis on S2 IDs (Core‑Five); optional H3
- IO: PyArrow/Arrow tables; later explore arro3 for geospatial arrays

High‑level flow (initial dev):
1) Ingest seed labels from DuckLake via DuckDB ATTACH
2) Load VHR chips (Core‑Five via HF + xarray; Satellogic EarthView via S3/Parquet)
3) Generate masks with segment‑geospatial (SAM/SAM‑HQ/SAM2), refine masks
4) Compute S2 indices and spectral indices for MSI chips
5) Build training samples and write to LanceDB
6) Train Lightning models, log to W&B; write validated outputs back to DuckLake later

