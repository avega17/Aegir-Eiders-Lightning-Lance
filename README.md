# Ægir Eider’s Lightning Lance

A geospatial computer-vision data engineering project focused on building a robust, reproducible data flywheel for PV segmentation: seed labels → Mask Factory (prompt‑driven VHR mask generation) → multispectral (MSI) sourcing and preprocessing → training → large‑scale inference → validation → backfill to improve the dataset. This repository emphasizes modern data stack patterns and LanceDB‑based operational storage for ML workloads.

## Why this repo
- Data‑centric thesis work: prioritize high‑quality, multi‑modal data generation, storage, and retrieval
- Operational datastore: LanceDB/Lance columnar format for training/inference data
- Composable pipelines: Apache Hamilton dataflows (function‑as‑node)
- Training scaffolds: PyTorch Lightning with Weights & Biases (W&B) experiment tracking
- Alignment with Ice‑mELT DuckLake: DuckLake (DuckDB+Postgres catalog) remains the system of record; this repo focuses on CV workflows and writes results back

## Architecture overview
- Data foundation
  - LanceDB at ./data/lancedb (local default)
  - Tables (initial): training_data (uuid, prompt_geometry, vhr_image_chip, generated_mask, msi_image_chip, metadata)
  - Metadata‑first filtering (S2/H3 level IDs, date ranges, sensor props)
- Core modules
  - Hamilton dataflows (structure‑only initially):
    - ingest_ducklake_labels: read prompts/labels from DuckLake (DuckDB ATTACH) and prepare for Lance
    - sam_mask_factory: prompt‑driven mask generation & refinement (segment‑geospatial: SAM/SAM‑HQ/SAM2)
    - vhr_core_five_loader: VHR/HR imagery via HF datasets + xarray (Core‑Five)
    - vhr_satellogic_earthview_loader: example loader for Satellogic EarthView (S3/Parquet, or STAC later)
    - s2_indexing: compute S2 IDs and square index metadata for chips/labels
    - spectral_indices: NDVI/NDBI/PVI etc. for MSI chips
    - training_sample_builder: assemble records for LanceDB
    - lance_writer: create/append to Lance tables
  - Training scaffolds: LightningDataModule (LanceDataset) + LightningModule skeleton; W&B logger

## Getting started
1) Python & uv
- Install uv (https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create environment + install deps from requirements.txt:
  - `uv venv -p 3.11`
  - `source .venv/bin/activate`
  - `uv pip install -r requirements.txt`

2) PyTorch
- Install PyTorch separately for your platform (CPU, CUDA, or MPS on macOS): https://pytorch.org/get-started/locally/
- Example (macOS, MPS): `pip install torch torchvision torchaudio`

3) Local data locations
- LanceDB root (default): `./data/lancedb`
- DuckLake access: Use DuckDB ATTACH to the DuckLake catalog (dev/prod commands from Ice‑mELT). Initial development is local only; keep Dev→Prod parity principles.

## Project layout
```
├── README.md
├── requirements.txt
├── docs/
│   ├── architecture.md
│   ├── arch_mermaid_diagram.mmd
│   ├── data_foundation_lance.md
|   └── lance/
│       ├── lanceDB_vs_lakehouse_roles.md
│       └── geospatial_embeddings_and_vector_search.md
├── src/
│   ├── dataflows/
│   │   ├── ingest_ducklake_labels.py
│   │   ├── sam_mask_factory.py
│   │   ├── vhr_core_five_loader.py
│   │   ├── vhr_satellogic_earthview_loader.py
│   │   ├── s2_indexing.py
│   │   ├── spectral_indices.py
│   │   ├── training_sample_builder.py
│   │   └── lance_writer.py
│   ├── training/
│   │   ├── datamodule.py
│   │   ├── litmodule.py
│   │   └── yolo_baseline.py  # skeleton only
│   ├── store/
│   │   └── lance_io.py
│   └── utils/
│       ├── geo.py
│       └── types.py
└── notebooks/
    ├── 01_mask_factory_demo.ipynb
    ├── 02_lance_training_data_browser.ipynb
    ├── 03_train_msi_smp_baselines.ipynb
    ├── 04_yolo_baseline_prep_and_eval.ipynb
    └── 05_inference_and_validation_loop.ipynb
```

Notes
- We’ll emphasize S2 indexing (Core‑Five), with optional H3 columns for compatibility with Ice‑mELT patterns.

## Running Hamilton (example pattern)
Hamilton modules follow the function‑as‑node pattern. We will run dataflows with a small wrapper or the Hamilton API. Example target outputs (to be implemented):
- `training_data` Arrow table → wrote/updated in LanceDB table `training_data`.

## Training scaffolds
- LightningDataModule will stream samples from LanceDB (via Lance’s torch dataset) into batches
- LightningModule will be model‑agnostic (SMP preferred initially); a YOLOv11‑Seg baseline skeleton is provided but not implemented yet
- Experiment tracking via Weights & Biases (W&B)

## References
- LanceDB/Lance: https://lancedb.github.io/lancedb/basic/ , https://lancedb.github.io/lancedb/python/python/
- Hamilton concepts: https://hamilton.incubator.apache.org/concepts/node/
- Segment‑Geospatial examples: 
  - Box prompts: https://samgeo.gishub.org/examples/sam2_box_prompts/
  - Point prompts: https://samgeo.gishub.org/examples/sam2_point_prompts/
  - SAM2/SAM‑HQ prompts: https://samgeo.gishub.org/examples/input_prompts_hq/
  - Automatic masks (HQ): https://samgeo.gishub.org/examples/automatic_mask_generator_hq/
- Core‑Five usage: https://huggingface.co/datasets/gajeshladhar/core-five#%F0%9F%A7%A0-usage
- Satellogic EarthView example: https://colab.research.google.com/github/satellogic/satellogic-earthview/blob/main/satellogic_earthview_exploration.ipynb
- S2 indexing: https://github.com/aaliddell/s2cell
- Apache Hamilton nodes: https://hamilton.incubator.apache.org/concepts/node/
- LanceDB quick start: https://lancedb.github.io/lancedb/basic/
- LanceDB Python: https://lancedb.github.io/lancedb/python/python/
- S2 indexing: https://github.com/aaliddell/s2cell
- Arrow/Geo: PyArrow docs; consider arro3 later: https://github.com/kylebarron/arro3

## Dev → Prod parity
This repo follows the same philosophy as Ice‑mELT DuckLake: keep configuration/environment patterns compatible for future Neon/MotherDuck/R2 setup. Early development stays local.