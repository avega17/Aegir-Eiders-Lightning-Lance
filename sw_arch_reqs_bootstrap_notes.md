# Software Architecture & Requirements: Ægir Eider's Lightning Lance

## 1. Vision & Objective

The "Ægir Eider's Lightning Lance" project will create a comprehensive, scalable, and iterative system for the global detection and semantic segmentation of photovoltaic (PV) solar panels. The core strategy is a **"Mask Factory" feedback loop**: leveraging a massive corpus of existing, lower-quality labels (points, bounding boxes) to generate high-quality segmentation masks using state-of-the-art foundation models finetuned for Earth Observation (EO) imagery. These newly created, higher-quality labels will then be used to train specialized, efficient multispectral segmentation models that can be deployed at scale for analytical tasks, such as estimating installed PV capacity and monitoring renewable energy adoption over time. As part of thesis results, we would compare the performance of smaller, specialized segmentation models that leverage additional spectral channels to pre-trained and foundation models like Ultralytics' YOLOv11-Seg and Meta AI's Segment Anything Model and its variants (SAM, SAM-HQ, SAM2), and the NASA-IBM's Prithvi-EO 2.0 that also supports multispectral inputs.

### The Data-Centric Thesis Contribution

This project's primary contribution is not the development of a state-of-the-art model architecture but rather the design and implementation of a robust, reproducible, and scalable **data engineering and MLOps platform for a geospatial computer vision problem**. It addresses a common gap in academic data science by prioritizing database-centric workflows, optimized data flows, and operational best practices. The architecture demonstrates how a modern data stack can solve the primary bottleneck in GeoAI: the scarcity of high-quality, large-scale training data. By systemizing the generation, storage, and retrieval of multi-modal data (vector labels, VHR imagery, multispectral data cubes), the project aims to create a flywheel that continuously improves model performance through better data, not just better algorithms.

---

## 2. Core Components & Software Architecture

The system is composed of four primary, interconnected modules built upon a unified data foundation.

### 2.1 Data Foundation: LanceDB & The DuckLake Ecosystem

This is the central nervous system of the project, moving beyond simple file storage to an integrated multi-modal database optimized for CV workflows.

* **Technology**: **LanceDB** and the **Lance (`.lance`) columnar format**.
* **Requirements**:
    * **Operational CV Datastore**: Within this repository's scope, LanceDB will serve as the primary operational database. All data actively used in training and inference workflows—including initial vector prompts, VHR image chips, and the generated segmentation masks—will be stored in LanceDB tables. This ensures a single, high-performance source for all CV tasks.
    * **Integration with Project Lakehouse**: It is critical to note that the broader project's data lakehouse (the "DuckLake" built on DuckDB and S3-compatible R2 cloud storage, with a NeonDB/pgstac catalog) remains the master source of truth. Ongoing work on the data lakehouse aspect can be found in the [Ice-mELT-Ducklake repository](https://github.com/avega17/Ice-mELT_DuckLake). LanceDB will be populated from the DuckLake, and final, validated inference results will be written *back* to the DuckLake for project-wide analytics.
    * **Optimized PyTorch Integration**: The data loading pipeline must directly leverage the `lance.torch.data.LanceDataset` class for ultra-fast, zero-copy data access during model training, bypassing common I/O bottlenecks.
    * **Metadata-First Querying**: The system must support efficient, SQL-based filtering on spatial and tabular metadata via the LanceDB-DuckDB integration. While LanceDB has powerful vector search capabilities, this project will **de-emphasize Approximate Nearest Neighbor (ANN) search**. The primary indexing and retrieval strategy will rely on performant filtering of geospatial metadata (e.g., location, H3 index), image properties (e.g., resolution, acquisition date), and contextual data (e.g., land cover class), as this is more reliable and interpretable for the defined tasks. ANN may be considered in the future for purely exploratory tasks on datasets lacking rich metadata, such as grouping visually similar panel installations.
    * example **Base Schema**: A primary `lance` table, `training_data`, will be designed to hold:
        * `uuid`: A unique identifier for the data sample.
        * `prompt_geometry`: The original weak label (point or bbox) in WKT or GeoArrow format.
        * `vhr_image_chip`: The raw VHR image pixel data used for mask generation.
        * `generated_mask`: The pixel data for the segmentation mask created by the Mask Factory.
        * `msi_image_chip`: The multispectral image chip sourced using the mask's footprint.
        * `metadata`: A struct containing source IDs, timestamps, CRS information, and contextual flags.

---

### 2.2 Module 1: The Mask Factory (Data Generation)

This module is responsible for converting low-quality labels into high-quality segmentation masks.

* **Primary Tool**: **`segment-geospatial` (SamGEO)**.
* **Requirements**:
    * **Geospatial Integration**: The process must be fully CRS-aware, seamlessly handling the projection of geographic labels (from the DuckLake) onto the pixel space of the source imagery. `segment-geospatial` is chosen specifically for its native handling of GeoTIFFs and geospatial data.
    * **Prompt-Driven Workflow**: The primary mode of operation will be **prompt-based segmentation**. The factory must be architected to systematically ingest point and bounding box prompts to generate precise masks for known objects. The automatic mask generation feature will be used sparingly for exploratory analysis only.
    * **Model Agnosticism**: The architecture must support different underlying segmentation models available through `segment-geospatial`, including the standard **SAM**, the higher-fidelity **SAM-HQ**, and the newer **SAM2**.
    * **Mask Refinement (Post-processing)**: After initial mask generation by the foundation model, the pipeline will include a mandatory sub-module for mask refinement. This will involve applying classical computer vision techniques (e.g., morphological operations, contour smoothing) to produce cleaner, more accurate boundaries for the training data.

---

### 2.3 Module 2: Data Sourcing & Preprocessing

This module acquires and prepares imagery, using the high-quality masks from Module 1 to source relevant multispectral data.

* **Primary Tools**: **`cubo`**, **Hugging Face Datasets**, custom scripts for VHR sources.
* **Requirements**:
    * **VHR Imagery Sourcing**: A dedicated component will handle fetching VHR imagery from sources like the Google Solar API and existing DOI datasets for use in the Mask Factory.
    * **Mask-to-MSI Workflow**: This is a critical link. The georeferenced footprints of the high-quality masks generated in Module 1 will be used as the precise areas of interest (AOIs) for sourcing corresponding multispectral imagery.
    * **Offline Development Dataset**: The **`gajeshladhar/core-five`** Hugging Face dataset will be the primary source for initial development and testing of the multispectral training pipeline. This allows for rapid iteration without network latency or API costs.
    * **On-Demand STAC Integration**: For scaling and accessing timely data, the **`cubo`** library will be the primary interface for acquiring analysis-ready multispectral data cubes from STAC catalogs. This component will be driven by the mask footprints.
    * **Spectral Index Calculation**: The preprocessing pipeline, likely implemented as a Hamilton dataflow, must be able to calculate and append PV-specific spectral indices (e.g., PVI) and relevant background indices (e.g., NDVI, NDBI) as additional channels to the multispectral data cubes.

---

### 2.4 Module 3: Model Training & Fine-Tuning

This module trains specialized, efficient models using the multi-modal data produced by the preceding modules.

* **Primary Tools**: **PyTorch Lightning**, **`segmentation-models-pytorch` (smp)**, and **`ultralytics` (YOLO)**.
* **Requirements**:
    * **Primary Training Framework**: The core focus will be on using PyTorch Lightning with models from `segmentation-models-pytorch`. This leverages your past work and provides maximum flexibility in experimenting with different encoders, decoders, and loss functions on multispectral inputs.
    * **Modular Trainer Design**: The training pipeline, orchestrated by PyTorch Lightning, will be designed to be model-agnostic. The `LightningDataModule` will source from LanceDB, while the `LightningModule` can flexibly encapsulate different model architectures.
    * **YOLO as a Comparative Baseline**: The Ultralytics framework (YOLOv11-Seg) will be integrated as a key point of comparison. The goal is to evaluate the performance of a highly optimized, pre-trained model against the more flexible `smp` approach. The data pipeline must include a robust converter to transform raster masks from LanceDB into the YOLO segmentation label format. A key investigation will be determining the feasibility and effectiveness of adapting YOLO models to accept and leverage multispectral inputs.
    * **Experiment Tracking**: Integrated experiment tracking is a firm requirement to manage the high number of experiments involving different models, data augmentations, and input band combinations.

---

### 3. The CV Feedback & Discovery Loop

The architecture is designed to support a continuous improvement "flywheel," a core objective of your thesis.

1.  **🌱 Seed**: The process begins with the **100K+ initial PV labels** curated in the DuckLake. These are used to query VHR imagery sources.
2.  **🏭 Generate (Mask Factory)**: The Mask Factory (Module 1) uses the VHR imagery and seed labels to generate a foundational set of **high-quality, VHR-derived segmentation masks**.
3.  **🛰️ Source & Train (Modules 2 & 3)**: The georeferenced footprints of these new masks are used to source corresponding **multispectral imagery chips** (from Core-Five or STAC). A specialized, multispectral segmentation model is then trained on this MSI data, using the VHR-derived masks as ground truth.
4.  **🗺️ Discover (Large-Scale Inference)**: The trained **multispectral model** is deployed for large-area inference. This is highly efficient as MSI provides wider coverage than VHR. This discovery process is guided by geospatial context from the DuckLake (e.g., Overture building data, ESA land cover) to focus on probable areas and reduce computational waste.
5.  **🔬 Validate & Refine**: New, high-confidence detections are flagged as candidate locations. To ensure quality, VHR imagery is then sourced for these *specific new locations* for verification.
6.  **🔄 Augment (The Flywheel Effect)**: These newly validated locations and their corresponding VHR imagery are fed *back* into the Mask Factory as new prompts. This generates more masks from diverse geographies, panel types, and environmental conditions, continuously enriching the LanceDB training set and improving the model's robustness and accuracy with each cycle.

This virtuous cycle transforms the system from a static model training pipeline into a dynamic data generation and refinement engine, directly addressing the core challenge of scalable, high-quality data creation in GeoAI.

---

### 4. References & Tooling

* **Mask Generation**:
    * `segment-geospatial`: [https://github.com/opengeos/segment-geospatial?tab=readme-ov-file](https://github.com/opengeos/segment-geospatial?tab=readme-ov-file)
* **Foundation Models**:
    * `segment-geospatial`: [https://github.com/opengeos/segment-geospatial?tab=readme-ov-file](https://github.com/opengeos/segment-geospatial?tab=readme-ov-file)
    * `segment-anything-model 2`: [https://github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
    * `sam-hq`: [https://github.com/SysCV/sam-hq?tab=readme-ov-file](https://github.com/SysCV/sam-hq?tab=readme-ov-file)
    * `Prithvi-EO-2.0: A Versatile Multi-Temporal Foundation Model for Earth Observation Applications`: [https://github.com/NASA-IMPACT/Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
* **Model Training**:
    * `ultralytics` (YOLOv11, SAM2): [https://docs.ultralytics.com/tasks/segment/](https://docs.ultralytics.com/tasks/segment/)
    * JSON2YOLO Converter: [https://github.com/ultralytics/JSON2YOLO](https://github.com/ultralytics/JSON2YOLO)
* **Data Sourcing**:
    * `cubo`: [https://github.com/ESDS-Leipzig/cubo](https://github.com/ESDS-Leipzig/cubo)
    * Google Solar API: [https://developers.google.com/maps/documentation/solar/geotiff](https://developers.google.com/maps/documentation/solar/geotiff), [https://developers.google.com/maps/documentation/solar/building-insights](https://developers.google.com/maps/documentation/solar/building-insights)
    * Core-Five Hugging Face Dataset: [https://huggingface.co/datasets/gajeshladhar/core-five](https://huggingface.co/datasets/gajeshladhar/core-five)
* **Data Foundation**:
    * LanceDB / Lance Format: [https://github.com/lancedb/lance?tab=readme-ov-file](https://github.com/lancedb/lance?tab=readme-ov-file)
    * LanceDB Integrations & Recipes:
        * PyTorch: [https://lancedb.github.io/lance/integrations/pytorch/](https://lancedb.github.io/lance/integrations/pytorch/)
        * DuckDB: [https://lancedb.github.io/lance/integrations/duckdb/](https://lancedb.github.io/lance/integrations/duckdb/)
        * Deep Learning Recipes Repo: [https://github.com/lancedb/lance-deeplearning-recipes](https://github.com/lancedb/lance-deeplearning-recipes)