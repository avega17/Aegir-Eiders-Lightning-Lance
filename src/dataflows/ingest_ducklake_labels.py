from typing import Iterable, Optional
import pyarrow as pa

# Skeleton for Hamilton: function-as-node; names describe outputs
# This module ingests seed PV labels from DuckLake (via DuckDB ATTACH) and
# prepares them as Arrow tables for downstream mask generation and LanceDB writes.

# Note: Implementation deferred; signatures only.


def ducklake_catalog_uri(dev_or_prod: str = "dev") -> str:
    """URI string or connection args for DuckLake catalog (DuckDB ATTACH)."""
    return "ducklake:postgres://..."  # to be configured via env


def seed_labels_table(catalog_uri: str = ducklake_catalog_uri()) -> pa.Table:
    """Arrow table of seed labels (points/bboxes/polygons) selected from DuckLake.
    Should include columns: uuid, geometry_wkt (or geoarrow), source_id, ts, crs.
    """
    return pa.table({})


def prompts_for_mask_factory(seed_labels: pa.Table) -> pa.Table:
    """Transform seed labels into prompt records for the Mask Factory.
    Include prompt type (point/box), geometry, and any selection metadata.
    """
    return seed_labels

