from typing import Optional
import pyarrow as pa

# Skeleton: Core-Five dataset via HuggingFace + xarray
# https://huggingface.co/datasets/gajeshladhar/core-five#%F0%9F%A7%A0-usage


def core_five_config(dataset_revision: Optional[str] = None) -> dict:
    """Configuration for HF dataset access (split, bands, spatial windowing)."""
    return {}


def msi_chips_from_core_five(prompts_or_masks: pa.Table, cfg: dict = core_five_config()) -> pa.Table:
    """Load MSI chips from Core-Five matching mask footprints or prompt AOIs.
    Output: uuid, msi_chip (ndarray/bytes), bands, timestamp, s2_id, sensor_meta.
    """
    return pa.table({})

