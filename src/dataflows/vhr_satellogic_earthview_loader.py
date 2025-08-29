from typing import Optional
import pyarrow as pa

# Skeleton: Satellogic EarthView example loader (S3/Parquet or STAC later)
# Reference notebook: Satellogic EarthView exploration (linked in docs).


def earthview_config(parquet_uri: Optional[str] = None) -> dict:
    """Connection/config details for EarthView labels/images."""
    return {}


def vhr_chips_from_earthview(prompts_or_masks: pa.Table, cfg: dict = earthview_config()) -> pa.Table:
    """Load VHR chips matching AOIs from EarthView.
    Output: uuid, chip, sensor_meta, timestamp.
    """
    return pa.table({})

