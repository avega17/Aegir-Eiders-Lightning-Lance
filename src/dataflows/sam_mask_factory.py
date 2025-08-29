from typing import Optional
import pyarrow as pa

# Skeleton for Hamilton: prompt-driven segmentation using segment-geospatial (SAM/SAM-HQ/SAM2)
# Implementation deferred; function signatures + docstrings only.


def vhr_chips_for_prompts(prompts: pa.Table) -> pa.Table:
    """Load VHR image chips corresponding to prompts.
    Downstream modules may source chips via HF datasets or other providers.
    Expected columns: uuid, chip (ndarray/bytes), geometry_wkt, crs, timestamp.
    """
    return pa.table({})


def generated_masks(vhr_chips: pa.Table, model_variant: str = "sam2") -> pa.Table:
    """Generate segmentation masks given VHR chips and prompt geometries.
    model_variant: one of [sam, sam-hq, sam2].
    Output columns: uuid, mask (ndarray/bytes), mask_quality, meta.
    """
    return vhr_chips


def refined_masks(masks: pa.Table) -> pa.Table:
    """Apply post-processing/refinement (e.g., morphology, smoothing) to masks.
    Output maintains alignment with original chip geometry.
    """
    return masks

