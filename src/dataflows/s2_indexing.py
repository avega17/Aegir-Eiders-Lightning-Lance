import pyarrow as pa

# Skeleton: Compute S2 cell indices for geometries or chip footprints
# https://github.com/aaliddell/s2cell


def s2_index_for_prompts(prompts_or_masks: pa.Table, level: int = 12) -> pa.Table:
    """Append S2 cell id(s) for each geometry.
    Output includes s2_id (string/int) and level used.
    """
    return prompts_or_masks

