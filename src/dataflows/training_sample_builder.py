import pyarrow as pa

# Skeleton: assemble training records for LanceDB from prompts/masks/chips


def training_records(prompts: pa.Table, masks: pa.Table, msi: pa.Table) -> pa.Table:
    """Join/align inputs and return Arrow table with columns for LanceDB write.
    Expected: uuid, prompt_geometry, vhr_image_chip, generated_mask, msi_image_chip, metadata
    """
    return pa.table({})

