from typing import TypedDict, Optional


class TrainingRecord(TypedDict, total=False):
    uuid: str
    prompt_geometry: str
    vhr_image_chip: bytes
    generated_mask: bytes
    msi_image_chip: bytes
    metadata: dict

