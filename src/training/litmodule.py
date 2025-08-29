# Skeleton: LightningModule stub for SMP-based segmentation models
# No implementation yet; outline only.


class LitSegmentationModel:
    """Wrap segmentation-models-pytorch (SMP) model; define training/val/test steps.
    Configure loss/metrics later.
    """

    def __init__(self, model_name: str = "unet", in_channels: int = 3, num_classes: int = 2) -> None:
        self.model_name = model_name
        self.in_channels = in_channels
        self.num_classes = num_classes

