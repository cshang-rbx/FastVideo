# SPDX-License-Identifier: Apache-2.0
"""
FastVideo dataset module with lazy imports to prevent early CUDA/Triton initialization.

This module uses lazy imports to defer loading of GPU-dependent modules
until they are actually accessed. This is critical for Ray compatibility,
where modules may be imported during deserialization before CUDA is ready.
"""
from torchvision import transforms
from torchvision.transforms import Lambda

# These imports are safe (don't trigger GPU-dependent code)
from fastvideo.dataset.parquet_dataset_map_style import (
    build_parquet_map_style_dataloader)
from fastvideo.dataset.transform import (CenterCropResizeVideo, Normalize255,
                                         TemporalRandomCrop)

# Lazy imports for classes that depend on torch.distributed.checkpoint
# (which requires CUDA to be initialized)
__all__ = [
    "build_parquet_map_style_dataloader", "ValidationDataset",
    "VideoCaptionMergedDataset", "TextDataset", "getdataset", "gettextdataset"
]


def __getattr__(name: str):
    """Lazy import for FastVideo dataset classes.

    This defers imports until they are actually accessed, preventing
    torch.distributed.checkpoint imports during Ray worker deserialization.
    """
    if name == "VideoCaptionMergedDataset":
        from .preprocessing_datasets import VideoCaptionMergedDataset
        return VideoCaptionMergedDataset
    elif name == "TextDataset":
        from .preprocessing_datasets import TextDataset
        return TextDataset
    elif name == "ValidationDataset":
        from .validation_dataset import ValidationDataset
        return ValidationDataset
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def getdataset(args):
    """Get VideoCaptionMergedDataset with lazy import."""
    from .preprocessing_datasets import VideoCaptionMergedDataset
    
    if args.do_temporal_sample:
        temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    else:
        temporal_sample = None
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    resize_topcrop = [
        CenterCropResizeVideo((args.max_height, args.max_width), top_crop=True),
    ]
    resize = [
        CenterCropResizeVideo((args.max_height, args.max_width)),
    ]
    transform = transforms.Compose([
        # Normalize255(),
        *resize,
    ])
    transform_topcrop = transforms.Compose([
        Normalize255(),
        *resize_topcrop,
        norm_fun,
    ])
    return VideoCaptionMergedDataset(data_merge_path=args.data_merge_path,
                                     args=args,
                                     transform=transform,
                                     temporal_sample=temporal_sample,
                                     transform_topcrop=transform_topcrop,
                                     seed=args.seed)
                                    

def gettextdataset(args):
    """Get TextDataset with lazy import."""
    from .preprocessing_datasets import TextDataset
    return TextDataset(data_merge_path=args.data_merge_path,
                       args=args,
                       seed=args.seed)
