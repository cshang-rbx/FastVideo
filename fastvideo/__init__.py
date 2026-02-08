"""FastVideo package exports.

Keep this module lightweight: importing fastvideo is common in Ray workers
before CUDA is ready. Avoid importing GPU/Triton-heavy modules at import time.
"""

from fastvideo.configs.pipelines import PipelineConfig
from fastvideo.configs.sample import SamplingParam
from fastvideo.version import __version__

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]


def __getattr__(name: str):
    if name == "VideoGenerator":
        # Lazy import to avoid Triton/CUDA initialization at package import time.
        from fastvideo.entrypoints.video_generator import VideoGenerator

        return VideoGenerator
    raise AttributeError(f"module 'fastvideo' has no attribute {name!r}")
