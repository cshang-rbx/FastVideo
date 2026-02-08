"""
FastVideo package with lazy imports to prevent early CUDA/Triton initialization.

This module uses lazy imports to defer loading of GPU-dependent modules
until they are actually accessed. This is critical for Ray compatibility,
where modules may be imported during deserialization before CUDA is ready.
"""

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]


def __getattr__(name: str):
    """Lazy import for FastVideo public API.

    This defers imports until they are actually accessed, preventing
    Triton kernel initialization during Ray worker deserialization.
    """
    if name == "VideoGenerator":
        from fastvideo.entrypoints.video_generator import VideoGenerator
        return VideoGenerator
    elif name == "PipelineConfig":
        from fastvideo.configs.pipelines import PipelineConfig
        return PipelineConfig
    elif name == "SamplingParam":
        from fastvideo.configs.sample import SamplingParam
        return SamplingParam
    elif name == "__version__":
        from fastvideo.version import __version__
        return __version__
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
