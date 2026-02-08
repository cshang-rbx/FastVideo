from fastvideo.version import __version__

__all__ = ["VideoGenerator", "PipelineConfig", "SamplingParam", "__version__"]


def __getattr__(name: str):
    """Lazy imports to avoid triggering heavy dependencies (Triton, CUDA)
    at package import time.  This is critical for Ray workers that
    deserialise modules before CUDA drivers are available."""
    if name == "VideoGenerator":
        from fastvideo.entrypoints.video_generator import VideoGenerator
        return VideoGenerator
    if name == "PipelineConfig":
        from fastvideo.configs.pipelines import PipelineConfig
        return PipelineConfig
    if name == "SamplingParam":
        from fastvideo.configs.sample import SamplingParam
        return SamplingParam
    raise AttributeError(f"module 'fastvideo' has no attribute {name!r}")
