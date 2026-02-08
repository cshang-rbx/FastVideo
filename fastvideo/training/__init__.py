"""
FastVideo training module with lazy imports to prevent early CUDA/Triton initialization.

This module uses lazy imports to defer loading of GPU-dependent modules
until they are actually accessed. This is critical for Ray compatibility,
where modules may be imported during deserialization before CUDA is ready.
"""

__all__ = ["TrainingPipeline", "WanTrainingPipeline", "DistillationPipeline"]


def __getattr__(name: str):
    """Lazy import for FastVideo training pipelines.

    This defers imports until they are actually accessed, preventing
    Triton kernel initialization during Ray worker deserialization.
    """
    if name == "TrainingPipeline":
        from .training_pipeline import TrainingPipeline
        return TrainingPipeline
    elif name == "WanTrainingPipeline":
        from .wan_training_pipeline import WanTrainingPipeline
        return WanTrainingPipeline
    elif name == "DistillationPipeline":
        from .distillation_pipeline import DistillationPipeline
        return DistillationPipeline
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
