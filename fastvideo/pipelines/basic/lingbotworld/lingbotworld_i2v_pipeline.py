from __future__ import annotations

from fastvideo.fastvideo_args import FastVideoArgs
from fastvideo.logger import init_logger
from fastvideo.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler)
from fastvideo.pipelines.composed_pipeline_base import ComposedPipelineBase
from fastvideo.pipelines.lora_pipeline import LoRAPipeline

# isort: off
from fastvideo.pipelines.stages import (
    CameraConditioningStage, LingbotWorldDenoisingStage,
    RefImageEncodingStage, ConditioningStage, DecodingStage,
    ImageVAEEncodingStage, InputValidationStage, LatentPreparationStage,
    TextEncodingStage, TimestepPreparationStage)
# isort: on

logger = init_logger(__name__)


class LingbotWorldI2VPipeline(LoRAPipeline, ComposedPipelineBase):
    """LingbotWorld camera-conditioned image-to-video pipeline.

    Stages:

    1. Input validation
    2. Text encoding
    3. Reference image encoding (CLIP)
    4. Camera conditioning (Plücker rays from poses / intrinsics)
    5. Conditioning (merge text + image embeddings)
    6. Timestep preparation
    7. Latent preparation
    8. Video VAE encoding (for V2V / I2V latent concat)
    9. Denoising (with camera conditioning)
    10. Decoding
    """

    _required_config_modules = [
        "text_encoder", "tokenizer", "vae", "transformer", "scheduler",
        "image_encoder", "image_processor"
    ]

    def initialize_pipeline(self, fastvideo_args: FastVideoArgs):
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=fastvideo_args.pipeline_config.flow_shift)

    def create_pipeline_stages(self, fastvideo_args: FastVideoArgs):
        """Set up pipeline stages with camera conditioning support."""

        self.add_stage(stage_name="input_validation_stage",
                       stage=InputValidationStage())

        self.add_stage(stage_name="prompt_encoding_stage",
                       stage=TextEncodingStage(
                           text_encoders=[self.get_module("text_encoder")],
                           tokenizers=[self.get_module("tokenizer")],
                       ))

        if (self.get_module("image_encoder") is not None
                and self.get_module("image_processor") is not None):
            self.add_stage(
                stage_name="ref_image_encoding_stage",
                stage=RefImageEncodingStage(
                    image_encoder=self.get_module("image_encoder"),
                    image_processor=self.get_module("image_processor"),
                ))

        # Camera conditioning — computes cam_plucker_emb from poses/intrinsics
        vae_arch = fastvideo_args.pipeline_config.vae_config.arch_config
        vae_stride = (vae_arch.scale_factor_temporal,
                      vae_arch.scale_factor_spatial,
                      vae_arch.scale_factor_spatial)
        patch_size = fastvideo_args.pipeline_config.dit_config.arch_config.patch_size
        self.add_stage(
            stage_name="camera_conditioning_stage",
            stage=CameraConditioningStage(
                vae_stride=vae_stride,
                patch_size=patch_size,
            ))

        self.add_stage(stage_name="conditioning_stage",
                       stage=ConditioningStage())

        self.add_stage(stage_name="timestep_preparation_stage",
                       stage=TimestepPreparationStage(
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="latent_preparation_stage",
                       stage=LatentPreparationStage(
                           scheduler=self.get_module("scheduler"),
                           transformer=self.get_module("transformer")))

        self.add_stage(stage_name="image_latent_preparation_stage",
                       stage=ImageVAEEncodingStage(vae=self.get_module("vae")))

        self.add_stage(stage_name="denoising_stage",
                       stage=LingbotWorldDenoisingStage(
                           transformer=self.get_module("transformer"),
                           transformer_2=self.get_module("transformer_2"),
                           scheduler=self.get_module("scheduler")))

        self.add_stage(stage_name="decoding_stage",
                       stage=DecodingStage(vae=self.get_module("vae")))


EntryClass = LingbotWorldI2VPipeline
