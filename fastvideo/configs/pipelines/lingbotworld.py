# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from collections.abc import Callable

import torch

from fastvideo.configs.models.dits import LingbotWorldConfig
from fastvideo.configs.models import DiTConfig, VAEConfig

from fastvideo.configs.models.vaes import WanVAEConfig
from fastvideo.configs.pipelines.wan import Wan2_2_I2V_A14B_Config


@dataclass
class LingbotWorldT2VBaseConfig(Wan2_2_I2V_A14B_Config):
    """Base configuration for LingbotWorld T2V Base pipeline architecture."""
    flow_shift: float | None = 10.0
    boundary_ratio: float | None = 0.900

    # DiT
    dit_config: DiTConfig = field(default_factory=LingbotWorldConfig)
    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)

    def __post_init__(self):
        super().__post_init__()
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
