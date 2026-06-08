from __future__ import annotations

from dataclasses import dataclass

from pipeline.config import PipelineConfig


@dataclass
class AIONSystemConfig:
    # AION-only Explorer integration. This restores DOE diagnostic signals for
    # router decisions without making DOE metadata a standalone Explorer input.
    enable_doe_router_signals: bool = True


AION_SYSTEM_CONFIG = AIONSystemConfig()


def apply_aion_system_config(config: PipelineConfig) -> PipelineConfig:
    if not bool(getattr(config, "aion_mode", False)):
        return config

    if config.explorer is not None:
        config.explorer.system.enable_doe_router_signals = bool(
            AION_SYSTEM_CONFIG.enable_doe_router_signals
        )

    return config
