from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from CAE_tool_interface.config import CAEConfig
from DOE.config import DOEConfig
from Explorer.config import ExplorerConfig
from Modeler.config import ModelerConfig

if TYPE_CHECKING:
    from Optimizer.config import OptimizerConfig


@dataclass
class PipelineTasks:
    # DOE 단계 실행 여부
    run_doe: bool = True
    # Modeler(모델러) 단계 실행 여부
    run_modeler: bool = True
    # Explorer(탐색기) 단계 실행 여부
    run_explorer: bool = True
    # Optimizer(최적화기) 단계 실행 여부
    run_optimizer: bool = False


@dataclass
class PipelineReusePolicy:
    # Explicit input path가 없을 때 기존 run_root의 DOE public CSV를 downstream input으로 쓸지 여부.
    use_existing_doe_csv: bool = True
    # 이번 호출에서 Modeler를 실행하지 않았을 때 기존 run_root의 Modeler public artifact를
    # Explorer optional model/selected-feature layer로 쓸지 여부.
    use_existing_modeler_artifacts: bool = True


@dataclass
class PipelineConfig:
    # CAE 설정
    cae: CAEConfig
    # DOE 설정 (선택)
    doe: DOEConfig | None
    # Modeler(모델러) 설정 (선택)
    modeler: ModelerConfig | None
    # Explorer(탐색기) 설정 (선택)
    explorer: ExplorerConfig | None
    # Optimizer(최적화기) 설정 (선택)
    optimizer: OptimizerConfig | None = None
    # 실행할 단계(task) 설정
    tasks: PipelineTasks = field(default_factory=PipelineTasks)
    # 기존 run_root 안의 public artifact 자동 재사용 정책.
    reuse: PipelineReusePolicy = field(default_factory=PipelineReusePolicy)
    # 기존 run을 이어서 사용할 때 backend가 넘기는 run root.
    # None이면 새 run context를 생성한다.
    run_root: str | None = None
    # AION mode enables pipeline-only integration features through
    # pipeline/aion_system_config.py. Default off for direct/standalone runs.
    aion_mode: bool = False
