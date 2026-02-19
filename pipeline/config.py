from dataclasses import dataclass, field

from CAE_tool_interface.config import CAEConfig
from DOE.config import DOEConfig
from Explorer.config import ExplorerConfig
from Modeler.config import ModelerConfig


@dataclass
class PipelineTasks:
    # DOE 단계 실행 여부
    run_doe: bool = True
    # Modeler(모델러) 단계 실행 여부
    run_modeler: bool = True
    # Explorer(탐색기) 단계 실행 여부
    run_explorer: bool = True


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
    # 실행할 단계(task) 설정
    tasks: PipelineTasks = field(default_factory=PipelineTasks)
