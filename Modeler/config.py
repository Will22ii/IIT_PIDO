from dataclasses import dataclass

from CAE_tool_interface.config import CAEConfig, CAEUserConfig


@dataclass
class ModelerUserConfig:
    # 모델 종류 이름 (예: "xgb")
    model_name: str = "xgb"
    # HPO 사용 여부
    use_hpo: bool = False
    # 목표 컬럼명
    target_col: str = "objective"


@dataclass
class ModelerSystemConfig:
    # HPO 상세 설정 (옵션)
    hpo_config: dict | None = None
    # 교차검증 K-fold 개수
    kfold_splits: int = 5
    # permutation 중요도 평가 샘플 수
    perm_sample_size: int = 1000
    # 폴드 통과 비율 기준
    perm_min_pass_rate: float = 0.6
    # permutation delta 정규화 임계값
    perm_epsilon: float = 0.04


@dataclass
class ModelerConfig:
    # 모델러 사용자 설정
    user: ModelerUserConfig
    # 모델러 시스템 설정
    system: ModelerSystemConfig
    # CAE 설정
    cae: CAEConfig
    # CAE 사용자 설정 (선택)
    cae_user: CAEUserConfig | None = None
    # DOE CSV 경로 (선택)
    doe_csv_path: str | None = None
    # DOE 메타데이터 경로 (선택)
    doe_metadata_path: str | None = None
