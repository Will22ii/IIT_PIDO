from dataclasses import dataclass, field


@dataclass
class CAEUserConfig:
    problem_name: str
    seed: int = 42
    objective_sense: str = "min"
    variables: list[dict] | None = None


@dataclass
class CAESystemConfig:
    use_timestamp: bool = False
    allow_latest_fallback: bool = False


@dataclass
class CAEConfig:
    user: CAEUserConfig
    system: CAESystemConfig = field(default_factory=CAESystemConfig)
