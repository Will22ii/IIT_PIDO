# Pipeline I/O Policy

이 문서는 현재 파이프라인의 입력, 출력, metadata, debug 산출물 정책을 정의한다.

## 핵심 원칙

`CAE_tool_interface`는 한 run의 문제 정의 기준이다. 문제 이름, 설계 변수, bounds, objective 방향, constraint 정의는 CAE context에서 온다.

각 task의 CSV는 데이터 전달용이지 schema 기준이 아니다. downstream task는 CSV column을 그대로 믿지 않고, CAE 문제 정의 또는 명시적인 selected feature list와 대조해야 한다.

Explorer는 임의 CSV의 numeric column을 보고 active feature를 추론하지 않는다. active feature는 아래 순서로 결정한다.

1. model bundle이 있으면 `feature_cols`
2. model bundle이 없고 `selected_features.csv`가 있으면 CSV의 selected feature rows
3. Modeler layer가 없을 때 CAE design features 전체

Explorer는 runtime 결정을 위해 DOE metadata나 Modeler metadata를 읽지 않는다. Explorer는 CAE context와 input CSV만 있으면 동작해야 하며, model과 selected feature는 optional layer다.

Debug 산출물은 단순하게 `debug_level = "on" | "off"`로 제어한다. 기본값은 `on`이다. batch 실행에서는 `--no-debug`로 debug 산출물을 끈다.

## Artifact Layer

모든 task는 같은 artifact 구조를 사용한다.

```text
<run_root>/<Task>/artifacts/public/
<run_root>/<Task>/artifacts/meta/
<run_root>/<Task>/artifacts/debug/
```

`public`은 사용자가 직접 보거나 downstream task가 공식 output으로 읽어도 되는 간단한 산출물이다. Public에는 compact result와 최종 user-facing artifact만 둔다.

`meta`는 resolved input, 내부 선택 결과, 구조화된 요약, analysis JSON, 재현성 정보를 담는다. meta는 debug dump가 아니므로 debug off에서도 기본적으로 유지한다. Backend는 필요하면 meta 중 일부를 선택적으로 UI에 보여줄 수 있지만, meta 자체가 user-facing public output은 아니다.

`debug`는 내부 trace, raw diagnostic table, plot, full history처럼 검증과 분석용으로 무겁거나 자세한 산출물을 둔다. `debug_level == "on"`일 때만 파일을 쓰고 metadata에 등록한다.

공통 `ResultSaver` 구조 때문에 debug off에서도 빈 `artifacts/debug/` 폴더가 생길 수 있다. 빈 폴더는 허용하지만, debug 파일과 debug metadata entry는 없어야 한다.

## Run Context 정책

pipeline run은 항상 `CAE_tool_interface` 실행과 CAE task metadata 저장으로 시작한다.

같은 run의 downstream task는 CAE context를 문제 정의 source of truth로 읽는다. 특정 task를 따로 이어서 실행할 때도 해당 run context에 CAE metadata가 이미 있어야 한다. CAE context가 없으면 fast-fail한다.

문제 정의, task list, input CSV, 중요한 task 설정이 바뀌면 기존 run을 조용히 재사용하지 않고 새 run을 만드는 것을 원칙으로 한다.

Backend가 기존 run을 이어서 실행하려면 `PipelineConfig.run_root`를 넘긴다. `run_root`가 있으면 `run_pipeline()`은 CAE를 다시 실행하지 않고 기존 run context의 CAE metadata를 검증한 뒤 선택된 task만 실행한다. `run_root`가 없으면 새 run context를 만들고 CAE task를 먼저 저장한다.

현재 검증 기준은 problem name, objective sense, 명시된 CAE variable override의 bounds다. 파일 변경, task 설정 변경, 데이터 변경을 새 run으로 강제할지 여부는 향후 project/run 정책에서 더 고도화한다.

기존 run artifact 자동 재사용은 `PipelineConfig.reuse`로 제어한다.

```text
PipelineReusePolicy.use_existing_doe_csv
PipelineReusePolicy.use_existing_modeler_artifacts
```

기본값은 둘 다 `True`다. Backend가 기존 `run_root`는 유지하되 기존 DOE CSV나 Modeler artifact를 의도적으로 쓰고 싶지 않으면 해당 값을 `False`로 넘긴다.

## 실행 진입점 정책

공식 실행 진입점은 pipeline layer에 둔다.

```text
pipeline/run_pipeline.py   # 단일 문제, task 선택형 실행
pipeline/run_AION.py       # 단일 문제, full preset 실행
pipeline/run_pipelines.py  # benchmark/batch 분석용
```

CLI는 `pipeline/run_pipelines.py`에만 둔다. `run_pipeline.py`와 `run_AION.py`는 서비스/API entrypoint이므로 JSON/dict config로만 호출한다.

```text
run_pipeline_from_dict(payload)
run_pipeline_from_json(config_path)
run_aion_from_dict(payload)
run_aion_from_json(config_path)
```

현재 backend/dev config contract 후보는 아래 top-level section을 사용한다.
최종 서비스 UI schema로 확정된 것은 아니며, 실제 parser 기준 계약은
`docs/pipeline_config_contract.md`에 별도로 정리한다.

```text
problem   # 문제명, seed, objective_sense, variables
run       # run_root, debug_level, use_timestamp
tasks     # doe/modeler/explorer/optimizer on/off (run_pipeline only)
reuse     # 기존 run artifact fallback 정책
inputs    # 외부 CSV/model/bounds path
doe       # DOE 사용자 입력
modeler   # Modeler 사용자 입력
explorer  # Explorer 사용자 입력
optimizer # Optimizer 사용자 입력
```

Config example은 아래에 둔다.

```text
pipeline/config_templates/run_pipeline.example.json
pipeline/config_templates/run_AION.example.json
```

Task별 `run_*.py` 파일은 직접 실행용이 아니라 내부 runner module이다.

```text
CAE_tool_interface/run_CAE.py
DOE/run_DOE.py
Modeler/run_Modeler.py
Explorer/run_Explorer.py
Optimizer/run_Optimizer.py
```

사용자 또는 backend는 task별 runner를 직접 실행하지 않고 `run_pipeline()` 또는 preset runner를 통해 조합 실행한다.

## CSV 정책

외부 CSV는 허용한다. 단, 현재 CAE 문제 정의와 맞는지 검증해야 한다.

### Objective column contract

Pipeline 내부 task는 objective를 항상 minimization score로 해석한다.

- `objective`: task 내부 계산용 canonical minimization score. 작을수록 좋다.
- `objective_raw`: 사용자/CAE/CAD가 낸 원래 objective 값.
- `objective_sense`: CAE context의 원래 문제 방향. `min` 또는 `max`.
- `canonical_objective_sense`: `min`.
- `objective_transform`: `identity` 또는 `negate`.

사용자가 제공하는 constraint, goal, report/UI에 노출되는 objective는 모두 raw 기준이다.
계산용 ranking, surrogate training, acquisition, bounds selection은 `objective` 기준이다.

`max` 문제에서 CAE/CAD가 raw objective `y`를 반환하면 ingestion boundary에서 아래와 같이 저장한다.

```text
objective_raw = y
objective     = -y
```

`min` 문제에서는 `objective_raw = objective = y`다.

실패 또는 invalid objective는 방향과 무관하게 `objective = inf`로 저장한다. `max` 문제라고 해서 실패값 `inf`를 `-inf`로 뒤집으면 best point처럼 보이는 치명적인 오류가 된다.

### External CSV objective policy

서비스/API/backend가 직접 넣는 external CSV는 raw objective를 보내는 것을 원칙으로 한다.
Pipeline 입구는 CAE context의 `objective_sense`를 보고 canonical `objective`를 만든다.

권장 external CSV schema:

- active feature column 전체
- `objective_raw`
- optional `objective`
- optional `success`
- optional `feasible`

외부 CSV에 `objective_raw`가 있으면 `objective_raw`가 source of truth다. 이때 `objective`가 없거나 현재 CAE context의 `objective_sense`와 맞지 않으면 pipeline은 `objective_raw`에서 `objective`를 다시 계산해야 한다.

외부 CSV에 `objective`만 있고 `objective_raw`가 없으면 legacy/raw CSV로 간주한다. 현재 CAE context가 `max`이면 ingestion boundary에서 `objective_raw = objective`, `objective = -objective_raw`로 변환한다. 현재 CAE context가 `min`이면 `objective_raw = objective`로 둔다.

이미 canonicalized된 external CSV를 넣고 싶다면 반드시 `objective_raw`와 metadata/flag로 그 사실을 명시해야 한다. raw인지 canonical인지 알 수 없는 `objective`만 있는 CSV를 그대로 downstream에 전달하면 `max` 문제에서 방향이 뒤집힐 수 있다.

Explorer에 필요한 column:

- `objective`
- `objective_raw` 권장
- active feature column 전체

Explorer에서 optional인 column:

- `success`
- `feasible`

`success` 또는 `feasible`이 없으면 기본적으로 row가 usable하다고 본다. column이 있으면 task logic에서 filtering에 사용할 수 있다.

CSV의 constraint column은 schema 입력으로 쓰지 않는다. constraint 정의는 CSV가 아니라 `CAE_tool_interface`에서 온다.

### Future public CSV / DB contract

현재 public CSV는 task별 구현 이력이 섞여 있어 `objective`의 의미가 완전히 동일하지 않을 수 있다.
예를 들어 DOE public CSV의 `objective`는 canonical minimization score이고, Optimizer public CSV의 `objective`는 user-facing effective objective에 가깝다. Downstream ingestion은 `objective_raw`를 source of truth로 다시 canonicalize하므로 계산상 문제는 없지만, 사람이 CSV를 직접 읽거나 backend가 그대로 DB에 적재하기에는 혼동될 수 있다.

향후 public CSV는 "유저가 다운로드한 뒤 다시 업로드할 수 있는 wide form"을 목표로 정리한다. Public CSV의 기본 source of truth는 raw objective다.

권장 public evaluation CSV:

```text
eval_id
task
iter
source
success
feasible
objective_raw
objective_internal
objective_effective_raw
objective_sense
x1
x2
...
xn
```

Column 의미:

- `objective_raw`: CAE/CAD/사용자 기준 원래 objective. UI/goal/constraint 기준.
- `objective_internal`: pipeline 계산용 canonical minimization score. `objective_sense=max`이면 `-objective_raw`.
- `objective_effective_raw`: post penalty 등을 raw scale에서 반영한 user-facing objective. penalty가 없으면 `objective_raw`와 같다.
- `objective_sense`: CAE context의 원래 문제 방향. `min` 또는 `max`.
- `x1...xn`: active design variable wide columns. 유저 upload/download 편의용.

Public CSV에서는 `objective`처럼 의미가 모호한 이름을 새 계약의 핵심 컬럼으로 쓰지 않는 방향을 선호한다. 하위 호환을 위해 남기더라도 `objective_raw` 또는 `objective_internal` 중 어느 것의 alias인지 metadata에 명확히 기록해야 한다.

Backend/DB 저장은 wide CSV를 그대로 테이블로 만들지 않는다. 변수 개수와 constraint 개수가 문제마다 달라지므로 normalized form을 기본으로 한다.

권장 DB 구조:

```text
evaluation
- eval_id
- run_id
- task
- iter
- source
- success
- feasible
- objective_raw
- objective_internal
- objective_effective_raw
- objective_sense

evaluation_variable
- eval_id
- variable_name
- value

evaluation_constraint
- eval_id
- constraint_id
- scope
- value
- feasible
- margin
```

사용자 외부 입력 CSV는 public CSV보다 더 단순해도 된다. 최소 입력은 active feature column 전체와 raw objective다.

```text
x1,x2,...,xn,objective_raw
```

호환 입력으로 `y`, `obj`, `objective` 같은 alias를 받을 수는 있지만, ingestion boundary에서 반드시 `objective_raw`로 표준화하고 CAE metadata의 `objective_sense`를 기준으로 `objective_internal`을 다시 계산한다. 사용자가 준 `objective_internal`은 기본적으로 신뢰하지 않는다.

## Downstream 입력 우선순위

`run_pipeline()`은 downstream data/model layer를 아래 우선순위로 연결한다.

### DOE/Input CSV

1. Task config에 명시된 explicit input CSV
   - `ModelerConfig.doe_csv_path`
   - `ExplorerConfig.doe_csv_path`
   - `OptimizerConfig.user.doe_csv_path`
   - `OptimizerConfig.doe_csv_path`
2. 이번 `run_pipeline()` 호출에서 새로 실행한 DOE public CSV
3. 기존 `run_root` 안의 DOE public CSV
   - `DOE/artifacts/public/doe_results.csv`

Explicit CSV가 있으면 가장 우선한다. 따라서 `run_doe=True`로 DOE를 새로 실행하더라도 downstream에는 explicit CSV가 기본 입력으로 전달된다. 새 DOE 결과를 downstream에 쓰고 싶으면 explicit CSV를 넘기지 않는다.

여러 task에 서로 다른 explicit CSV가 동시에 들어오면 현재 정책에서는 fast-fail한다. Task별로 다른 CSV를 의도적으로 쓰는 정책은 나중에 별도 옵션으로 확장한다.

`PipelineReusePolicy.use_existing_doe_csv=False`이면 3번 fallback은 사용하지 않는다.

### Modeler layer

Explorer의 model/selected feature layer는 아래 우선순위로 연결한다.

1. Explorer config에 명시된 explicit model/selected feature path
   - `ExplorerConfig.model_pkl_path`
   - `ExplorerConfig.selected_features_csv_path`
2. 이번 `run_pipeline()` 호출에서 새로 실행한 Modeler public artifacts
   - `Modeler/artifacts/public/modeler_selected_models.pkl`
   - `Modeler/artifacts/public/selected_features.csv`
3. 기존 `run_root` 안의 Modeler public artifacts

Modeler layer가 없으면 Explorer는 CAE design features 전체와 input CSV의 objective data로 동작한다. 이때 prediction model 기반 candidate generation과 prediction cluster는 비활성화된다.

`PipelineReusePolicy.use_existing_modeler_artifacts=False`이면 3번 fallback은 사용하지 않는다.

## AION preset 정책

`pipeline/run_AION.py`는 task 선택권이 없는 full preset runner다. Backend/Python code에서 `run_aion(config=AIONConfig(...))` 형태로 호출한다.

AION preset은 아래 task를 모두 실행한다.

```text
CAE -> DOE(additional on) -> Modeler(primary selection only) -> Explorer(S4_dual) -> Optimizer
```

AION은 실험/운영에서 리소스를 더 사용하더라도 최선의 결과를 노리는 기본 조합이다. `additional DOE`와 `Modeler`는 필수이며, Explorer strategy는 `S4_dual`로 고정한다. Secondary selection은 사용하지 않는다. 세부 preset은 `AIONConfig`와 `build_aion_pipeline_config()`에서 관리한다.

일반 `run_pipeline()`의 기본 task config 정책은 AION과 다르다.

```text
DOE additional: optional, default off
Modeler: primary selection on, secondary selection off
Explorer: S4_obj default
Optimizer: 연결은 유지하되 향후 고도화 대상
```

## Feature 정책

CAE design features는 가능한 전체 feature universe다.

Modeler는 model bundle의 `feature_cols`를 통해 더 작은 active feature set을 제공할 수 있다. `selected_features.csv`는 사용자가 볼 수 있는 feature-selection report이며, model bundle이 없을 때만 downstream selected-feature fallback으로 사용한다. 이 selected feature list는 CAE 문제 정의를 바꾸는 것이 아니라, downstream feature-reduced operation을 위한 override다.

Selected feature가 CAE design feature에 없는 이름을 포함하면 fast-fail한다.

Input CSV에 active feature column이 하나라도 없으면 fast-fail한다.

Model bundle을 사용하는 경우, bundle의 `feature_cols`가 active selected feature list의 권위 소스다.

## Debug 정책

공식 값:

```text
debug_level = "on"
debug_level = "off"
```

기본값:

```text
debug_level = "on"
```

Legacy 값:

```text
debug_level = "full"
```

`full`은 기존 config 호환을 위해 `on` alias로만 허용한다. 새 config와 문서는 `on`을 사용한다.

Batch runner:

```text
python -m pipeline.run_pipelines             # debug on
python -m pipeline.run_pipelines --no-debug  # debug off
```

Debug off일 때:

- DOE internal CSV를 쓰지 않는다.
- Modeler raw FI debug table과 FI plot을 유지하지 않는다.
- Explorer plot을 쓰지 않는다.
- Optimizer full history CSV를 쓰지 않는다.

## Optimizer artifact 정책

Optimizer의 public artifact는 최종 사용자/프론트가 바로 볼 수 있는 결과만 남긴다.

```text
OPT/artifacts/public/opt_results.csv
OPT/artifacts/public/best_point.json
```

`opt_results.csv`는 Optimizer가 새로 CAE 평가한 점만 기록한다. 입력 DOE/CSV archive row는 포함하지 않는다.

아래 파일은 실행 검증, 재현, backend audit용이므로 meta에 둔다.

```text
OPT/artifacts/meta/selected_features.csv
OPT/artifacts/meta/optimizer_inputs.json
OPT/artifacts/meta/focus_regions.json
OPT/artifacts/meta/focus_bounds.json
```

`selected_features.csv`와 `optimizer_inputs.json`은 사용자가 넣은 값을 다시 보여주기 위한 public report가 아니라, Optimizer input resolve가 어떤 feature/bounds/path를 실제로 채택했는지 검증하는 metadata다.

`focus_regions.json`과 `focus_bounds.json`은 Focus2 내부 selected-bounds generator와 region manager의 audit artifact다. Explorer/user selected bounds가 없어도 Focus2가 generated bounds를 만들면 `focus_bounds.json`에 기록되고, Focus3는 이 bounds를 in-memory로 사용한다. 이 파일은 필요 시 backend가 선택적으로 UI에 노출할 수 있지만 기본 public 결과는 아니다.

## Task별 문서

Task별 상세 정책은 아래 문서에 둔다.

```text
docs/tasks/cae_tool_interface.md
docs/tasks/doe.md
docs/tasks/modeler.md
docs/tasks/explorer.md
docs/tasks/optimizer.md
```
