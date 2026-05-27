# Explorer Policy

Explorer는 실행된 data를 읽고, optional Modeler output을 layer로 사용할 수 있다. CAE context와 input CSV만으로 standalone objective-data operation이 가능해야 한다.

## Inputs

필수 입력:

- CAE context
- `objective`와 active feature columns를 포함한 input CSV

Optional 입력:

- `selected_features.csv`
- model bundle PKL
- post-feasibility model PKL
- FI scores path

Explorer는 runtime 동작을 위해 DOE metadata나 Modeler metadata를 요구하지 않는다.

Input resolution은 `Explorer/executor/input_workflow.py`가 담당한다. Strategy alias/source-mode 결정은 `Explorer/executor/strategy_workflow.py`가 담당한다. Bounds/array/acquisition helper는 `Explorer/executor/math_workflow.py`가 담당한다. Debug plot 생성은 `Explorer/executor/plot_workflow.py`가 담당한다. Report metadata 조립은 `Explorer/executor/report_workflow.py`가 담당한다. Public/meta/debug artifact 저장은 `Explorer/executor/output_workflow.py`가 담당한다. Orchestrator는 resolved input bundle을 받아 candidate generation과 refinement 계산을 수행한 뒤 plot/output workflow로 후처리를 위임한다.

## Input CSV 정책

Explorer input CSV는 아래 경로 중 하나에서 올 수 있다.

- `ExplorerConfig.doe_csv_path`
- 현재 run의 DOE public CSV: `DOE/artifacts/public/doe_results.csv`
- user-provided external CSV

필수 column:

- `objective`
- active feature columns 전체

Optional column:

- `success`
- `feasible`

`success` 또는 `feasible`이 없어도 허용한다. Column이 있으면 filtering에 사용할 수 있다.

CSV의 constraint column은 schema로 읽지 않는다. Constraints는 CAE에서 온다.

## Active Feature Resolution

Explorer는 active feature를 아래 우선순위로 결정한다.

1. model bundle `feature_cols`
2. model bundle이 없을 때 `selected_features.csv`의 selected rows
3. CAE design features

Resolved features는 CAE design features의 subset이어야 한다.

Input CSV에는 모든 active feature column이 있어야 한다. 빠진 column이 있으면 fast-fail한다.

Model bundle이 있으면 bundle의 `feature_cols`가 active features의 권위 소스다. `selected_features.csv`는 사용자-facing feature-selection report이며 FI score 입력으로 참조될 수 있다.

## Model Layer 정책

Model layer는 optional이다.

Model bundle이 있으면 Explorer는 LHC/boundary candidate와 prediction cluster를 만들 수 있다.

Model bundle이 없으면 prediction candidate generation과 prediction cluster 생성을 건너뛴다. Objective-data strategy는 input CSV만으로 동작해야 한다.

Public strategy는 `S4_obj`와 `S4_dual`만 둔다.

Model layer 없이 `S4_dual`을 요청하면 prediction cluster를 만들 수 없으므로 `S4_obj`로 degrade한다.

## Public Outputs

주요 public artifacts:

```text
Explorer/artifacts/public/<strategy>/explorer_results_<strategy>.csv
Explorer/artifacts/public/<strategy>/selected_bounds.json
```

`selected_bounds.json`은 downstream-facing public artifact이며 Optimizer가 사용할 수 있다.

## Benchmark Diagnostic

Explorer의 역할은 downstream Optimizer가 사용할 수 있는 좋은 selected bounds를 만드는 것이다.

`run_pipelines.py` benchmark에서는 known optimum inclusion을 Explorer bounds 품질 진단값으로 볼 수 있다. 하지만 실제 `run_pipeline`/`run_AION` 서비스성 실행에서는 known optimum 개념이 없으므로, Explorer runtime contract가 아니다.

Explorer가 좋은 bounds를 만들면 Optimizer가 더 강한 goal을 달성할 가능성이 높아진다. 반대로 Optimizer-only 실행에서는 Explorer bounds가 없으므로 known optimum inclusion이 Optimizer의 성공 조건이 아니다.

## Debug Outputs

`debug_level == "on"`이고 `save_plot == True`일 때 Explorer는 아래 plot을 쓸 수 있다.

- pairwise dual cluster plots
- DOE-vs-optimum plots

DOE-vs-optimum plot은 같은 run context의 `DOE/artifacts/debug/doe_results_internal.csv`가 있으면 그 파일을 사용한다. 이 파일이 없으면 input CSV로 fallback하며, DOE metadata는 읽지 않는다.

실제 plot 조건:

```text
plots_enabled = debug_level == "on" and save_plot
```

`debug_level == "off"`이면 Explorer plot을 쓰거나 debug artifact로 등록하지 않는다.

## Metadata

Explorer metadata는 provenance 용도로 아래 input reference를 기록할 수 있다.

- input CSV
- model bundle
- selected feature CSV

이 reference들은 schema authority가 아니다.
