# Modeler Policy

Modeler는 optional surrogate/model artifact를 만들고, active selected feature list를 제공할 수 있다.

Modeler는 optional task다. Downstream task는 strategy가 허용하는 경우 Modeler output 없이도 objective-data-only 방식으로 동작해야 한다.

## Inputs

Modeler 실행 시 필수 입력:

- CAE context
- DOE dataframe 또는 DOE/public input CSV

Modeler는 CAE design variables를 전체 feature universe로 사용한다.

## Public Outputs

주요 public artifacts:

```text
Modeler/artifacts/public/modeler_selected_models.pkl
Modeler/artifacts/public/selected_features.csv
```

Optional public artifact:

```text
Modeler/artifacts/public/modeler_feas_models.pkl
```

Model bundle은 trained models와 `feature_cols`를 포함해야 한다. Model-backed downstream task에서는 이 `feature_cols`가 active feature schema의 권위 소스다.

`selected_features.csv`는 사람이 보기 좋은 feature-selection report다. 반드시 `feature` column을 포함해야 하며, `selected` column이 있으면 selected row가 active selected features를 뜻한다.

## Selected Feature 정책

Modeler selected features는 downstream feature-reduced operation에서 CAE full feature list보다 우선한다.

Selected features는 CAE design features의 subset이어야 한다.

Model bundle과 selected features report를 같이 사용하는 경우, model bundle의 `feature_cols`를 downstream model input schema로 사용한다.

## Debug Outputs

`debug_level == "on"`일 때 Modeler는 아래 debug 산출물을 유지할 수 있다.

- raw permutation importance tables
- raw score-drop tables
- feature-selection plots
- secondary-selection diagnostic plots

`debug_level == "off"`이면 raw FI debug table과 plot을 유지하거나 debug artifact로 등록하지 않는다.

## Metadata

Modeler는 processed analysis summary를 아래 위치에 저장할 수 있다.

```text
Modeler/artifacts/meta/
```

이 summary는 debug dump가 아니다. Batch analysis와 재현성을 위해 유지할 수 있다.

Explorer는 runtime에 Modeler metadata를 요구하지 않는다. 명시적으로 전달되었거나 같은 run에서 생성된 Modeler public artifact만 optional layer로 사용할 수 있다.
