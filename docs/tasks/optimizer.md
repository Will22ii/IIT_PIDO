# Optimizer Policy

Optimizer는 CAE context, optional DOE data, optional Explorer bounds, optional Modeler feasibility 정보를 사용해 optimized point를 제안한다.

## Inputs

필수 입력:

- CAE context

Optional 입력:

- initial/archive point 용 DOE dataframe 또는 DOE CSV
- Explorer selected bounds
- Modeler selected features
- Modeler post-feasibility model

## Current Behavior

Optimizer는 현재 optional input resolve를 위해 DOE, Explorer, Modeler metadata를 읽을 수 있다.

- DOE metadata: DOE dataframe fallback
- Explorer metadata: `selected_bounds.json`
- Modeler metadata: selected features와 feasibility model path

이는 Explorer standalone 정책보다 느슨한 현재 예외다.

## Preferred Direction

Optimizer도 장기적으로 Explorer와 같은 layered input 구조를 따르는 것이 좋다.

1. CAE context는 필수다.
2. Data CSV는 optional이다.
3. selected bounds는 optional이며 public artifact path 또는 direct config path로 받아야 한다.
4. selected features는 optional이며 `selected_features.csv` 또는 direct config path로 받아야 한다.
5. feasibility model은 optional이며 direct public artifact path로 받아야 한다.

장기 목표는 runtime 결정을 위해 Modeler 또는 Explorer metadata를 필수로 요구하지 않는 것이다.

## Public Outputs

주요 public artifacts:

```text
OPT/artifacts/public/opt_results.csv
OPT/artifacts/public/best_point.json
```

Public optimizer CSV는 compact하고 downstream/user-facing이어야 한다.

## Debug Outputs

`debug_level == "on"`일 때 Optimizer는 아래 파일을 쓴다.

```text
OPT/artifacts/debug/optimizer_history_full.csv
```

`debug_level == "off"`이면 full history CSV를 쓰거나 metadata에 등록하지 않는다.

## Validation 정책

Selected features는 CAE bounds 기준으로 resolve되어야 한다.

DOE data가 제공되었지만 selected feature column이 하나도 없으면 Optimizer는 그 DOE data를 initial source로 쓰지 않는다.

Partial DOE feature match는 현재 matched features만 사용하는 방식으로 허용된다. 이 동작은 Optimizer input policy를 더 엄격하게 정리할 때 재검토해야 한다.

