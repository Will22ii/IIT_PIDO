# Optimizer Policy

Optimizer는 CAE context, optional initial data, optional Modeler model layer, optional Explorer bounds를 사용해 optimized point를 제안한다.

## Inputs

필수 입력:

- CAE context

Optional 입력:

- initial/archive point 용 DOE dataframe 또는 외부 CSV
- Modeler model bundle PKL
- selected features CSV
- Explorer selected bounds JSON
- Modeler post-feasibility model

## Target Input Definition

Optimizer의 주요 입력 레이어는 3개다.

```text
1. initial data layer
   - DOE output 또는 외부 CSV
   - objective column이 있으면 warm-start archive로 사용한다.
   - objective column이 없으면 현재 BO 학습 데이터로는 직접 쓰기 어렵다.

2. model layer
   - Modeler PKL 또는 직접 제공된 model bundle
   - 모델과 feature schema를 포함할 수 있다.
   - selected_features.csv를 별도로 받을 수 있다.

3. bounds layer
   - Explorer selected_bounds.json 또는 직접 제공된 bounds
   - Optimizer search space를 좁히는 용도다.
```

`CAE context`는 항상 문제 정의 source of truth다. 변수 이름, 전체 bounds, objective 방향, constraint 정의는 CAE에서 온다.

## Feature Resolution Policy

Optimizer에서 사용할 feature set은 명시적인 user/task input이 가장 우선한다. 장기 정책은 task마다 사용할 feature list를 config로 받을 수 있게 하는 것이다.

현재 정리 대상 우선순위:

```text
1. Optimizer user-selected feature list
2. selected_features.csv
3. model bundle feature_cols
4. input CSV feature columns
5. CAE design features 전체
```

단, model bundle을 실제 surrogate evaluation에 사용할 때는 model의 `feature_cols`와 실행 feature list가 호환되어야 한다. 사용자가 넣은 selected feature CSV가 model bundle의 feature schema와 다르면 fast-fail 또는 명시적 fallback 정책이 필요하다.

Explorer bounds는 feature 선택을 결정하지 않는다. Bounds는 선택된 feature 중 해당 변수의 search bounds를 좁히는 레이어다. Bounds가 없는 feature는 CAE 전체 bounds를 사용한다.

## Initial Data Policy

DOE output과 외부 CSV는 Optimizer 관점에서 initial guess 또는 warm-start archive다.

필수에 가까운 column:

- active feature columns
- `objective`

`objective`가 있으면:

- BO 초기 archive로 사용할 수 있다.
- top-k warm-start와 source mixture의 `topk` source에 사용할 수 있다.

`objective`가 없으면:

- 현재 정책에서는 BO 학습 데이터로 쓰기 어렵다.
- 추후 "planned points" 또는 "candidate hints"로 처리하는 별도 정책을 검토한다.

CSV column에서 파생 objective를 조합하는 편의 기능은 후순위 backlog로 둔다.

## Bounds Policy

Explorer selected bounds는 Optimizer search space를 제한한다.

정책:

- bounds JSON은 변수별 `{lb, ub}`를 제공한다.
- bounds가 있는 feature는 해당 bounds를 사용한다.
- bounds가 없는 selected feature는 CAE 전체 bounds를 사용한다.
- bounds에 있지만 Optimizer selected feature가 아닌 변수는 무시한다.
- bounds 값은 CAE 전체 bounds 안에 있어야 하며, 벗어나면 clip보다 fast-fail을 우선 검토한다.

## Current Behavior

Optimizer는 현재 optional input resolve를 위해 DOE, Explorer, Modeler metadata를 읽을 수 있다.

- DOE metadata: DOE dataframe fallback
- Explorer metadata: `selected_bounds.json`
- Modeler metadata: selected features와 feasibility model path

이는 Explorer standalone 정책보다 느슨한 현재 예외다.

현재 BO focus 명칭 정책:

- DOE objective가 있으면 바로 `focus3 / point_converge`에서 시작한다.
- DOE objective가 없고 `n_samples < no_doe_mode_threshold`이면 `focus1 -> focus2`만 돈다.
- DOE objective가 없고 `n_samples >= no_doe_mode_threshold`이면 `focus1 -> focus2 -> focus3`이 돈다.

기존 `phase` 컬럼은 compatibility 용도로 유지한다. 공식 새 컬럼은 `opt_focus_level`, `opt_focus_name`이다.

## Preferred Direction

Optimizer도 장기적으로 Explorer와 같은 layered input 구조를 따르는 것이 좋다.

1. CAE context는 필수다.
2. Data CSV는 optional이다.
3. selected bounds는 optional이며 public artifact path 또는 direct config path로 받아야 한다.
4. selected features는 optional이며 `selected_features.csv` 또는 direct config path로 받아야 한다.
5. feasibility model은 optional이며 direct public artifact path로 받아야 한다.

장기 목표는 runtime 결정을 위해 Modeler 또는 Explorer metadata를 필수로 요구하지 않는 것이다.

공식 용어:

```text
opt_focus_level = 1
opt_focus_name  = space_scan

opt_focus_level = 2
opt_focus_name  = region_focus

opt_focus_level = 3
opt_focus_name  = point_converge

opt_focus_level = 4
opt_focus_name  = final_verify  # reserved
```

`phase`라는 단어는 DOE additional에서도 쓰이므로 Optimizer의 공식 진행 상태는 focus로 표현한다. Public/debug output에는 기존 호환용 `phase`를 잠시 유지한다.

## Focus3 Budget Policy

`focus3 / point_converge`는 Explorer boundary, DOE objective, optional Modeler layer가 이미 있는 상태에서 하나의 점으로 수렴하는 마지막 focus다.

Focus3 내부 전략은 Optimizer budget과 active feature 수의 비율로 coarse하게 나눈다.

```text
focus3_budget_ratio = n_opt / max(p_dim, 1)

ultra_low : ratio < 3
low       : 3 <= ratio < 8
normal    : 8 <= ratio < 20
rich      : 20 <= ratio
```

현재 1차 구현은 기존 `topk / boundary / random` source mixture를 budget class별로 조정한다. 작은 budget에서는 random을 강하게 줄이고 DOE top 근처 후보를 우선한다.

```text
ultra_low : topk 0.85 / boundary 0.10 / random 0.05
low       : topk 0.75 / boundary 0.15 / random 0.10
normal    : topk 0.60 / boundary 0.25 / random 0.15
rich      : topk 0.45 / boundary 0.35 / random 0.20
```

이 정책은 실제 CAE objective 평가를 유지한다. Modeler surrogate-only objective는 아직 사용하지 않는다.

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
