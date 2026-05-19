# Optimizer Backlog

이 문서는 Optimizer 고도화 중 당장 구현하지 않고 보류한 기능과 정책 결정을 기록한다.

## Initial Data / CSV

- Objective 없는 CSV 처리 정책 정의
  - 현재 BO warm-start에는 `objective`가 필요하다.
  - Objective 없는 CSV는 "planned points", "candidate hints", "initial evaluation plan" 중 어떤 의미로 볼지 결정해야 한다.

- CSV column 기반 derived objective 기능
  - CSV에 `objective` 대신 `y`, `mass`, `stress`, `constraint margin` 같은 column이 있을 때 사용자 식으로 objective를 조합하는 편의 기능.
  - 예: `objective = mass + penalty * max(stress - limit, 0)`.
  - 서비스 UI에서 식 입력/검증이 필요하므로 후순위.

- CSV schema strictness 결정
  - Partial feature match를 허용할지 fast-fail할지 재검토.
  - 현재 Optimizer는 partial match 시 matched feature만 사용하는 느슨한 동작이 있다.

## Model Layer

- Model bundle 기반 surrogate-only BO
  - 실제 CAE 평가 없이 Modeler PKL로 objective를 예측하며 BO를 돌리는 모드.
  - 현재 `surrogate_only_mode` config는 있지만 실제 평가는 CAE objective로 전환되어 있다.
  - Model uncertainty, ensemble std, post feasibility model 결합 정책이 필요하다.

- Model bundle feature schema 검증
  - 사용자가 직접 넣은 selected_features.csv와 model bundle `feature_cols`가 다를 때 정책 필요.
  - 후보 정책:
    - strict fast-fail
    - model 사용 비활성화 후 CAE-only mode
    - feature intersection 사용
  - 모델 입력 차원 오류를 막기 위해 기본은 strict fast-fail 후보.

- selected_features.csv 직접 입력 정책
  - `selected` column이 있으면 True row만 사용.
  - `selected` column이 없으면 `feature` column 전체 사용.
  - CAE design features에 없는 feature는 fast-fail.

## Bounds Layer

- Explorer bounds strict validation
  - bounds가 CAE 전체 bounds 밖이면 clip할지 fast-fail할지 결정.
  - 추천 기본값은 fast-fail.

- Partial bounds 정책
  - selected feature 중 bounds가 없는 변수는 CAE 전체 bounds를 쓰는 방향.
  - bounds에 있지만 selected feature가 아닌 변수는 무시.

- 사용자 직접 bounds override
  - 서비스 UI에서 Optimizer 전용 bounds를 직접 줄 수 있게 할지 결정.
  - Explorer bounds보다 사용자 직접 입력 bounds가 우선해야 한다.

## Naming / Focus

- `phase` legacy 제거 시점 결정
  - DOE additional에서도 phase를 쓰므로 Optimizer는 `opt_focus_level`, `opt_focus_name`을 공식 컬럼으로 사용한다.
  - 기존 `phase` 컬럼은 compatibility 용도로 일정 기간 유지.

- 추천 mapping

```text
current condition                         opt_focus_level          opt_focus_name
DOE objective 있음                         3                        point_converge
DOE objective 없음, small budget focus1    1                        space_scan
DOE objective 없음, small budget focus2    2                        region_focus
DOE objective 없음, large budget focus1    1                        space_scan
DOE objective 없음, large budget focus2    2                        region_focus
DOE objective 없음, large budget focus3    3                        point_converge
Reserved future focus4                     4                        final_verify
```

## Algorithm

- DOE warm-start source mixture 재설계
  - 현재 topk/boundary/random 확률 기반 source mixture를 사용한다.
  - Explorer bounds와 Modeler model layer가 있을 때 source weight를 어떻게 조정할지 재검토.
  - 1차 구현에서는 focus3 budget class별 확률 조정만 적용했다.
  - 다음 단계는 `random` source를 단순 uniform이 아니라 trust-region/model-agreement/uncertainty source로 대체하는 것이다.

- Cold-start 간소화 전략
  - Optimizer만 단독 실행할 때 주어진 budget 안에서 최소한 돌아가는 전략.
  - 현재는 no-DOE bootstrap 후 two-focus 또는 three-focus로 분기한다.
  - budget이 작을 때 bootstrap evaluation 개수와 BO iteration 개수의 균형 재검토.

- Pre-constraint handling
  - 현재 `enforce_pre_constraints` 기본값은 off.
  - 실제 CAD/CAE 연동에서 pre-constraint가 강한 문제의 후보 생성 실패/재시도 정책 필요.

- Post feasibility penalty
  - Modeler feasibility model이 있을 때만 활성화 가능.
  - penalty strength, hard penalty, objective sense별 score 처리 재검토.
