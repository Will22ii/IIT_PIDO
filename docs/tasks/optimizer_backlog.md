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
  - Legacy `surrogate_only_mode` config는 제거되었고, 현재 실제 평가는 CAE objective로 고정되어 있다.
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

- Optimizer 단계 명칭은 focus로 통일한다.
  - DOE additional의 `phase`와 혼동하지 않도록 Optimizer output/debug/metadata에서는 `phase`를 사용하지 않는다.
  - 공식 컬럼은 `segment`, `opt_focus_level`, `source_mode`이다.

## Config Cleanup

- Compatibility/default BO config 제거 후보
  - `acq_type`, `kappa_start`, `kappa_end`
  - `starts_per_iter`, `random_starts_ratio`
  - 조건: Focus3 acquisition scheduling과 fallback path가 모두 `focus3_*` config로 이동한 뒤 제거.

- Focus3 source config rename 후보
  - `source_pool_size`
  - `source_topk_fraction`
  - `source_topk_perturb_sigma`
  - `source_boundary_near_ratio`
  - 현재 이름은 shared처럼 보이지만 실제 책임은 Focus3 source-pool 생성이다.
  - 후보 rename: `focus3_source_pool_size`, `focus3_source_topk_fraction`, `focus3_source_topk_perturb_sigma`, `focus3_source_boundary_near_ratio`.

- 중복 path config 통합 후보
  - `OptimizerUserConfig.doe_csv_path`와 `OptimizerConfig.doe_csv_path`
  - `OptimizerUserConfig.explorer_bounds_path`와 `OptimizerConfig.explorer_bounds_path`
  - 장기적으로 user/input layer 하나만 남긴다.

- Metadata path fallback legacy 후보
  - `doe_metadata_path`, `explorer_metadata_path`, `modeler_metadata_path`
  - service mode에서는 `run_context`와 explicit artifact path로 연결하고, task metadata fallback은 internal resume/debug 호환용으로만 남긴다.

- `objective_sense_override` 제거 후보
  - CAE metadata objective sense가 source of truth다.
  - 반대 방향 benchmark 실험이 필요하면 별도 experiment-only config로 격리한다.

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
