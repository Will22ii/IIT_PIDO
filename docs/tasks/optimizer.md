# Optimizer Policy

Optimizer는 CAE context, optional initial data, optional Modeler model layer, optional Explorer bounds를 사용해 optimized point를 제안한다.

## Goal Policy

Optimizer의 목표는 할당된 CAE evaluation budget 안에서 문제별 objective goal threshold를 달성하는 것이다.

Benchmark에서 known optimum을 알고 있더라도, Optimizer는 Focus2/Focus3 bounds 안에 known optimum 좌표를 반드시 포함할 의무가 없다. 실제 문제에서는 global optimum 위치를 알 수 없고, 사용 가능한 판단 기준은 목표 objective 수준과 관측된 best point다.

따라서 Optimizer-only benchmark에서는 다음처럼 판단한다.

- best objective가 goal threshold에 도달하면 성공이다.
- known optimum이 generated bounds 밖에 있어도, 다른 region에서 goal을 달성하면 성공이다.
- known optimum marker는 `run_pipelines.py` benchmark/debug 분석용이다.

Explorer까지 실행한 pipeline에서는 Explorer가 selected bounds를 제공하므로, Optimizer는 그 bounds 안에서 더 높은 품질의 solution을 찾는 것을 목표로 한다. 이 경우에도 Optimizer의 직접 목표는 known optimum inclusion이 아니라 downstream objective 개선이다.

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

현재 기본 Focus BO 정책은 `Optimizer.algorithms.focus_bo.focus_pipeline`에서 만든 plan이 결정한다.

Optimizer의 공식 진행 컬럼은 `segment`, `opt_focus_level`, `source_mode`이다.
`phase`는 DOE additional 전용 용어로 두고 Optimizer output에서는 사용하지 않는다.

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
segment = focus0
opt_focus_level = 0

segment = focus1
opt_focus_level = 1

segment = focus2
opt_focus_level = 2

segment = focus3
opt_focus_level = 3
```

`phase`라는 단어는 DOE additional에서만 사용한다. Optimizer의 단계 정의와 output schema는 focus로 통일한다.

## Algorithm Registry

Optimizer task의 공통 layer는 input resolve와 output save다. 실제 최적화 알고리즘은 registry에서 선택한다.

```text
optimizer.system.algorithm_id = "focus_bo"
```

현재 기본 제공 알고리즘은 `focus_bo`다. Focus0~3, focus budget orchestration, global/local GP, Focus2 region manager, Focus3 local BO는 모두 `focus_bo` 알고리즘의 내부 구현이다. Optimizer 전체 task의 고정 계약이 아니다.

다른 알고리즘은 같은 Optimizer input/output contract를 지키면 registry에 추가할 수 있다.

```text
Optimizer/algorithms/
  registry.py
  base.py
  result.py
  focus_bo/
  custom/
```

서비스에서 사용자가 올린 optimizer 알고리즘은 `Optimizer/algorithms/custom/*.py`
형태로 배치한다. Registry는 Optimizer 시작 시 이 폴더를 자동 discovery한다.
Custom 파일은 `ALGORITHM_ID`와 `Algorithm` class를 제공해야 한다.

```python
ALGORITHM_ID = "my_algorithm"


class Algorithm:
    def run(self, *, runtime, config, resolved):
        for _ in range(config.user.n_samples):
            x = runtime.sample_uniform(1)[0]
            runtime.evaluate(x, source_mode=ALGORITHM_ID, segment="custom")
            if runtime.should_stop:
                break
        return runtime.build_result(algorithm_id=ALGORITHM_ID)
```

Custom 알고리즘은 CAE를 직접 호출하거나 산출물을 직접 저장하지 않는다.
후보점 생성만 담당하고, evaluation은 `runtime.evaluate()` 또는
`runtime.evaluate_batch()`를 통해 수행한다. Runtime이 archive/history,
best tracking, pre-constraint check, optional goal early stop, 표준 output
저장을 공통으로 처리한다.

`focus_bo`는 성능과 debug artifact 호환성을 위해 아직 전체 loop를
`OptimizerRuntime`으로 이관하지 않는다. 대신 1차 이관으로
`Optimizer.executor.evaluation_core`의 공통 evaluator를 공유한다.

```text
SelectedFeatureMapper
  selected feature vector -> full CAE variable vector

CaeObjectiveEvaluator
  CAE 호출 + sanitize + objective 반환
```

따라서 FocusBO와 custom Runtime은 동일한 feature mapping/objective
evaluation 코어를 사용한다. 2차 이관으로 pre-constraint raw evaluation과
constraint debug field 생성도 `evaluation_core`에 공통 helper로 둔다.
3차 이관으로 custom Runtime과 FocusBO가 공유하는 history row base builder를
`Optimizer.executor.history_core`에 만들었다. FocusBO는 공통 base row 위에
focus-specific diagnostics를 붙인다. Focus2/Focus3 region debug,
timing, plot 산출물은 현재 FocusBO engine 내부에 유지한다.
4차 이관으로 `Optimizer.executor.archive_core`에 archive append와
raw/effective best 갱신 helper를 분리했다. FocusBO는 기존 archive array와
debug loop를 유지하되, 새 평가점 추가와 best update 계산은 공통 helper를
사용한다.
5차 이관으로 `Optimizer.executor.history_core`에 goal monitor update와
공통 evaluation history row 생성을 묶은 helper를 추가했다. Custom Runtime과
FocusBO가 같은 goal/history 기본 필드를 만들고, FocusBO는 그 위에
Focus별 debug 진단값만 추가한다.
6차 이관으로 `Optimizer.executor.result_core`에 `OptimizerAlgorithmResult`
생성 helper를 추가했다. FocusBO와 Custom Runtime은 각자의 archive/history와
알고리즘별 summary를 준비한 뒤, 공통 result envelope은 같은 helper로 만든다.

교체 알고리즘도 아래 입력을 공통으로 받는다.

```text
CAE context
optional objective CSV/archive
optional selected bounds
optional selected features/model layer
n_samples
goal / goal_objective (optional)
algorithm_params (optional)
```

그리고 `Optimizer.algorithms.result.OptimizerAlgorithmResult` 공통 result contract를 반환해야 한다.

```text
history_df
archive_df
best_point
best_objective
n_iterations
algorithm/focus summary
```

Backend가 알고리즘 선택 UI를 구성해야 하면
`Optimizer.algorithms.describe_optimizer_algorithms()`로 현재 사용 가능한
알고리즘과 custom discovery error를 조회할 수 있다.
사용자가 업로드한 `.py` 파일을 교체한 직후 같은 프로세스 안에서 다시 조회해야 하면
`discover_custom_optimizer_algorithms(force=True)`를 호출해 registry를 재탐색한다.

## Focus Pipeline

`focus_bo`는 더 이상 `objective archive / p`만 보고 Focus1 또는 Focus3를 자동 선택하지 않는다. 어떤 focus를 실행할지는 `Optimizer.algorithms.focus_bo.focus_pipeline`에서 만든 plan을 따른다.

```text
optimizer.system.focus_pipeline = "auto"
```

또는 명시적으로 지정할 수 있다.

```text
focus_pipeline = "focus0,focus1"
focus_pipeline = "focus0,focus1,focus3"
focus_pipeline = ["focus3"]
```

`auto` 정책:

```text
Explorer/user/override selected bounds 있음:
  objective archive < focus0 target:
    focus0 추가

  objective archive < focus1 target:
    focus1 추가

  항상:
    focus3 추가

  Focus2는 auto로 추가하지 않음.
  외부 selected bounds는 downstream contract로 신뢰하고,
  Optimizer는 그 bounds 안에서 final Focus3만 수행한다.

selected bounds 없음(bounds_source=cae):
  focus0/focus1로 objective archive 보강
  n_samples >= focus2_min_total_budget and focus2_enabled:
    focus2가 내부 region bounds 생성
    focus3가 Focus2 generated bounds로 실행
  Focus2 budget이 없거나 region 생성 실패:
    Focus3 자동 진입 금지
```

Focus3는 외부 selected bounds 또는 Focus2 generated bounds가 있을 때만 실행할 수 있다. `bounds_source=cae` 상태에서 `focus3`만 명시하면 fast-fail한다. 전체 CAE bounds는 문제 정의일 뿐, Focus3가 수렴할 selected region으로 직접 취급하지 않는다.

Focus pipeline은 stage 선택뿐 아니라 budget도 배분한다. `bo_engine`은 plan이 준 target/budget을 실행한다.

```text
focus0 target:
  max(focus0_min_np_ratio * p_dim, focus0_min_points)

focus1 target:
  focus1_target_np_ratio * p_dim

focus0 budget:
  focus0 target까지 부족한 objective count

focus1 budget:
  focus1 target까지 부족한 objective count
  단 focus1_max_budget_fraction으로 상한
  focus3가 있으면 focus3 minimum budget을 침범하지 않음

focus2 budget:
  n_samples * focus2_budget_fraction
  단 focus3 minimum budget을 침범하지 않음
  auto에서는 selected bounds가 없고 n_samples >= focus2_min_total_budget일 때만 stage 후보가 됨
  selected bounds가 있어도 focus_pipeline에 focus2를 명시하면 실행 가능

focus3 budget:
  focus0/focus1/focus2 이후 남은 budget
```

기본 budget 계수:

```text
focus1_max_budget_fraction = 0.50
focus2_min_total_budget = 10
focus2_budget_fraction = 0.30
focus3_min_budget_np_ratio = 2.0
focus3_min_budget = 5
```

Focus0/1/2가 target을 만족하거나 early stop하면 남은 budget은 뒤 focus로 넘어간다. Focus2가 region bounds를 만들면 Focus3는 외부 Explorer bounds가 없어도 그 generated bounds를 사용해 실행한다. 실제 plan은 metadata의 `focus_pipeline_summary`에 `stages`, `budgets`, `targets`, `reasons`, `input_state`로 저장한다.

## Config Ownership

Optimizer config는 아래처럼 나눈다.

```text
User/service-facing:
  user.n_samples
  user.doe_csv_path
  user.explorer_bounds_path
  system.focus_pipeline
  system.debug_level

System/preset tuning:
  focus0_*
  focus1_*
  focus2_*
  focus3_*
  init_train_*
  gp_train_recent_fraction
  gp_refit_every
```

Legacy `no_doe_*` fallback, `surrogate_only_mode`, and unused `n_restarts` compatibility config는 제거되었고, 공식 Focus 실행 선택은 `focus_pipeline`이 담당한다.

## Legacy / Duplicate Config Cleanup Candidates

현재 제거 후보는 기능을 바로 삭제하지 않고 compatibility alias로 유지한다. 삭제 조건은 smoke/benchmark에서 대체 필드가 충분히 검증된 뒤다.

| 후보 | 현재 상태 | 정리 방향 |
| --- | --- | --- |
| `acq_type`, `kappa_start`, `kappa_end` | generic/default acquisition scaffold와 Focus3 fallback/recovery에서 아직 사용 | Focus3 acquisition scheduling이 `focus3_acq_type`, `focus3_recover_*`로 완전히 이동하면 제거 |
| `starts_per_iter`, `random_starts_ratio` | legacy default BO start 생성에 사용 | Focus3 plan-pool/refine path만 남으면 제거 |
| `source_topk_prob`, `source_boundary_prob`, `source_random_prob` | `focus3_budget_policy_enabled=False` fallback | budget-class source policy를 고정하면 제거 |
| `focus3_plan_pool_per_source` | 과거 hard override alias | `focus3_plan_pool_min_per_source`, `focus3_plan_pool_per_dim`, `focus3_plan_pool_max_per_source`만 남기고 제거 |
| `source_stagnation_*` | `focus3_recover_*`와 개념 중복 | benchmark 후 recovery policy로 병합 |
| `source_pool_size`, `source_topk_fraction`, `source_topk_perturb_sigma`, `source_boundary_near_ratio` | 이름은 shared지만 실제로 Focus3 source pool에서 주로 사용 | `focus3_source_*`로 rename 후 old alias 제거 |
| `objective_sense_override` | CAE objective sense를 덮어쓸 수 있음 | CAE metadata를 source of truth로 고정하면 제거 또는 benchmark-only로 격리 |
| `OptimizerConfig.doe_csv_path` vs `OptimizerUserConfig.doe_csv_path` | duplicate path source | user/input-layer path 하나로 통일 |
| `OptimizerConfig.explorer_bounds_path` vs `OptimizerUserConfig.explorer_bounds_path` | duplicate path source | user/input-layer path 하나로 통일 |
| `doe_metadata_path`, `explorer_metadata_path`, `modeler_metadata_path` | standalone/sequential fallback용 | service mode에서는 run_context + explicit artifact path로 대체하고 legacy 처리 |

이미 제거된 항목:

```text
no_doe_* fallback
surrogate_only_mode
n_restarts
phase output naming
```

## Focus0 Initial DOE

Focus0는 active bounds 안에 GP 학습을 시작할 최소 objective archive가 없을 때 실행하는 초기 데이터 생성 단계다.

```text
target_count = max(focus0_min_np_ratio * p_dim, focus0_min_points, 2)
default focus0_min_np_ratio = 1
default focus0_min_points = 3
```

Focus0는 `n_samples` 예산 안에서 실제 CAE 평가를 수행한다. 샘플링 논리는 DOE initial sampling과 같은 방향이다.

```text
LHS 중심 candidate 생성
corner 일부 포함
pre-constraint가 켜져 있으면 feasible filtering
pre-constraint가 있으면 LHS feasible 후보 중 일부는 constraint margin 근처 우선
```

기본값:

```text
focus0_initial_corner_ratio = 0.10
focus0_initial_corner_adaptive_enabled = true
focus0_margin_ratio = 0.30
focus0_margin_subset_enabled = true
focus0_probe_multiplier = 2.0
focus0_filter_safety = 1.2
```

Corner adaptive 정책은 DOE initial DOE와 같은 논리를 따른다.

```text
np_ratio = focus0_budget / p_dim

np_ratio <= 10 and p_dim <= 6:
  effective_corner_ratio = min(base_corner_ratio, 0.05)
else:
  effective_corner_ratio = base_corner_ratio
```

Constraint가 있는 경우 regular LHS 후보는 `probe_multiplier * filter_safety`만큼 더 생성한 뒤 pre-constraint feasible 후보만 남긴다. 그중 `focus0_margin_ratio`만큼은 constraint margin이 작은 후보를 우선 선택하고, 나머지는 feasible 후보에서 채운다.

Focus0가 끝나면 같은 archive를 그대로 Focus1/Focus3가 사용한다.

## Focus1 Active Classification

Focus1은 active bounds 안의 objective archive가 부족할 때 사용하는 데이터 보강 단계다. 목적은 최적점을 확정하는 것이 아니라, Focus3 또는 이후 Region Manager가 사용할 수 있을 만큼 전역 surrogate evidence를 채우는 것이다.

```text
target_count = focus1_target_np_ratio * p_dim
default focus1_target_np_ratio = 10
```

active bounds 안의 objective row 수가 target_count보다 작으면 Focus1이 `n_samples` 예산 안에서 CAE 평가를 수행한다. Focus1 평가는 예산 밖 bootstrap이 아니며 `opt_results.csv`에 그대로 기록된다. GP 학습이 가능하지 않으면 Focus1은 실행되지 않고 Focus0가 먼저 archive를 채운다.

Focus1은 batch 단위로 움직인다.

```text
default focus1_batch_size = 10
class ratio = good 20% / uncertain 60% / bad 20%
```

GP 학습이 가능하면 Focus1은 큰 candidate pool을 만들고 GP 예측으로 class를 나눈다.

```text
candidate_pool = clamp(100 * p_dim, 2048, 20000)
tau = observed objective 20% quantile  # minimize 기준
lower = mu - beta * sigma
upper = mu + beta * sigma

good      : upper <= tau
bad       : lower > tau
uncertain : lower <= tau < upper
```

class별 선택 기준:

```text
good:
  LCB = mu - beta * sigma 낮은 순서

uncertain:
  straddle = beta * sigma - abs(mu - tau) 높은 순서

bad:
  lower - tau 작은 순서
```

Focus1은 기존 archive 및 같은 batch에서 이미 고른 후보와 너무 가까운 점을 피한다. 거리는 각 feature를 active bounds 기준으로 0~1 normalize한 뒤 RMS 거리로 계산한다.

```text
default focus1_min_rms_distance = 0.03
```

Focus1 early stop은 classification 지표만으로 판단하지 않는다. 데이터가 너무 적으면 GP가 우연히 안정되어 보일 수 있으므로, 최소 archive 밀도를 만족해야 조기 종료를 허용한다.

```text
early stop 조건:
  archive_count_after_batch / p_dim >= focus1_early_stop_min_data_ratio
  uncertain_ratio <= focus1_early_stop_uncertain_ratio
  good_count >= focus1_early_stop_min_good_candidates

default:
  focus1_early_stop_min_data_ratio = 10.0
  focus1_early_stop_uncertain_ratio = 0.25
  focus1_early_stop_min_good_candidates = 10
```

따라서 p=5 문제는 최소 50개 archive를 확보하기 전에는 Focus1이 classification 안정 신호만으로 Focus2/3에 넘기지 않는다.

## Focus3 Budget Policy

`focus3`는 Explorer boundary, DOE objective, optional Modeler layer가 이미 있는 상태에서 하나의 점으로 수렴하는 마지막 focus다.

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

Focus3 execution은 single BO다. 한 iteration에서 실제 CAE 평가는 1개만 수행한다. 대신 내부 후보 탐색은 `X_plan -> refine starts -> X_exec` 구조를 사용한다.

```text
topk / boundary / random source별 X_plan pool 생성
GP acquisition으로 source별 pool scoring
source 비율대로 총 focus3_refine_starts개 선택
선택된 starts만 L-BFGS-B multi-start refine
최종 1개 x_next만 CAE 평가
```

기본값:

```text
focus3_plan_pool_per_source = clamp(80 * p_dim, 1024, 4096)
focus3_refine_starts = 30
```

`n_samples`는 CAE 평가 횟수이고, Focus3 plan pool 크기와 직접 연결하지 않는다.

Focus3 acquisition은 `auto`가 기본이다. `auto`는 selected region이 충분히 믿을 만하면 EI를 쓰고, 데이터가 부족하거나 GP fallback이 발생했거나 bounds가 아직 넓으면 LCB로 fallback한다.

```text
focus3_acq_type = auto
focus3_auto_ei_min_data_ratio = 15.0
focus3_auto_ei_max_volume_ratio = 0.25
focus3_auto_ei_max_mean_width_ratio = 0.75
```

명시적으로 `focus3_acq_type = EI` 또는 `LCB`를 설정하면 그 값을 그대로 사용한다. `auto`에서 EI를 쓰는 조건은 Focus3가 최종 수렴 단계라는 전제에 맞춘 것이다. 다만 25% volume bounds라도 고차원에서는 축별 폭이 넓을 수 있으므로, mean width ratio도 같이 보고 넓으면 LCB로 유지한다.

Source 비율은 budget class를 base로 하고, 아래 신호로 adaptive 보정한다.

```text
data_ratio = n_train / p_dim

data_ratio < focus3_data_ratio_low:
  boundary/random 증가

focus3_data_ratio_low <= data_ratio < focus3_data_ratio_good:
  random 소폭 증가

gp_fallback_used:
  boundary/random 증가

recent best 개선:
  topk 증가

recent best 정체:
  boundary/random 증가
```

Random source는 pure random이 아니라 coverage-aware random으로 동작한다. random pool을 넉넉히 만든 뒤 acquisition이 나쁘지 않은 후보만 남기고, 그 안에서 기존 archive와 normalized distance가 먼 후보를 refine start로 우선 선택한다.

```text
random pool 생성
acquisition score 상위 focus3_random_cover_acq_quantile 비율 유지
archive 기준 normalized nearest-distance 계산
거리 먼 후보를 greedy 선택
```

기본값:

```text
focus3_random_cover_enabled = true
focus3_random_cover_acq_quantile = 0.70
focus3_random_cover_reference_max = 1000
```

이 정책은 GP 신뢰도가 낮거나 최근 개선이 정체되어 random 비율이 커졌을 때, 완전 무작위 탐색 대신 아직 덜 본 영역 중 acquisition이 완전히 나쁘지 않은 곳을 보게 하기 위한 장치다.

Focus3는 최종 후보가 기존 archive와 너무 가까우면 같은 refine pool에서 acquisition score가 좋은 다음 후보를 찾는다. 그래도 없으면 제한 횟수 안에서 random fallback을 시도한다.

```text
focus3_dedup_enabled = true
focus3_min_candidate_rms_distance = 0.01
focus3_dedup_random_attempts = 64
```

Focus3가 일정 window 동안 개선을 만들지 못하면 recovery 모드가 켜진다. Recovery는 boundary/random source 비율을 일시적으로 올리고 LCB kappa를 키워 현재 best 주변에만 붙는 현상을 완화한다.

```text
focus3_recover_enabled = true
focus3_recover_window = 5
focus3_recover_boundary_bonus = 0.15
focus3_recover_random_bonus = 0.10
focus3_recover_kappa_multiplier = 1.50
```

Focus3 diagnostics는 이후 adaptive source control을 위해 아래 값을 history에 남긴다.

```text
source별 selected refine starts
source별 best plan acquisition score
최종 x_next에 가장 가까운 refine source
random cover 후보 수/거리
recover active/reason
dedup applied/fallback/archive distance
GP prediction error
raw/effective objective improvement
```

## GP Train Set Policy

Optimizer는 입력 CSV/DOE objective row와 새 BO 평가 row를 모두 archive에 보존한다.

GP 학습에는 archive 전체를 그대로 쓰지 않고, active feature 수 `p_dim` 기준 cap을 둔다. 여기서 `p_dim`은 Optimizer가 실제로 받은 `selected_features` 개수다. Modeler selected features가 적용되면 그 개수가 `p_dim`이 된다.

```text
gp_train_cap = min(init_train_np_ratio * p_dim, init_max_points)
default: init_train_np_ratio = 20, init_max_points = 500
```

archive row 수가 cap 이하이면 boundary 안의 모든 CSV/DOE row를 GP 초기 학습에 사용한다.

archive row 수가 cap을 초과하면 다음 우선순위로 GP train subset을 만든다.

```text
1. 최근 BO 평가점 일부 보존
2. objective 우수점 보존
3. 남은 슬롯은 diversity 기준으로 채움
```

새 BO 평가점은 항상 archive에 추가된다. 이후 GP train subset은 archive에서 다시 선택되므로, 새 점은 다음 refit부터 학습 후보가 된다.

## Focus2 TuRBO-Lite Region Policy

Focus1은 global GP로 good / uncertain / bad를 분류하고, 그 결과에서 `focus_regions.json` 후보를 만든다. Focus2는 이 region list를 받아 region별 local GP 경쟁을 수행한다. Focus2는 Focus1으로 되돌아가지 않고, 자기 budget 안에서 region을 평가하고 나쁜 region을 drop한다.

Focus2 활성화 조건:

```text
focus_pipeline contains focus2
n_opt >= focus2_min_total_budget
focus2_enabled == true
```

Focus1/region builder는 archive 전체와 global GP를 사용해 region 후보를 만든다.

```text
candidate pool 생성
GP로 good / uncertain / bad 분류
good + uncertain 상위 일부를 region seed로 사용
seed 주변 cluster에서 region bounds 생성
archive support / good candidate / volume filter 통과 region만 채택
```

기본 region filter:

```text
focus2_region_min_archive_points = 5
focus2_region_min_good_candidates = 10
focus2_region_min_volume_ratio = 0.05
```

즉, region은 반드시 실제 CAE archive 점을 5개 이상 포함해야 한다. Candidate만 좋아 보이는 region은 local GP 초기화 근거가 약하므로 Focus2로 넘기지 않는다. Region size는 전체 active design space 대비 5% 이상이면 허용한다.

Focus2 TuRBO-lite loop:

```text
1. active region마다 내부 archive 점을 모은다.
2. region별 local GP를 fit한다.
3. 각 local GP가 자기 bounds 안에서 topk / boundary / random pool을 만든다.
4. pool 전체를 LCB로 scoring하고 region별 후보 1개를 제안한다.
5. region별 후보를 `focus2_pending` queue에 넣는다.
6. main loop가 queue를 하나씩 소비하며 CAE 평가한다.
7. 평가 결과로 해당 region의 success/no_improve count를 업데이트한다.
8. queue가 비면 최신 archive로 local GP를 다시 fit하고 다음 batch를 만든다.
9. no_improve_count가 tolerance 이상이면 region을 drop한다.
10. budget이 끝나거나 active region이 없어지면 종료한다.
```

1차 구현은 복잡도를 낮추기 위해 shrink / expand / merge를 하지 않는다.

```text
focus2_region_no_improve_tolerance = 3
focus2_local_pool_size = 512
focus2_local_topk_ratio = 0.30
focus2_local_boundary_ratio = 0.35
focus2_local_random_ratio = 0.35
focus2_min_candidate_rms_distance = 0.02
focus2_kappa_start = 2.50
focus2_kappa_end = 1.20
```

Focus2는 region 검증 단계이므로 region끼리 acquisition 경쟁을 시켜 1개만 고르지 않는다. Active region마다 후보 1개씩 batch로 계획하고 queue처럼 순차 평가한다. 현재 CAE 호출은 sequential이지만, 구조상 나중에 CAE batch submit으로 바꾸기 쉽다.

Focus2 batch를 만들 때는 region scheduler가 먼저 이번 batch에 참여할 region을 고른다. 첫 round는 starvation을 막기 위해 아직 평가되지 않은 active region을 우선 평가한다. 이후에는 region priority를 계산해 budget이 부족할 때 우선순위 높은 region만 후보를 만든다.

```text
region_priority =
  archive_weight * archive_best_score
  + support_weight * archive_support
  + success_weight * success_score
  + exploration_weight * under_eval_score
  + active_bounds_weight * active_bounds_score
  - no_improve_penalty
  - duplicate_penalty

default:
  archive_weight = 0.50
  support_weight = 0.15
  success_weight = 0.15
  exploration_weight = 0.10
  active_bounds_weight = 0.10
```

Scheduler 결과는 debug history의 `focus2_scheduler_priority`와 `focus_regions.json`의 `turbo_lite.scheduler_history`에 기록한다.

Focus2 batch 후보는 normalized RMS distance `0.02` 기준으로 기존 archive와 batch 내부 후보 모두에 대해 near-duplicate를 제거한다. Duplicate로 제거된 region은 failure로 보지 않고 `skip_duplicate_count`를 누적한다. 평가된 점이 다른 region에도 들어가거나 충분히 가까우면 shared evidence로 반영해 `shared_eval_count` / `shared_success_count`를 업데이트한다. 반복 duplicate는 향후 region merge 신호로 사용한다.

여기서 `no_improve_count`는 CAE 실행 실패가 아니라, 실제 CAE 평가는 성공했지만 해당 region의 best objective를 개선하지 못한 횟수다. CAE 실행 실패/invalid objective는 별도 오류 처리 대상이다.

Focus2 merge는 bounds overlap을 핵심 trigger로 쓰지 않는다. Focus1 clustering 이후 axis-aligned bounds에서 생긴 overlap은 보조 진단값일 뿐이다. 1차 merge trigger는 아래 두 가지다.

```text
strong merge:
  candidate_duplicate_pair_count >= 2

moderate merge:
  shared_pair_count >= 2
  AND same_basin_score >= 0.75
```

`same_basin_score`는 best point proximity와 best objective similarity를 합친 보조 점수다. 위치가 가까워도 objective가 크게 다르면 narrow basin/ridge 가능성이 있으므로 merge를 보수적으로 본다.

```text
point_score = 1 - normalized_best_point_distance / 0.05
objective_score = 1 - normalized_objective_gap / 0.10
same_basin_score = 0.70 * point_score + 0.30 * objective_score
```

Merge는 `focus2_pending` queue가 비었을 때만 검사한다. Merge 후 survivor region의 `no_improve_count`는 두 region count의 `min`으로 둔다. Absorbed region은 `status = merged`, `merged_into = survivor_region_id`로 기록한다.

Focus2 active bounds는 Focus1이 준 parent bounds를 안전 울타리로 두고, batch가 끝날 때 archive evidence 기반으로 갱신한다. 기존 `bounds_lb/ub`는 active bounds이고, `parent_lb/ub`는 최초 region bounds다.

```text
focus2_active_bounds_enabled = True
focus2_active_bounds_min_archive_points = 5
focus2_active_bounds_top_fraction = 0.50
focus2_active_bounds_min_top_points = 3
focus2_active_bounds_quantile_low = 0.10
focus2_active_bounds_quantile_high = 0.90
focus2_active_bounds_expand_ratio = 1.20
focus2_active_bounds_min_volume_ratio = 0.05
focus2_active_bounds_max_volume_ratio = 0.25
focus2_active_bounds_update_rate = 0.35
```

업데이트는 region parent bounds 안의 archive 점을 모으고, objective 상위 점들의 q10~q90 bounds를 계산한 뒤 expand ratio를 적용한다. 새 bounds는 parent bounds 안으로 clip되고, 전체 design space 기준 최소 10%, 최대 25% volume을 지킨다. 최종 적용은 기존 active bounds와 proposed bounds를 `update_rate`로 smoothing한다. Active bounds shrink는 region별 실제 평가 evidence가 충분할 때만 수행한다.

Focus2는 L-BFGS-B refine을 하지 않는다. 후보를 많이 뿌린 뒤 LCB로 scoring하는 discrete search만 사용한다. L-BFGS-B refine은 최종 point convergence인 Focus3에 남긴다.

Focus2 평가 행의 `source_mode`는 `focus2_topk`, `focus2_boundary`, `focus2_random`처럼 local pool source를 기록한다. 선택된 region id와 queue 정보는 debug history의 `focus2_selected_region_id`, `focus2_batch_id`, `focus2_batch_size`, `focus2_batch_remaining` 및 `focus_regions.json`에 기록한다.

Focus2가 끝나면 단일 best region만 과신하지 않는다. 기본 정책은 surviving region 중 최종 점수 상위 2개를 union하고, 그 union bounds를 1.2배 확장한 뒤 전체 design space 기준 최소 25% volume을 보장해 Focus3 bounds로 적용한다. Surviving region이 1개뿐이면 그 region을 쓰되 같은 최소-volume 확장을 적용한다. Surviving region이 없으면 dropped region까지 포함해 후보를 평가한다. 그래도 region이 없으면 Focus3는 실행하지 않고 현재까지의 exploratory best만 남긴다.

Focus2 global-GP region builder가 region을 만들지 못해도 objective archive support가 충분하면 archive 기반 fallback region을 하나 만든다. Fallback region은 objective 상위점과 best 주변 archive support를 이용해 q10~q90 bounds를 만든다. 최종 Focus3 전달 시에는 다른 generated bounds와 동일하게 보수적 union/expand/min-volume 정책을 거친다. 이 fallback은 Focus3가 실행될 내부 selected bounds generator 역할을 한다.

최종 region selector는 Focus1 prior score를 쓰지 않는다. Focus2가 새로 만든 실제 CAE evidence를 중심으로 판단한다.

```text
final_score =
  archive_weight * normalized_archive_best
  + support_weight * archive_support
  + success_weight * focus2_success
  - no_improve_penalty
  - dropped_penalty

default weights:
  archive = 0.70
  support = 0.15
  success = 0.10
```

선택된 최종 bounds는 Focus3에 in-memory로 전달되고, audit용 meta artifact로 저장한다.

```text
OPT/artifacts/meta/focus_bounds.json
```

Debug 산출물:

```text
OPT/artifacts/debug/focus2/focus2_bounds_evolution.csv
OPT/artifacts/debug/focus2/focus2_region_events.csv
OPT/artifacts/debug/focus2/focus2_region_timeline.png
OPT/artifacts/debug/focus2/focus2_bounds_evolution_<feature_i>_<feature_j>.png
```

Timeline plot은 region별 CAE 평가, scheduler selected/skipped, active bounds update, merge, final merged/dropped 상태를 한 축에서 확인하기 위한 debug artifact다.

Focus2 region output:

```text
OPT/artifacts/meta/focus_regions.json
```

Region output에는 feature list를 별도 필드로 반복하지 않는다. Feature list는 Optimizer input resolve의 공통 context이며, region bounds의 key로만 자연스럽게 나타난다.

## Public Outputs

Optimizer artifact layer 정책:

```text
public:
OPT/artifacts/public/opt_results.csv
OPT/artifacts/public/best_point.json

meta:
OPT/artifacts/meta/selected_features.csv
OPT/artifacts/meta/optimizer_inputs.json
OPT/artifacts/meta/optimizer_algorithm.json
OPT/artifacts/meta/optimizer_system_config.json
OPT/artifacts/meta/focus_regions.json   # focus_bo only
OPT/artifacts/meta/focus_bounds.json    # focus_bo only

debug:
OPT/artifacts/debug/optimizer_history_full.csv
OPT/artifacts/debug/focus2/*            # focus_bo only
OPT/artifacts/debug/focus3/*            # focus_bo only
```

Public optimizer CSV는 compact하고 downstream/user-facing이어야 한다.

`best_point.json`은 알고리즘 종류와 결과 상태를 함께 기록한다. `focus_bo`는 Focus3까지 수렴 최적화를 시도했는지와, bounds 없이 데이터 보강만 수행했는지를 구분한다. Runtime/custom 알고리즘은 Focus3를 갖지 않으므로, 평가를 1회 이상 수행하면 Focus3 여부와 무관하게 `result_status=optimized`로 기록한다.

```json
{
  "result_status": "optimized | exploratory_best",
  "converged": true,
  "final_focus": "focus3",
  "algorithm_id": "focus_bo",
  "algorithm_engine": "focus_bo",
  "algorithm_kind": "focus",
  "selected_bounds_available": true,
  "generated_bounds_available": false,
  "optimization_bounds_available": true,
  "focus3_executed": true,
  "optimizer_status_basis": "focus3_and_bounds"
}
```

`optimization_bounds_available=false`이거나 Focus3가 실행되지 않은 경우 `result_status=exploratory_best`다. 이때 best point는 최적화 완료 해가 아니라, 입력 archive와 이번 Optimizer 평가점 중 현재까지 관측된 best다. `selected_bounds_available=false`라도 Focus2가 generated bounds를 만들고 Focus3가 실행되면 `generated_bounds_available=true`, `result_status=optimized`가 될 수 있다.

위 Focus3 기준은 `focus_bo`에만 적용된다. Custom 알고리즘은 `optimizer_algorithm.json`에 algorithm summary와 schema 정보를 남기고, `focus_regions.json`/`focus_bounds.json` 및 Focus2/Focus3 debug plot은 생성하지 않는다.

`opt_results.csv`는 모든 focus가 같은 schema를 쓴다. 이 파일은 실제 Optimizer가 추가로 CAE 평가한 점만 기록하며, DOE seed/archive row는 포함하지 않는다.

```text
iter
segment
opt_focus_level
source_mode
success
feasible
objective
objective_raw
p_feasible
<selected feature columns...>
```

Focus별 진단값(`focus1_tau`, pool count, Focus3 source count, GP train count 등)은 public CSV에 넣지 않고 debug full history에만 둔다.

`selected_features.csv`와 `optimizer_inputs.json`은 Optimizer input resolve 결과를 기록하므로 public이 아니라 meta다. Feature list와 initial bounds는 Focus0/1/3이 각각 정하는 값이 아니라 Optimizer 시작 시점에 확정된 input layer 산출물이다.

`optimizer_algorithm.json`은 모든 알고리즘에 대해 생성되는 algorithm-neutral meta artifact다. `optimizer_system_config.json`도 모든 알고리즘에 대해 생성되며, 기존 전체 dataclass snapshot과 별도로 system 설정을 `common`, selected `algorithm`, `focus_bo`, `compatibility` 영역으로 나눈 audit artifact다. 이 파일은 custom/runtime 알고리즘 이관 시 어떤 설정이 공통 계약이고 어떤 설정이 FocusBO 전용인지 확인하는 기준이다.

구현상 `OptimizerSystemConfig`는 아직 flat dataclass로 유지한다. 기존 pipeline JSON override와 성능 실험 config가 field 이름에 의존하므로, 물리 분리는 성능 안정화 이후 별도 compatibility window에서만 진행한다. 현재 이관 경계는 `optimizer_system_config_view()`와 `split_optimizer_system_config()`가 제공하는 view를 기준으로 판단한다.

`focus_regions.json`과 `focus_bounds.json`은 `focus_bo` 전용이며, 사용자 최종 결과가 아니라 Focus2 내부 region manager와 generated bounds audit 정보다. Backend가 필요하면 meta에서 선택적으로 읽어 UI에 노출할 수 있지만, 기본 public artifact는 아니다.

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
