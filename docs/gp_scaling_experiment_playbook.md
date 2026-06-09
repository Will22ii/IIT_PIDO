# GP Scaling Experiment Playbook

이 문서는 GP 입력 X scaling 실험의 목적, 우선순위, 해석 기준을 기록한다. 현재 관심사는 Optimizer와 Explorer에서 GP가 raw coordinate를 그대로 쓰는 것이 좋은지, bounds-normalized coordinate를 쓰는 것이 좋은지 검증하는 것이다.

## 핵심 개념

GP X scaling은 CAE 평가 좌표를 바꾸는 것이 아니다. GP 학습과 예측 내부에서만 active bounds 기준으로 X를 변환한다.

```text
x_scaled = (x - lb) / (ub - lb)
```

모드는 아래처럼 해석한다.

```text
off
  raw X 그대로 GP 학습/예측

auto
  active bounds span ratio가 threshold 이상일 때만 bounds scaling 적용

bounds
  span ratio와 무관하게 항상 bounds scaling 적용
```

현재 auto threshold는 `3.0`이다.

```text
span_ratio = max(ub - lb) / min(ub - lb)

span_ratio >= 3.0 -> scaling 적용
span_ratio < 3.0  -> raw X 유지
```

## 현재 코드 기준

현재 GP X scaling은 Optimizer FocusBO GP에만 적용되어 있다.

Optimizer에서는 원래 CAE 전체 변수 차원이 아니라, Optimizer에 실제 전달된 `selected_features`와 `selected_bounds` 기준으로 scaling 여부를 판단한다.

따라서 AION full pipeline에서 Modeler primary selection이 dummy를 제거하면, Optimizer GP scaling은 `no_dummy` optimizer-only와 비슷한 feature/bounds 관점으로 작동한다. Primary selection이 dummy를 제거하지 못한 run은 최종 joint score상 앞단 실패에 가깝기 때문에, 초기 scaling 해석의 핵심 대상에서 제외한다.

2026-06-09 변경 이후 Explorer GP에도 auto scaling을 적용한다. Explorer의 `gp_pred`, `gp_obj`는 같은 GP fit wrapper를 타므로 둘 다 동일한 policy를 사용한다.

Explorer scaling diagnostics는 Explorer metadata와 batch try stats에 아래 필드로 남긴다.

```text
explorer_gp_x_scaling_applied_pred
explorer_gp_x_scaling_applied_obj
explorer_gp_x_scaling_span_ratio_pred
explorer_gp_x_scaling_span_ratio_obj
explorer_gp_uncertainty_dim_weights
```

## 1차 실험: Optimizer Standalone

가장 먼저 볼 실험은 optimizer-only `no_dummy` standalone이다.

목적:

- Optimizer GP scaling 자체가 성능에 도움이 되는지 확인한다.
- `auto` threshold 3.0이 너무 보수적인지 확인한다.
- Explorer/AION 영향을 배제하고 GP scaling 효과만 먼저 본다.

실험 조건:

```text
BATCH_AION_MODE = False
BATCH_TASKS = {
  doe: False,
  modeler: False,
  explorer: False,
  optimizer: True
}
ACTIVE_PROBLEM_CASES = *_nodummy suite
```

우선 비교:

```text
gp_x_scaling_mode = off
gp_x_scaling_mode = bounds
```

기존 `auto` 결과는 baseline으로 쓰되, 코드 변경이 많거나 seed/repeat를 맞추고 싶으면 같은 조건으로 `auto`도 다시 돌린다.

문제별 기대 해석:

- `cantilever_beam_nodummy`: span ratio가 커서 auto가 켜지는 문제다. `off` 대비 scaling 자체가 좋은지 확인한다.
- `rosenbrock_nodummy`: span ratio가 작아 auto가 꺼지는 문제다. `bounds` 강제 scaling이 narrow valley 수렴에 도움이 되는지 본다.
- `goldstein_price_nodummy`: span ratio가 작아 auto가 꺼지는 문제다. `bounds`가 multi-modal 탐색을 돕는지, 혹은 해치는지 본다.
- `six_hump_camel_nodummy`: span ratio가 작아 auto가 꺼지는 문제다. 현재 성능이 좋으면 `bounds`가 불필요한 변동을 만드는지 확인한다.

판단 기준:

- `bounds > auto/off`: threshold 3.0이 보수적이거나, GP에는 상시 bounds-normalization이 유리할 수 있다.
- `bounds < auto/off`: 현재 auto 정책을 유지한다.
- 문제별로 갈리면 AION full pipeline에서 selected bounds span ratio와 문제 유형을 같이 본다.

## Dummy 포함 Optimizer-only 실험

dummy 포함 optimizer-only는 필수 실험이 아니다.

이 실험은 dummy/ARD/scaling 가설을 진단하는 데는 의미가 있다. 하지만 AION 목표에서는 primary selection이 dummy를 제거한 뒤 Optimizer에 들어오는 경로가 더 중요하다.

따라서 우선순위는 낮다.

dummy 변수는 objective에 영향을 주지 않으므로, objective goal 기준에서는 dummy 값이 꼭 0일 필요가 없다. real variables가 goal을 만족하면 dummy 좌표와 무관하게 optimizer goal hit가 가능하다. 다만 dummy 차원이 GP ARD를 방해할 수 있으므로, 별도 진단 실험으로는 의미가 있다.

## 2차 실험: AION Optimizer GP Scaling

Optimizer standalone 결과를 본 뒤 AION full pipeline에서 Optimizer GP scaling을 비교한다.

실험 조건:

```text
BATCH_AION_MODE = True
BATCH_TASKS = {
  doe: True,
  modeler: True,
  explorer: True,
  optimizer: True
}
```

비교 후보:

```text
Optimizer GP auto
Optimizer GP off
Optimizer GP bounds
```

처음에는 `auto`를 baseline으로 둔다. `off`와 `bounds`는 진단용이다.

분석할 때 반드시 분해할 항목:

- `modeler_all_real_only`
- `explorer_bounds_pass`
- `explorer_joint_pass`
- `optimizer_goal_hit`
- `optimizer_final_pass`
- Focus3에 실제 적용된 `gp_x_scaling_applied`
- Focus3 bounds span ratio
- `explorer_joint_pass=true`인데 `optimizer_goal_hit=false`인 run

AION에서는 Explorer bounds, DOE archive, Modeler selected features가 함께 작용한다. 따라서 Optimizer GP scaling 실험은 `optimizer_goal_hit`만 보지 말고 최종 joint score와 Focus3 직접 실패를 분리해야 한다.

## 3차 실험: Explorer GP Scaling

Explorer GP scaling은 장기적으로 generalized Explorer를 위해 검토한다. 다만 현재 Explorer는 raw-GP 기준으로 여러 threshold와 보정값이 튜닝되어 있을 가능성이 크므로 Optimizer보다 더 조심해야 한다.

Explorer에서 GP가 쓰이는 주요 위치:

1. S4/S4_dual anchor refinement
2. pred-side/objective-side acquisition refinement
3. `uncertainty_aware` bounds expansion의 dimension weight 계산

Refinement GP에는 scaling이 도움이 될 가능성이 크다. 변수 span 차이가 큰 문제에서 acquisition surface가 안정화될 수 있기 때문이다.

반면 `uncertainty_aware`는 더 위험하다. 이 로직은 GP posterior sigma를 차원별로 비교해서 bounds expansion weight로 사용한다.

현재 uncertainty-aware 흐름:

```text
1. selected_bounds center 계산
2. 각 차원의 lb/ub boundary point에서 GP sigma 측정
3. gp_pred/gp_obj sigma 평균 계산
4. weights = sigmas / sigmas.mean()
5. bounds_weight_clip_min/max로 clip
6. weights 합이 dimension 수가 되도록 normalize
7. sigma가 큰 차원을 더 많이 확장
```

현재 주요 값:

```text
bounds_weight_clip_min = 0.7
bounds_weight_clip_max = 1.5
bounds_expansion_high_crate_threshold = 0.85
```

AION/S4_dual에서는 제약 문제가 있고 `constraint_rate_hat < 0.85`이면 `uncertainty_aware`로 갈 가능성이 있다. 특히 cantilever 같은 constraint/boundary-sensitive 문제에서 GP sigma weight가 selected bounds에 직접 영향을 줄 수 있다.

## Explorer Scaling 실험 순서

Explorer scaling은 한 번에 전부 켜지 않는다.

권장 순서:

```text
A. baseline
   Explorer GP raw
   Optimizer GP auto

B. refine-only scaling
   Explorer anchor refinement GP auto scaling
   uncertainty-aware weight는 기존 raw GP 기준 유지
   Optimizer GP auto

C. full Explorer GP scaling
   Explorer refinement GP auto scaling
   uncertainty-aware GP auto scaling
   Optimizer GP auto
```

이렇게 나눠야 성능 변화 원인을 분리할 수 있다.

## Explorer Uncertainty-aware 우려

generalized Explorer 관점에서는 uncertainty-aware에도 auto scaling을 넣는 것이 이론적으로 맞다.

이유:

- GP kernel은 좌표 scale에 민감하다.
- raw sigma는 진짜 uncertainty와 변수 단위 효과가 섞일 수 있다.
- bounds-normalized GP sigma가 차원별 불확실성 비교에는 더 자연스럽다.

하지만 현재 benchmark 성능은 raw-GP 기준 튜닝값 위에서 맞춰졌을 수 있다. scaling을 넣으면 sigma의 차원별 상대 순위가 바뀌고, 그 결과 selected bounds가 다른 축으로 열리거나 닫힐 수 있다.

따라서 Explorer uncertainty-aware scaling은 아래 보호장치와 함께 실험한다.

초기 보호장치 후보:

- AION Explorer uncertainty weight clip을 더 보수적으로 둔다.

```text
기존: 0.7 ~ 1.5
후보: 0.85 ~ 1.20
```

- scaled weight가 과도하게 뾰족하면 uniform fallback한다.

```text
max(weight) / min(weight) > threshold -> uniform fallback
```

- raw-vs-scaled shadow diagnostics를 저장한다.

기록할 진단값:

- raw uncertainty weights
- scaled uncertainty weights
- weight L1 difference
- max/min weight ratio
- weight rank change 여부
- Explorer selected bounds volume ratio
- Explorer bounds pass
- downstream optimizer final pass

## 현재 우선순위

지금 당장 집중할 것은 Explorer scaling이 아니다.

현재 우선순위:

1. Optimizer standalone `no_dummy`에서 `off`와 `bounds` 비교
2. 기존 `auto` baseline과 비교
3. Optimizer GP scaling 방향 결정
4. AION full pipeline에서 Optimizer GP `auto/off/bounds` 비교
5. 그 다음 Explorer GP scaling 설계 및 실험

Explorer GP scaling은 generalized 방향으로는 필요하지만, 현재 단계에서 함께 넣으면 결과 해석이 어려워진다.

## 해석 원칙

- scaling 실험은 run mode를 섞어서 해석하지 않는다.
- optimizer-only, AION full pipeline, Explorer scaling 실험은 서로 분리한다.
- AION 최종 점수는 `modeler_all_real_only * explorer_bounds_pass * optimizer_goal_hit` 구조로 본다.
- Explorer bounds가 틀린 run을 Optimizer scaling 실패로 보지 않는다.
- primary selection이 dummy를 남긴 run은 Optimizer scaling 분석의 핵심 대상에서 제외한다.
- Explorer uncertainty-aware scaling은 성능이 떨어져도 즉시 폐기하지 말고, raw-vs-scaled diagnostics로 원인을 본다.
