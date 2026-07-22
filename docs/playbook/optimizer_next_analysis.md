# Optimizer 다음 분석 체크리스트

다음 `run_pipelines.py` batch가 끝나면 이 항목을 기준으로 본다. Rosenbrock 한 run만 보고 튜닝하지 않는다.

## 1차 성공 기준

- Optimizer-only 성공은 [optimizer_experiment_goals.md](./optimizer_experiment_goals.md)의 문제별 goal threshold를 best observed objective가 달성했는지로 판단한다.
- Known optimum inclusion은 benchmark 진단값이다. Optimizer Focus2/3 bounds가 known optimum을 포함하지 않아도 goal을 달성하면 성공이다.

## Batch-Level 확인

- 문제별/repeat별 best objective, goal hit/miss, runtime, final focus
- GP, RB, SC, 이후 CB를 분리해서 goal hit rate 비교
- 실패 원인이 Focus2 generated bounds인지, Focus3 convergence인지, Focus0/1 data 부족인지 분리
- Goal을 달성한 run은 goal 최초 달성 iteration을 기록
- Goal 달성 이후 best가 추가 개선된 횟수와 마지막 개선 iteration을 확인

## Focus2 확인

- Region timeline에서 scheduler/eval/drop timing이 자연스러운지 확인
- Active/dropped region의 best objective 비교
- Dropped region을 남겼다면 goal에 도달했을 가능성이 있었는지 확인
- Focus2 boundary/topk/random 후보의 실제 개선 기여도 확인
- Known optimum을 generated bounds에 강제로 포함시키려 하지 말고, goal hit와 region objective quality 기준으로 판단

## Focus3 확인

- Source/mode 기여도: topk, boundary, random, discrete, refine
- `focus3_refine_every = 2`가 goal hit rate와 runtime 사이에서 더 나은지 확인
- `gp_refit_every = 1`은 유지한 상태에서 비교
- Runtime split: GP fit, plan scoring, refine, debug plot
- Recovery active rate가 계속 90% 이상이면 recovery가 사실상 default mode가 된 것이므로 조건 재검토
- Dedup rate가 높으면 후보가 archive 주변으로 과도하게 몰리는지 확인

## 후보 튜닝 항목

- Focus3 dedup distance
  - 너무 작으면 거의 같은 후보가 반복될 수 있다.
  - 너무 크면 좋은 basin 근처의 local refinement를 막을 수 있다.
  - 다음 batch에서는 dedup rate와 best improvement source를 보고 조정한다.

- Boundary ratio
  - Constraint 없는 문제에서 boundary source의 개선 기여도가 낮으면 낮춘다.
  - Constraint 문제까지 포함하기 전에는 전체 기본값을 성급히 낮추지 않는다.

- Focus2 merge
  - Duplicate/shared evidence가 반복되는데 region이 분리되어 있으면 merge trigger를 강화한다.
  - 단순 overlap만으로 merge하지 않는다.
  - Best point proximity와 objective similarity를 함께 본다.

- Focus3 recovery
  - Recovery가 너무 자주 켜지면 `recover_window`, `recover_tol`, bonus 크기를 재검토한다.
  - 개선이 있는 run에서도 recovery가 계속 켜지면 recent improvement 기준이 너무 엄격한 것이다.

- Mode scheduling
  - 현재 후보: `discrete 1회 + refine 1회`
  - 추가 후보: recovery active일 때 refine 빈도를 높이고, 안정 구간에서는 discrete를 늘리는 adaptive schedule
  - 문제별로 고정하지 않고 data/recovery/dedup 신호 기반으로 조정한다.

- Early stop
  - Goal 달성 후 best 개선이 거의 없으면 조기 종료할 수 있다.
  - Benchmark 분석에서는 goal 최초 달성 iteration과 이후 추가 개선 횟수를 먼저 확인한 뒤 적용한다.
  - 실제 score run에서 early stop을 켜기 전에는 hit rate가 유지되는지 검증한다.

## 지금 바로 할 것과 보류할 것

이번 반영:

- Focus2 final bounds plot에 실제 Focus3로 전달된 final bounds를 함께 표시한다.
- Focus2 final union에서 dropped region도 penalty를 받은 상태로 경쟁에 남긴다. merged region은 제외한다.
- Focus2 final union bounds를 Focus3로 넘기기 전에 union 내부 archive 우수점 기반 q10~q90 trim을 조건부로 적용한다.
  - 충분한 archive support가 없으면 trim하지 않는다.
  - trim bounds로 완전 대체하지 않고 union bounds의 center/width를 약하게만 보정한다.
  - 이후 margin/min-volume 정책을 다시 적용한다.
- Focus3 recovery는 최소 history가 쌓인 뒤에만 켜지도록 한다.
- Focus3 recovery는 mild/strong 2단으로 나누고, 짧은 정체에서는 boundary/random/kappa 증가를 약하게 적용한다.
- 30회 batch 분석 결과 RB/GP에서 boundary source 개선 기여가 낮고 recovery strong이 과도하게 켜지는 것으로 확인했다.
- Constraint가 없는 Focus3 run에서는 final source mixture에 boundary cap/topk floor/random floor를 적용한다.
  - 기본값: boundary <= 0.18, topk >= 0.55, random >= 0.10
- Constraint가 없는 Focus3 run에서는 boundary 후보가 topk/random/best_local보다 충분히 acquisition 우위일 때만 최종 선택 후보로 허용한다.
- Focus3 strong recovery 기준을 `20` no-improve step에서 `50` no-improve step으로 늦춘다.
- Mild recovery의 boundary/kappa 증가는 더 약하게 조정한다.
- Constraint가 없는 recovery에서는 boundary bonus를 제거하고, random/topk 중심으로 회복하도록 조정한다.
- Focus3에 `best_local` source를 추가한다.
  - 현재 best 상위 소수 점 주변을 더 작은 sigma로 샘플링한다.
  - 기본적으로 Focus3 history와 archive 밀도가 어느 정도 쌓인 뒤 topk quota 일부를 가져온다.
- Focus2 fallback region은 evidence가 약하므로 final bounds를 더 보수적으로 유지한다.
  - archive trim을 끄거나 약하게 적용하고, expand/min-volume을 더 크게 둔다.
- Focus budget scheduler를 adaptive로 전환한다.
  - Focus3 최소 budget을 먼저 보장하고, selected/generated bounds 여부에 따라 Focus1/2 비중을 조정한다.
  - Focus0/1/2가 조기 종료하면 기존처럼 남은 budget은 Focus3로 넘어간다.
- Focus3 acquisition auto policy에 `MEAN` exploitation 상태를 추가한다.
  - 충분한 archive 밀도, 작은 bounds, `best_local` 활성 조건을 만족할 때만 평균 예측값 기준 exploitation을 쓴다.
  - recovery가 켜지면 `EI/MEAN` 모두 `LCB`로 되돌린다.
- Focus2 region generation은 fallback 전에 relaxed retry를 한 번 수행한다.
  - uncertain seed 비율, cluster radius, expand ratio를 키우고 good 후보 최소 조건을 낮춰 classifier 기반 region 생성을 한 번 더 시도한다.

지금 바로:

- `gp_refit_every = 1` 유지
- `focus3_refine_every = 2` 실험
- 다음 batch 결과에서 `best_local`이 RB goal hit rate를 올리는지 확인
- 다음 batch 결과에서 boundary gate가 RB/GP boundary 최종 선택률을 충분히 낮추는지 확인
- 다음 batch 결과에서 fallback bounds run의 boundary 선택률과 GP 실패 run을 분리해서 확인
- 다음 batch 결과에서 `focus3_refine_filter_skipped` 비율과 runtime 감소량을 확인
- 다음 batch 결과에서 `focus3_refine_filter_best_plan_source`가 boundary일 때 refine 생략이 goal hit를 해치지 않는지 확인
- 다음 batch 결과에서 `focus3_boundary_score_penalty_applied` 이후 RB/GP의 boundary best_plan 비중이 낮아졌는지 확인
- 다음 batch 결과에서 `focus3_recover_level=strong` 비율이 RB/GP에서 낮아졌는지 확인
- 다음 batch 결과에서 goal hit iteration과 이후 개선 횟수 분석

추가 반영:

- 무제약 Focus3 boundary cap/gate를 더 강하게 조정했다.
  - boundary cap 기본값을 `0.18`에서 `0.10`으로 낮췄다.
  - boundary gate margin을 `0.02`에서 `0.08`로 올렸다.
- `best_local` source를 강화했다.
  - 기본 quota를 `0.15`에서 `0.25`로 올리고 max를 `0.35`로 확장했다.
  - sigma를 `0.025`에서 `0.015`로 줄여 best 주변을 더 촘촘히 탐색한다.
- Focus2 fallback bounds일 때 Focus3 source policy를 별도로 적용한다.
  - boundary를 최대 `0.05`로 제한하고 topk/best_local/random floor를 둔다.
- Focus3 refine source filter를 추가했다.
  - 무제약 문제에서 best plan source가 `boundary`이면 L-BFGS-B refine을 생략하고 discrete GP scoring 후보를 사용한다.
  - `random`은 좋은 basin 근처의 시작점 역할을 할 수 있어 refine 대상으로 되돌렸다.
  - 목적은 RB/GP runtime의 가장 큰 병목인 Focus3 refine 시간을 줄이는 것이다.
- Recovery를 더 늦고 약하게 조정했다.
  - strong recovery 기준을 `50`에서 `120` no-improve step으로 늦췄다.
  - 무제약 recovery는 topk 중심으로 더 기울이고 random scale은 낮췄다.
- Recovery 판정 window를 `20`으로 늘리고 objective scale 기반 relative tolerance를 추가했다.
  - 너무 작은 수치 변화까지 정체로 보던 문제를 줄이기 위한 변경이다.
- 무제약 boundary best_plan 과다 선택을 줄이기 위해 boundary acquisition score penalty를 추가했다.
  - constraint가 없는 Focus3에서만 boundary score에 penalty를 더한다.
  - constraint 문제의 boundary 탐색에는 적용하지 않는다.

보류:

- Dedup distance 변경
- Focus2 merge trigger 변경
- Early stop 적용

이 항목들은 다음 batch 결과를 보고 문제별 공통 신호가 확인되면 적용한다.
