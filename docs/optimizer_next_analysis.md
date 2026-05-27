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

지금 바로:

- `gp_refit_every = 1` 유지
- `focus3_refine_every = 2` 실험
- 다음 batch 결과에서 goal hit iteration과 이후 개선 횟수 분석

보류:

- Boundary ratio 변경
- Dedup distance 변경
- Focus2 merge trigger 변경
- Early stop 적용

이 항목들은 다음 batch 결과를 보고 문제별 공통 신호가 확인되면 적용한다.
