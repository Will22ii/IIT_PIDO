# Optimizer FocusBO Analysis Playbook

이 문서는 Optimizer 개선 실험을 분석할 때의 기준이다. 목적은 benchmark 정답을 코드 로직에 넣는 것이 아니라, 실행 결과를 일관되게 읽고 다음 개선 방향을 고르는 것이다.

## 기본 원칙

- 분석 대상은 기본적으로 Optimizer task만 활성화한 FocusBO run이다.
- DOE, Modeler, Explorer까지 모두 켠 full pipeline run은 별도 요청이 없는 한 Optimizer score 분석에서 제외한다.
- `index.json`의 tasks가 `CAE`, `OPT`만 있는 run을 optimizer-only 기준으로 본다.
- benchmark의 known optimum은 분석용 기준으로만 사용한다. Optimizer 로직은 known optimum, 정답 좌표, benchmark 함수명을 이용해 풀면 안 된다.
- 결과 리포트 파일은 사용자가 요청하지 않는 한 새로 만들지 않고, 대화에서 핵심 통계와 해석을 설명한다.

## Run 보관 위치

과거 optimizer-only 실행 결과는 아래 폴더에 보관한다.

```text
result/optimizer_only_runs/
```

이 폴더의 `run_*` 디렉터리는 이전 FocusBO optimizer-only 실험 archive로 취급한다. 새로 `result/run_*`에 쌓이는 실행 결과와 섞어서 최신 batch로 판단하지 않는다.

분석할 때는 목적에 따라 대상을 명확히 나눈다.

- 과거 optimizer-only 재분석: `result/optimizer_only_runs/run_*`
- 방금 새로 실행한 optimizer-only 결과: `result/run_*` 중 `index.json` tasks가 `CAE`, `OPT`만 있는 run
- full pipeline 결과: 별도 보관 위치 또는 `index.json`에 DOE/Modeler/Explorer가 포함된 run

`result/optimizer_only_runs`는 과거 성능 기준선과 개선 히스토리를 확인하는 용도다. 최신 개선 효과를 볼 때는 새 실행 batch와 이 archive를 구분해서 비교한다.

## 항상 먼저 볼 것

문제별 최신 optimizer-only run을 분리한 뒤 아래 순서로 본다.

1. 문제별 hit rate
2. best objective 분포: 평균, 중앙값, best, worst
3. 실패값이 goal 근처인지, 완전히 멀리 남았는지
4. early stop 수와 평균 budget 사용량
5. Focus0~3 segment별 사용량
6. Focus3 source별 평가 수와 개선 수
7. `focus3_best_plan_source`, `focus3_nearest_refine_source`, `raw_improved`
8. `focus3_recover_active`, `focus3_recover_no_improve_count`
9. `focus3_selected_*`, `focus3_best_score_*`
10. dedup, refine skip, boundary gate, random gate, source performance policy가 실제로 작동했는지

## Focus3를 가장 중요하게 보는 이유

현재 benchmark 실험은 optimizer-only로 돌리기 때문에 Focus0 -> Focus1 -> Focus2 -> Focus3 전체 내부 pipeline이 사용될 수 있다.

하지만 실제 상위 task가 모두 작동하는 시나리오에서는 DOE, Modeler, Explorer가 데이터와 boundary를 만든 뒤 Optimizer가 최종 bounds를 받아 Focus3 중심으로 수렴하게 된다. 즉 최종 정확도는 결국 Focus3가 받은 bounds 안에서 얼마나 안정적으로 좋은 값을 찾는지에 크게 달려 있다.

따라서 개선 우선순위는 다음처럼 둔다.

1. Focus3 수렴 안정성
2. Focus3 source mixture와 recovery
3. Focus3 refine/discrete 선택 품질
4. Focus3가 받은 bounds가 맞을 때 확실히 수렴하는지
5. Focus2 bounds가 틀렸을 때 실패 양상이 어떻게 나타나는지
6. Focus0~2의 데이터 보강, region 생성, bounds 전달 품질

## Boundary 분석 기준

분석할 때는 Focus3가 받은 bounds를 별도로 평가한다.

중요한 구분은 두 가지다.

- Focus3 bounds 안에 known optimum이 있는 경우
- Focus3 bounds 안에 known optimum이 없는 경우

이 구분은 코드 로직에 넣지 않는다. 분석할 때만 benchmark known optimum을 사용해서 사후 통계로 본다.

보고 싶은 통계는 다음이다.

- 문제별 전체 hit rate
- bounds contains known optimum = true인 run의 hit rate
- bounds contains known optimum = false인 run의 hit rate
- contains true인데 실패한 run의 best objective 거리
- contains false인데 성공한 run의 이유: goal threshold가 global optimum 근처가 아니거나, bounds 밖 optimum 없이도 goal hit 가능한지
- Focus2 generated bounds와 Explorer/user selected bounds를 구분
- bounds volume ratio, mean width ratio
- Focus3 budget, first hit iteration, early stop 여부

이 분석의 목적은 Focus2를 벌주는 것이 아니라 실패 원인을 분리하는 것이다. Focus2가 틀린 bounds를 줄 수 있고, margin/expand 정책으로 Focus3에 전달되는 최종 boundary가 달라질 수 있다. 따라서 Focus3는 "받은 bounds가 충분히 맞다"는 조건에서 먼저 안정적이어야 하고, 그 다음 Focus2/Explorer boundary 품질을 개선한다.

## 현재까지의 개선 히스토리

주요 개선 방향은 Focus3 exploitation과 recovery 안정화였다.

- 무제약 Focus3에서 boundary source가 과하게 이기는 문제를 줄이기 위해 boundary cap/gate/score penalty를 적용했다.
- random source가 넓은 coverage만으로 best plan을 자주 이기는 현상을 줄이기 위해 random pool ratio와 gate를 조정했다.
- `best_local` source를 추가해서 현재 best와 elite 주변을 더 촘촘히 탐색하게 했다.
- recovery를 mild/strong으로 나누고, 무제약 문제에서는 boundary보다 topk/best_local 쪽을 더 보강했다.
- source performance policy로 최근 random refine 개선률이 낮으면 random quota 일부를 best_local/topk로 이동하게 했다.
- refine source filter를 추가해서 boundary/random 중심 후보에는 비싼 L-BFGS-B refine을 생략하고, topk/best_local 중심 후보를 refine하게 했다.
- Rosenbrock near-hit 실패를 줄이기 위해 `local_probe` source를 추가했다.
- 최근 개선에서는 `local_probe`를 refine 허용 source에 포함했고, recovery 정체 상태에서 local_probe refine quota floor를 보장했다.
- `local_probe`는 더 작은 step scale을 포함해 goal 근처에서 미세하게 파도록 조정했다.

## 최근 관찰

Cantilever와 Six-hump는 optimizer-only 기준으로 안정적이다. 현재 우선 개선 대상은 아니다.

Rosenbrock은 여전히 핵심 병목이다. 실패값이 goal 근처인 경우가 많아 완전히 못 찾는 문제가 아니라 Focus3 후반 exploitation과 recovery가 부족한 양상이다.

Goldstein은 batch에 따라 흔들린다. local_probe가 Goldstein에서 강한 개선원은 아니었고, topk/best_local/random 균형이 더 중요해 보인다.

최근 source 통계에서는 Rosenbrock에서 best_local 개선률이 가장 높고, local_probe는 작동은 하지만 아직 보조적이다. 따라서 local_probe는 무작정 늘리기보다 recovery 후반/near-hit 상황에서 refine으로 연결되게 하는 것이 맞다.

## 다음 분석에서 꼭 확인할 것

다음 batch를 보면 아래를 우선 확인한다.

- Rosenbrock hit rate가 이전 `17/50`, 최신 `5/10` 대비 올랐는지
- Rosenbrock 실패값 중 `1.20~1.25` near-hit 실패가 줄었는지
- `focus3_nearest_refine_source=local_probe` 평가 수가 늘었는지
- local_probe의 `raw_improved` 수와 개선률이 늘었는지
- local_probe가 늘면서 Goldstein hit rate를 해치지 않았는지
- `focus3_local_probe_quota_floor_applied`가 실제로 켜졌는지
- local_probe가 active인데도 `selected_local_probe`가 거의 0인 run이 있는지
- Focus3 refine skip이 hit 실패와 상관되는지
- 실패 run들이 full budget을 다 쓰는지, early stop이 잘못 작동하는지
- bounds contains known optimum 조건별 hit rate가 어떻게 갈리는지

## 개선 가능성이 높은 방향

Focus3 쪽에서 가능성이 높은 후보:

- near-hit 상태 감지 후 step scale을 더 미세하게 전환
- local_probe와 best_local의 source별 recent improvement rate를 같이 보고 quota를 동적으로 조정
- local_probe가 실제 개선을 못 만들면 다시 best_local/topk로 quota를 회수
- refine start를 source별로 뽑은 뒤, 최종 `x_next`가 어느 source에 귀속되는지 더 정확히 추적
- L-BFGS-B refine 결과가 시작 source에서 너무 멀리 벗어나면 source attribution을 별도로 기록
- Focus3 acquisition auto policy에서 recovery 상태별 LCB/EI/MEAN 전환 조건 재조정
- dedup fallback이 near-hit 후보를 과하게 밀어내는지 확인
- Focus3 bounds volume과 data density에 따라 best_local sigma와 local_probe step을 자동 조정
- failed full-budget run에서 마지막 200 iter의 source별 개선률을 보고 recovery policy를 바꾸기

Boundary/Focus2 쪽에서 가능성이 높은 후보:

- Focus2 final bounds가 known optimum을 포함하는지 사후 통계화
- contains false 실패와 contains true 실패를 분리해서 Focus2 문제인지 Focus3 문제인지 판정
- Focus2 union/expand/min-volume 정책이 너무 넓거나 좁은지 문제별로 확인
- fallback bounds 사용 run과 정상 region bounds 사용 run을 분리
- Focus2 selected region이 하나뿐일 때 과신하는지 확인
- Focus2가 준 bounds 안에서 archive support가 부족하면 Focus3 초반 source mixture를 다르게 시작

Focus0~1 쪽에서 가능성이 높은 후보:

- low-budget 문제에서 Focus0/1이 너무 많은 budget을 쓰는지 확인
- Focus1 good/uncertain/bad sampling 비율이 Focus2 bounds 품질에 미치는 영향 분석
- Focus1 early stop이 너무 빠르게 걸려 Focus2 region evidence가 부족해지는지 확인
- global archive diversity가 부족한 run에서 Focus2/3가 한쪽으로 몰리는지 확인

완전히 새로운 관점의 후보:

- 단순 hit rate 외에 time-to-hit, budget efficiency, near-hit recovery rate를 별도 metric으로 관리
- source를 고정 확률이 아니라 bandit처럼 최근 개선률 기반으로 선택
- Focus3 내부에서 trust-region 크기를 명시적으로 관리하는 TuRBO-lite 변형
- GP uncertainty calibration이 나쁜 구간을 감지해서 acquisition을 강제로 LCB로 돌리는 정책
- 여러 acquisition 후보를 동시에 점수화한 뒤 disagreement가 큰 경우 exploration을 늘리는 정책
- objective landscape 유형을 직접 맞히지는 않되, history shape로 "narrow valley", "multi-modal", "boundary optimum" 상태를 분류
- 성공 run의 마지막 100 iter 패턴과 실패 run의 마지막 100 iter 패턴을 비교해 recovery trigger를 학습

## 보고 방식

사용자가 "결과 분석"을 요청하면 md/csv 리포트를 만들지 말고 대화로 설명한다.

권장 응답 구조:

1. 분석 대상과 제외 기준
2. 문제별 hit/early stop/budget 요약
3. 병목 문제 지정
4. Focus3 source별 해석
5. boundary contains known optimum 조건별 해석이 가능하면 포함
6. 다음 개선 후보 2~4개
7. 지금 바로 고칠지, 한 번 더 돌릴지 판단

핵심은 점수만 보는 것이 아니라 실패 원인을 분리하는 것이다. 특히 앞으로는 Focus3가 받은 bounds가 맞는 경우에도 실패하는지, bounds 자체가 틀려서 실패하는지를 반드시 나눠서 본다.
