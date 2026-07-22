# Full Pipeline Focus3 Optimization Playbook

이 문서는 DOE, Modeler, Explorer, Optimizer를 모두 켠 full pipeline run을 분석하고 개선할 때의 기준이다. 목적은 상위 task가 만든 feature, data, bounds를 Optimizer가 받아서 Focus3 중심으로 안정적으로 goal을 달성하게 만드는 것이다.

## 목표

최종 목표는 full pipeline 기준 최종 score 95% 이상이다.

우선순위는 아래처럼 둔다.

1. Optimizer Focus3가 Explorer selected bounds 안에서 goal을 안정적으로 달성한다.
2. `optimizer_final_pass_pct`를 95% 이상으로 올린다.
3. Focus3 실패 원인을 source, acquisition, refine, recovery, dedup 관점에서 분해한다.
4. Explorer, Modeler, Additional DOE 개선은 Focus3보다 낮은 비중으로 다루되, 최종 점수에 영향을 주는 경우 원인을 기록한다.

## Score Gate 정의

점수는 세 단계로 본다.

```text
1. primary selection:
   modeler_all_real_only

2. Explorer boundary:
   explorer_bounds_pass

3. Optimizer:
   optimizer_goal_hit
```

각 단계의 raw score는 따로 볼 수 있지만, downstream joint score는 앞단 gate를 모두 통과해야 pass로 센다. 즉 하나라도 틀리면 최종 pass는 false다.

현재 CSV 구현 기준:

```text
modeler_all_real_only =
  selected_features == real_variables

explorer_bounds_pass =
  survivor_optimum_included
  * volume_cap_pass

explorer_joint_pass =
  modeler_all_real_only
  * explorer_bounds_pass

optimizer_final_pass =
  explorer_joint_pass
  * optimizer_goal_hit
```

따라서 primary selection이 틀리면 Explorer가 known optimum을 포함해도 `explorer_joint_pass = false`다. Explorer가 틀리면 Optimizer가 goal을 찾아도 `optimizer_final_pass = false`다. Optimizer 자체 점수는 `optimizer_goal_hit_pct`로 별도 확인한다.

주의: 현재 `explorer_bounds_pass`는 known optimum inclusion만이 아니라 volume cap도 포함한다. 공식 Explorer 점수를 "boundary에 known optimum 포함 여부"만으로 볼지, "known optimum 포함 + volume cap"으로 볼지는 분석 시작 시 확인한다.

최종 score는 아래 연쇄 조건으로 본다.

```text
optimizer_final_pass =
  modeler_all_real_only
  * explorer_bounds_pass
  * optimizer_goal_hit
```

따라서 Focus3만 좋아져도 앞단 score가 낮으면 최종 score는 100점이 될 수 없다. 이 불리함은 분석에서 감안하되, 기본 개선 책임은 Focus3에 둔다.

## Run 보관 위치

새 full pipeline run은 기본적으로 `result/run_*`와 batch summary CSV에 쌓인다.

과거 full pipeline 실행 결과는 아래 폴더에 보관한다.

```text
result/full_pipeline_runs/
```

이 폴더는 과거 run archive로 취급한다. 새로 실행한 최신 결과와 섞어서 최신 batch로 판단하지 않는다.

분석 대상은 목적에 따라 분리한다.

- 최신 full pipeline 분석: `result/explorer_strategy_stats/*`의 최신 CSV와 `result/run_*`
- 과거 full pipeline 재분석: `result/full_pipeline_runs/run_*`
- optimizer-only 분석: `result/optimizer_only_runs/run_*` 또는 optimizer-only playbook 기준

## 기본 전제

- full pipeline run에서는 `BATCH_TASKS`의 `doe`, `modeler`, `explorer`, `optimizer`가 모두 `True`여야 한다.
- Explorer selected bounds가 있으면 Optimizer auto pipeline은 Focus2로 가지 않고 Focus3로 간다.
- Archive가 부족하면 Focus0/Focus1이 objective archive를 보강할 수 있지만, Explorer bounds를 다시 Focus2로 줄이거나 넓히지 않는다.
- Known optimum은 benchmark 분석용으로만 사용한다. Optimizer 로직에 benchmark 정답 좌표나 함수명을 넣지 않는다.
- 결과 분석을 시작할 때 바로 코드 수정하지 않는다. 먼저 실패 원인과 개선 후보를 설명한다.

## 출력 CSV 기준

full pipeline batch가 끝나면 아래 6개 CSV를 본다.

```text
result/explorer_strategy_stats/fi_primary_try_stats_*.csv
result/explorer_strategy_stats/fi_primary_problem_summary_*.csv

result/explorer_strategy_stats/explorer_strategy_try_stats_*.csv
result/explorer_strategy_stats/explorer_strategy_problem_summary_*.csv

result/explorer_strategy_stats/optimizer_try_stats_*.csv
result/explorer_strategy_stats/optimizer_problem_summary_*.csv
```

가장 먼저 볼 파일은 `optimizer_problem_summary_*.csv`다.

핵심 컬럼:

- `optimizer_final_pass_pct`: 최종 score
- `optimizer_goal_hit_pct`: Focus3/Optimizer 자체 goal hit
- `explorer_joint_pass_pct`: feature selection과 Explorer bounds를 모두 통과한 비율
- `explorer_bounds_pass_pct`: known optimum inclusion과 volume cap만 본 Explorer bounds score
- `modeler_all_real_only_pct`: feature selection strict score
- `problem = __overall__`: 전체 평균 score

문제별 상세 실패 원인은 `optimizer_try_stats_*.csv`와 `explorer_strategy_try_stats_*.csv`를 같이 본다.

## 분석 순서

분석은 항상 score를 곱셈 구조로 분해해서 시작한다.

1. `optimizer_problem_summary_*.csv`에서 `__overall__`의 `optimizer_final_pass_pct` 확인
2. 문제별 `optimizer_final_pass_pct` 확인
3. 문제별로 아래 병목을 분리
   - `modeler_all_real_only_pct`가 낮은가
   - `explorer_bounds_pass_pct`가 낮은가
   - `optimizer_goal_hit_pct`가 낮은가
4. Focus3 책임 구간은 `explorer_joint_pass = true`인데 `optimizer_goal_hit = false`인 run이다.
5. Focus3 분석 대상 run을 골라 OPT debug history와 metadata를 읽는다.
6. 실패 run의 source, acquisition, refine, recovery, dedup, budget 사용량을 성공 run과 비교한다.

## Focus3 우선 분석 기준

Focus3 개선은 아래 지표를 우선 본다.

- 문제별 `optimizer_goal_hit_pct`
- `explorer_joint_pass = true` 조건에서의 `optimizer_goal_hit_pct`
- 실패 run의 best objective가 goal 근처인지, 멀리 있는지
- full budget을 다 쓰고 실패했는지
- early stop이 잘못 걸렸는지
- `focus_pipeline_stages`가 기대대로 `focus3` 중심인지
- `focus_pipeline_bounds_source = explorer`인지
- `focus3_best_plan_source`
- `focus3_nearest_refine_source`
- `raw_improved`
- `focus3_recover_active`
- `focus3_recover_level`
- `focus3_recover_no_improve_count`
- `focus3_refine_filter_skipped`
- `focus3_dedup_applied`
- `focus3_best_local_applied`
- `focus3_local_probe_*` 계열 진단값이 있으면 반드시 확인

문제별로 볼 관점:

- Rosenbrock: narrow valley, near-hit 실패, best_local/local_probe/refine 품질
- Goldstein-Price: multi-modal, source mixture 균형, random/topk 균형
- Cantilever Beam: constraint 및 boundary 근처 수렴 안정성
- Six-Hump Camel: low-dimensional multi-modal에서 빠른 goal hit와 과도한 탐색 여부

## 실패 유형 분류

full pipeline 실패는 아래처럼 분류한다.

```text
feature_fail:
  modeler_all_real_only = false

explorer_fail:
  modeler_all_real_only = true
  explorer_bounds_pass = false

optimizer_fail:
  explorer_joint_pass = true
  optimizer_goal_hit = false

pipeline_pass:
  optimizer_final_pass = true
```

Focus3 고도화의 직접 대상은 `optimizer_fail`이다.

`feature_fail`과 `explorer_fail`은 기록하되, Focus3 tuning의 성공/실패 판단에 섞지 않는다. 다만 전체 최종 score가 95%에 못 미치면 앞단 개선 후보도 별도 항목으로 제시한다.

## 개선 우선순위

Focus3 쪽 개선 후보를 우선한다.

1. `explorer_joint_pass = true`인데 goal hit 실패한 run의 source별 개선률 분석
2. near-hit 실패가 많으면 best_local/local_probe/refine 정책 우선 조정
3. 멀리 실패하면 source mixture, acquisition mode, recovery trigger를 우선 조정
4. refine skip이 실패와 연결되면 allowed source 또는 cooldown 정책 재검토
5. dedup fallback이 좋은 후보를 밀어내면 distance threshold와 fallback source를 재검토
6. budget이 충분한데 후반 개선이 멈추면 recovery strong/mild 전환과 quota 이동을 재검토

앞단 개선은 낮은 비중으로 보되 아래 조건이면 분석에 포함한다.

- `modeler_all_real_only_pct < 95`
- `explorer_bounds_pass_pct < 95`
- 특정 problem에서 feature/explorer failure가 전체 score를 지배
- Additional DOE가 Explorer bounds 품질에 직접 영향을 주는 패턴이 확인됨

## 개선안 제시 방식

사용자가 분석을 요청하면 바로 수정하지 않는다.

응답은 아래 순서로 한다.

1. 분석 대상 파일과 run 범위
2. 전체 score와 문제별 score
3. 병목 task 분해
4. Focus3 직접 실패 run 수와 실패 양상
5. Focus3 source/refine/recovery 해석
6. 개선 후보 2~5개
7. 추천 우선순위와 예상 리스크
8. 수정 여부 확인

코드 수정은 사용자가 명시적으로 시작하라고 한 뒤에만 한다.

## 개선 히스토리

아직 full pipeline Focus3 고도화 분석은 시작 전이다.

기록 방식:

```text
YYYY-MM-DD
- 분석 대상:
- 전체 score:
- 문제별 병목:
- Focus3 실패 패턴:
- 변경 제안:
- 실제 수정:
- 재실험 결과:
- 다음 액션:
```

## 현재 기준선

아직 기준선 run은 등록하지 않았다.

첫 full pipeline run이 끝나면 아래 항목을 채운다.

- active problem suite
- repeats
- DOE budget
- Optimizer budget
- HPO 사용 여부
- Explorer strategy
- `optimizer_final_pass_pct`
- `optimizer_goal_hit_pct`
- `explorer_joint_pass_pct`
- `modeler_all_real_only_pct`

## 주의할 점

- `optimizer_goal_hit_pct`만 보고 성공으로 판단하지 않는다. 최종 점수는 feature selection과 Explorer bounds까지 곱해진다.
- `optimizer_final_pass_pct`가 낮아도 Focus3가 병목이 아닐 수 있다. 반드시 분해한다.
- Focus3 tuning은 benchmark 정답을 직접 사용하면 안 된다.
- Explorer bounds가 틀린 run을 Focus3 실패로 벌주지 않는다.
- 반대로 Explorer bounds가 맞았는데 Focus3가 실패한 run은 가장 중요한 개선 대상이다.
- full pipeline 결과와 optimizer-only 결과를 같은 표에서 섞지 않는다.
