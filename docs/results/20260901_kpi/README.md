# 20260901 KPI 측정 배치

국책 KPI 6개 항목을 전부 통과한 배치의 원본 stats다. `result/`가 gitignore되어
있어 산출물이 로컬에만 남는 문제를 피하려고 요약·상세 CSV를 여기에 보존한다.
과거 점수(96.35, 106/110 등)가 커밋 메시지 문구로만 남아 재검증이 불가능했던
전례를 반복하지 않기 위한 조치다.

- 배치 경로: `result/bench/aion/20260901_091126/`
- 실행 코드: 이 디렉토리를 추가한 커밋 (batch_meta.json의 `git_commit`은 배치
  시작 시점 HEAD인 `2edbc7e`로 찍혀 있으나, 실제 실행된 코드는 그 시점의
  미커밋 작업 트리이며 그 내용이 이 커밋에 담겨 있다)
- 110 runs / AION mode / `optimizer_goal_mode=off` / debug=on / fast_mode 미사용

## 결과

### 1. 국책 KPI

| 주요 성능지표 | 최종 개발목표 | 결과 | 판정 |
| --- | ---: | ---: | :-: |
| Feature Importance 성공률 | 95 이상 | 100.00 | PASS |
| 유효 설계공간 탐색 성공률 | 95 이상 | 95.45 | PASS |
| 벤치마크1 Goldstein-Price | 3.60 이하 | 3.0003 | PASS |
| 벤치마크2 Rosenbrock | 1.20 이하 | 1.0254 | PASS |
| 벤치마크3 Cantilever Beam | 114.66 이하 | 95.5957 | PASS |
| 벤치마크4 Six-Hump Camel | -0.86 이하 | -1.0181 | PASS |

벤치마크 4종은 run 평균으로 판정한다(`점수측정체계.md` §4.1).

### 2. task 점수 (110 runs micro 평균)

| task | 지표 | 결과 | 목표 |
| --- | --- | ---: | ---: |
| FS | `fi_real_only_success_pct` | 100.00 (110/110) | >= 95 |
| DSE | `explorer_bounds_pass_pct` | 95.45 (105/110) | >= 95 |
| opt | `optimizer_goal_hit_pct` | 99.09 (109/110) | >= 95 |

### 3. run 단위 판정

95.45% (105/110). 엄격판(`modeler_all_real_only` 기준)과 동일하다 — FS가 100%라
`all_real_only`와 `all_real_included`가 같은 값이기 때문이다.

## 실패한 5 run

전부 DSE 실패이며 4건이 cantilever_beam에 몰려 있다.

| problem | repeat | seed | DSE | opt | best objective |
| --- | ---: | ---: | :-: | :-: | ---: |
| cantilever_beam | 7 | 6042 | FAIL | pass | 94.0034 |
| cantilever_beam | 13 | 12042 | FAIL | FAIL | 134.1546 |
| cantilever_beam | 20 | 19042 | FAIL | pass | 93.6161 |
| cantilever_beam | 24 | 23042 | FAIL | pass | 93.7401 |
| six_hump_camel | 7 | 6045 | FAIL | pass | -0.8660 |

opt 실패 1건도 DSE 실패에서 파생된 것이다(경계가 optimum을 잘라내 134까지 튐).
즉 남은 개선 여지는 사실상 전부 DSE, 그중에서도 cantilever_beam에 있다.

## 재현

```
venv/Scripts/python.exe -m pipeline.run_pipelines
```

`ACTIVE_PROBLEM_CASES` 4종, `BATCH_FAST_OPTIMIZER_MODE=False`,
`BATCH_OPTIMIZER_GOAL_MODE="off"`, `BATCH_OPTIMIZER_SYSTEM_OVERRIDES={}` 상태에서
`--base-seed` 없이 실행하면 같은 seed 배열이 재현된다. 파이프라인은 결정적이므로
동일 코드·동일 seed면 결과가 소수점까지 일치한다(20260831 배치 2회로 확인).
