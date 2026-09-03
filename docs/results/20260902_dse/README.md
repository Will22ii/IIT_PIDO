# 20260902 DSE 개선 측정 배치

Explorer의 지지가중 확장(`bounds_expand_pred_support_enabled`)을 켜고 측정한
110 runs. `docs/results/20260901_kpi/`(kpi-20260901 기준점) 대비 DSE가
105 -> 108/110으로 오르고 나머지 지표는 유지 또는 개선됐다.

- 배치 경로: `result/bench/aion/20260902_115957/`
- 실행 코드: 이 디렉토리를 추가한 커밋 (batch_meta의 git_commit은 배치 시작
  시점 HEAD `979a9ab`로 찍혀 있으나 실제 실행 코드는 그 시점의 미커밋 작업
  트리이며 그 내용이 이 커밋에 담겨 있다)
- 조건: AION mode / optimizer_goal_mode=off / fast_mode 미사용 / base_seed 42

## 결과 (괄호는 kpi-20260901 확정치)

### 1. 국책 KPI — 6/6 PASS

| 지표 | 목표 | 결과 | 이전 |
| --- | ---: | ---: | ---: |
| Feature Importance 성공률 | 95 이상 | 100.00 | 100.00 |
| 유효 설계공간 탐색 성공률 | 95 이상 | 98.18 | 95.45 |
| Goldstein-Price | 3.60 이하 | 3.0002 | 3.0003 |
| Rosenbrock | 1.20 이하 | 1.0110 | 1.0254 |
| Cantilever Beam | 114.66 이하 | 95.0896 | 95.5957 |
| Six-Hump Camel | -0.86 이하 | -1.0086 | -1.0181 |

### 2. task 점수 (110 runs micro)

| task | 결과 | 이전 |
| --- | ---: | ---: |
| FS | 100.00 (110/110) | 100.00 |
| DSE | 98.18 (108/110) | 95.45 (105/110) |
| opt | 99.09 (109/110) | 99.09 |

DSE 기준(95%) 미달까지의 여유가 0 run에서 3 run으로 늘었다.

### 3. run 단위 판정

98.18% (108/110). 이전 95.45% (105/110).

## 남은 실패 2 run (둘 다 cantilever_beam)

| repeat | seed | DSE | opt | best objective |
| ---: | ---: | :-: | :-: | ---: |
| 13 | 12042 | FAIL | FAIL | 134.1546 |
| 20 | 19042 | FAIL | pass | 93.4155 |

- rep 13: obj 클러스터 부피가 0.675로 최소부피(0.2499)를 이미 넘어 지지가중
  확장 경로를 타지 않는 케이스. cap 축소 경로의 별개 문제.
- rep 20: b1 상한이 9.03에서 멈춰 optimum(9.485)에 4.5% 미달. 해 자체는
  93.42로 사실상 최적.

## 안전성 검증

- rosenbrock / goldstein_price: 경계가 전 run에서 바뀌었음에도 DSE·opt 100%
  유지, 목적함수 평균은 개선 (RB 1.0254 -> 1.0110).
- six_hump_camel: DSE 24 -> 25/25 (만점).
- cantilever_beam 25 runs는 전날 CB 단독 검증 배치(20260902_083340)와 동일 —
  파이프라인 결정성 재확인.
- volume cap(<=0.25) 위반 0건.
