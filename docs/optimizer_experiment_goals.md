# Optimizer 실험 목표

이 문서는 기본 제공 Optimizer 알고리즘인 `focus_bo`의 단기 benchmark 목표를 정리한다. 여기의 목표는 `run_pipelines.py`에서 고정한 evaluation budget 기준의 score target이다. 실행 시간을 줄이기 위해 `n_samples`나 `repeat`를 줄이면 안 된다.

## 범위

초기 목표:

- Optimizer만 실행한다.
- selected bounds가 없으면 내부 Focus0 -> Focus1 -> Focus2 -> Focus3 pipeline을 사용한다.
- 문제별 Optimizer budget과 repeat 수는 고정한다.
- Optimizer-only 성공 여부는 아래 문제별 goal threshold를 달성했는지로 판단한다.

이후 목표:

- DOE/Modeler/Explorer까지 upstream task를 모두 실행한 뒤, 그 결과 bounds/model/data를 사용해 Focus3를 실행한다.
- 이 문서의 Optimizer-only baseline과 비교한다. Upstream evidence가 있는 경우에는 이후 더 강한 goal을 따로 둘 수 있다.

## Known Optimum과 Optimizer Goal

아래 값들은 모두 minimization objective 기준이다.

`known optimum`은 benchmark 분석용 정보이지 Optimizer runtime 요구사항이 아니다. 실제 서비스 run에서는 보통 global optimum 위치를 알 수 없고, target quality goal만 있다. 따라서 Optimizer-only run은 `Optimizer-only goal`을 달성했는지로 평가한다.

| Problem | Known optimum objective | Optimizer-only goal | Relative gap vs `abs(optimum)` |
| --- | ---: | ---: | ---: |
| Goldstein-Price | 3.0000 | 3.6000 | 20.00% |
| Rosenbrock shifted | 1.0000 | 1.2000 | 20.00% |
| Cantilever Beam | 92.7700 | 114.6600 | 23.60% |
| Six-Hump Camel | -1.0316 | -0.8600 | 16.63% |

메모:

- Rosenbrock는 objective에 `+1` shift가 들어가 있다. 수학적 원래 optimum은 `0.0`이지만, benchmark objective optimum은 `1.0`이다.
- Rosenbrock의 단기 goal은 `1.2`로 둔다. Shifted optimum `1.0` 대비 20% gap이다.
- 장기 benchmark 목표는 positive/negative objective 모두에 일관적인 metric을 정한 뒤 known optimum objective 기준 약 10% 이내로 들어오는 것이다.

## Task별 책임

Optimizer:

- 목표는 할당된 evaluation budget 안에서 문제별 goal threshold를 달성하는 것이다.
- Focus2 또는 Focus3 bounds 안에 benchmark known optimum을 반드시 포함할 의무는 없다.
- 다른 region에서 goal을 달성하면 Optimizer-only benchmark에서는 성공이다.

Explorer:

- 목표는 downstream solution potential을 보존하는 selected bounds를 만드는 것이다.
- Benchmark 분석에서는 known optimum inclusion을 Explorer bounds 품질 진단값으로 볼 수 있다.
- 실제 run에서는 known optimum이 없으므로, Explorer는 downstream solution quality와 robustness로 판단해야 한다.

DOE / additional DOE:

- 목표는 Explorer와 Optimizer가 좋은 region을 찾을 수 있도록 objective/feasibility data를 충분히 제공하는 것이다.
- Additional DOE는 Explorer bounds가 좋은 region을 포함할 가능성을 높인다. 하지만 known optimum inclusion은 benchmark diagnostic이지 runtime contract가 아니다.

## Runtime 단축 정책

Runtime 개선은 benchmark 조건을 보존해야 한다.

- `n_samples`를 줄이지 않는다.
- `repeat`를 줄이지 않는다.
- Score run 중 Focus3 pool size, acquisition refinement, GP update policy 같은 후보 품질 knob를 조용히 낮추지 않는다.

허용되는 최적화 방향:

- Focus, GP fit, candidate scoring, acquisition refinement, CAE evaluation, debug plotting 단위 timing/profiling log 추가
- 후보 선택이 바뀌지 않는 범위에서 normalization, bounds, duplicate-distance, archive slicing 계산 cache
- 최종 selected candidate가 동일하게 유지되는 범위에서 독립 candidate scoring/refinement vectorization 또는 parallelization
- 독립 problem/repeat run의 batch-runner level parallelization
- Debug plot은 매 Optimizer iteration이 아니라 task 종료 시점에 생성

명시적인 benchmark 검증이 필요한 위험한 최적화 방향:

- `gp_refit_every` 증가
- Focus3 plan pool size 감소
- `focus3_refine_starts` 감소
- Focus2 local candidate pool size 변경
- Exact GP behavior를 approximate surrogate로 대체
