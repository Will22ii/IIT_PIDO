# AION 파이프라인

CAE → DOE → Modeler(FS) → Explorer(DSE) → Optimizer 5단계.
run 산출물은 `run_<problem>_<timestamp>_<hash>/` 아래 task별로 쌓인다.

## 실행 환경
- Python: `venv/Scripts/python.exe` — 시스템 Python 금지
- **파이프라인 실행은 유저가 직접 한다.** "분석/검사/확인" 요청은
  `result/` 기존 산출물을 읽어서 수행하고, 절대 새로 돌리지 않는다.

## 작업 규칙
- **코드 변경 전 계획을 항목별로 설명하고 승인받는다.** auto-accept은 사전 승인이 아니다.
  분석·조사·파일 읽기는 자유. Edit/Write는 승인된 항목만.
- 한 항목을 승인받았다고 인접한 "당연해 보이는" 수정까지 하지 않는다.

## 절대 제약
1. `CAE_model/` benchmark 함수 수정 금지
2. known optimum은 **평가 전용**. 탐색·분기 로직에서 정답 좌표 참조 금지
3. **benchmark 이름 기반 분기 금지.** 모든 로직은 상태/특성 기반 일반화로 설계
4. 사후 라벨(`joint_pass`, `fail_type`, optimum 포함 여부) 실시간 입력 사용 금지
5. ProblemCase별 `n_samples`·`repeat`는 고정값. 실행 시간 단축 목적으로도 변경 금지
   (기준: `pipeline/run_pipelines.py`, 근거: `docs/goals/optimizer_experiment_goals.md`)
6. 앞단 task metadata 경유 금지. 문제 정의는 CAE metadata에서 직접 읽는다.
   task 간에는 데이터(CSV/DataFrame, pkl)만 전달한다.

## 구조

| task | 코드 | result 하위 |
|---|---|---|
| CAE | `CAE_tool_interface/` | `CAE/` |
| DOE | `DOE/{executor,gate,doe_algorithm}/` | `DOE/` |
| Modeler (FS) | `Modeler/{executor,feature_selection,Models}/` | `Modeler/` |
| Explorer (DSE) | `Explorer/executor/`, `strategy_presets.py` | `Explorer/` |
| Optimizer | `Optimizer/{executor,algorithms/focus_bo}/` | `OPT/` |

- config: `pipeline/aion_system_config.py` + 각 모듈 `config.py`. 파서는 `pipeline/config_io.py`
- 산출물 계층: `<task>/artifacts/{public,meta,debug}/`
  - `meta/analysis*.json` — **분석 1순위**
  - Explorer는 전략별로 분리 (`analysis_<strategy>.json`, `metadata_<strategy>.json`)
  - 상세: `docs/policy/pipeline_io_policy.md`

## 진입점과 저장 위치

| 진입점 | 성격 | tries | 저장 위치 |
|---|---|---|---|
| `pipeline/run_pipelines.py` | **배치 시뮬레이터** (점수 책정) | 있음 | `result/bench/<mode>/<batch_ts>/` |
| `pipeline/run_pipeline.py` | 단일 run 엔진 | 없음 | `result/service/<run_id>/` |
| `pipeline/run_AION.py` | AION 프리셋 래퍼 (백엔드 호출용) | 없음 | `result/service/<run_id>/` |

`pipeline/run_context.py`는 진입점이 아니라 run 디렉터리/`index.json` 관리 모듈이다.

```
result/
  bench/<mode>/<batch_ts>/      # mode: aion | full | opt_only | <task조합>
    runs/                       # run_* 디렉터리
    stats/                      # CSV 6종 (fi_primary / explorer_strategy / optimizer × try_stats·problem_summary)
    batch_meta.json             # mode, 스키마 버전, git commit, problem_cases
  bench/_legacy/stats/          # 구조 변경 이전 배치의 stats. mode 미상
  service/<run_id>/             # 서비스 단일 run
  best/                         # 승격 스냅샷
  reports/                      # 분석 산출물
```

mode는 `aion_mode`와 켜진 task 조합에서 자동 도출한다(`_resolve_batch_mode`). **서로 다른 mode의 결과를 같은 표에서 섞지 않는다.**

## 결과 분석 규약

**bench**(110~1,100 runs)는 컨텍스트 예산 때문에 반드시 계층적으로 읽는다.

1. `<batch>/stats/*problem_summary*.csv`(3개, ~3.6 KB)에서 시작한다. 여기서 곱셈 체인을 분해해 병목 task를 지목한다.
2. `try_stats`는 **필요한 컬럼만 투영**한다. `explorer_strategy_try_stats`는 158컬럼이므로 통째로 읽지 않는다.
3. `runs/` 디렉터리는 **특정된 run에 한해서만** 연다. 전수 순회 금지.
4. 열람 대상은 실패뿐 아니라 **아슬아슬한 성공**(goal 대비 마진이 얇은 run)과 **매칭 성공 대조군**을 포함한다. 실패만 보면 원인 규명도, 안정성 진단도 불가능하다.
5. 10건을 넘는 드릴다운은 서브에이전트에 위임하고 표만 회수한다. 파일 원문을 반환받지 않는다.
6. **숫자 집계를 LLM이 하지 않는다.** 집계는 `run_pipelines.py`가 배치 종료 시 만든 stats CSV 또는 스크립트가 한다.

**service**는 run이 1개뿐이라 summary가 없다. `<run>/index.json`과 `artifacts/meta/analysis*.json`을 직접 읽는다.

최종 점수는 곱셈 게이트다 — 상세는 `docs/playbook/full_pipeline_focus3_optimization_playbook.md`.

```
optimizer_final_pass = modeler_all_real_only × explorer_bounds_pass × optimizer_goal_hit
```

각 지표가 95를 넘어도 곱하면 95 미만일 수 있다. 목표는 **110 runs micro 평균**이며 benchmark 동등가중(macro)이 아니다.

## 지표 / 목표
- **FS**: `fi_real_only_success_pct` ≥ 95
- **DSE**: `joint_pass_pct_macro` ≥ 95
  = `survivor_optimum_included` ∧ `volume_cap_pass`(`volume_ratio` ≤ 0.25)
  실패 유형: `over_shrink_fail` / `over_wide_fail` / `both_fail`
- **Optimizer**: `focus_bo` — `docs/goals/optimizer_experiment_goals.md`

## 문서 라우팅
작업 전 해당 문서를 읽는다. 폴더 판별 기준은 `docs/README.md`.

| 찾는 것 | 위치 |
|---|---|
| 현재 코드가 지켜야 할 계약 | `docs/policy/` |
| 결과를 읽고 다음을 고르는 절차 | `docs/playbook/` |
| 목표 수치 | `docs/goals/` |
| **미구현** 설계·보류 항목 | `docs/planned/` ← 구현됐다고 착각 금지 |

분석 스킬: `/dse` `/fs` `/score-analyze` (`.claude/skills/`, 재작성 예정)
