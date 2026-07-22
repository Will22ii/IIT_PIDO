# docs

AION 파이프라인 문서. 성격에 따라 4개로 나눈다.

| 폴더 | 성격 | 판별 질문 |
|---|---|---|
| `policy/` | 현재 시스템의 계약. **이미 구현됨** | "코드가 이걸 지켜야 하는가" |
| `playbook/` | 분석·개선 절차 | "결과를 어떻게 읽고 다음을 고르는가" |
| `goals/` | 목표·타겟 수치 | "무엇을 달성해야 하는가" |
| `planned/` | **미구현** 설계와 보류 항목 | "아직 코드에 없는가" |

문서를 새로 추가할 때는 위 판별 질문으로 폴더를 고른다.
특히 **구현되지 않은 것은 반드시 `planned/`**에 둔다. `policy/`에 있는 문서는
"현재 코드가 이렇게 동작한다"는 뜻이므로, 미구현 설계가 섞이면 계약을 신뢰할 수 없게 된다.

---

## policy/ — 현재 시스템 계약

| 파일 | 내용 |
|---|---|
| `pipeline_config_contract.md` | config JSON/dict 스키마. `pipeline/config_io.py`가 실제 파서 기준 |
| `pipeline_io_policy.md` | 단계 간 입출력, metadata, artifact 계층(public/meta/debug), objective 정규화 규약 |
| `tasks/cae_tool_interface.md` | CAE task 정책. 모든 run의 선행 task |
| `tasks/doe.md` | DOE task 정책 |
| `tasks/modeler.md` | Modeler task 정책. feature 스키마의 권위 |
| `tasks/explorer.md` | Explorer task 정책. standalone 동작 요구사항 포함 |
| `tasks/optimizer.md` | Optimizer task 정책. focus 단계 정의 |

## playbook/ — 분석·개선 절차

스킬(`.claude/skills/`)이 사용하는 방법론이 여기 있다.

| 파일 | 내용 |
|---|---|
| `explorer_dse_analysis_playbook.md` | DSE 점수(`joint_pass`) 분석 기준, 허용/금지 개선 목록, 데이터 정합성 규칙 |
| `optimizer_focus_analysis_playbook.md` | Optimizer 개선 실험 분석 기준 |
| `full_pipeline_focus3_optimization_playbook.md` | 전 단계를 켠 full pipeline run 분석. 곱셈 게이트 체인 정의 |
| `gp_scaling_experiment_playbook.md` | GP 입력 X scaling 실험 설계와 해석 기준 |
| `optimizer_next_analysis.md` | 다음 batch 종료 후 확인할 체크리스트 |

## goals/ — 목표

| 파일 | 내용 |
|---|---|
| `optimizer_experiment_goals.md` | `focus_bo` 단기 benchmark 목표. `n_samples`/`repeat`는 고정값이며 줄이면 안 됨 |

## planned/ — 미구현

`planned/README.md`에 항목별 상태가 정리되어 있다.

| 파일 | 상태 |
|---|---|
| `doe_pre_equality_projection.md` | 설계 확정, 미구현 |
| `optimizer_backlog.md` | 보류 결정 모음 |
