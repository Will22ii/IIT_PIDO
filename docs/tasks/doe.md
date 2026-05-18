# DOE Policy

DOE는 실행된 design sample을 만들고, downstream 데이터 입력용으로 간단한 public CSV를 저장한다.

## Inputs

필수 입력:

- CAE context
- DOE configuration

DOE는 CAE의 variables, bounds, objective sense, constraints를 문제 정의로 사용한다.

## Public Outputs

주요 public artifact:

```text
DOE/artifacts/public/doe_results.csv
```

Public DOE CSV는 downstream에 필요한 compact column만 포함한다.

- `id`
- `objective`
- `feasible`
- `success`
- CAE design variable columns

Public CSV에 아래 내부/debug field는 넣지 않는다.

- `source`
- `round`
- `exec_scope`
- `constraint_*`
- margins
- raw constraint detail JSON

## Debug Outputs

`debug_level == "on"`일 때 DOE는 아래 파일을 쓴다.

```text
DOE/artifacts/debug/doe_results_internal.csv
```

Internal CSV에는 아래 정보를 포함할 수 있다.

- `source`
- `round`
- `exec_scope`
- `constraint_*`
- `feasible_pre`
- `feasible_post`
- margins
- `constraint_details_json`

`debug_level == "off"`이면 internal debug CSV를 쓰지 않고 metadata에도 등록하지 않는다.

## Metadata

DOE는 analysis metadata를 아래 위치에 저장할 수 있다.

```text
DOE/artifacts/meta/
```

이 metadata는 summary, diagnostics, batch analysis 용도다. Explorer runtime은 DOE metadata에 의존하지 않는다.

## Downstream Contract

Explorer는 `doe_results.csv`를 input CSV로 사용할 수 있다. 단, column은 CAE feature 또는 selected feature list와 대조해야 한다.

Explorer는 feature schema나 constraints를 결정하기 위해 DOE metadata를 읽지 않는다.

