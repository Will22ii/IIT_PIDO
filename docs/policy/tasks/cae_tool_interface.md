# CAE Tool Interface Policy

`CAE_tool_interface`는 유효한 pipeline run에서 반드시 먼저 실행되는 task다.

## 책임

CAE는 최적화 문제를 정의한다. 아래 항목의 source of truth는 CAE output이다.

- problem name
- random seed
- objective sense
- design variable names
- design variable bounds
- pre-constraint definitions
- post-constraint definitions

Downstream task는 자신의 입력을 CAE 문제 정의와 대조해야 한다.

## Required Outputs

CAE task는 DOE, Modeler, Explorer, Optimizer 실행 전에 run context에 metadata를 저장해야 한다.

Downstream에서 기대하는 대표 field:

- `problem`
- `inputs.variables`
- `inputs.constraint_defs`
- `resolved_params.objective_sense`
- seed 정보

Seed 정보는 resolved params, inputs, user config snapshot 중 하나에서 찾을 수 있어야 한다.

## Runtime Contract

CAE는 optional task가 아니다. Downstream task가 현재 run context에서 CAE metadata를 찾지 못하면 fast-fail한다.

Downstream task 안에서 조용히 새 CAE context를 만들지 않는다. run에는 이미 CAE context가 있어야 한다.

## CAE가 소유하지 않는 것

CAE는 실행된 sample data를 소유하지 않는다. Sample data는 DOE public CSV 또는 외부 CSV에서 온다.

CAE는 Modeler selected feature를 소유하지 않는다. Modeler는 selected feature list를 만들 수 있지만, 그 list는 항상 CAE design variable의 subset이어야 한다.

