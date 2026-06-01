# Pipeline Config Contract

This document describes the current development/backend config contract.
It is not the final service UI schema.

The source of truth is the parser code:

- `pipeline.config_io.pipeline_config_from_dict` for `run_pipeline`
- `pipeline.run_AION.aion_config_from_dict` for `run_AION`

Example JSON files under `pipeline/config_templates/` are executable examples of
this contract, not a frozen product API.

## Shared Policy

- CLI entrypoints are disabled by policy.
- Backend/Python code should call `run_pipeline_from_dict/json` or
  `run_aion_from_dict/json`.
- Unknown config keys fail fast, except explicitly ignored removed compatibility
  keys such as Optimizer `n_restarts`.
- `debug_level` is controlled at `run.debug_level` and propagated to task system
  configs.
- `run.run_root` is optional. If provided, backend is asking to continue/reuse an
  existing run context. If omitted or `null`, a new run context is created.

## `run_pipeline` Contract

Top-level keys accepted by `pipeline.config_io`:

```text
problem
run
tasks
reuse
inputs
doe
modeler
explorer
optimizer
```

### `problem`

Required.

```json
{
  "name": "cantilever_beam",
  "seed": 42,
  "objective_sense": "min",
  "variables": null
}
```

- `name` or `problem_name` is required.
- `seed` defaults to `42`.
- `objective_sense` defaults to `"min"`.
- `variables` is optional CAE variable override.
- `known_optimum` is not part of the service-style `run_pipeline` contract. It
  is used only by `run_pipelines.py` benchmark/debug cases.

### `run`

Optional.

```json
{
  "run_root": null,
  "debug_level": "on",
  "use_timestamp": false
}
```

- `debug_level`: `"on"` or `"off"`.
- `run_root`: optional existing run root supplied by backend.
- `use_timestamp`: controls result saver timestamp behavior.

### `tasks`

Optional. Defaults:

```json
{
  "doe": true,
  "modeler": true,
  "explorer": true,
  "optimizer": false
}
```

Task execution is decided only by this section. A task config section may exist
while the task is disabled. In that case the task is configured but not executed.

### `reuse`

Optional.

```json
{
  "use_existing_doe_csv": true,
  "use_existing_modeler_artifacts": true
}
```

This controls whether downstream tasks can reuse public artifacts from an
existing `run_root` when a previous task is not executed in the current call.

### `inputs`

Optional standalone/direct artifact paths.

```json
{
  "doe_csv_path": null,
  "selected_features_csv_path": null,
  "model_pkl_path": null,
  "fi_scores_path": null,
  "explorer_bounds_path": null
}
```

These paths are task inputs, not task outputs.

### Task Sections

Each task section supports direct keys plus optional nested `user` and `system`
objects. Direct keys are convenience aliases for the common user/system fields.

DOE:

```json
{
  "n_samples": 90,
  "algo_name": "lhs",
  "use_additional": false
}
```

Modeler:

```json
{
  "model_name": "xgb",
  "use_hpo": false,
  "use_primary_selection": true,
  "use_secondary_selection": false,
  "target_col": "objective"
}
```

Explorer:

```json
{
  "strategy_id": "S4_obj",
  "save_plot": true
}
```

Optimizer:

```json
{
  "n_samples": 30,
  "goal": null,
  "doe_csv_path": null,
  "explorer_bounds_path": null,
  "system": {
    "algorithm_id": "focus_bo",
    "focus_pipeline": "auto",
    "algorithm_params": {}
  }
}
```

`algorithm_id` selects the Optimizer algorithm implementation from the Optimizer
registry. The built-in default is `focus_bo`.

Custom Optimizer algorithms are discovered from `Optimizer/algorithms/custom`.
They must use the common runtime API for evaluation, goal early-stop, archive
tracking, and standard output generation. Backend code can inspect
`Optimizer.algorithms.describe_optimizer_algorithms()` to list available
algorithms and discovery errors.

Optimizer official progress naming inside `focus_bo` is focus-only:

```text
focus0
focus1
focus2
focus3
```

Focus0~3 are internal stages of the built-in `focus_bo` algorithm. Other
registered algorithms do not need to use focus stages as long as they consume the
common Optimizer inputs and return the common Optimizer result contract.

`phase` is reserved for DOE additional internals and is not part of Optimizer
output/debug/metadata schema.

## `run_AION` Contract

`run_AION` intentionally has a narrower schema. It does not accept `tasks` or
`inputs`; task choices are fixed by AION policy.

Top-level keys accepted by `pipeline.run_AION`:

```text
problem
run
reuse
doe
optimizer
explorer
```

Required `problem.name` or `problem.problem_name`.

Recognized controls:

```json
{
  "problem": {
    "name": "cantilever_beam",
    "seed": 42,
    "objective_sense": "min",
    "variables": null
  },
  "run": {
    "run_root": null,
    "debug_level": "on",
    "use_timestamp": false
  },
  "reuse": {
    "use_existing_doe_csv": true,
    "use_existing_modeler_artifacts": true
  },
  "doe": {
    "n_samples": 90
  },
  "optimizer": {
    "n_samples": 30
  }
}
```

AION ignores user task selection because it is a fixed full-policy runner.

## Validation Status

Current templates validated against the parsers:

- `pipeline/config_templates/run_pipeline.example.json`
- `pipeline/config_templates/run_AION.example.json`
