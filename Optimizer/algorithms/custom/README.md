# Custom Optimizer Algorithms

Drop user-provided optimizer algorithm files in this directory. Files are
discovered automatically when Optimizer starts. If a backend replaces files
while the Python process is alive, call
`discover_custom_optimizer_algorithms(force=True)` before listing/selecting
algorithms again.

Bad custom files are skipped and reported through
`describe_optimizer_algorithms()["errors"]` and
`list_optimizer_algorithm_errors()`. A broken upload must not prevent built-in
`focus_bo` or other valid custom algorithms from running.

Each `.py` file must define:

```python
ALGORITHM_ID = "my_algorithm"


class Algorithm:
    def run(self, *, runtime, config, resolved):
        params = runtime.algorithm_params
        for _ in range(config.user.n_samples):
            x = runtime.sample_uniform(1)[0]
            runtime.evaluate(x, source_mode="my_algorithm", segment="custom")
            if runtime.should_stop:
                break
        return runtime.build_result(algorithm_id=ALGORITHM_ID)
```

Select it with:

```json
{
  "optimizer": {
    "n_samples": 100,
    "goal": 1.2,
    "system": {
      "algorithm_id": "my_algorithm",
      "algorithm_params": {
        "example_param": 1.0
      }
    }
  }
}
```

Contract:

- The algorithm proposes candidate points only.
- Do not call `CAE_model` or task output writers directly.
- Always evaluate candidates through `runtime.evaluate(x)` or
  `runtime.evaluate_batch(X)`.
- `x` may be a vector in `runtime.feature_names` order or a
  `{feature: value}` dict.
- Keep candidates inside `runtime.lb` / `runtime.ub` or `runtime.bounds`.
- For pre-inequality constraints, use `runtime.filter_pre_inequality_candidates(X)`
  or `runtime.is_pre_inequality_feasible(x)` before calling `runtime.evaluate(x)`.
- Pre-equality constraints are handled by the runtime as an effective-objective
  penalty. `RuntimeEvaluation.objective` and `runtime.best_objective` are
  search/internal values; saved `objective`, `objective_raw`, `best_objective`,
  and `best_objective_raw` stay raw CAE objective values.
- Check `runtime.should_stop` after each evaluation.
- Return `runtime.build_result(algorithm_id=ALGORITHM_ID)`.
- Read custom knobs from `runtime.algorithm_params` or
  `config.system.algorithm_params`.
- Do not add FocusBO-specific fields to the common runtime contract. Algorithm
  parameters belong under `optimizer.system.algorithm_params`.

Runtime owns:

- CAE objective evaluation
- pre-constraint evaluation
- archive/history updates
- best point tracking
- `goal` / early stop
- standard Optimizer outputs (`opt_results.csv`, `best_point.json`,
  `metadata.json`)
- algorithm-neutral meta outputs (`optimizer_algorithm.json`,
  `optimizer_system_config.json`)

Custom/runtime algorithms do not emit FocusBO-only artifacts:

- `focus_regions.json`
- `focus_bounds.json`
- Focus2/Focus3 debug plots

Useful Runtime API:

- `runtime.feature_names`
- `runtime.n_dim`
- `runtime.bounds`
- `runtime.lb`, `runtime.ub`
- `runtime.rng`
- `runtime.algorithm_params`
- `runtime.max_evaluations`
- `runtime.remaining_budget`
- `runtime.sample_uniform(n)`
- `runtime.pre_inequality_active`
- `runtime.pre_equality_penalty_active`
- `runtime.is_pre_inequality_feasible(x)`
- `runtime.check_pre_inequality(x)`
- `runtime.filter_pre_inequality_candidates(X)`
- `runtime.evaluate(x, source_mode="...", segment="custom")`
- `runtime.evaluate_batch(X, source_mode="...", segment="custom")`
- `runtime.archive_df`
- `runtime.history_df`
- `runtime.best_point`
- `runtime.best_objective`
- `runtime.goal_state`
- `runtime.should_stop`

`evaluate_batch(X)` evaluates rows sequentially and stops early if the common
goal/budget stop condition is reached.
