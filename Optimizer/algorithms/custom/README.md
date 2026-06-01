# Custom Optimizer Algorithms

Drop user-provided optimizer algorithm files in this directory. Files are
discovered automatically when Optimizer starts. If a backend replaces files
while the Python process is alive, call
`discover_custom_optimizer_algorithms(force=True)` before listing/selecting
algorithms again.

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
- Check `runtime.should_stop` after each evaluation.
- Return `runtime.build_result(algorithm_id=ALGORITHM_ID)`.
- Read custom knobs from `runtime.algorithm_params` or
  `config.system.algorithm_params`.

Runtime owns:

- CAE objective evaluation
- pre-constraint evaluation
- archive/history updates
- best point tracking
- `goal` / early stop
- standard Optimizer outputs (`opt_results.csv`, `best_point.json`,
  `metadata.json`)

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
