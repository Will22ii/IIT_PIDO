# Explorer DSE Analysis Playbook

## Purpose

This document defines the working plan for Explorer = DSE analysis and improvement.

The immediate goal is to recover and exceed the historical 110-run score where Explorer DSE reached 95+.
The broader goal is to generalize beyond the 110-run benchmark so the same Explorer logic can keep 95+ even when the experiment count is larger, such as 1100 runs or more.

The target metric is Explorer `joint_pass`, not Optimizer performance.

## Primary Target

- Phase 1 target: 110-run benchmark `joint_pass_pct_macro >= 95`.
- Phase 2 target: larger-run benchmark, including 1100+ runs, also `joint_pass_pct_macro >= 95`.
- Hard volume target: selected bounds `volume_ratio <= 0.25`.
- The improvement target is Explorer/DSE only.

## Scoring Definition

Explorer pass must be measured with the following rule:

```text
joint_pass =
  modeler_all_real_included
  AND survivor_optimum_included
  AND volume_ratio <= 0.25
```

Definitions:

- `modeler_all_real_included`: primary selection found every real variable. Dummy variables may also be selected.
- `modeler_all_real_only`: primary selection selected all real variables and no dummy variables. This is a feature-selection quality score, not the Explorer bounds score.
- `survivor_optimum_included`: at least one known optimum is inside selected bounds.
- `volume_ratio`: selected bounds volume divided by the real-variable design-space volume.
- `volume_cap_pass`: `volume_ratio <= 0.25`.

Important distinction:

- If real variables are missing, Explorer cannot be credited because the real optimum cannot be represented correctly.
- If all real variables are included and dummy variables are also included, Explorer can still pass if the real-variable known optimum is included and the selected bounds volume is within 25%.
- Dummy-free selection should improve `modeler_all_real_only`, but dummy presence alone must not fail Explorer `joint_pass`.

## Non-Negotiable Constraints

- Known optimum is for evaluation only. Do not use known optimum coordinates in Explorer runtime logic, routing, or branch conditions.
- Do not modify benchmark functions under `CAE_model/`.
- Do not add benchmark-name-based branches.
- All new rules must be state/feature based and generalizable.
- Modeler/FS output is an input to Explorer; do not retroactively alter Modeler output to improve Explorer scores.
- Do not use post-run labels such as `joint_pass`, `fail_type`, or optimum inclusion as live routing inputs.
- Improvement work should target Explorer/DSE logic. Pipeline changes are allowed only for Explorer scoring, reporting, and analysis support.

## Current Context

Historical evidence says Explorer previously exceeded 95 on the 110-run setup.
Current runs do not exceed 95.

Runs may come from different execution environments. Linux and Windows results can differ slightly because of OS/runtime numerical differences, including math-library behavior. Every new run batch must record:

- execution environment: Linux or Windows
- git commit or commit range
- whether the run is before or after Explorer logic changes such as feasibility-aware floor expansion
- batch size and problem mix
- whether stats were copied from another machine

Do not attribute a score drop to a code change until environment and commit are separated as variables.

AION-only integration switches are managed in `pipeline/aion_system_config.py`.
`run_AION.py` always builds `PipelineConfig(aion_mode=True)`.
`run_pipelines.py` should only toggle `BATCH_AION_MODE`; it should not duplicate the individual AION sub-options.

Current AION sub-option:

- `enable_doe_router_signals`: when true, Explorer may read DOE metadata diagnostics from the current `run_context` and pass them to the router. This is off for standalone/direct pipeline mode by default.

The first analysis task is therefore regression analysis:

1. Find the historical high-score state.
2. Compare it with the current Explorer code and config.
3. Separate harmless refactors from actual logic changes.
4. Identify which Explorer logic changes likely affected score.
5. Restore equivalent behavior when the old logic was better.
6. If logic is already equivalent, search for new generalizable improvements.

Recent git comparison indicated that the most direct Explorer performance-affecting change is feasibility-aware floor expansion. That change must be analyzed against historical results rather than assumed good or bad.

## Data Sources

Use current and historical data in this priority order.

Current and recent run data:

- `result/explorer_strategy_stats/`
- `result/full_pipeline_runs/`
- `result/run_*/Explorer/artifacts/meta/analysis_<strategy>.json`
- `result/run_*/Explorer/artifacts/public/<strategy>/selected_bounds.json`

Known strong historical data:

- `result/best/dse/`
- `result/past/`
- `linux_dse` if available in the local workspace or imported result set
- `explorer_strategy` if available in the local workspace or imported result set

Existing policy reference:

- `.claude/skills/dse/SKILL.md`
- `docs/tasks/explorer.md`

Path handling rule:

- Do not trust OS-specific absolute prefixes from old stats files.
- Use run directory basename, such as `run_<problem>_<timestamp>_<hash>`, as the run key.
- Map old `run_root` references back to local `result/<run_key>/...` when possible.

## Valid Dataset Selection

For stats-based analysis:

- Use matching timestamp pairs only:
  - `explorer_strategy_try_stats_<timestamp>.csv`
  - `explorer_strategy_problem_summary_<timestamp>.csv`
- Both files must exist and be non-empty.
- If latest stats are partial or empty, use the latest complete non-empty pair.
- Do not mix `try_stats` and `problem_summary` from different timestamps.

For run-output analysis:

- A complete Explorer run should include all expected strategies for that experiment.
- Partial runs are allowed for pattern observation only.
- Partial runs must not be used as denominator for official score.
- If analysis metadata is missing, exclude that run from metadata-driven diagnosis.

## Analysis Workflow

### Step 1. Establish Score Baseline

For the current run set and the historical best run set:

- Compute benchmark-level `joint_pass_pct`.
- Compute macro average across benchmarks.
- Compute micro average across all tries.
- Report:
  - `modeler_all_real_included_pct`
  - `modeler_all_real_only_pct`
  - `survivor_optimum_included_pct`
  - `volume_cap_pass_pct`
  - `joint_pass_pct`
  - `volume_ratio_pct_mean`

Use the corrected Explorer scoring rule from this document.

### Step 2. Split Failure Modes

Classify every failed try:

- `feature_miss_fail`: real variables not all included.
- `over_shrink_fail`: all real variables included, volume pass, optimum missed.
- `over_wide_fail`: all real variables included, optimum included, volume fail.
- `both_bounds_fail`: all real variables included, optimum missed and volume fail.

Interpretation:

- `feature_miss_fail` is a Modeler/FS input issue. Track it, but do not count it as an Explorer bounds logic failure.
- `over_shrink_fail` means Explorer bounds are too narrow or centered incorrectly.
- `over_wide_fail` means Explorer found the right region but failed the 25% volume cap.
- `both_bounds_fail` usually means region selection and volume policy are both unstable.

### Step 3. Compare Historical Good Runs

Compare high-score historical data against current data:

- Problem by problem.
- Strategy by strategy.
- 110-run first.
- Larger-run data second.

For each benchmark and strategy, compare:

- selected bounds volume ratio
- optimum inclusion
- fail type distribution
- selected feature count
- selected real count
- selected dummy count
- Explorer `p_dim`, `usable_n`, `usable_n_over_p`
- `has_pre_constraints`, `has_post_constraints`
- `pred_*` analysis fields
- `dual_*` analysis fields
- selected bounds center and normalized widths if available

The first objective is to find why past logic scored 95+ and current logic does not.

### Step 4. Git-Based Logic Diff

Use git history to compare:

- The known high-score commit or date range.
- Current `HEAD`.
- Intermediate commits where score changed.

Focus on Explorer logic only:

- `Explorer/config.py`
- `Explorer/executor/explorer_orchestrator.py`
- `Explorer/executor/math_workflow.py`
- `Explorer/executor/strategy_workflow.py`
- `Explorer/executor/routing.py`
- `Explorer/strategy_presets.py`
- Pipeline scoring code in `pipeline/run_pipelines.py`

Separate changes into:

- Refactor only: function split, output formatting, metadata movement.
- Scoring/reporting only: CSV and summary definitions.
- Runtime logic: candidate generation, clustering, acquisition, expansion, shrinking, cap policy, routing, constraints.
- Config/default changes: thresholds, strategy presets, debug, sampling counts.

Only runtime logic and config/default changes should be treated as possible Explorer performance causes.

## 110-Run First, Larger-Run Second

The 110-run benchmark is the first gate.

Required 110-run analysis:

- Determine whether the current score drop comes from only a few benchmark/seed failures or a broad shift.
- Check whether failures were present in historical 110-run data.
- If the old behavior was better, identify the exact logic or config responsible.
- Restore or emulate the old behavior only through general state-based logic.

After 110-run passes 95:

- Test larger run counts.
- Check if the 110-run fix overfits small sample behavior.
- Preserve 95+ target under larger run counts by adding generalized stability rules, not benchmark-specific exceptions.

## Improvement Rules

Allowed Explorer improvements:

- State-based bounds expansion.
- State-based bounds shrink/cap policy.
- Robust center selection for selected bounds.
- Constraint-aware side policy if it uses only runtime-observable constraint feasibility statistics.
- Strategy preset adjustment if it is not benchmark-name based.
- Additional analysis metadata that helps post-run diagnosis.

Disallowed improvements:

- Directly using known optimum coordinates.
- Branching on benchmark names.
- Modifying CAE benchmark functions.
- Changing Modeler output after the fact.
- Using post-run labels as live inputs.

## Feasibility-Aware Floor Expansion Review

Current logic includes feasibility-aware floor expansion.

Analysis questions:

- Which benchmarks activate it?
- Which dimensions are overridden?
- Does it reduce `over_shrink_fail`?
- Does it increase `over_wide_fail`?
- Is the effect stable across 110-run and larger-run data?
- Did historical high-score logic have an equivalent behavior?

If it helps only one pattern but hurts another, refine the activation gate using observable state:

- `has_pre_constraints`
- `p_dim`
- side feasibility rate gap
- selected bounds pre-cap volume
- boundary touch count
- usable data ratio

Do not gate by benchmark name.

## Expected Output For Each Analysis Cycle

Every DSE analysis cycle should produce:

1. Dataset integrity report.
2. Current score table.
3. Historical best score table.
4. Current vs historical delta.
5. Failure mode breakdown.
6. Git logic diff summary.
7. Candidate root causes.
8. Proposed Explorer-only changes.
9. Risk assessment for each change.
10. Decision: apply, reject, or collect more data.

Use this final review table:

```text
change | type | expected gain | affected problems | risk | accepted
```

## Success Criteria

The work is successful when:

- Corrected Explorer `joint_pass_pct_macro >= 95` on the 110-run benchmark.
- `volume_ratio <= 0.25` is preserved for passing cases.
- Larger-run tests do not reveal obvious overfitting.
- Any restored historical behavior is implemented as general Explorer logic.
- `modeler_all_real_only` remains a separate FS quality metric.
- Explorer pass is not penalized by dummy variables when all real variables are present and bounds are correct.
