import pandas as pd
import glob
import json
import os

REAL_FEATURES = {'x1', 'x2'}
DUMMY_FEATURES = {'d1', 'd2', 'd3'}

def classify_run(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, None, str(e)

    selected = df[df['selected'] == True]['feature'].tolist()
    selected_set = set(selected)

    has_dummy = bool(selected_set & DUMMY_FEATURES)
    missing_real = bool(REAL_FEATURES - selected_set)

    if not has_dummy and not missing_real:
        cat = 'SUCCESS'
    elif has_dummy and missing_real:
        cat = 'TYPE_AB'
    elif has_dummy:
        cat = 'TYPE_A'
    else:
        cat = 'TYPE_B'

    return cat, df, None

def simulate_gap_logic(df, n_samples=50):
    selected_df = df[df['selected'] == True].sort_values('final_score_adj', ascending=False).copy()
    selected_features = selected_df['feature'].tolist()
    scores = selected_df['final_score_adj'].tolist()
    global_scores = dict(zip(df['feature'], df['global_score']))

    if len(selected_features) < 2:
        return {
            'current_removes_dummy': False,
            'proposed_removes_dummy': False,
            'current_removes_real': False,
            'proposed_removes_real': False,
            'current_removed': set(),
            'proposed_removed': set(),
            'dummy_details': [],
            'gaps': [],
            'selected_features': selected_features,
            'scores': scores
        }

    gap_threshold = 0.08 if n_samples < 55 else 0.12
    global_score_threshold = 0.79

    # Compute gaps between adjacent selected features (sorted desc by final_score_adj)
    gaps = []
    for i in range(len(scores) - 1):
        gap_size = scores[i] - scores[i+1]
        gaps.append({
            'position': i,
            'above_feat': selected_features[i],
            'below_feat': selected_features[i+1],
            'gap_size': gap_size,
            'above_score': scores[i],
            'below_score': scores[i+1],
            'n_above': i + 1
        })

    # --- CURRENT LOGIC: single best gap ---
    current_removed = set()
    if gaps:
        best_gap = max(gaps, key=lambda g: g['gap_size'])
        if best_gap['gap_size'] >= gap_threshold and best_gap['n_above'] >= 2:
            for j in range(best_gap['position'] + 1, len(selected_features)):
                feat = selected_features[j]
                if global_scores.get(feat, 1.0) < global_score_threshold:
                    current_removed.add(feat)

    # --- PROPOSED LOGIC: multi-gap fallback ---
    proposed_removed = set()
    if gaps:
        sorted_gaps = sorted(gaps, key=lambda g: g['gap_size'], reverse=True)
        for gap in sorted_gaps:
            if gap['gap_size'] >= gap_threshold and gap['n_above'] >= 2:
                for j in range(gap['position'] + 1, len(selected_features)):
                    feat = selected_features[j]
                    if global_scores.get(feat, 1.0) < global_score_threshold:
                        proposed_removed.add(feat)
                break

    dummy_details = []
    fsa_map = dict(zip(selected_df['feature'], selected_df['final_score_adj']))
    for feat in selected_features:
        if feat in DUMMY_FEATURES:
            dummy_details.append({
                'feature': feat,
                'global_score': global_scores.get(feat, None),
                'final_score_adj': fsa_map.get(feat, None),
                'removed_by_current': feat in current_removed,
                'removed_by_proposed': feat in proposed_removed
            })

    return {
        'current_removes_dummy': bool(current_removed & DUMMY_FEATURES),
        'proposed_removes_dummy': bool(proposed_removed & DUMMY_FEATURES),
        'current_removes_real': bool(current_removed & REAL_FEATURES),
        'proposed_removes_real': bool(proposed_removed & REAL_FEATURES),
        'current_removed': current_removed,
        'proposed_removed': proposed_removed,
        'dummy_details': dummy_details,
        'gaps': gaps,
        'selected_features': selected_features,
        'scores': scores
    }

# ============================================================
# SCAN ALL RUNS
# ============================================================
results = {'current': [], 'past': []}

current_paths = sorted(glob.glob('C:/python/project/result/run_six_hump_camel_*/Modeler/artifacts/public/selected_features.csv'))
for p in current_paths:
    pn = p.replace('\\', '/')
    run_name = pn.split('/result/')[1].split('/Modeler')[0]
    cat, df, err = classify_run(p)
    if cat is None:
        continue
    results['current'].append({'run': run_name, 'category': cat, 'df': df, 'path': p})

past_paths = sorted(glob.glob('C:/python/project/result/past/additional=True/run_six_hump_camel_*/Modeler/artifacts/public/selected_features.csv'))
for p in past_paths:
    pn = p.replace('\\', '/')
    run_name = pn.split('additional=True/')[1].split('/Modeler')[0]
    cat, df, err = classify_run(p)
    if cat is None:
        continue
    results['past'].append({'run': run_name, 'category': cat, 'df': df, 'path': p})

# ============================================================
# SUMMARY REPORT
# ============================================================
print("=" * 80)
print("FEATURE SELECTION VERIFICATION REPORT")
print("=" * 80)

for label in ['current', 'past']:
    runs = results[label]
    total = len(runs)
    cats = {'SUCCESS': 0, 'TYPE_A': 0, 'TYPE_B': 0, 'TYPE_AB': 0}
    for r in runs:
        cats[r['category']] += 1

    success_pct = (cats['SUCCESS'] / total * 100) if total > 0 else 0

    print()
    print("-" * 60)
    print("  {} DIRECTORY  ({} runs)".format(label.upper(), total))
    print("-" * 60)
    print("  SUCCESS  (exactly x1,x2):  {:>4}  ({:5.1f}%)".format(cats['SUCCESS'], cats['SUCCESS']/total*100))
    print("  TYPE_A   (dummy included):  {:>4}  ({:5.1f}%)".format(cats['TYPE_A'], cats['TYPE_A']/total*100))
    print("  TYPE_B   (real missing):    {:>4}  ({:5.1f}%)".format(cats['TYPE_B'], cats['TYPE_B']/total*100))
    print("  TYPE_AB  (both problems):   {:>4}  ({:5.1f}%)".format(cats['TYPE_AB'], cats['TYPE_AB']/total*100))
    print("  fi_real_only_success_pct:   {:5.1f}%".format(success_pct))

# ============================================================
# TYPE_A FAILURE ANALYSIS
# ============================================================
print()
print("=" * 80)
print("TYPE_A FAILURE ANALYSIS - MULTI-GAP FALLBACK SIMULATION")
print("=" * 80)

type_a_results = []
for label in ['current', 'past']:
    runs = results[label]
    type_a_runs = [r for r in runs if r['category'] == 'TYPE_A']

    if not type_a_runs:
        print()
        print("  [{}] No TYPE_A failures to analyze.".format(label.upper()))
        continue

    print()
    print("-" * 60)
    print("  [{}] TYPE_A failures: {}".format(label.upper(), len(type_a_runs)))
    print("-" * 60)

    current_fix_count = 0
    proposed_fix_count = 0

    for r in type_a_runs:
        sim = simulate_gap_logic(r['df'], n_samples=50)

        if sim['current_removes_dummy']:
            current_fix_count += 1
        if sim['proposed_removes_dummy']:
            proposed_fix_count += 1

        type_a_results.append({
            'label': label,
            'run': r['run'],
            'sim': sim
        })

        gap_strs = []
        for g in sim['gaps']:
            gap_strs.append("{:.4f} (after {})".format(g['gap_size'], g['above_feat']))

        print()
        print("  Run: {}".format(r['run']))
        print("    Selected: {}".format(sim['selected_features']))
        print("    Scores:   {}".format(["%.4f" % s for s in sim['scores']]))
        print("    Gaps:     {}".format(gap_strs))
        for dd in sim['dummy_details']:
            print("    Dummy '{}': global_score={:.4f}, final_score_adj={:.4f}".format(
                dd['feature'], dd['global_score'], dd['final_score_adj']))
            print("      Current logic removes: {}".format(dd['removed_by_current']))
            print("      Proposed logic removes: {}".format(dd['removed_by_proposed']))

    print()
    print("  SUMMARY [{}]:".format(label.upper()))
    print("    Current gap logic would fix:  {}/{} TYPE_A failures".format(current_fix_count, len(type_a_runs)))
    print("    Proposed gap logic would fix:  {}/{} TYPE_A failures".format(proposed_fix_count, len(type_a_runs)))

# ============================================================
# FALSE POSITIVE CHECK on SUCCESS runs
# ============================================================
print()
print("=" * 80)
print("FALSE POSITIVE CHECK - Would proposed logic damage SUCCESS runs?")
print("=" * 80)

for label in ['current', 'past']:
    runs = results[label]
    success_runs = [r for r in runs if r['category'] == 'SUCCESS']

    false_positives = 0
    fp_details = []
    for r in success_runs:
        sim = simulate_gap_logic(r['df'], n_samples=50)
        if sim['proposed_removes_real']:
            false_positives += 1
            fp_details.append((r['run'], sim))

    print()
    print("  [{}] {} SUCCESS runs checked".format(label.upper(), len(success_runs)))
    print("    False positives (proposed removes x1/x2): {}".format(false_positives))
    if fp_details:
        for run_name, sim in fp_details:
            print("      PROBLEM: {}".format(run_name))
            print("        Selected: {}".format(sim['selected_features']))
            print("        Scores: {}".format(sim['scores']))
            removed_reals = sim['proposed_removed'] & REAL_FEATURES
            print("        Would remove: {}".format(removed_reals))

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
