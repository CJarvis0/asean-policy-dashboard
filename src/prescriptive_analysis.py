"""
=============================================================================
PRESCRIPTIVE ANALYSIS: RECOMMENDATION ENGINE (Story 5) & WHAT-IF SIMULATION (Story 6)
Dataset: Panel Dataset with DTM/ETM Stages (360 obs, 15 countries, 2000-2023)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 0. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
df = pd.read_csv('C:/Users/Emma/Desktop/CIS5450/Final_Panel_Dataset_with_DTM_ETM.csv')

# Rename columns for convenience
df.columns = [c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct') for c in df.columns]
# Fill missing trade_percent_gdp with country-level median
df['trade_percent_gdp'] = df.groupby('country')['trade_percent_gdp'].transform(lambda x: x.fillna(x.median()))

print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"  Rows: {len(df)} | Countries: {df['country'].nunique()} | Years: {df['year'].min()}–{df['year'].max()}")
print(f"  DTM stages: {sorted(df['dtm_stage'].unique())}  |  ETM stages: {sorted(df['etm_stage'].unique())}")
print(f"  Income groups: {list(df['income_group'].unique())}\n")


# ─────────────────────────────────────────────────────────────────────────────
# STORY 5 — RECOMMENDATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
print("="*70)
print("STORY 5: RECOMMENDATION ENGINE — SCORING FRAMEWORK")
print("="*70)

# ── 5.1  Define indicator weights per priority dimension ──────────────────────
DIMENSION_WEIGHTS = {
    'inequality':      {'gini_index': -0.40, 'bottom_10pct_income_share_pct': 0.35,
                        'top_10pct_income_share_pct': -0.25},
    'health':          {'life_expectancy': 0.40, 'infant_mortality': -0.35,
                        'infectious_disease_rate': -0.15, 'noncommunicable_disease_rate': -0.10},
    'demographics':    {'crude_birth_rate': -0.20, 'crude_death_rate': -0.30,
                        'natural_increase_rate': 0.25, 'pop_growth': 0.25},
    'economic':        {'gdp_per_capita': 0.45, 'average_income_usd': 0.35,
                        'trade_percent_gdp': 0.20},
}

# ── 5.2  Min-max normalize each indicator ────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(['year','dtm_stage','etm_stage'])
df_norm = df.copy()
for col in numeric_cols:
    mn, mx = df[col].min(), df[col].max()
    df_norm[col] = (df[col] - mn) / (mx - mn + 1e-9)

# ── 5.3  Compute composite dimension scores ──────────────────────────────────
for dim, indicators in DIMENSION_WEIGHTS.items():
    score = sum(df_norm[col] * w for col, w in indicators.items() if col in df_norm.columns)
    # Normalise to 0-100
    df[f'score_{dim}'] = (score - score.min()) / (score.max() - score.min() + 1e-9) * 100

# Overall composite score (equal weight across dimensions)
score_cols = [f'score_{d}' for d in DIMENSION_WEIGHTS]
df['composite_score'] = df[score_cols].mean(axis=1)

# ── 5.4  Classify countries into Recommendation Tiers ────────────────────────
def assign_tier(score):
    if score >= 65:  return 'Tier 1 – Sustain & Lead'
    elif score >= 45: return 'Tier 2 – Optimise'
    elif score >= 25: return 'Tier 3 – Transition'
    else:            return 'Tier 4 – Critical Intervention'

df['recommendation_tier'] = df['composite_score'].apply(assign_tier)

# Latest year snapshot for recommendations
latest = df[df['year'] == df['year'].max()].copy()
latest = latest.sort_values('composite_score', ascending=False)

print("\n── 5.4  Country Recommendation Tiers (Latest Year: 2023) ──")
print(f"{'Country':<8} {'DTM':>4} {'ETM':>4} {'Composite':>10} {'Tier'}")
print("-"*65)
for _, row in latest.iterrows():
    print(f"  {row['country']:<8} {row['dtm_stage']:>4}   {row['etm_stage']:>4}   "
          f"{row['composite_score']:>8.1f}   {row['recommendation_tier']}")


# ── 5.5  Per-country rule-based recommendations ──────────────────────────────
RECOMMENDATIONS = {
    # (dtm_stage, etm_stage) → list of policy recommendations
    (2, 1): ["Accelerate public health investment to reduce infant mortality and infectious disease.",
             "Expand access to primary education to drive demographic transition.",
             "Introduce conditional cash-transfer programmes to address extreme inequality."],
    (2, 2): ["Strengthen maternal health services to reduce birth rates sustainably.",
             "Invest in vocational training to build a productive workforce.",
             "Diversify trade partnerships to reduce commodity dependence."],
    (3, 2): ["Scale preventive healthcare infrastructure to shift disease burden.",
             "Introduce progressive taxation to narrow income inequality (Gini >0.5).",
             "Target FDI in manufacturing to accelerate GDP per-capita growth."],
    (3, 5): ["Redirect fiscal space from population growth management to ageing preparedness.",
             "Strengthen non-communicable disease programmes (NCDs are the primary burden).",
             "Invest in innovation and R&D for sustained economic competitiveness."],
    (4, 5): ["Prioritise pension and elderly-care systems given advanced demographic stage.",
             "Monitor and manage inequality as economic maturity can widen income gaps.",
             "Lead on climate-aligned trade and sustainable development policies."],
}

def get_recommendations(dtm, etm):
    key = (dtm, etm)
    return RECOMMENDATIONS.get(key, ["Monitor key indicators and benchmark against peer countries.",
                                      "Conduct detailed structural diagnostics.",
                                      "Engage international partners for tailored advisory support."])

print("\n── 5.5  Sample Detailed Recommendations by Stage ──")
for country in ['NGA', 'IND', 'ZAF', 'MEX', 'USA']:
    row = latest[latest['country'] == country].iloc[0]
    recs = get_recommendations(row['dtm_stage'], row['etm_stage'])
    print(f"\n  [{country}] DTM={row['dtm_stage']} | ETM={row['etm_stage']} | "
          f"Score={row['composite_score']:.1f} | {row['recommendation_tier']}")
    for i, r in enumerate(recs, 1):
        print(f"    {i}. {r}")


# ─────────────────────────────────────────────────────────────────────────────
# STORY 6 — WHAT-IF SIMULATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n\n" + "="*70)
print("STORY 6: WHAT-IF SIMULATION — SCENARIO ANALYSIS FRAMEWORK")
print("="*70)

# ── 6.1  Define scenario levers (% change applied to raw indicator value) ─────
SCENARIOS = {
    'Baseline (No Change)': {},
    'Scenario A – Health Push':
        {'life_expectancy': +5.0, 'infant_mortality': -20.0,
         'infectious_disease_rate': -15.0},
    'Scenario B – Inequality Reduction':
        {'gini_index': -0.08, 'bottom_10pct_income_share_pct': +3.0,
         'average_income_usd': +15.0},
    'Scenario C – Economic Acceleration':
        {'gdp_per_capita': +25.0, 'trade_percent_gdp': +10.0,
         'average_income_usd': +20.0},
    'Scenario D – Comprehensive Reform':
        {'life_expectancy': +3.0, 'infant_mortality': -15.0,
         'gini_index': -0.05, 'gdp_per_capita': +15.0,
         'average_income_usd': +10.0, 'infectious_disease_rate': -10.0},
}

def apply_scenario(row, lever_dict):
    """Apply scenario delta (absolute units) to a country row and recompute score."""
    r = row.copy()
    for col, delta in lever_dict.items():
        if col in r.index:
            r[col] = r[col] + delta
    return r

def recompute_score(row):
    """Recompute composite score for a modified row using same weighting as original."""
    dim_scores = []
    for dim, indicators in DIMENSION_WEIGHTS.items():
        s = 0
        for col, w in indicators.items():
            if col in df.columns:
                mn, mx = df[col].min(), df[col].max()
                norm_val = (row[col] - mn) / (mx - mn + 1e-9)
                norm_val = np.clip(norm_val, 0, 1)
                s += norm_val * w
        dim_scores.append(s)
    # Average across dimensions; scale to match composite_score range
    composite_raw = np.mean(dim_scores)
    # Build baseline raw scores to scale properly
    return composite_raw  # Return raw; rescale after building full matrix

# ── 6.2  Run simulations for each country (latest year) ──────────────────────
# First pass: collect raw scores, then rescale to 0-100
raw_results = {}
for _, row in latest.iterrows():
    country = row['country']
    raw_results[country] = {}
    for sname, levers in SCENARIOS.items():
        modified_row = apply_scenario(row, levers)
        raw_results[country][sname] = recompute_score(modified_row)

# Rescale: use min/max across all raw values
all_raw = [v for d in raw_results.values() for v in d.values()]
r_min, r_max = min(all_raw), max(all_raw)

scenario_results = {}
for country, scores in raw_results.items():
    scenario_results[country] = {
        s: np.clip((v - r_min) / (r_max - r_min + 1e-9) * 100, 0, 100)
        for s, v in scores.items()
    }

print("\n── 6.2  Scenario Score Projections by Country (2023 Baseline) ──")
print(f"\n{'Country':<8}", end='')
for sname in SCENARIOS:
    label = sname[:20].ljust(22)
    print(f"  {label}", end='')
print()
print("-"*130)

for country in scenario_results:
    row_str = f"  {country:<8}"
    for sname in SCENARIOS:
        sc = scenario_results[country][sname]
        row_str += f"  {sc:>6.1f}            "
    print(row_str)

# ── 6.3  Score uplift ranking ─────────────────────────────────────────────────
print("\n── 6.3  Score Uplift vs Baseline — Best Reform Scenario per Country ──")
print(f"\n  {'Country':<8} {'Best Scenario':<40} {'Baseline':>10} {'New Score':>10} {'Uplift':>8}")
print("  " + "-"*78)
for country, scores in scenario_results.items():
    baseline = scores['Baseline (No Change)']
    best_scenario = max(
        {k: v for k, v in scores.items() if k != 'Baseline (No Change)'},
        key=lambda k: scores[k]
    )
    best_score = scores[best_scenario]
    uplift = best_score - baseline
    print(f"  {country:<8} {best_scenario:<40} {baseline:>10.1f} {best_score:>10.1f} {uplift:>+8.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLORS = {
    'Tier 1 – Sustain & Lead':     '#2ecc71',
    'Tier 2 – Optimise':           '#3498db',
    'Tier 3 – Transition':         '#f39c12',
    'Tier 4 – Critical Intervention': '#e74c3c',
}
SCENARIO_COLORS = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71', '#9b59b6']

fig = plt.figure(figsize=(22, 28))
fig.patch.set_facecolor('#0f1117')
gs = GridSpec(4, 2, figure=fig, hspace=0.42, wspace=0.30)

# ── Plot 1: Composite Score Bar Chart (2023) ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#1a1d2e')
colors = [TIER_COLORS[t] for t in latest['recommendation_tier']]
bars = ax1.barh(latest['country'], latest['composite_score'], color=colors, edgecolor='none', height=0.6)
for bar, score in zip(bars, latest['composite_score']):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{score:.1f}', va='center', ha='left', color='white', fontsize=9)
ax1.set_xlim(0, 115)
ax1.set_xlabel('Composite Score (0–100)', color='#cccccc', fontsize=10)
ax1.set_title('Story 5 — Composite Recommendation Score by Country (2023)',
              color='white', fontsize=13, fontweight='bold', pad=12)
ax1.tick_params(colors='#cccccc')
ax1.spines[:].set_visible(False)
# Legend
patches = [mpatches.Patch(color=c, label=t) for t, c in TIER_COLORS.items()]
ax1.legend(handles=patches, loc='lower right', framealpha=0.2,
           labelcolor='white', fontsize=8)

# ── Plot 2: Dimension Radar for selected countries ────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#1a1d2e')
dims = list(DIMENSION_WEIGHTS.keys())
focus_countries = ['NGA', 'IND', 'MEX', 'USA', 'DEU']
dim_palette = ['#e74c3c','#f39c12','#3498db','#2ecc71','#9b59b6']
x = np.arange(len(dims))
width = 0.15
for i, (c, color) in enumerate(zip(focus_countries, dim_palette)):
    row = latest[latest['country'] == c].iloc[0]
    vals = [row[f'score_{d}'] for d in dims]
    ax2.bar(x + i * width, vals, width, label=c, color=color, alpha=0.85)
ax2.set_xticks(x + width * 2)
ax2.set_xticklabels([d.capitalize() for d in dims], color='#cccccc', fontsize=9)
ax2.set_ylabel('Dimension Score (0–100)', color='#cccccc', fontsize=9)
ax2.set_title('Story 5 — Dimension Scores by Country (Selected)', color='white',
              fontsize=10, fontweight='bold')
ax2.legend(framealpha=0.2, labelcolor='white', fontsize=8)
ax2.tick_params(colors='#cccccc')
ax2.spines[:].set_visible(False)
ax2.set_ylim(0, 110)

# ── Plot 3: DTM/ETM Stage Heatmap ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#1a1d2e')
pivot = latest.set_index('country')[['dtm_stage','etm_stage','composite_score']]
scatter = ax3.scatter(pivot['dtm_stage'], pivot['etm_stage'],
                      c=pivot['composite_score'], cmap='RdYlGn', s=260,
                      vmin=0, vmax=100, zorder=3, edgecolors='white', linewidths=0.5)
for country, row in pivot.iterrows():
    ax3.annotate(country, (row['dtm_stage'], row['etm_stage']),
                 textcoords="offset points", xytext=(6, 4),
                 color='white', fontsize=7.5)
ax3.set_xlabel('DTM Stage', color='#cccccc', fontsize=9)
ax3.set_ylabel('ETM Stage', color='#cccccc', fontsize=9)
ax3.set_title('Story 5 — Composite Score by DTM/ETM Stage', color='white',
              fontsize=10, fontweight='bold')
plt.colorbar(scatter, ax=ax3, label='Composite Score').ax.yaxis.label.set_color('white')
ax3.tick_params(colors='#cccccc')
ax3.set_facecolor('#1a1d2e')
ax3.spines[:].set_visible(False)
ax3.grid(True, color='#333355', alpha=0.4)

# ── Plot 4: What-If Scenario Chart (NGA) ─────────────────────────────────────
for plot_i, (country_code, ax_pos) in enumerate(
    zip(['NGA', 'IND', 'MEX', 'USA'], [(2, 0), (2, 1), (3, 0), (3, 1)])):
    ax = fig.add_subplot(gs[ax_pos])
    ax.set_facecolor('#1a1d2e')
    snames = list(SCENARIOS.keys())
    sscores = [scenario_results[country_code][s] for s in snames]
    short_labels = ['Baseline', 'Health\nPush', 'Inequality\nRedn.', 'Economic\nAccel.', 'Compre-\nhensive']
    bar_colors = SCENARIO_COLORS
    bars = ax.bar(short_labels, sscores, color=bar_colors, edgecolor='none', width=0.6)
    for bar, sc in zip(bars, sscores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{sc:.1f}', ha='center', va='bottom', color='white', fontsize=8.5)
    baseline_val = scenario_results[country_code]['Baseline (No Change)']
    ax.axhline(baseline_val, color='#aaaaaa', linestyle='--', linewidth=1, alpha=0.6)
    max_score = max((s for s in sscores if not np.isnan(s)), default=1)
    ax.set_ylim(0, max_score * 1.18)
    ax.set_title(f'Story 6 — What-If Scenarios: {country_code}',
                 color='white', fontsize=10, fontweight='bold')
    ax.set_ylabel('Projected Composite Score', color='#cccccc', fontsize=8)
    ax.tick_params(colors='#cccccc', labelsize=8)
    ax.spines[:].set_visible(False)

fig.suptitle('Prescriptive Analysis: Recommendation Engine & What-If Simulation\nPanel Dataset — 15 Countries, 2000–2023',
             color='white', fontsize=15, fontweight='bold', y=0.99)

plt.savefig('C:/Users/Emma/Desktop/CIS5450/prescriptive_analysis.png',
            dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("\n\n✅  Visualisation saved → C:/Users/Emma/Desktop/CIS5450/prescriptive_analysis.png")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────
output_rows = []
for country, scores in scenario_results.items():
    base = scores['Baseline (No Change)']
    row_data = latest[latest['country'] == country].iloc[0]
    for sname, score in scores.items():
        output_rows.append({
            'country': country,
            'dtm_stage': row_data['dtm_stage'],
            'etm_stage': row_data['etm_stage'],
            'income_group': row_data['income_group'],
            'recommendation_tier': row_data['recommendation_tier'],
            'scenario': sname,
            'projected_score': round(score, 2),
            'uplift_vs_baseline': round(score - base, 2),
        })

results_df = pd.DataFrame(output_rows)
results_df.to_csv('C:/Users/Emma/Desktop/CIS5450/prescriptive_results.csv', index=False)
print("✅  Results table saved → C:/Users/Emma/Desktop/CIS5450/prescriptive_results.csv")

print("\n" + "="*70)
print("IMPLEMENTATION PLAN SUMMARY")
print("="*70)
print("""
Story 5 — Recommendation Engine
─────────────────────────────────
  Step 1  Feature engineering: 4 composite dimension scores
          (Inequality, Health, Demographics, Economic) from 13 raw indicators.
  Step 2  Min-max normalisation across the full panel (2000–2023).
  Step 3  Weighted aggregation → composite_score [0-100].
  Step 4  Tier classification: 4 tiers (Sustain → Critical Intervention).
  Step 5  Stage-based rule engine: 5 DTM/ETM stage-pair combinations
          → contextualised policy recommendations (3 per country).
  Step 6  Output: per-country recommendation card with tier + action list.

Story 6 — What-If Simulation
──────────────────────────────
  Step 1  Define 4 intervention scenarios (Health Push, Inequality Reduction,
          Economic Acceleration, Comprehensive Reform) as indicator delta vectors.
  Step 2  Apply deltas to latest-year country snapshot.
  Step 3  Recompute composite score using same normalisation framework.
  Step 4  Rank scenarios by score uplift per country.
  Step 5  Identify best reform pathway and magnitude of impact.
  Step 6  Output: scenario score matrix + uplift table + visualisations.
""")