import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.formula.api as smf

# Styling for publication-grade figures matching the Hindi analysis dashboard
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 1.0

PRED_ORDER = ["Equational/Nominal", "Adjectival", "Locative/Existential"]
SUB_ORDER = ["Film Dialogue", "Literary Dialogue", "Literary Narrative"]

def load_telugu_datasets(
    auto_csv="Telugu_copula_dataset_automatic.csv",
    manual_csv="Telugu_copula_dataset_manual.csv"
):
    """
    Loads Telugu automatic and manual gold datasets with dynamic path resolution:
    - Resolves paths relative to the script directory to prevent FileNotFoundError.
    - Sets Copula_Deleted_Clean = 1 when Copula_Status_NLP == 0 (zero copula).
    - Ensures uniform factor levels across registers and predicate categories.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    auto_path = os.path.join(script_dir, auto_csv) if not os.path.isabs(auto_csv) else auto_csv
    manual_path = os.path.join(script_dir, manual_csv) if not os.path.isabs(manual_csv) else manual_csv

    if not os.path.exists(auto_path) or not os.path.exists(manual_path):
        raise FileNotFoundError(f"Missing Telugu CSVs in '{script_dir}'. Expected '{auto_csv}' and '{manual_csv}'.")

    df_auto = pd.read_csv(auto_path, encoding='utf-8-sig')
    df_manual = pd.read_csv(manual_path, encoding='utf-8-sig')

    for df in [df_auto, df_manual]:
        df['Copula_Deleted_Clean'] = (df['Copula_Status_NLP'] == 0).astype(int)

    return df_auto, df_manual

def compute_contingency_stats(df_sub, group_col, target_col='Copula_Deleted_Clean'):
    """Computes Chi-Square (or Fisher exact for 2x2 tables), p-value, df, and Cramer's V."""
    ct = pd.crosstab(df_sub[group_col], df_sub[target_col]).reindex(columns=[0, 1], fill_value=0)
    chi2, p_val, dof, exp = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
    fisher_p = fisher_exact(ct.values)[1] if ct.shape == (2, 2) else None
    return chi2, p_val, dof, cramers_v, fisher_p, ct

def evaluate_hypotheses(df, label):
    """Prints rate tables and tests H1 (Register), H2 (Predicate Type), and H3 (Polarity)."""
    print("\n" + "=" * 85)
    print(f" {label.upper()} (N = {len(df)} Valid Non-Verbal Clauses)")
    print("=" * 85)

    # Sub-Register breakdown
    reg_summary = df.groupby(['Register', 'Sub_Register']).agg(
        Total=('Copula_Deleted_Clean', 'count'),
        Copula_Present=('Copula_Deleted_Clean', lambda x: (x == 0).sum()),
        Zero_Copula=('Copula_Deleted_Clean', 'sum'),
        Zero_Rate_Pct=('Copula_Deleted_Clean', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    print(reg_summary.to_string(index=False))

    # H1: Register Test (Film Dialogue vs Literary Narrative)
    h1_df = df[df['Sub_Register'].isin(['Film Dialogue', 'Literary Narrative'])]
    chi2_1, p1, dof1, v1, fp1, _ = compute_contingency_stats(h1_df, 'Sub_Register')
    print(f"\n[H1: Register Test (Film Dialogue vs. Literary Narrative)]")
    print(f"Chi2 = {chi2_1:.4f}, df = {dof1}, p = {p1:.4e}, Cramer's V = {v1:.4f}" + (f", Fisher exact p = {fp1:.4e}" if fp1 is not None else ""))

    # H2: Predicate Type Hierarchy
    print("\n[H2: Predicate Type Breakdown]")
    pred_summary = df.groupby('Predicate_Type_Final_NLP').agg(
        Total=('Copula_Deleted_Clean', 'count'),
        Copula_Present=('Copula_Deleted_Clean', lambda x: (x == 0).sum()),
        Zero_Copula=('Copula_Deleted_Clean', 'sum'),
        Zero_Rate_Pct=('Copula_Deleted_Clean', lambda x: round(x.mean() * 100, 2))
    ).reindex(PRED_ORDER).reset_index()
    print(pred_summary.to_string(index=False))

    chi2_2, p2, dof2, v2, _, _ = compute_contingency_stats(df, 'Predicate_Type_Final_NLP')
    print(f"Chi2 = {chi2_2:.4f}, df = {dof2}, p = {p2:.4e}, Cramer's V = {v2:.4f}")

    # H3: Polarity Effect
    print("\n[H3: Polarity Breakdown]")
    pol_summary = df.groupby('Polarity_NLP').agg(
        Total=('Copula_Deleted_Clean', 'count'),
        Copula_Present=('Copula_Deleted_Clean', lambda x: (x == 0).sum()),
        Zero_Copula=('Copula_Deleted_Clean', 'sum'),
        Zero_Rate_Pct=('Copula_Deleted_Clean', lambda x: round(x.mean() * 100, 2))
    ).reset_index().sort_values('Zero_Rate_Pct', ascending=False)
    print(pol_summary.to_string(index=False))

    chi2_3, p3, dof3, v3, fp3, _ = compute_contingency_stats(df, 'Polarity_NLP')
    print(f"Chi2 = {chi2_3:.4f}, df = {dof3}, p = {p3:.4e}, Cramer's V = {v3:.4f}" + (f", Fisher exact p = {fp3:.4e}" if fp3 is not None else ""))

    # Multivariate Binomial Logistic Regression on Envelope of Variation
    # In Dravidian Telugu:
    # 1. Negative polarity obligatorily recruits overt negators (కాదు / లేదు) -> 0% zero copula.
    # 2. Affirmative equational clauses categorically possess no overt present copula -> 100% zero copula.
    # Therefore, the variationist envelope is modeled on Affirmative Adjectival + Locative clauses.
    aff = df[(df['Polarity_NLP'] == 'Affirmative') & (df['Predicate_Type_Final_NLP'] != 'Equational/Nominal')].copy()
    aff['Spoken'] = (aff['Register'] == 'Spoken Subtitle').astype(int)

    print(f"\n--- Multivariate Logit on Envelope of Variation (N = {len(aff)}) ---")
    try:
        logit_m = smf.logit(
            "Copula_Deleted_Clean ~ Spoken + C(Predicate_Type_Final_NLP, Treatment(reference='Locative/Existential'))",
            data=aff
        ).fit(disp=False)
        print(logit_m.summary())

        or_table = pd.DataFrame({
            'Odds_Ratio': np.exp(logit_m.params),
            'CI_2.5%': np.exp(logit_m.conf_int()[0]),
            'CI_97.5%': np.exp(logit_m.conf_int()[1]),
            'p_value': logit_m.pvalues
        })
        print("\n--- Odds Ratios (OR = exp(Beta)) ---")
        print(or_table.round(4).to_string())
    except Exception as e:
        print(f"Model estimation note: {e}")

def run_full_telugu_pipeline():
    df_auto, df_manual = load_telugu_datasets()
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    # Stage 1: Automatic Pipeline Evaluation
    evaluate_hypotheses(df_auto, "Stage 1: Automatic Pipeline Corpus")

    # Stage 2: Manual Gold Set Evaluation
    evaluate_hypotheses(df_manual, "Stage 2: Manually Verified Gold Set")

    # Generate Publication-Grade Visualizations (4-Panel Dashboard)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    pct = lambda x: np.mean(x) * 100

    # Plot 1: Register Comparison Across Automatic Pipeline vs Gold Set (H1)
    comp_df = pd.concat([
        df_auto.assign(Pipeline_Stage='Automatic Pipeline (N=759)'),
        df_manual.assign(Pipeline_Stage='Manual Gold Set (N=116)')
    ])
    sns.barplot(data=comp_df, x='Sub_Register', y='Copula_Deleted_Clean', hue='Pipeline_Stage',
                order=SUB_ORDER, estimator=pct, errorbar=None, ax=axes[0, 0], palette=['#95a5a6', '#2ecc71'])
    axes[0, 0].set_title("A. Telugu: Zero-Copula Rate by Register & Corpus Stage (H1)", fontweight='bold')
    axes[0, 0].set_ylabel("Zero-Copula Rate (%)")
    axes[0, 0].set_xlabel("Sub-Register")
    axes[0, 0].set_ylim(0, 105)
    for c in axes[0, 0].containers:
        axes[0, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot 2: Predicate Type Hierarchy by Sub-Register (Gold Set, H2)
    sns.barplot(data=df_manual, x='Predicate_Type_Final_NLP', y='Copula_Deleted_Clean', hue='Sub_Register',
                order=PRED_ORDER, hue_order=SUB_ORDER, estimator=pct, errorbar=None, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title("B. Telugu: Predicate Hierarchy by Sub-Register (Gold Set, H2)", fontweight='bold')
    axes[0, 1].set_ylabel("Zero-Copula Rate (%)")
    axes[0, 1].set_xlabel("Predicate Category")
    axes[0, 1].set_ylim(0, 105)
    for c in axes[0, 1].containers:
        axes[0, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    # Plot 3: Source-Level Zero-Copula Rate Across All 6 Sources (N=759)
    src_summary = df_auto.groupby(['Source_Title', 'Register'])['Copula_Deleted_Clean'].mean().mul(100).reset_index().sort_values('Copula_Deleted_Clean', ascending=False)
    sns.barplot(data=src_summary, x='Source_Title', y='Copula_Deleted_Clean', hue='Register', dodge=False, ax=axes[1, 0], palette=['#3498db', '#e67e22'])
    axes[1, 0].set_title("C. Telugu: Zero-Copula Rate Across All 6 Sources (N=759)", fontweight='bold')
    axes[1, 0].set_ylabel("Zero-Copula Rate (%)")
    axes[1, 0].set_xlabel("Source Title")
    axes[1, 0].tick_params(axis='x', rotation=20)
    axes[1, 0].set_ylim(0, 105)
    for c in axes[1, 0].containers:
        axes[1, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot 4: Polarity x Sub-Register Interaction (Gold Set, H3)
    sns.barplot(data=df_manual, x='Polarity_NLP', y='Copula_Deleted_Clean', hue='Sub_Register',
                hue_order=SUB_ORDER, estimator=pct, errorbar=None, ax=axes[1, 1], palette='magma')
    axes[1, 1].set_title("D. Telugu: Polarity Categorical Block (Gold Set, H3)", fontweight='bold')
    axes[1, 1].set_ylabel("Zero-Copula Rate (%)")
    axes[1, 1].set_xlabel("Clausal Polarity")
    axes[1, 1].set_ylim(0, 105)
    for c in axes[1, 1].containers:
        axes[1, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    plt.tight_layout()
    chart_output = os.path.join(script_dir, "telugu_copula_analysis_dashboard.png")
    plt.savefig(chart_output, dpi=300)
    print(f"\n[SUCCESS] Generated '{chart_output}' successfully.")

if __name__ == "__main__":
    run_full_telugu_pipeline()