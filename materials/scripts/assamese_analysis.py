import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# Visualization styling matching the Hindi publication dashboard
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 1.0

def load_and_audit_assamese_corpus(formal_path="formal_assamese.csv", informal_path="informal_assamese.csv"):
    """
    Ingests and audits Assamese formal and informal datasets:
    1. Standardizes headers and removes empty metadata rows.
    2. Identifies Zero-Copula eligibility (dashes '——' indicate deletion is blocked).
    3. Normalizes fine-grained predicate sub-types into 3 core Universal categories:
       Equational/Nominal, Adjectival, and Locative/Existential.
    4. Filters out dynamic action verbs to isolate pure non-verbal copular clauses.
    5. Extracts clausal polarity (Affirmative vs. Negative).
    """
    # 1. Load Formal Prose Dataset
    df_formal = pd.read_csv(formal_path, skiprows=1, encoding='utf-8')
    df_formal = df_formal.dropna(subset=['Type']).copy()
    if 'Unnamed: 0' in df_formal.columns:
        df_formal = df_formal.drop(columns=['Unnamed: 0'])
    df_formal.columns = [c.strip() for c in df_formal.columns]
    df_formal['Register'] = 'Formal Prose'

    # 2. Load Informal Dialogue Dataset
    df_informal = pd.read_csv(informal_path, skiprows=2, encoding='utf-8')
    type_col = [c for c in df_informal.columns if 'Type' in c][0]
    df_informal = df_informal.dropna(subset=[type_col]).copy()
    df_informal.columns = [c.strip() for c in df_informal.columns]
    df_informal['Register'] = 'Informal Dialogue'

    # Combine
    combined = pd.concat([df_formal, df_informal], ignore_index=True)

    # 3. Determine Copula Deletion
    # A dashed entry ('——', '————', '——-') means deletion is grammatically blocked (Overt verb required = 0)
    def check_zero_copula(val):
        s = str(val).strip()
        if not s or s in ['——', '————', '——-', '—', '-']:
            return 0  # Overt verb required / deletion blocked
        return 1      # Zero Copula permitted / observed

    combined['Copula_Deleted'] = combined['Zero Copula Form'].apply(check_zero_copula)

    # 4. Standardize Predicate Categories to Curnow (1999) Universal Types
    def map_predicate(raw_type):
        t = str(raw_type).strip().lower()
        if 'verbal' in t and 'non copular' not in t:
            return 'Verbal Action'
        if any(k in t for k in ['nominal', 'defenitional', 'identify', 'role', 'kinship', 'proffesion', 'choice', 'classification']):
            return 'Equational/Nominal'
        if any(k in t for k in ['adjectival', 'adjevtival', 'quality', 'taste', 'weather', 'texture', 'behaviour', 'condition']):
            return 'Adjectival'
        if any(k in t for k in ['locative', 'existential', 'exiatential', 'posses', 'state']):
            return 'Locative/Existential'
        return 'Other'

    combined['Predicate_Type'] = combined['Type'].apply(map_predicate)

    # 5. Determine Clausal Polarity
    def detect_polarity(raw_type, overt_str):
        t, o = str(raw_type).strip().lower(), str(overt_str).strip()
        if 'negative' in t or any(m in o for m in ['নাই', 'নহয়', 'নহয়নে', 'নকৰে', 'নোৱাৰে', 'নাথাকে']):
            return 'Negative'
        return 'Affirmative'

    combined['Polarity'] = [detect_polarity(t, o) for t, o in zip(combined['Type'], combined['Overt Form'])]

    # 6. Exclude dynamic action verbal clauses (Methodological consistency with Hindi)
    clean_df = combined[~combined['Type'].str.contains('Verbal', case=False, na=False)].copy()
    clean_df = clean_df[clean_df['Predicate_Type'].isin(['Equational/Nominal', 'Locative/Existential', 'Adjectival'])].copy()

    return combined, clean_df

def compute_stats(df_sub, group_col, target_col='Copula_Deleted'):
    """Calculates Chi-Square, p-value, and Cramér's V effect size."""
    ct = pd.crosstab(df_sub[group_col], df_sub[target_col])
    chi2, p_val, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
    return chi2, p_val, dof, cramers_v, ct

def run_assamese_analysis():
    _, df = load_and_audit_assamese_corpus()
    total_n = len(df)
    
    # Save clean dataset
    df.to_csv("assamese_clean_nonverbal.csv", index=False, encoding='utf-8-sig')

    print("=" * 80)
    print(f" ASSAMESE CORPUS EMPIRICAL EVALUATION (N = {total_n})")
    print("=" * 80)

    # Hypothesis 1: Register Effect
    print("\n--- [H1] Register Breakdown ---")
    reg_summary = df.groupby('Register')['Copula_Deleted'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(reg_summary.to_string(index=False))

    chi2_r, p_r, dof_r, v_r, _ = compute_stats(df, 'Register')
    print(f"H1 Test: Chi2 = {chi2_r:.4f}, df = {dof_r}, p = {p_r:.4e}, Cramer's V = {v_r:.4f}")

    # Hypothesis 2: Predicate Hierarchy
    print("\n--- [H2] Predicate Hierarchy Breakdown ---")
    pred_summary = df.groupby('Predicate_Type')['Copula_Deleted'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pred_summary.to_string(index=False))

    chi2_p, p_p, dof_p, v_p, _ = compute_stats(df, 'Predicate_Type')
    print(f"H2 Test: Chi2 = {chi2_p:.4f}, df = {dof_p}, p = {p_p:.4e}, Cramer's V = {v_p:.4f}")

    # Hypothesis 3: Polarity Effect
    print("\n--- [H3] Polarity Breakdown ---")
    pol_summary = df.groupby('Polarity')['Copula_Deleted'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pol_summary.to_string(index=False))

    chi2_pol, p_pol, dof_pol, v_pol, _ = compute_stats(df, 'Polarity')
    print(f"H3 Test: Chi2 = {chi2_pol:.4f}, df = {dof_pol}, p = {p_pol:.4e}, Cramer's V = {v_pol:.4f}")

    # Multivariate Binomial Logistic Regression
    print("\n" + "=" * 80)
    print(" MULTIVARIATE LOGIT MODEL (PREDICTING ASSAMESE COPULA DELETION)")
    print("=" * 80)
    logit_model = smf.logit(
        "Copula_Deleted ~ C(Register, Treatment(reference='Formal Prose')) "
        "+ C(Predicate_Type, Treatment(reference='Locative/Existential')) "
        "+ C(Polarity, Treatment(reference='Affirmative'))",
        data=df
    ).fit(disp=False)
    print(logit_model.summary())

    or_table = pd.DataFrame({
        'Odds_Ratio': np.exp(logit_model.params),
        'CI_2.5%': np.exp(logit_model.conf_int()[0]),
        'CI_97.5%': np.exp(logit_model.conf_int()[1]),
        'p_value': logit_model.pvalues
    })
    print("\n--- Odds Ratios (exp(Beta)) ---")
    print(or_table.round(4).to_string())

    # Generate 4-Panel Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot A: Register
    sns.barplot(data=df, x='Register', y='Copula_Deleted', estimator=lambda x: np.mean(x) * 100, errorbar=None, ax=axes[0, 0], palette=['#e74c3c', '#2ecc71'])
    axes[0, 0].set_title("A. Assamese: Zero-Copula Rate by Register (H1)", fontweight='bold')
    axes[0, 0].set_ylabel("Zero Copula Rate (%)")
    axes[0, 0].set_ylim(0, 105)
    for c in axes[0, 0].containers:
        axes[0, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot B: Predicate Hierarchy by Register
    sns.barplot(data=df, x='Predicate_Type', y='Copula_Deleted', hue='Register', estimator=lambda x: np.mean(x) * 100, errorbar=None, ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title("B. Assamese: Predicate Type Hierarchy (H2)", fontweight='bold')
    axes[0, 1].set_ylabel("Zero Copula Rate (%)")
    axes[0, 1].set_ylim(0, 105)
    for c in axes[0, 1].containers:
        axes[0, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    # Plot C: Overall Predicate Deletion
    pred_df = df.groupby('Predicate_Type')['Copula_Deleted'].mean().reset_index()
    pred_df['Copula_Deleted'] *= 100
    pred_df = pred_df.sort_values('Copula_Deleted', ascending=False)
    sns.barplot(data=pred_df, x='Predicate_Type', y='Copula_Deleted', ax=axes[1, 0], palette=['#2980b9', '#8e44ad', '#d35400'])
    axes[1, 0].set_title("C. Assamese: Overall Deletion across Predicates", fontweight='bold')
    axes[1, 0].set_ylabel("Zero Copula Rate (%)")
    axes[1, 0].set_ylim(0, 105)
    for c in axes[1, 0].containers:
        axes[1, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot D: Polarity Interaction
    sns.barplot(data=df, x='Polarity', y='Copula_Deleted', hue='Register', estimator=lambda x: np.mean(x) * 100, errorbar=None, ax=axes[1, 1], palette='magma')
    axes[1, 1].set_title("D. Assamese: Polarity Interaction (H3)", fontweight='bold')
    axes[1, 1].set_ylabel("Zero Copula Rate (%)")
    axes[1, 1].set_ylim(0, 105)
    for c in axes[1, 1].containers:
        axes[1, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    plt.tight_layout()
    plt.savefig("assamese_copula_dashboard.png", dpi=300)
    print("\n[SUCCESS] Saved 'assamese_clean_nonverbal.csv' and 'assamese_copula_dashboard.png'")

if __name__ == "__main__":
    run_assamese_analysis()