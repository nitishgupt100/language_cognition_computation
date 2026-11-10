import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# Publication-grade styling matching the Hindi analysis dashboard
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 1.0

def load_and_audit_tamil_corpus(formal_path="formal_tamil_2.csv", informal_path="informal_tamil_2.csv"):
    """
    Ingests and audits Tamil formal prose and informal dialogue datasets:
    1. Resolves paths relative to the script directory to prevent FileNotFoundError.
    2. Handles preamble lines (reads headers from row 4).
    3. Normalizes zero-copula realization: '——' indicates deletion is syntactically blocked (0),
       while a valid text realization indicates zero-copula is permitted (1).
    4. Normalizes predicate categories into Universal Curnow (1999) types:
       Equational/Nominal, Adjectival, and Locative/Existential.
    5. Tags clausal polarity (Affirmative vs. Negative).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    f_path = os.path.join(script_dir, formal_path) if not os.path.isabs(formal_path) else formal_path
    inf_path = os.path.join(script_dir, informal_path) if not os.path.isabs(informal_path) else informal_path

    # Fallback if files do not have the '_2' suffix
    if not os.path.exists(f_path):
        f_alt = f_path.replace("formal_tamil_2.csv", "formal_tamil.csv")
        if os.path.exists(f_alt):
            f_path = f_alt
    if not os.path.exists(inf_path):
        inf_alt = inf_path.replace("informal_tamil_2.csv", "informal_tamil.csv")
        if os.path.exists(inf_alt):
            inf_path = inf_alt

    if not os.path.exists(f_path) or not os.path.exists(inf_path):
        raise FileNotFoundError(f"Missing input files. Looked for '{f_path}' and '{inf_path}'.")

    # 1. Load Formal Prose (Line 4 contains column headers)
    df_f = pd.read_csv(f_path, skiprows=3, encoding='utf-8').dropna(subset=['Type']).copy()
    df_f.columns = [c.strip() for c in df_f.columns]
    df_f['Register'] = 'Literary Prose'
    df_f['Sub_Register_Corrected'] = 'Formal Prose'

    # 2. Load Informal Dialogue
    df_inf = pd.read_csv(inf_path, skiprows=3, encoding='utf-8').dropna(subset=['Type']).copy()
    df_inf.columns = [c.strip() for c in df_inf.columns]
    df_inf['Register'] = 'Spoken Subtitle'
    df_inf['Sub_Register_Corrected'] = 'Informal Dialogue'

    combined = pd.concat([df_f, df_inf], ignore_index=True)

    # 3. Determine Copula Deletion
    # A dashed entry ('——', '————', '——-') means deletion is grammatically blocked (Overt mandatory = 0)
    def check_zero_copula(val):
        s = str(val).strip()
        if not s or s in ['——', '————', '——-', '—', '-']:
            return 0  # Overt verb mandatory / deletion blocked
        return 1      # Zero copula permitted / observed

    combined['Copula_Deleted_Clean'] = combined['Zero Copula Form'].apply(check_zero_copula)

    # 4. Standardize Predicate Category to Universal UD Types
    def map_predicate(raw_type):
        t = str(raw_type).strip().lower()
        if 'possessive' in t or ('possession' in t and any(k in t for k in ['locative', 'existential', 'கிட்ட'])):
            return 'Locative/Existential'
        if any(k in t for k in ['locative', 'existential', 'dative', 'state', 'event']):
            return 'Locative/Existential'
        if any(k in t for k in ['nominal', 'definitional', 'person', 'classification', 'identity', 'kinship', 'possession', 'choice', 'social relation', 'role', 'profession']):
            return 'Equational/Nominal'
        if any(k in t for k in ['adjectival', 'quality', 'taste', 'weather', 'texture', 'behaviour', 'condition', 'evaluation']):
            return 'Adjectival'
        return 'Locative/Existential' if 'possess' in t else 'Equational/Nominal'

    combined['Predicate_Type_Final_NLP'] = combined['Type'].apply(map_predicate)

    # 5. Extract Clausal Polarity
    def detect_polarity(raw_type, overt_str):
        t = str(raw_type).strip().lower()
        o = str(overt_str).strip()
        if 'negative' in t or any(m in o for m in ['இல்லை', 'இல்ல', 'அல்ல', 'அல்லர்', 'அல்லவா', 'நல்லதல்ல']):
            return 'Negative'
        return 'Affirmative'

    combined['Polarity_NLP'] = [detect_polarity(t, o) for t, o in zip(combined['Type'], combined['Overt Form'])]
    combined['Is_Valid_NonVerbal'] = True

    return combined

def compute_contingency_stats(df_sub, group_col, target_col='Copula_Deleted_Clean'):
    """Calculates Chi-Square, p-value, degrees of freedom, and Cramér's V."""
    ct = pd.crosstab(df_sub[group_col], df_sub[target_col])
    chi2, p_val, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
    return chi2, p_val, dof, cramers_v, ct

def run_tamil_dashboard_pipeline():
    df = load_and_audit_tamil_corpus()
    total_n = len(df)
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

    # Save cleaned corpus
    df.to_csv(os.path.join(script_dir, "tamil_clean_nonverbal.csv"), index=False, encoding='utf-8-sig')

    print("=" * 80)
    print(f" TAMIL CORPUS EMPIRICAL EVALUATION (N = {total_n})")
    print("=" * 80)

    # Hypothesis 1: Register Effect
    print("\n--- [H1] Register Breakdown ---")
    reg_summary = df.groupby('Sub_Register_Corrected')['Copula_Deleted_Clean'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(reg_summary.to_string(index=False))

    chi2_r, p_r, dof_r, v_r, _ = compute_contingency_stats(df, 'Sub_Register_Corrected')
    print(f"H1 Register: Chi2 = {chi2_r:.4f}, df = {dof_r}, p = {p_r:.4e}, Cramer's V = {v_r:.4f}")

    # Hypothesis 2: Predicate Hierarchy
    print("\n--- [H2] Predicate Hierarchy Breakdown ---")
    pred_summary = df.groupby('Predicate_Type_Final_NLP')['Copula_Deleted_Clean'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pred_summary.to_string(index=False))

    chi2_p, p_p, dof_p, v_p, _ = compute_contingency_stats(df, 'Predicate_Type_Final_NLP')
    print(f"H2 Predicate: Chi2 = {chi2_p:.4f}, df = {dof_p}, p = {p_p:.4e}, Cramer's V = {v_p:.4f}")

    # Hypothesis 3: Polarity Effect
    print("\n--- [H3] Polarity Breakdown ---")
    pol_summary = df.groupby('Polarity_NLP')['Copula_Deleted_Clean'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pol_summary.to_string(index=False))

    chi2_pol, p_pol, dof_pol, v_pol, _ = compute_contingency_stats(df, 'Polarity_NLP')
    print(f"H3 Polarity: Chi2 = {chi2_pol:.4f}, df = {dof_pol}, p = {p_pol:.4e}, Cramer's V = {v_pol:.4f}")

    # Multivariate Binomial Logistic Regression (Affirmative Subset)
    # Negative polarity in Tamil categorically forbids zero copula (100% overt negative verbs: இல்லை, அல்ல)
    df_aff = df[df['Polarity_NLP'] == 'Affirmative'].copy()
    logit_model = smf.logit(
        "Copula_Deleted_Clean ~ C(Sub_Register_Corrected, Treatment(reference='Formal Prose')) "
        "+ C(Predicate_Type_Final_NLP, Treatment(reference='Locative/Existential'))",
        data=df_aff
    ).fit(disp=False)
    print("\n--- Logistic Regression (Affirmative Subset) ---")
    print(logit_model.summary())

    # Generate 4-Panel Visualization Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Sub-Register (H1)
    reg_plot = df.groupby('Sub_Register_Corrected')['Copula_Deleted_Clean'].mean().reset_index()
    reg_plot['Copula_Deleted_Clean'] *= 100
    reg_plot = reg_plot.sort_values('Copula_Deleted_Clean', ascending=False)
    sns.barplot(data=reg_plot, x='Sub_Register_Corrected', y='Copula_Deleted_Clean', ax=axes[0, 0], palette=['#2ecc71', '#e74c3c'])
    axes[0, 0].set_title("A. Tamil: Zero-Copula Rate by Register (H1)", fontweight='bold')
    axes[0, 0].set_ylabel("Zero Copula Rate (%)")
    axes[0, 0].set_xlabel("Register")
    axes[0, 0].set_ylim(0, 105)
    for c in axes[0, 0].containers:
        axes[0, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot 2: Predicate Hierarchy by Register (H2 Interaction)
    pred_reg_plot = df.groupby(['Predicate_Type_Final_NLP', 'Sub_Register_Corrected'])['Copula_Deleted_Clean'].mean().reset_index()
    pred_reg_plot['Copula_Deleted_Clean'] *= 100
    sns.barplot(data=pred_reg_plot, x='Predicate_Type_Final_NLP', y='Copula_Deleted_Clean', hue='Sub_Register_Corrected', ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title("B. Tamil: Predicate Hierarchy by Register (H2)", fontweight='bold')
    axes[0, 1].set_ylabel("Zero Copula Rate (%)")
    axes[0, 1].set_xlabel("Predicate Category")
    axes[0, 1].set_ylim(0, 105)
    for c in axes[0, 1].containers:
        axes[0, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    # Plot 3: Overall Predicate Deletion (H2 Universal Trend)
    pred_overall = df.groupby('Predicate_Type_Final_NLP')['Copula_Deleted_Clean'].mean().reset_index()
    pred_overall['Copula_Deleted_Clean'] *= 100
    pred_overall = pred_overall.sort_values('Copula_Deleted_Clean', ascending=False)
    sns.barplot(data=pred_overall, x='Predicate_Type_Final_NLP', y='Copula_Deleted_Clean', ax=axes[1, 0], palette=['#2980b9', '#8e44ad', '#d35400'])
    axes[1, 0].set_title("C. Tamil: Overall Deletion across Predicate Types", fontweight='bold')
    axes[1, 0].set_ylabel("Zero Copula Rate (%)")
    axes[1, 0].set_xlabel("Predicate Category")
    axes[1, 0].set_ylim(0, 105)
    for c in axes[1, 0].containers:
        axes[1, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot 4: Polarity Interaction (H3)
    pol_reg_plot = df.groupby(['Polarity_NLP', 'Sub_Register_Corrected'])['Copula_Deleted_Clean'].mean().reset_index()
    pol_reg_plot['Copula_Deleted_Clean'] *= 100
    sns.barplot(data=pol_reg_plot, x='Polarity_NLP', y='Copula_Deleted_Clean', hue='Sub_Register_Corrected', ax=axes[1, 1], palette='magma')
    axes[1, 1].set_title("D. Tamil: Polarity Categorical Block (H3)", fontweight='bold')
    axes[1, 1].set_ylabel("Zero Copula Rate (%)")
    axes[1, 1].set_xlabel("Clausal Polarity")
    axes[1, 1].set_ylim(0, 105)
    for c in axes[1, 1].containers:
        axes[1, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    plt.tight_layout()
    chart_output = os.path.join(script_dir, "tamil_copula_dashboard.png")
    plt.savefig(chart_output, dpi=300)
    print(f"\n[SUCCESS] Generated 'tamil_clean_nonverbal.csv' and '{chart_output}'")

if __name__ == "__main__":
    run_tamil_dashboard_pipeline()