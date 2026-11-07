import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# Visualization styling matching Hindi and Assamese dashboards
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 1.0

def run_bengali_analysis(csv_filename="Bengali_Cross_Register_Nonverbal.csv"):
    # Resolve relative path dynamically to avoid directory mismatch errors
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    csv_path = os.path.join(script_dir, csv_filename) if not os.path.isabs(csv_filename) else csv_filename
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find Bengali CSV at: '{csv_path}'")
        
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Filter for verified non-verbal copular clauses
    df_clean = df[df['Is_Valid_NonVerbal'] == True].copy()
    total_n = len(df_clean)

    print("=" * 80)
    print(f" BENGALI CORPUS EMPIRICAL EVALUATION (Clean Non-Verbal N = {total_n})")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. HYPOTHESIS 1: REGISTER & SUB-REGISTER EFFECT
    # -------------------------------------------------------------
    print("\n--- [H1] Sub-Register Breakdown ---")
    subreg_summary = df_clean.groupby('Sub_Register_Corrected')['Copula_Deleted_Clean'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(subreg_summary.to_string(index=False))

    ct_sub = pd.crosstab(df_clean['Sub_Register_Corrected'], df_clean['Copula_Deleted_Clean'])
    chi2_s, p_s, dof_s, _ = chi2_contingency(ct_sub)
    v_s = np.sqrt(chi2_s / (total_n * (min(ct_sub.shape) - 1)))
    print(f"Sub-Register Test: Chi2 = {chi2_s:.4f}, df = {dof_s}, p = {p_s:.4e}, Cramer's V = {v_s:.4f}")

    ct_reg = pd.crosstab(df_clean['Register'], df_clean['Copula_Deleted_Clean'])
    chi2_r, p_r, dof_r, _ = chi2_contingency(ct_reg)
    v_r = np.sqrt(chi2_r / (total_n * (min(ct_reg.shape) - 1)))
    print(f"Broad Register (Spoken vs. Literary): Chi2 = {chi2_r:.4f}, df = {dof_r}, p = {p_r:.4e}, Cramer's V = {v_r:.4f}")

    # -------------------------------------------------------------
    # 2. HYPOTHESIS 2: PREDICATE HIERARCHY
    # -------------------------------------------------------------
    print("\n--- [H2] Predicate Type Hierarchy Breakdown ---")
    pred_summary = df_clean.groupby('Predicate_Type_Final_NLP')['Copula_Deleted_Clean'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pred_summary.to_string(index=False))

    ct_p = pd.crosstab(df_clean['Predicate_Type_Final_NLP'], df_clean['Copula_Deleted_Clean'])
    chi2_p, p_p, dof_p, _ = chi2_contingency(ct_p)
    v_p = np.sqrt(chi2_p / (total_n * (min(ct_p.shape) - 1)))
    print(f"Predicate Hierarchy Test: Chi2 = {chi2_p:.4f}, df = {dof_p}, p = {p_p:.4e}, Cramer's V = {v_p:.4f}")

    # -------------------------------------------------------------
    # 3. HYPOTHESIS 3: POLARITY EFFECT
    # -------------------------------------------------------------
    print("\n--- [H3] Polarity Breakdown ---")
    pol_summary = df_clean.groupby('Polarity_NLP')['Copula_Deleted_Clean'].agg(
        Total='count',
        Deleted='sum',
        Deletion_Rate_Pct=lambda x: round(x.mean() * 100, 2)
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pol_summary.to_string(index=False))

    ct_pol = pd.crosstab(df_clean['Polarity_NLP'], df_clean['Copula_Deleted_Clean'])
    chi2_pol, p_pol, dof_pol, _ = chi2_contingency(ct_pol)
    v_pol = np.sqrt(chi2_pol / (total_n * (min(ct_pol.shape) - 1)))
    print(f"Polarity Test: Chi2 = {chi2_pol:.4f}, df = {dof_pol}, p = {p_pol:.4e}, Cramer's V = {v_pol:.4f}")

    # -------------------------------------------------------------
    # 4. MULTIVARIATE BINOMIAL LOGISTIC REGRESSION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" MULTIVARIATE LOGIT MODEL: BENGALI (AFFIRMATIVE SUBSET)")
    print("=" * 80)
    # Note on Bengali: Negative polarity categorically forbids zero copulas (100% overt: নয়, নেই, নাই).
    # Since Negative has 0% deletion, we evaluate the logit model on Affirmative clauses where variation occurs.
    df_aff = df_clean[df_clean['Polarity_NLP'] == 'Affirmative'].copy()
    
    logit_model = smf.logit(
        "Copula_Deleted_Clean ~ C(Sub_Register_Corrected, Treatment(reference='Literary Narrative')) "
        "+ C(Predicate_Type_Final_NLP, Treatment(reference='Locative/Existential'))",
        data=df_aff
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

    # -------------------------------------------------------------
    # 5. GENERATE 4-PANEL PUBLICATION DASHBOARD
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot A: Sub-Register
    subreg_plot = df_clean.groupby('Sub_Register_Corrected')['Copula_Deleted_Clean'].mean().reset_index()
    subreg_plot['Copula_Deleted_Clean'] *= 100
    subreg_plot = subreg_plot.sort_values('Copula_Deleted_Clean', ascending=False)
    sns.barplot(data=subreg_plot, x='Sub_Register_Corrected', y='Copula_Deleted_Clean', ax=axes[0, 0], palette='Blues_r')
    axes[0, 0].set_title("A. Bengali: Zero-Copula Rate by Sub-Register (H1)", fontweight='bold')
    axes[0, 0].set_ylabel("Zero-Copula Rate (%)")
    axes[0, 0].set_xlabel("Sub-Register")
    axes[0, 0].tick_params(axis='x', rotation=15)
    axes[0, 0].set_ylim(0, 105)
    for c in axes[0, 0].containers:
        axes[0, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot B: Predicate Hierarchy by Register
    pred_reg_plot = df_clean.groupby(['Predicate_Type_Final_NLP', 'Register'])['Copula_Deleted_Clean'].mean().reset_index()
    pred_reg_plot['Copula_Deleted_Clean'] *= 100
    sns.barplot(data=pred_reg_plot, x='Predicate_Type_Final_NLP', y='Copula_Deleted_Clean', hue='Register', ax=axes[0, 1], palette='viridis')
    axes[0, 1].set_title("B. Bengali: Predicate Hierarchy by Register (H2)", fontweight='bold')
    axes[0, 1].set_ylabel("Zero-Copula Rate (%)")
    axes[0, 1].set_xlabel("Predicate Category")
    axes[0, 1].set_ylim(0, 105)
    for c in axes[0, 1].containers:
        axes[0, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    # Plot C: Overall Predicate Deletion
    pred_overall = df_clean.groupby('Predicate_Type_Final_NLP')['Copula_Deleted_Clean'].mean().reset_index()
    pred_overall['Copula_Deleted_Clean'] *= 100
    pred_overall = pred_overall.sort_values('Copula_Deleted_Clean', ascending=False)
    sns.barplot(data=pred_overall, x='Predicate_Type_Final_NLP', y='Copula_Deleted_Clean', ax=axes[1, 0], palette=['#2980b9', '#8e44ad', '#d35400'])
    axes[1, 0].set_title("C. Bengali: Overall Deletion across Predicates", fontweight='bold')
    axes[1, 0].set_ylabel("Zero-Copula Rate (%)")
    axes[1, 0].set_xlabel("Predicate Category")
    axes[1, 0].set_ylim(0, 105)
    for c in axes[1, 0].containers:
        axes[1, 0].bar_label(c, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot D: Polarity Interaction
    pol_reg_plot = df_clean.groupby(['Polarity_NLP', 'Register'])['Copula_Deleted_Clean'].mean().reset_index()
    pol_reg_plot['Copula_Deleted_Clean'] *= 100
    sns.barplot(data=pol_reg_plot, x='Polarity_NLP', y='Copula_Deleted_Clean', hue='Register', ax=axes[1, 1], palette='magma')
    axes[1, 1].set_title("D. Bengali: Polarity Categorical Block (H3)", fontweight='bold')
    axes[1, 1].set_ylabel("Zero-Copula Rate (%)")
    axes[1, 1].set_xlabel("Clausal Polarity")
    axes[1, 1].set_ylim(0, 105)
    for c in axes[1, 1].containers:
        axes[1, 1].bar_label(c, fmt='%.1f%%', fontsize=9, padding=2)

    plt.tight_layout()
    output_fig = os.path.join(script_dir, "bengali_copula_dashboard.png")
    plt.savefig(output_fig, dpi=300)
    print(f"\n[SUCCESS] Dashboard saved to: '{output_fig}'")

if __name__ == "__main__":
    run_bengali_analysis()