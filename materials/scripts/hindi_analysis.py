import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Styling for publication-grade figures matching all language dashboards
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 1.0

def load_and_audit_corpus(csv_path="Cross_register.csv"):
    """
    Loads Cross_register.csv and applies multi-stage linguistic auditing to correct:
    1. Em-dash / ellipsis attached copulas (e.g., 'हैं—', 'हैं…!')
    2. Literary OCR copula transcriptions ('हे', 'हें', 'हैे' at clause boundaries)
    3. Unmarked literary dialogue turns (em-dash '—', '--', curly quotes)
    4. Past perfective, habitual, inceptive, imperative, subjunctive, and future action verbs
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    
    # Check current directory, script directory, and common subfolders
    if not os.path.exists(csv_path):
        candidate_paths = [
            os.path.join(script_dir, csv_path),
            os.path.join(script_dir, "6 source analysis", csv_path),
            os.path.join(os.getcwd(), "sem_courses", "sem3_courses", "hs5652", "6 source analysis", csv_path)
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                csv_path = p
                break

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not locate '{csv_path}'. Please check directory path.")

    print(f"Loading Hindi dataset from: '{csv_path}'")
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    # Extended punctuation splitter (includes Devanagari danda, em-dash, en-dash, ellipsis, curly quotes)
    punct_chars = string.punctuation + "।॥‘’“”«»—–‐−…\n\r\t"
    punct_split_re = re.compile(r'[\s' + re.escape(punct_chars) + r']+')
    
    # Standard present copulas + Literary OCR variants
    standard_copulas = {"है", "हैं", "हूँ", "हूं", "हो"}
    ocr_literary_copulas = {"हे", "हें", "हैे"}
    
    # Comprehensive verbal / non-present / action-root exclusion lists
    past_and_action_tokens = {
        # Past auxiliaries & typos
        "था", "थी", "थे", "थीं", "थां", "थें",
        # Perfective / light verbs / past actions
        "गया", "गयी", "गये", "गई", "गए", "हुआ", "हुई", "हुए", "हुईं",
        "लिया", "लिये", "लिए", "ली", "लीं", "दिया", "दिये", "दिए", "दी", "दीं",
        "किया", "किये", "किए", "की", "आया", "आयी", "आये", "आई", "आए", "आयीं",
        "लगा", "लगी", "लगे", "लगीं", "पड़ा", "पड़ी", "पड़े", "पड़ीं",
        "सका", "सकी", "सके", "सकीं", "चुका", "चुकी", "चुके",
        "रहा", "रही", "रहे", "रहीं", "ठहरा", "ठहरी", "ठहरे",
        "बोला", "बोली", "बोले", "कहा", "कही", "कहे", "पूछा", "पूछी",
        "देखा", "देखी", "देखे", "सुना", "सुनी", "सुने", "बचा", "बची", "बचे",
        "चला", "चली", "चले", "दौड़ा", "दौड़ी", "दौड़े", "भागा", "भागी", "भागे",
        "मारा", "मारी", "मारे", "उठा", "उठी", "उठे", "बैठा", "बैठी", "बैठे",
        "निकला", "निकली", "निकले", "पहुँचा", "पहुँची", "पहुँचे", "रूका", "रुका",
        "रोया", "रोयी", "रोए", "हँसा", "हँसी", "हँसे", "मुस्कराया", "चौंका", "चौंकी",
        # Imperatives, modals, subjunctives
        "चाहिए", "चाहिये", "कीजिए", "दीजिए", "लीजिए", "सुनिए", "बैटिए", "बैठिए",
        "जाओ", "आओ", "बताओ", "दिखाओ", "बजाओ", "सोचो", "देखो", "सुनो", "चलो", "मरो", "करो", "लो", "दो"
    }
    
    # Verbal suffixes (Future variants with ऍं/यें, habitual ता/ती/ते, subjunctive एँ/ऍं)
    verbal_suffixes = (
        "एगा", "एगी", "एंगे", "एंगी", "ऊंगा", "ऊँगा", "ऊंगी", "ऊँगी", "ओगे", "ओगी",
        "ऍंगे", "ऍंगी", "ऍंगें", "येंगे", "येंगी", "येगा", "येगी", "ड़ेगा", "ड़ेगी", "ड़ेंगे", "देगें", "पकड़ेगें",
        "ऊँ", "जाऍं", "आऍं", "लाऍं", "खाऍंगे", "चुराऍंगे"
    )
    
    # Common verbal roots tagged by Stanza in NLP_Root_Word_Notes
    action_roots = {
        "कहा", "बोला", "पूछा", "देखा", "किया", "दिया", "लिया", "आया", "गया", "लगा", "लगी", "लगे", "लगीं",
        "दे", "ले", "कर", "मार", "पटक", "फेंक", "पकड़", "छोड़", "सोचो", "देखो", "जाने", "भाग", "भागे",
        "दौड़ा", "दौड़ी", "चला", "चले", "बना", "बनी", "रहा", "रही", "रहे", "सकता", "सका", "चाहिए", "चाहिये",
        "पहुँचेंगे", "पूछूँगा", "देगा", "लूँगा", "करता", "करते", "खाता", "देखता", "समझता", "आती", "जाती", "जाते",
        "होता", "होती", "होते", "रहता", "रहती", "रहते", "पड़ता", "पड़ती", "पड़ते", "कहता", "कहती", "कहते",
        "बजा", "बजती", "सुनायी", "बोली", "गिरा", "तोड", "मारा", "गिरी", "निकली", "हटा", "बढा", "चूम"
    }

    corrected_status = []
    corrected_subreg = []
    is_valid_nonverbal = []
    exclusion_reason = []

    for idx, row in df.iterrows():
        raw_sent = str(row['Raw_Sentence']).strip()
        reg = row['Register']
        subreg = row['Sub_Register']
        orig_status = int(row['Copula_Status_NLP'])
        root_note = str(row.get('NLP_Root_Word_Notes', '')).replace('Root Word:', '').strip(punct_chars)
        
        tokens = [t for t in punct_split_re.split(raw_sent) if t]
        token_set = set(tokens)
        
        # 1. Fix Copula Status (recover em-dash attached copulas & Literary OCR 'हे'/'हें'/'हैे')
        new_status = orig_status
        if not token_set.isdisjoint(standard_copulas):
            new_status = 1
        elif reg == 'Literary' and not token_set.isdisjoint(ocr_literary_copulas):
            # Exclude vocative/exclamation 'हें हें'
            if "हें हें" not in raw_sent:
                new_status = 1
        corrected_status.append(new_status)
        
        # 2. Fix Sub-Register in Literary texts (detect em-dash & quote dialogues)
        new_subreg = subreg
        if reg == 'Literary':
            has_speech_marker = (
                bool(re.search(r'["\'‘’“”«»—]|--', raw_sent)) or
                any(s in raw_sent for s in ["ने कहा", "बोला", "पूछा", "चिढ़ाया", "समझाया"])
            )
            if has_speech_marker:
                new_subreg = "Literary Dialogue"
        corrected_subreg.append(new_subreg)
        
        # 3. Audit eligibility as a Present-Tense Non-Verbal Copular Clause
        # Check A: Script headers / English / metadata noise
        if "Subtitles by" in raw_sent or "https://" in raw_sent or len(tokens) < 2:
            is_valid_nonverbal.append(False)
            exclusion_reason.append("Metadata/Fragment")
            continue
            
        # Check B: Contains past/action/imperative/modal verb tokens (when no present copula is present)
        if new_status == 0:
            if not token_set.isdisjoint(past_and_action_tokens):
                is_valid_nonverbal.append(False)
                exclusion_reason.append("Past/Action/Imperative Verb")
                continue
            if any(t.endswith(verbal_suffixes) for t in tokens):
                is_valid_nonverbal.append(False)
                exclusion_reason.append("Future/Subjunctive Suffix")
                continue
            if root_note in action_roots or root_note.endswith(("ना", "ता", "ती", "ते", "या", "यी", "ये", "ऊँगा", "एगा", "ेंगे")):
                is_valid_nonverbal.append(False)
                exclusion_reason.append("Verbal Root")
                continue
        else:
            # Even when copula is present (1), exclude present continuous/habitual verbal clauses (e.g. 'कर रहा है', 'जाता है')
            verbal_aux_neighbors = {"रहा", "रही", "रहे", "ता", "ती", "ते", "चुका", "चुकी", "चुके", "सकता", "सकती", "सकते"}
            if not token_set.isdisjoint(verbal_aux_neighbors) or any(
                t.endswith(("ता", "ती", "ते")) and t not in {"माता", "पिता", "भ्राता", "नेता", "दाता", "कुत्ता", "पत्ता", "रास्ता", "नाश्ता", "सस्ता", "महँगा", "बेटा", "छोटा", "मोटा"}
                for t in tokens[:-1]
            ):
                if root_note in action_roots:
                    is_valid_nonverbal.append(False)
                    exclusion_reason.append("Present Verbal Clause (Habitual/Progressive)")
                    continue

        is_valid_nonverbal.append(True)
        exclusion_reason.append("Valid Non-Verbal Copular Clause")

    df['Copula_Status_Corrected'] = corrected_status
    df['Copula_Deleted_Raw'] = (df['Copula_Status_NLP'] == 0).astype(int)
    df['Copula_Deleted_Clean'] = (df['Copula_Status_Corrected'] == 0).astype(int)
    df['Sub_Register_Corrected'] = corrected_subreg
    df['Is_Valid_NonVerbal'] = is_valid_nonverbal
    df['Audit_Classification'] = exclusion_reason
    
    return df

def compute_contingency_stats(df_sub, group_col, target_col='Copula_Deleted_Clean'):
    """Computes Chi-Square, p-value, degrees of freedom, and Cramer's V."""
    ct = pd.crosstab(df_sub[group_col], df_sub[target_col])
    chi2, p_val, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0
    return chi2, p_val, dof, cramers_v, ct

def run_full_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    
    df = load_and_audit_corpus("Cross_register.csv")
    
    # Save audited full file & filtered non-verbal file directly to script directory
    df.to_csv(os.path.join(script_dir, "Cross_register_audited_full.csv"), index=False, encoding='utf-8-sig')
    df_clean = df[df['Is_Valid_NonVerbal']].copy()
    df_clean.to_csv(os.path.join(script_dir, "Cross_register_clean_nonverbal.csv"), index=False, encoding='utf-8-sig')
    
    print("=" * 80)
    print(" STAGE 1: RAW UNFILTERED DATASET (AS UPLOADED IN Cross_register.csv)")
    print("=" * 80)
    raw_summary = df.groupby(['Register', 'Sub_Register']).agg(
        Total_Rows=('Copula_Deleted_Raw', 'count'),
        Copula_Present=('Copula_Deleted_Raw', lambda x: (x == 0).sum()),
        Copula_Deleted=('Copula_Deleted_Raw', 'sum'),
        Raw_Deletion_Pct=('Copula_Deleted_Raw', lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    print(raw_summary.to_string(index=False))
    
    print("\n--- Why Literary Deletion Was Inflated (Audit Breakdown of Literary Rows) ---")
    lit_audit = df[df['Register'] == 'Literary']['Audit_Classification'].value_counts()
    print(lit_audit.to_string())

    print("\n" + "=" * 80)
    print(" STAGE 2: CLEANED PRESENT-TENSE NON-VERBAL COPULAR CORPUS")
    print("=" * 80)
    clean_reg_summary = df_clean.groupby(['Register', 'Sub_Register_Corrected']).agg(
        NonVerbal_Clauses=('Copula_Deleted_Clean', 'count'),
        Copula_Present=('Copula_Deleted_Clean', lambda x: (x == 0).sum()),
        Copula_Deleted=('Copula_Deleted_Clean', 'sum'),
        Deletion_Rate_Pct=('Copula_Deleted_Clean', lambda x: round(x.mean() * 100, 2))
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(clean_reg_summary.to_string(index=False))

    # Hypothesis 1: Register Comparison (Spoken Subtitle vs Literary Narrative)
    h1_df = df_clean[df_clean['Sub_Register_Corrected'].isin(['Film Dialogue', 'Literary Narrative'])]
    chi2_h1, p_h1, dof_h1, v_h1, ct_h1 = compute_contingency_stats(h1_df, 'Sub_Register_Corrected')
    print(f"\n[H1: Register Test (Film Dialogue vs. Literary Narrative)]")
    print(f"Chi2 = {chi2_h1:.4f}, df = {dof_h1}, p = {p_h1:.4e}, Cramer's V = {v_h1:.4f}")

    # Hypothesis 2: Predicate Type Comparison
    print("\n[H2: Predicate Type Breakdown (Cleaned Non-Verbal Corpus)]")
    pred_summary = df_clean.groupby('Predicate_Type_Final_NLP').agg(
        Total=('Copula_Deleted_Clean', 'count'),
        Deleted=('Copula_Deleted_Clean', 'sum'),
        Deletion_Rate_Pct=('Copula_Deleted_Clean', lambda x: round(x.mean() * 100, 2))
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pred_summary.to_string(index=False))
    chi2_h2, p_h2, dof_h2, v_h2, _ = compute_contingency_stats(df_clean, 'Predicate_Type_Final_NLP')
    print(f"Chi2 = {chi2_h2:.4f}, df = {dof_h2}, p = {p_h2:.4e}, Cramer's V = {v_h2:.4f}")

    # Hypothesis 3: Polarity Comparison
    print("\n[H3: Polarity Breakdown (Cleaned Non-Verbal Corpus)]")
    pol_summary = df_clean.groupby('Polarity_NLP').agg(
        Total=('Copula_Deleted_Clean', 'count'),
        Deleted=('Copula_Deleted_Clean', 'sum'),
        Deletion_Rate_Pct=('Copula_Deleted_Clean', lambda x: round(x.mean() * 100, 2))
    ).reset_index().sort_values('Deletion_Rate_Pct', ascending=False)
    print(pol_summary.to_string(index=False))
    chi2_h3, p_h3, dof_h3, v_h3, _ = compute_contingency_stats(df_clean, 'Polarity_NLP')
    print(f"Chi2 = {chi2_h3:.4f}, df = {dof_h3}, p = {p_h3:.4e}, Cramer's V = {v_h3:.4f}")

    # Multivariate Binomial Logistic Regression
    print("\n" + "=" * 80)
    print(" MULTIVARIATE BINOMIAL LOGISTIC REGRESSION (PREDICTING COPULA DELETION)")
    print("=" * 80)
    logit_model = smf.logit(
        "Copula_Deleted_Clean ~ C(Sub_Register_Corrected, Treatment(reference='Literary Narrative')) "
        "+ C(Predicate_Type_Final_NLP, Treatment(reference='Locative/Existential')) "
        "+ C(Polarity_NLP, Treatment(reference='Affirmative'))",
        data=df_clean
    ).fit(disp=False)
    print(logit_model.summary())
    
    # Odds Ratios
    or_table = pd.DataFrame({
        'Odds_Ratio': np.exp(logit_model.params),
        'CI_2.5%': np.exp(logit_model.conf_int()[0]),
        'CI_97.5%': np.exp(logit_model.conf_int()[1]),
        'p_value': logit_model.pvalues
    })
    print("\n--- Odds Ratios (exp(Beta)) ---")
    print(or_table.round(4).to_string())

    # Generate Publication Visualizations (4-Panel Hindi Dashboard)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # Plot 1: Before vs After Adversity Cleaning by Sub-Register (H1)
    raw_sub = df.groupby('Sub_Register')['Copula_Deleted_Raw'].mean() * 100
    clean_sub = df_clean.groupby('Sub_Register_Corrected')['Copula_Deleted_Clean'].mean() * 100
    comp_df = pd.DataFrame({'Raw Unfiltered CSV': raw_sub, 'Cleaned Non-Verbal Corpus': clean_sub}).reset_index()
    comp_melt = comp_df.melt(id_vars='index', var_name='Pipeline Stage', value_name='Deletion Rate (%)')
    sns.barplot(data=comp_melt, x='index', y='Deletion Rate (%)', hue='Pipeline Stage', ax=axes[0, 0], palette=['#e74c3c', '#2ecc71'])
    axes[0, 0].set_title("A. Hindi: Impact of Linguistic Cleaning on Register Deletion Rates (H1)", fontweight='bold')
    axes[0, 0].set_ylabel("Zero-Copula Rate (%)")
    axes[0, 0].set_xlabel("Sub-Register")
    axes[0, 0].set_ylim(0, 75)
    for container in axes[0, 0].containers:
        axes[0, 0].bar_label(container, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot 2: Predicate Type x Register Interaction (H2)
    sns.barplot(
        data=df_clean, x='Predicate_Type_Final_NLP', y='Copula_Deleted_Clean',
        hue='Sub_Register_Corrected', estimator=lambda x: np.mean(x) * 100,
        errorbar=None, ax=axes[0, 1], palette='viridis'
    )
    axes[0, 1].set_title("B. Hindi: Zero-Copula Rate by Predicate Type & Sub-Register (H2)", fontweight='bold')
    axes[0, 1].set_ylabel("Zero-Copula Rate (%)")
    axes[0, 1].set_xlabel("Predicate Category")
    axes[0, 1].set_ylim(0, 65)
    for container in axes[0, 1].containers:
        axes[0, 1].bar_label(container, fmt='%.1f%%', fontsize=9, padding=2)

    # Plot 3: Source-Level Comparison Across All 6 Texts
    src_summary = df_clean.groupby(['Source_Title', 'Register'])['Copula_Deleted_Clean'].mean().reset_index()
    src_summary['Copula_Deleted_Clean'] *= 100
    src_summary = src_summary.sort_values('Copula_Deleted_Clean', ascending=False)
    sns.barplot(data=src_summary, x='Source_Title', y='Copula_Deleted_Clean', hue='Register', dodge=False, ax=axes[1, 0], palette=['#3498db', '#e67e22'])
    axes[1, 0].set_title("C. Hindi: Cleaned Zero-Copula Rate Across All 6 Sources", fontweight='bold')
    axes[1, 0].set_ylabel("Zero-Copula Rate (%)")
    axes[1, 0].set_xlabel("Source Title")
    axes[1, 0].tick_params(axis='x', rotation=20)
    axes[1, 0].set_ylim(0, 65)
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, fmt='%.1f%%', fontweight='bold', padding=3)

    # Plot 4: Polarity x Register (H3)
    sns.barplot(
        data=df_clean, x='Polarity_NLP', y='Copula_Deleted_Clean',
        hue='Sub_Register_Corrected', estimator=lambda x: np.mean(x) * 100,
        errorbar=None, ax=axes[1, 1], palette='magma'
    )
    axes[1, 1].set_title("D. Hindi: Zero-Copula Rate by Polarity & Sub-Register (H3)", fontweight='bold')
    axes[1, 1].set_ylabel("Zero-Copula Rate (%)")
    axes[1, 1].set_xlabel("Polarity")
    axes[1, 1].set_ylim(0, 65)
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, fmt='%.1f%%', fontsize=9, padding=2)

    plt.tight_layout()
    
    # Save with both standardized filenames in the same folder as the script
    output_png1 = os.path.join(script_dir, "hindi_copula_dashboard.png")
    output_png2 = os.path.join(script_dir, "advanced_copula_analysis_dashboard.png")
    
    plt.savefig(output_png1, dpi=300)
    plt.savefig(output_png2, dpi=300)
    
    print(f"\n[SUCCESS] Saved Cleaned Data to '{os.path.join(script_dir, 'Cross_register_clean_nonverbal.csv')}'")
    print(f"[SUCCESS] Saved Dashboard Plot to: '{output_png1}'")

if __name__ == "__main__":
    run_full_pipeline()