import os
import pandas as pd

# Automatically locate the directory where cross_register.py resides
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

file_mapping = {
    "3_Idiots": (os.path.join(BASE_DIR, "model_comparison_subtitles_3_idiots.csv"), "Spoken Subtitle", "Film Dialogue"),
    "Secret_Superstar": (os.path.join(BASE_DIR, "model_comparison_subtitles_secret_superstar.csv"), "Spoken Subtitle", "Film Dialogue"),
    "Taare_Zameen_Par": (os.path.join(BASE_DIR, "model_comparison_subtitles_taare_zameen_par.csv"), "Spoken Subtitle", "Film Dialogue"),
    "Idgah": (os.path.join(BASE_DIR, "formal_idgah.csv"), "Literary", "Literary Narrative / Dialogue"),
    "Pahlwan_Ki_Dholak": (os.path.join(BASE_DIR, "formal_pahalwa_ki_dholak.csv"), "Literary", "Literary Narrative / Dialogue"),
    "Do_Bailon_Ki_Katha": (os.path.join(BASE_DIR, "formal_katha.csv"), "Literary", "Literary Narrative / Dialogue")
}

master_dfs = []

for title, (filepath, register, sub_reg) in file_mapping.items():
    if not os.path.exists(filepath):
        print(f"Skipping missing file: {filepath}")
        continue

    df = pd.read_csv(filepath, encoding='utf-8-sig')

    df['Source_Title'] = title
    df['Register'] = register
    if 'Sub_Register' not in df.columns:
        df['Sub_Register'] = sub_reg

    columns_to_keep = [
        "Sentence_ID", "Register", "Sub_Register", "Source_Title", "Raw_Sentence",
        "Copula_Status_NLP", "Polarity_NLP", "Predicate_Type_Final_NLP", "NLP_Root_Word_Notes"
    ]
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    master_dfs.append(df[existing_cols])

if not master_dfs:
    print("Error: No valid CSV files were loaded.")
else:
    master_corpus = pd.concat(master_dfs, ignore_index=True)
    master_corpus['Master_Sentence_ID'] = [f"CORPUS_{i:04d}" for i in range(1, len(master_corpus) + 1)]

    output_path = os.path.join(BASE_DIR, "Cross_register.csv")
    master_corpus.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("=" * 60)
    print(f"SUCCESS: Master corpus created with {len(master_corpus)} rows.")
    print(f"Saved at: '{output_path}'")