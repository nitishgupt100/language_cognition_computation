import os
import pandas as pd

# Point to hs5652 subfolder where files reside
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_mapping = {
    "Mirzapur": (
        os.path.join(BASE_DIR, "extracted_subtitles_nlp4.csv"), 
        "Spoken Subtitle", 
        "Film Dialogue"
    ),
    "Poos_Ki_Raat": (
        os.path.join(BASE_DIR, "formal_poos_ki_raat.csv"), 
        "Literary", 
        "Literary Narrative / Dialogue"
    )
}

compiled_dfs = []

for title, (filepath, register, sub_reg) in file_mapping.items():
    if not os.path.exists(filepath):
        print(f"Skipping missing file: {filepath}")
        continue

    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['Source_Title'] = title
    df['Register'] = register
    if 'Sub_Register' not in df.columns:
        df['Sub_Register'] = sub_reg

    # Map column names if they lack the '_NLP' suffix
    col_rename = {
        'Copula_Status': 'Copula_Status_NLP',
        'Polarity': 'Polarity_NLP',
        'Predicate_Type_Final': 'Predicate_Type_Final_NLP',
        'Notes': 'NLP_Root_Word_Notes'
    }
    df = df.rename(columns=col_rename)

    columns_to_keep = [
        "Sentence_ID", "Register", "Sub_Register", "Source_Title", "Raw_Sentence",
        "Copula_Status_NLP", "Polarity_NLP", "Predicate_Type_Final_NLP", "NLP_Root_Word_Notes"
    ]
    existing_cols = [c for c in columns_to_keep if c in df.columns]
    compiled_dfs.append(df[existing_cols])

if not compiled_dfs:
    print("Error: No valid CSV files were loaded.")
else:
    verification_df = pd.concat(compiled_dfs, ignore_index=True)
    verification_df['Master_Sentence_ID'] = [f"VERIFY_{i:04d}" for i in range(1, len(verification_df) + 1)]

    output_path = os.path.join(BASE_DIR, "verification.csv")
    verification_df.to_csv(output_path, index=False, encoding='utf-8-sig')

    print("=" * 60)
    print(f"SUCCESS: Compiled verification dataset created with {len(verification_df)} rows.")
    print(f"Saved at: '{output_path}'")
    print(verification_df[['Master_Sentence_ID', 'Source_Title', 'Copula_Status_NLP', 'Polarity_NLP', 'Predicate_Type_Final_NLP']].head())
    print("=" * 60)