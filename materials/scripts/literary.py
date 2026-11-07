import string
import os
import re
import pandas as pd
import stanza
from indicnlp.tokenize import sentence_tokenize

def process_hindi_literary_txt(txt_path, output_csv_path="formal_idgah.csv", story_title="idgah"):
    """
    Parses a UTF-8 encoded Hindi literary text file (.txt), separates direct character dialogue 
    from authorial narrative prose, filters present-tense sentences, and uses Stanza NLP 
    to tag Copula Status, Polarity, and Predicate Types for formal register analysis.
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Text file not found at: '{txt_path}'")

    print(f"Step 1: Reading clean Devanagari text from '{txt_path}'...")
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Diagnostic check: Print first 200 characters
    print("\n" + "="*50)
    print("DIAGNOSTIC - First 200 characters from .txt file:")
    print(repr(full_text[:200]))
    print("="*50 + "\n")

    # Step 2: Initialize Stanza Hindi NLP Pipeline
    print("Step 2: Loading Stanza Hindi NLP Pipeline into memory...")
    stanza.download('hi', processors='tokenize,pos,lemma,depparse', verbose=False)
    nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma,depparse', verbose=False)

    # Step 3: Sentence Tokenization using Indic NLP
    raw_sentences = sentence_tokenize.sentence_split(full_text, lang='hi')

    # Linguistic Marker Sets for Tense Filtering
    present_copulas = {"है", "हैं", "हूँ", "हुं", "हो"}
    past_markers = {"था", "थी", "थे", "थीं", "हुआ", "हुई", "हुए", "हुईं", "गया", "गयी", "गये", "गई"}
    future_auxiliaries = {"होगा", "होगी", "होंगे", "होगीं", "होऊंगा", "होओगे"}
    negative_markers = {"नहीं", "न", "ना", "मत", "बिना", "बगैर"}
    locative_postpositions = {
        "में", "पर", "पे", "तक", "के पास", "के अंदर", "के भीतर", 
        "के बाहर", "के ऊपर", "के नीचे", "के सामने", "के पीछे", 
        "के बीच", "के यहाँ", "की तरफ", "की ओर"
    }

    hindi_punctuation = string.punctuation + "।॥''\"\"“”«»"

    filtered_candidates = []

    print("Step 3: Filtering present-tense sentences & categorizing dialogue vs narrative...")
    for raw in raw_sentences:
        sentence = raw.strip()
        
        # Detect if sentence is direct character speech inside quotation marks
        is_dialogue = bool(re.search(r'["\'“«].*?["\'”»]', sentence)) or sentence.startswith(('"', "'", "“", "«"))
        
        raw_words = sentence.split()
        words = [w.strip(hindi_punctuation) for w in raw_words if w.strip(hindi_punctuation)]

        # Skip short fragments or title noise (< 3 words)
        if len(words) < 3:
            continue

        word_set = set(words)

        # Exclude past and future tenses to isolate present copular clauses
        if not word_set.isdisjoint(past_markers) or not word_set.isdisjoint(future_auxiliaries):
            continue
        if any(w.endswith(("एगा", "एगी", "एंगे", "ेंगी", "ऊंगा", "ओगे")) for w in words):
            continue

        filtered_candidates.append((sentence, word_set, is_dialogue))

    print(f"Extracted {len(filtered_candidates)} candidate present-tense sentences.")

    columns = [
        "Sentence_ID", "Register", "Sub_Register", "Movie_Title", "Raw_Sentence",
        "Copula_Status_Base", "Copula_Status_NLP", "Copula_Status_Match",
        "Polarity_Base", "Polarity_NLP", "Polarity_Match",
        "Predicate_Type_Hint_Base", "Predicate_Type_Hint_NLP",
        "Predicate_Type_Final_Base", "Predicate_Type_Final_NLP", "NLP_Root_Word_Notes"
    ]

    if not filtered_candidates:
        print("WARNING: No present-tense sentences matched the criteria.")
        df = pd.DataFrame(columns=columns)
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        return df

    print("Step 4: Running Stanza Neural Dependency Parsing...")
    in_docs = [stanza.Document([], text=item[0]) for item in filtered_candidates]
    out_docs = nlp(in_docs)

    annotated_rows = []

    for sentence_id, ((sentence, word_set, is_dialogue), doc) in enumerate(zip(filtered_candidates, out_docs), start=1):
        copula_status = 1 if not word_set.isdisjoint(present_copulas) else 0

        # Polarity detection via Stanza dependency tree
        is_negative = any(
            token.deprel == 'neg' or token.text in negative_markers 
            for sent in doc.sentences for token in sent.words
        )
        polarity = "Negative" if is_negative else "Affirmative"

        # Predicate classification via Universal Dependencies
        root_token = None
        for sent in doc.sentences:
            for token in sent.words:
                if token.deprel == 'root':
                    root_token = token
                    break

        is_locative = any(postpos in sentence for postpos in locative_postpositions)

        if root_token:
            if is_locative or root_token.deprel == 'obl' or any(w.upos == 'ADP' for sent in doc.sentences for w in sent.words if w.head == root_token.id):
                predicate_type = "Locative/Existential"
            elif root_token.upos == 'ADJ':
                predicate_type = "Adjectival"
            elif root_token.upos in ['NOUN', 'PROPN', 'PRON']:
                predicate_type = "Equational/Nominal"
            elif root_token.upos == 'VERB':
                predicate_type = "Adjectival" if any(w.upos == 'ADJ' for sent in doc.sentences for w in sent.words) else "Equational/Nominal"
            else:
                predicate_type = "Equational/Nominal"
        else:
            predicate_type = "Locative/Existential" if is_locative else "Equational/Nominal"

        text_sub_type = "Literary Dialogue" if is_dialogue else "Literary Narrative"

        annotated_rows.append({
            "Sentence_ID": f"LIT_{sentence_id:04d}",
            "Register": "Literary",
            "Sub_Register": text_sub_type,
            "Movie_Title": story_title,
            "Raw_Sentence": sentence,
            "Copula_Status_Base": copula_status,
            "Copula_Status_NLP": copula_status,
            "Copula_Status_Match": True,
            "Polarity_Base": "Negative" if not word_set.isdisjoint(negative_markers) else "Affirmative",
            "Polarity_NLP": polarity,
            "Polarity_Match": ("Negative" if not word_set.isdisjoint(negative_markers) else "Affirmative") == polarity,
            "Predicate_Type_Hint_Base": "Locative/Existential" if is_locative else "Nominal/Adjectival (Check Manually)",
            "Predicate_Type_Hint_NLP": f"NLP Parsed ({root_token.upos if root_token else 'UNK'})",
            "Predicate_Type_Final_Base": "",
            "Predicate_Type_Final_NLP": predicate_type,
            "NLP_Root_Word_Notes": f"Root Word: {root_token.text if root_token else 'N/A'}"
        })

    # Save CSV with UTF-8 BOM encoding for proper Devanagari display in Excel/VS Code
    df = pd.DataFrame(annotated_rows, columns=columns)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print("="*60)
    print(f"SUCCESS: Processed {len(df)} clean Devanagari sentences.")
    print(f"Output saved to: '{output_csv_path}'")

    narrative_df = df[df['Sub_Register'] == 'Literary Narrative']
    dialogue_df = df[df['Sub_Register'] == 'Literary Dialogue']
    
    print(f"\n--- Summary for '{story_title}' ---")
    print(f"Total Sentences Extracted: {len(df)}")
    print(f"  • Authorial Narrative Prose: {len(narrative_df)} sentences")
    print(f"  • Direct Character Dialogue: {len(dialogue_df)} sentences")

    return df

if __name__ == "__main__":
    process_hindi_literary_txt(
        txt_path=r"C:\Users\nitis\Downloads\poos_ki_raat.txt",
        output_csv_path="formal_poos_ki_raat.csv",
        story_title="poos_ki_raat"
    )
