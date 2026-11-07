import string
import pysrt
import pandas as pd
from indicnlp.tokenize import sentence_tokenize

def process_hindi_subtitles_pysrt(srt_file_path, output_csv_path="extracted_subtitles3.csv", movie_title="Movie_01"):
    """
    Parses an SRT file, strips punctuation, excludes past/future tenses using expanded 
    marker sets, auto-detects present copula status, polarity, and locative heuristics.
    """
    # 1. Parse SRT file using pysrt
    subs = pysrt.open(srt_file_path, encoding='utf-8')
    clean_lines = [sub.text_without_tags.strip() for sub in subs if sub.text_without_tags.strip()]
    full_text = " ".join(clean_lines)

    # 2. Split into sentences using Indic NLP Library
    raw_sentences = sentence_tokenize.sentence_split(full_text, lang='hi')

    # 3. Exhaustive Linguistic Marker Sets
    present_copulas = {"है", "हैं", "हूँ", "हुं", "हो"}
    
    past_markers = {
        "था", "थी", "थे", "थीं", 
        "हुआ", "हुई", "हुए", "हुईं", 
        "गया", "गयी", "गये", "गई"
    }
    
    future_auxiliaries = {
        "होगा", "होगी", "होंगे", "होगीं", "होऊंगा", "होओगे"
    }
    
    negative_markers = {"नहीं", "न", "ना", "मत", "बिना", "बगैर"}
    
    locative_postpositions = [
        "में", "पर", "पे", "तक", "के पास", "के अंदर", "के भीतर", 
        "के बाहर", "के ऊपर", "के नीचे", "के सामने", "के पीछे", 
        "के बीच", "के यहाँ", "की तरफ", "की ओर"
    ]

    # Hindi/Devanagari punctuation to strip from token boundaries
    hindi_punctuation = string.punctuation + "।॥''\"\"“”"

    annotated_rows = []
    sentence_id = 1

    for raw in raw_sentences:
        sentence = raw.strip()
        
        # Tokenize and strip punctuation from individual words
        raw_words = sentence.split()
        words = [w.strip(hindi_punctuation) for w in raw_words if w.strip(hindi_punctuation)]

        # Skip noise, exclamations, or short fragments (< 3 words)
        if len(words) < 3:
            continue

        word_set = set(words)

        # Exclude past tense sentences
        if not word_set.isdisjoint(past_markers):
            continue

        # Exclude future tense sentences (standalone auxiliaries or verb suffixes)
        if not word_set.isdisjoint(future_auxiliaries):
            continue
        if any(w.endswith(("एगा", "एगी", "एंगे", "ेंगी", "ऊंगा", "ओगे")) for w in words):
            continue

        # Rule 1: Copula Status (1 if any present copula is in word_set, else 0)
        copula_status = 1 if not word_set.isdisjoint(present_copulas) else 0

        # Rule 2: Polarity (Negative vs. Affirmative)
        polarity = "Negative" if not word_set.isdisjoint(negative_markers) else "Affirmative"

        # Rule 3: Locative/Existential Heuristic
        is_locative = any(postpos in sentence for postpos in locative_postpositions)
        predicate_hint = "Locative/Existential" if is_locative else "Nominal/Adjectival (Check Manually)"

        annotated_rows.append({
            "Sentence_ID": f"SUB_{sentence_id:03d}",
            "Register": "Subtitle",
            "Movie_Title": movie_title,
            "Raw_Sentence": sentence,
            "Copula_Status": copula_status,
            "Polarity": polarity,
            "Predicate_Type_Hint": predicate_hint,
            "Predicate_Type_Final": "",  # Manual Review: Equational, Adjectival, Locative
            "Notes": ""
        })
        sentence_id += 1

    # 4. Export to CSV with UTF-8 BOM encoding for Excel support
    df = pd.DataFrame(annotated_rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"Extraction complete! Processed {len(df)} candidate sentences into '{output_csv_path}'.")
    return df


# Execution call
if __name__ == "__main__":
    process_hindi_subtitles_pysrt(
        srt_file_path=r"C:\Users\nitis\Downloads\TaareZameenPar.2.42.32.SHUDDH.Hindi.srt",
        output_csv_path="extracted_subtitles3.csv",
        movie_title="Taare_zameen_par"
    )