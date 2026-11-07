import string
import pysrt
import pandas as pd
import stanza
from indicnlp.tokenize import sentence_tokenize

# 1. Initialize Stanza Pipeline ONCE globally (no repeated downloads)
print("Initializing Stanza Hindi Pipeline into memory...")
nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma,depparse', verbose=False, use_gpu=False)

def process_hindi_subtitles_fast(srt_file_path, output_csv_path="extracted_subtitles_nlp4.csv", movie_title="Movie_01"):
    # 2. Parse SRT file using pysrt
    subs = pysrt.open(srt_file_path, encoding='utf-8')
    clean_lines = [sub.text_without_tags.strip() for sub in subs if sub.text_without_tags.strip()]
    full_text = " ".join(clean_lines)

    # 3. Split into sentences using Indic NLP Library
    raw_sentences = sentence_tokenize.sentence_split(full_text, lang='hi')

    # Linguistic Marker Sets for Fast Filtering
    present_copulas = {"है", "हैं", "हूँ", "हुं", "हो"}
    past_markers = {"था", "थी", "थे", "थीं", "हुआ", "हुई", "हुए", "हुईं", "गया", "गयी", "गये", "गई"}
    future_auxiliaries = {"होगा", "होगी", "होंगे", "होगीं", "होऊंगा", "होओगे"}
    negative_markers = {"नहीं", "न", "ना", "मत", "बिना", "बगैर"}
    locative_postpositions = {
        "में", "पर", "पे", "तक", "के पास", "के अंदर", "के भीतर", 
        "के बाहर", "के ऊपर", "के नीचे", "के सामने", "के पीछे", 
        "के बीच", "के यहाँ", "की तरफ", "की ओर"
    }

    hindi_punctuation = string.punctuation + "।॥''\"\"“”"

    filtered_candidates = []

    print("Pre-filtering present-tense candidates...")
    for raw in raw_sentences:
        sentence = raw.strip()
        raw_words = sentence.split()
        words = [w.strip(hindi_punctuation) for w in raw_words if w.strip(hindi_punctuation)]

        if len(words) < 3:
            continue

        word_set = set(words)

        if not word_set.isdisjoint(past_markers) or not word_set.isdisjoint(future_auxiliaries):
            continue
        if any(w.endswith(("एगा", "एगी", "एंगे", "ेंगी", "ऊंगा", "ओगे")) for w in words):
            continue

        filtered_candidates.append((sentence, word_set))

    print(f"Found {len(filtered_candidates)} candidate present-tense sentences. Running batched Stanza NLP...")

    # 4. Prepare Stanza Documents for BATCH PROCESSING
    in_docs = [stanza.Document([], text=item[0]) for item in filtered_candidates]
    out_docs = nlp(in_docs)  # Runs all neural parsing in a single batched operation!

    annotated_rows = []

    for sentence_id, ((sentence, word_set), doc) in enumerate(zip(filtered_candidates, out_docs), start=1):
        copula_status = 1 if not word_set.isdisjoint(present_copulas) else 0

        # Polarity
        is_negative = False
        for sent in doc.sentences:
            for token in sent.words:
                if token.deprel == 'neg' or token.text in negative_markers:
                    is_negative = True
                    break
        polarity = "Negative" if is_negative else "Affirmative"

        # Predicate Classification
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

        annotated_rows.append({
            "Sentence_ID": f"SUB_{sentence_id:03d}",
            "Register": "Subtitle",
            "Movie_Title": movie_title,
            "Raw_Sentence": sentence,
            "Copula_Status": copula_status,
            "Polarity": polarity,
            "Predicate_Type_Hint": f"NLP Parsed ({root_token.upos if root_token else 'UNK'})",
            "Predicate_Type_Final": predicate_type,
            "Notes": f"Root Word: {root_token.text if root_token else 'N/A'}"
        })

    df = pd.DataFrame(annotated_rows)
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    print(f"Extraction complete! Saved {len(df)} tagged sentences to '{output_csv_path}'.")
    return df


if __name__ == "__main__":
    process_hindi_subtitles_fast(
        srt_file_path=r"C:\Users\nitis\Downloads\Mirzapur.The.Movie.2026.Extended.1080p.AMZN.WEB-DL.DD+5.1.H.264-playWEB.hin.srt",
        output_csv_path="extracted_subtitles_nlp4.csv",
        movie_title="Mirzapur"
    )