# Cross-Linguistic Computational Analysis of Copula Deletion: A Cross-Register Investigation of Indo-Aryan and Dravidian Languages

[![Course](https://img.shields.io/badge/Course-HS5652%20Language%2C%20Cognition%20%26%20Computation-blue.svg)](https://github.com/nitishgupt100/language_cognition_computation)
[![Institution](https://img.shields.io/badge/Institution-IIT%20Madras-orange.svg)](https://www.iitm.ac.in)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-green.svg)](requirements.txt)
[![Status](https://img.shields.io/badge/Paper%20Status-Complete%20(48%20Pages)-success.svg)](Report/report.pdf)

> **Course:** HS5652 &mdash; Language, Cognition and Computation  
> **Department:** Department of Humanities and Social Sciences, Indian Institute of Technology Madras  
> **Authors:** Sairam &middot; Varsha &middot; Arya &middot; Nitish &middot; Swapnil &middot; Karthik  
> **Repository Link:** [https://github.com/nitishgupt100/language_cognition_computation](https://github.com/nitishgupt100/language_cognition_computation)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Key Deliverables & Quick Links](#key-deliverables--quick-links)
3. [Research Hypotheses](#research-hypotheses)
4. [Master Empirical Results Table](#master-empirical-results-table)
5. [Presentation Feedback & Linguistic Auditing](#presentation-feedback--linguistic-auditing)
6. [Repository Structure](#repository-structure)
7. [Step-by-Step Reproduction Guide](#step-by-step-reproduction-guide)
8. [Theoretical Mapping to HS5652 Course Concepts](#theoretical-mapping-to-hs5652-course-concepts)
9. [References](#references)

---

## Project Overview

In natural language syntax, a **copula** is a linking element&mdash;prototypically a verb such as English *be (is, am, are)* or Hindi होना (*hona: hai, hain, hoon, ho*)&mdash;that connects a subject noun phrase to a non-verbal predicate complement (predicate nominals, predicate adjectives, and locative/existential phrases). Unlike canonical transitive or intransitive verbs, a pure copula contributes minimal independent lexical meaning, serving primarily as a morphosyntactic carrier for tense, aspect, mood, person-number agreement, and polarity.

While languages across the world vary in whether they require an overt copula or license a bare juxtaposition (**zero-copula**), South Asian languages present an exceptionally rich empirical landscape:
* **Indo-Aryan Languages:** Hindi (Central Indo-Aryan), Assamese (Eastern Magadhan), and Bengali (Eastern Magadhan).
* **Dravidian Languages:** Telugu (South-Central Dravidian) and Tamil (Southern Dravidian).

This repository contains the complete codebase, datasets, executed Jupyter notebooks, publication figures, and comprehensive research paper report for our computational and corpus investigation across all five languages ($N_{\text{clean}} = 3,781$ clauses, plus $1,063$ unseen Hindi held-out verification clauses).

---

## Key Deliverables & Quick Links

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Complete Research Paper** | [`Report/report.pdf`](Report/report.pdf) | Full academic paper (14-page core article + 34-page sentence verification compendium) formatted in publication standard with full statistical tests, methodology, course correlation table, and references. |
| **Sentence-Level Verification Compendium** | [`materials/Copula_Deletion_Cross_Linguistic_Sample_Sentences.pdf`](materials/Copula_Deletion_Cross_Linguistic_Sample_Sentences.pdf) | Standalone reference PDF containing verified sample sentences across all 5 languages, annotated with H1 (Register), H2 (Predicate Type), H3 (Polarity), and assigned Copula Status. |
| **Notebook 1: Extraction & Compilation** | [`materials/01_extraction_and_compilation.ipynb`](materials/01_extraction_and_compilation.ipynb) | End-to-end extraction pipeline: subtitle SRT parser, Wikisource scrapers, Stanza NLP dependency tagger, and multi-source CSV compiler for Hindi and Telugu. |
| **Notebook 2: Cross-Linguistic Analysis** | [`materials/02_cross_linguistic_analysis.ipynb`](materials/02_cross_linguistic_analysis.ipynb) | Complete statistical analysis, contingency tables ($\chi^2$, Cram&eacute;r's $V$, Fisher's exact), multivariate logistic regressions, and 300-DPI dashboards for each language in order: Hindi, Assamese, Bengali, Telugu, and Tamil. |
| **Master Summary Metrics** | [`materials/data/cross_linguistic_summary_metrics.csv`](materials/data/cross_linguistic_summary_metrics.csv) | CSV containing exact cross-linguistic deletion percentages across all five languages for H1, H2, and H3. |
| **Publication Dashboards** | [`materials/figures/`](materials/figures/) | Six 300-DPI publication figures (`01_hindi_copula_dashboard.png` through `06_cross_linguistic_summary.png`). |

---

## Research Hypotheses

Our cross-linguistic investigation tests three central structural and sociolinguistic hypotheses:

1. **H1 &middot; Register & Diglossic Conditioning:**  
   Informal conversational dialogue (*Spoken Film Subtitles / Informal Dialogue*) exhibits a higher rate of present-tense copula deletion than formal written prose (*Literary Narrative / Formal Prose*).
2. **H2 &middot; Universal Semantic Predicate Type Hierarchy (Curnow, 1999; Stassen, 1997):**  
   Copula deletion varies systematically across predicate classes according to lexical contentfulness:
   $$\texttt{Equational / Nominal} \;\approx\; \texttt{Adjectival} \;\gg\; \texttt{Locative / Existential}$$
3. **H3 &middot; Polarity & Morphological Negation-Copula Fusion:**  
   Negative clauses exhibit a categorical or significant resistance to copula deletion compared to affirmative clauses due to synthetic morphological fusion between negation and the copular head.

---

## Master Empirical Results Table

The following table summarizes the verified empirical distributions across all five languages ($N_{\text{clean}} = 3,781$). Every count, numerator, and denominator is mathematically consistent across our analysis scripts, notebooks, figures, and research paper:

| # | Language & Family | Clean $N$ | Overall Deletion | H1: Register Breakdown (Informal vs. Formal) | H2: Predicate Type Hierarchy (Equational / Adjectival / Locative) | H3: Polarity Effect (Affirmative vs. Negative) | Negative Copula Mechanism |
| :-: | :--- | :-: | :-: | :--- | :--- | :--- | :--- |
| **1** | **Hindi**<br>*Indo-Aryan (Central)* | $3,089$<br>*(+1,063 verif.)* | **$34.2\%$**<br>($1,056/3,089$) | Film Dialogue: **$36.3\%$** ($977/2,692$)<br>Lit. Dialogue: **$26.8\%$** ($26/97$)<br>Lit. Narrative: **$17.7\%$** ($53/300$)<br>$\chi^2 = 40.66, p = 1.81\times 10^{-10}$ | Equational: **$39.7\%$** ($785/1,977$)<br>Locative: **$26.8\%$** ($191/714$)<br>Adjectival: **$20.1\%$** ($80/398$)<br>$\chi^2 = 79.42, p = 5.68\times 10^{-18}$ | Affirmative: **$34.3\%$** ($878/2,561$)<br>Negative: **$33.7\%$** ($178/528$)<br>$\chi^2 = 0.04, p = 0.840$ | Standalone particle नहीं (*nahin*) carries negation independently, licensing free drop of है (*hai*). |
| **2** | **Assamese**<br>*Indo-Aryan (Eastern)* | $97$<br>*(from 101 raw)* | **$71.1\%$**<br>($69/97$) | Informal Dialogue: **$84.3\%$** ($43/51$)<br>Formal Prose: **$56.5\%$** ($26/46$)<br>$\chi^2 = 7.79, p = 0.0052, V = 0.283$<br>*(Locative Gap: $+66.0\text{ pp}$)* | Equational: **$84.8\%$** ($39/46$)<br>Adjectival: **$84.0\%$** ($21/25$)<br>Locative: **$34.6\%$** ($9/26$)<br>$\chi^2 = 23.08, p = 9.76\times 10^{-6}$ | Affirmative: **$86.2\%$** ($69/80$)<br>Negative: **$0.0\%$** ($0/17$)<br>Fisher exact $p = 5.68\times 10^{-12}$ | Categorical block: Fused synthetic negative copulas নহয় (*nohoy*) and নাই (*nai*) cannot be deleted. |
| **3** | **Bengali**<br>*Indo-Aryan (Eastern)* | $373$<br>*(8 sources)* | **$52.0\%$**<br>($194/373$) | Spoken Subtitles: **$43.6\%$** ($44/101$)<br>Literary Prose: **$55.1\%$** ($150/272$)<br>$\chi^2 = 3.51, p = 0.061$<br>*(Grammaticalized in affirmatives)* | Equational: **$78.2\%$** ($104/133$)<br>Adjectival: **$67.8\%$** ($61/90$)<br>Locative: **$19.3\%$** ($29/150$)<br>$\chi^2 = 109.67, p = 1.53\times 10^{-24}$ | Affirmative: **$73.8\%$** ($194/263$)<br>Negative: **$0.0\%$** ($0/110$)<br>Fisher exact $p = 5.15\times 10^{-47}$ | Categorical block: Fused negative copulas নয় (*noy*) and নেই/নাই (*nei/nai*) are obligatorily overt. |
| **4** | **Telugu**<br>*Dravidian (South-Central)* | $116$ Manual<br>*(759 Auto)* | **$47.4\%$**<br>($55/116$)<br>*(Auto: 60.3%)* | Film Dialogue: **$42.9\%$** ($21/49$)<br>Lit. Dialogue: **$55.6\%$** ($20/36$)<br>Lit. Narrative: **$45.2\%$** ($14/31$)<br>Fisher exact $p = 1.000$ | Equational: **$79.2\%$** ($42/53$; $100\%$ aff)<br>Adjectival: **$45.8\%$** ($11/24$)<br>Locative: **$5.1\%$** ($2/39$)<br>$\chi^2 = 49.53, p = 1.76\times 10^{-11}$ | Affirmative: **$65.5\%$** ($55/84$)<br>Negative: **$0.0\%$** ($0/32$)<br>Fisher exact $p = 9.56\times 10^{-12}$ | Categorical block: Mandatory negative copulas కాదు (*kaadu*) and లేదు (*ledu*) block deletion. |
| **5** | **Tamil**<br>*Dravidian (Southern)* | $106$<br>*(from 121 raw)* | **$70.8\%$**<br>($75/106$) | Informal Dialogue: **$81.1\%$** ($43/53$)<br>Formal Prose: **$60.4\%$** ($32/53$)<br>$\chi^2 = 4.56, p = 0.0327, V = 0.207$<br>*(Locative Gap: $+58.8\text{ pp}$)* | Adjectival: **$85.7\%$** ($24/28$)<br>Equational: **$84.1\%$** ($37/44$)<br>Locative: **$41.2\%$** ($14/34$)<br>$\chi^2 = 21.19, p = 2.51\times 10^{-5}$ | Affirmative: **$85.2\%$** ($75/88$)<br>Negative: **$0.0\%$** ($0/18$)<br>Fisher exact $p = 2.14\times 10^{-12}$ | Categorical block: Fused negative copulas அல்ல (*alla*) and இல்லை (*illai*) cannot be omitted. |

---

## Presentation Feedback & Linguistic Auditing

During our classroom project presentation, the professor correctly raised an intuitive concern: **why did the initial raw NLP pipeline indicate that formal Hindi literary prose deleted the copula more frequently ($59.8\%$) than casual film subtitles ($52.7\%$)?** Native Hindi speakers immediately recognize that dropping *hai* is standard in rapid informal speech and rare in formal prose.

Our subsequent 4-stage linguistic audit resolved this discrepancy completely:
1. **The Root Cause:** Standard dependency parsers (Stanza Universal Dependencies) search for overt copula lemmas (*hai, hain, hoon, ho*). In literary prose, authors narrate actions in the past/perfective aspect (*gaya, hua, bola, dekha*), habitual aspect (*karta tha*), or subjunctive/imperative moods lacking *hai*. The raw script tagged any sentence lacking *hai* as `Copula_Deleted = 1`, mistakenly treating $320$ finite action verbs and $27$ unquoted dialogue lines as "deleted copulas".
2. **The Corrected Distribution:** When sentences were audited to restrict analysis strictly to **verified present-tense non-verbal clauses** ($N_{\text{clean}} = 3,089$), the artifact evaporated:
   $$\text{Film Dialogue } (36.3\%) > \text{ Literary Dialogue } (26.8\%) > \text{ Literary Narrative } (17.7\%) \quad (\chi^2 = 40.66, p = 1.81\times 10^{-10})$$
3. **Out-of-Sample Validation:** This distribution was independently validated on an unseen held-out corpus ($N=1,063$) consisting of *Mirzapur* subtitles ($26.4\%$ deletion) vs. Premchand's *Poos Ki Raat* narrative prose ($4.5\%$ deletion).
4. **Telugu Dual-Track:** A parallel issue in Telugu automated heuristic parsing (defaulting unparsed clauses to Equational $88.4\%$) was rectified through our manually verified reference set ($N=116$), which revealed the true hierarchy ($79.2\%$ Equational $>$ $45.8\%$ Adjectival $>$ $5.1\%$ Locative).

Full sentence-level verification tables are provided in [Appendix A of `Report/report.pdf`](Report/report.pdf#page=15) and [`materials/Copula_Deletion_Cross_Linguistic_Sample_Sentences.pdf`](materials/Copula_Deletion_Cross_Linguistic_Sample_Sentences.pdf).

---

## Repository Structure

```text
language_cognition_computation/
├── README.md                                                 # Project overview, reproduction guide, and master results
├── requirements.txt                                          # Python package dependencies
│
├── Report/                                                   # Final Research Paper & Verification Report
│   └── report.pdf                                            # Complete 48-page publication PDF with Appendix A
│
└── materials/                                                # Core project assets, data, and scripts
    ├── README.md                                             # Technical reference guide for materials
    ├── 01_extraction_and_compilation.ipynb                   # Notebook 1: Extraction & CSV compilation pipeline
    ├── 02_cross_linguistic_analysis.ipynb                    # Notebook 2: 5-language statistical & visual analysis
    ├── Copula_Deletion_Cross_Linguistic_Sample_Sentences.pdf # Standalone sample sentences compendium PDF
    │
    ├── figures/                                              # 300-DPI publication figures
    │   ├── 01_hindi_copula_dashboard.png                     # Hindi 4-panel dashboard (N = 3,089)
    │   ├── 01b_hindi_verification_dashboard.png              # Hindi held-out verification dashboard (N = 1,063)
    │   ├── 02_assamese_copula_dashboard.png                  # Assamese 4-panel dashboard (N = 97)
    │   ├── 03_bengali_copula_dashboard.png                   # Bengali 4-panel dashboard (N = 373)
    │   ├── 04_telugu_copula_dashboard.png                    # Telugu dual-track dashboard (N = 759 vs 116)
    │   ├── 05_tamil_copula_dashboard.png                     # Tamil 4-panel dashboard (N = 106)
    │   └── 06_cross_linguistic_summary.png                   # Master 3-panel cross-linguistic synthesis
    │
    ├── reference/                                            # Presentation slides & initial proposal
    │   ├── Copula Deletion Assignment Proposal.pdf
    │   └── HS5652_Assignment_Copula_Deletion.pdf
    │
    ├── scripts/                                              # Standalone Python analysis modules
    │   ├── setup_stanza.py                                   # Stanza Hindi pipeline installer
    │   ├── subtitles_nlp.py                                  # Hindi subtitle parser (.srt -> NLP CSV)
    │   ├── subtitles_base.py                                 # Hindi rule-based extractor
    │   ├── literary.py                                       # Hindi literary text segmenter (.txt -> CSV)
    │   ├── cross_register.py                                 # Compiles 6 Hindi sources into Cross_register.csv
    │   ├── cross_register_hypothesis_verify.py               # Compiles verification corpus
    │   ├── hindi_analysis.py                                 # Auditing & stats for Hindi 6-source corpus
    │   ├── hindi_verification_analysis.py                    # Verification stats for Mirzapur & Poos Ki Raat
    │   ├── assamese_analysis.py                              # Assamese statistical analysis pipeline
    │   ├── bengali_analysis.py                               # Bengali statistical analysis pipeline
    │   ├── telugu_analysis.py                                # Telugu automated vs manual pipeline
    │   └── tamil_analysis.py                                 # Tamil diglossic analysis pipeline
    │
    └── data/                                                 # Raw text, subtitle SRTs, and clean CSVs
        ├── cross_linguistic_summary_metrics.csv              # Master synthesis table (H1, H2, H3 metrics)
        ├── hindi/                                            # Hindi raw texts, extracted CSVs, clean datasets
        ├── assamese/                                         # Formal prose, informal dialogue, clean dataset
        ├── bengali/                                          # Curated 8-source Bengali non-verbal dataset
        ├── telugu/                                           # Automated (759) and manual verified (116) datasets
        └── tamil/                                            # Formal prose, colloquial dialogue, clean dataset
```

---

## Step-by-Step Reproduction Guide

### 1. Environment Setup

Clone the repository and install the required dependencies inside a clean virtual environment:

```bash
# Clone the repository
git clone https://github.com/nitishgupt100/language_cognition_computation.git
cd language_cognition_computation

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Running Language Analysis Pipelines

All individual language analysis scripts can be executed standalone from the root directory:

```bash
# Hindi Main Corpus Analysis (generates Cross_register_clean_nonverbal.csv & figures)
python materials/scripts/hindi_analysis.py

# Hindi Held-Out Verification Corpus Analysis (Mirzapur + Poos Ki Raat)
python materials/scripts/hindi_verification_analysis.py

# Assamese Analysis (audits 97 clauses and prints contingency stats & odds ratios)
python materials/scripts/assamese_analysis.py

# Bengali Analysis (evaluates 373 clauses across 8 sources)
python materials/scripts/bengali_analysis.py

# Telugu Dual-Track Analysis (evaluates 759 automated vs. 116 manual clauses)
python materials/scripts/telugu_analysis.py

# Tamil Diglossic Analysis (evaluates 106 clauses across formal & informal registers)
python materials/scripts/tamil_analysis.py
```

### 3. Running Interactive Jupyter Notebooks

Launch Jupyter Notebook to inspect or re-execute the complete interactive pipeline:

```bash
jupyter notebook
```

- Open **`materials/01_extraction_and_compilation.ipynb`** to inspect the data extraction and compilation from raw SRT and text files.
- Open **`materials/02_cross_linguistic_analysis.ipynb`** to execute the statistical models, generate summary tables, and render all publication plots inline.

---

## Theoretical Mapping to HS5652 Course Concepts

In accordance with course objectives, our computational and empirical findings connect directly to core cognitive and linguistic theories taught in **HS5652: Language, Cognition and Computation**:

| Course Concept | Theoretical Framework in Class | Manifestation in Our Copula Deletion Study |
| :--- | :--- | :--- |
| **Competence vs. Performance (I-Language vs. E-Language)** | Chomsky (1965, 1986); underlying syntactic knowledge vs. observable surface production. | Zero-copula in Dravidian (Telugu, Tamil) and Eastern Indo-Aryan (Bengali, Assamese) is an I-language competence rule for affirmative nominal identity, whereas in Hindi it is an E-language stylistic performance variable conditioned by speech rate and register. |
| **Child Language & Telegraphic Speech** | Omission of functional categories (determiners, inflections, copulas) in early acquisition. | In Hindi, adults selectively omit the copular functional head *hai* in informal spoken register, mirroring child telegraphic speech where content words (*Pahlwan, sher, dhol*) are prioritized over functional carriers. |
| **Usage-Based Linguistics & Entrenchment** | Bybee (2006), Tomasello (2003); grammatical constructions emerge through frequency and cognitive routinization. | Conversational idioms and frequent collocations (e.g., Hindi *kuch parvaah nahin*, Tamil *naan veetla*, Bengali *gramgulo shob kothay*) achieve cognitive entrenchment as bare zero-copula constructions. |
| **Working Memory & Processing Efficiency** | Gibson (1998), Hawkins (2004); speakers minimize cognitive production and parsing cost. | Rapid spoken dialogue significantly favors zero-copula because communicators omit predictable linking morphemes under time pressure when semantic predication is already transparent. |
| **Construal & Perspective Taking** | Langacker (1987), Croft (2001); grammatical alternations package the same conceptual scene from different perspectives. | In Tamil and Assamese, choosing between zero vs. overt copula encodes permanent inherent identity (zero) vs. temporary state or dynamic manifestation (overt *aagum / hoy*). |
| **Ontological Categorization & LOTH** | Fodor (1975), Jackendoff (1983); Language of Thought and ontological boundary between attribution and existence. | All five Indian languages strictly bifurcate conceptual attribution (nominal/adjectival zero copula) from physical spatial existence (which demands an overt existential verb *ase, aachhe, undi, irukku*). |
| **Descriptive Accuracy vs. Psychological Reality** | Chomskyan critique of naive corpus statistics; raw surface frequencies can mislead without theory. | Explains the presentation discrepancy: raw neural dependency parsing mistakenly counted 320 past-tense action verbs as zero copulas, yielding a psychologically unreal $59.8\%$ literary deletion rate until theory-guided auditing corrected it. |

---

## References

1. **Benmamoun, E.** (2000). *The Feature Structure of Functional Categories: A Comparative Study of Arabic Dialects*. Oxford University Press.
2. **Chatterji, S. K.** (1926). *The Origin and Development of the Bengali Language*. Calcutta University Press.
3. **Curnow, T. J.** (1999). Towards a cross-linguistic typology of copula constructions. In *Proceedings of the 1999 Conference of the Australian Linguistic Society*.
4. **Emeneau, M. B.** (1956). India as a linguistic area. *Language*, 32(1), 3&ndash;16.
5. **Kachru, Y.** (2006). *Hindi*. London Oriental and African Language Library, Vol. 12. John Benjamins.
6. **Krishnamurti, B.** (2003). *The Dravidian Languages*. Cambridge University Press.
7. **Lehmann, T.** (1989). *A Grammar of Modern Tamil*. Pondicherry Institute of Linguistics and Culture.
8. **Masica, C. P.** (1991). *The Indo-Aryan Languages*. Cambridge University Press.
9. **Pustet, R.** (2003). *Copulas: Universals in the Categorization of the Lexicon*. Oxford University Press.
10. **Qi, P., et al.** (2020). Stanza: A Python natural language processing toolkit for many human languages. In *Proceedings of the 58th Annual Meeting of the ACL: System Demonstrations*, 101&ndash;108.
11. **Stassen, L.** (1997). *Intransitive Predication*. Oxford University Press.
12. **Stassen, L.** (2013). Zero copula for predicate nominals. In *The World Atlas of Language Structures Online*. Max Planck Institute for Evolutionary Anthropology.

---

*For queries or reproduction support, please refer to the project repository:* [https://github.com/nitishgupt100/language_cognition_computation](https://github.com/nitishgupt100/language_cognition_computation)
