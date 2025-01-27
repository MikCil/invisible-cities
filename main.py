#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: calvino_extractor_italian.py

Description:
    This script reads in a JSON file of extracted Calvino text data, standardizes category
    labels (removing trailing numbers), uses SpaCy for robust Italian tokenization and
    lemmatization, and uses MultiWordNet (Open Multilingual WordNet in NLTK) to gather
    the relevant Italian lexemes for certain semantic fields (color, building, location, etc.).
    It then builds CSV files for network analysis (e.g., with Gephi).

    The script will generate:
        - nodes_all.csv and edges_all.csv for the entire dataset
        - nodes_<category>.csv and edges_<category>.csv for each category

Usage:
    python calvino_extractor_italian.py --input calvino_data.json --output out_directory

Prerequisites:
    - python -m spacy download it_core_news_sm
    - pip install nltk spacy
    - nltk.download('wordnet')
    - nltk.download('omw-1.4')
    
    (Ensure you have the necessary data installed for NLTK and SpaCy.)
"""

import json
import re
import os
import argparse
from collections import Counter, defaultdict

# NLTK imports
import nltk
from nltk.corpus import wordnet as wn

# We will use Italian language model from SpaCy
import spacy

# -----------------------------------------------------------------------------
# 1. LOAD SPACY FOR ITALIAN
# -----------------------------------------------------------------------------
# Make sure you have downloaded a SpaCy language model for Italian, e.g.:
#   python -m spacy download it_core_news_sm
try:
    nlp = spacy.load("it_core_news_lg")
except OSError:
    raise RuntimeError(
        "SpaCy Italian model not found. Please install via:\n"
        "  python -m spacy download it_core_news_sm"
    )

# -----------------------------------------------------------------------------
# 2. ACCENT CORRECTION HELPER
# -----------------------------------------------------------------------------
def accent_correction(text):
    """
    A simple approach to fix common accent confusion in Italian.
    E.g., 'piú' -> 'più', 'perche' -> 'perché', etc.
    This is by no means exhaustive; extend as needed.
    """
    # We’ll do a few targeted replacements. Adjust as desired.
    replacements = {
        "à": "à",  # keep
        "è": "è",  # keep
        "é": "é",  # keep
        "ì": "ì",  # keep
        "ò": "ò",  # keep
        "ù": "ù",  # keep
        # Wrong to correct:
        "piú": "più",
        "perche": "perché",
        "perchè": "perché",  # unify
        "cosi": "così",
        "cosi'": "così",
        "così'": "così",
    }
    # We apply them in a naive way. If one word is replaced, the next might be missed,
    # so do them all in one pass:
    for wrong, right in replacements.items():
        text = re.sub(r"\b{}\b".format(wrong), right, text, flags=re.IGNORECASE)
    return text

# -----------------------------------------------------------------------------
# 3. GATHER RELEVANT ITALIAN SYNSETS FROM OPEN MULTILINGUAL WORDNET
# -----------------------------------------------------------------------------

def build_synsets_dict():
    """
    Define the main conceptual domains you care about (color, building, location, etc.)
    using BOTH:
      1) Multiple Italian root words (via wn.synsets(..., lang="ita"))
      2) English offsets (e.g. 'color.n.01') to get coverage from the English-based hierarchy.
    
    Returns a dictionary:
      { 'lemma_in_italian' : 'label_for_domain' }
    """

    # 1. A utility to gather all Italian lemmas from an English synset and its hyponyms
    def gather_ita_lemmas_from_english_offset(eng_synset):
        """
        Traverse 'eng_synset' and all its hyponyms in English WordNet,
        collect all associated Italian lemmas from the Open Multilingual WordNet.
        Return a set of lemma strings in lowercase.
        """
        to_visit = [eng_synset]
        visited = set()
        ita_lemmas = set()

        while to_visit:
            current_syn = to_visit.pop()
            if current_syn in visited:
                continue
            visited.add(current_syn)

            # Gather all *Italian* lemmas for this synset
            for lemma_obj in current_syn.lemmas(lang="ita"):
                ita_lemmas.add(lemma_obj.name().lower().replace("_", " "))

            # Add hyponyms to the queue
            for hypo in current_syn.hyponyms():
                to_visit.append(hypo)

        return ita_lemmas

    # 2. A utility to gather all Italian lemmas from ALL senses (synsets) of a given Italian root word
    def gather_lemmas_from_italian_root(root_ita_word):
        """
        Returns the union of all lemmas found by exploring each synset
        that has 'root_ita_word' as an Italian lemma.
        """
        union_lemmas = set()

        # All synsets that contain this lemma in Italian
        candidate_synsets = wn.synsets(root_ita_word, lang="ita")
        for syn in candidate_synsets:
            union_lemmas.update(get_all_italian_hyponyms(syn, lang="ita"))

        return union_lemmas

    # 3. A helper that recursively collects Italian lemmas of a synset and its hyponyms
    def get_all_italian_hyponyms(synset, lang="ita"):
        results = set()
        # Include this synset's own lemmas (in Italian)
        for lemma_obj in synset.lemmas(lang=lang):
            results.add(lemma_obj.name().lower().replace("_", " "))
        
        # Recursively gather hyponyms
        for hyponym in synset.hyponyms():
            results.update(get_all_italian_hyponyms(hyponym, lang=lang))
        return results

    # 4. Now define your categories (labels) and the seeds:
    #    (a) English offsets (e.g. color.n.01) 
    #    (b) Italian root words for each domain
    category_defs = {
        "color": {
            "english_offset": "color.n.01",
            "italian_roots": ["colore", "tinta", "pigmento", "tonalità"]
        },
        "building": {
            "english_offset": "building.n.01",
            "italian_roots": ["edificio", "costruzione", "fabbricato"]
        },
        "location": {
            "english_offset": "location.n.01",
            "italian_roots": ["luogo", "posto", "sito"]
        },
        "scene": {
            "english_offset": "scene.n.01",
            "italian_roots": ["scena", "visione"]
        },
        "shape": {
            "english_offset": "shape.n.02",
            "italian_roots": ["forma", "figura"]
        },

        "time": {
            "english_offset": "time_period.n.01",
            "italian_roots": ["periodo"]
        },
        # Add more categories if needed
    }

    lemma_dict = {}  # { lemma -> label }

    # 5. For each category, merge:
    #    1) Lemmas from the English offset + hyponyms
    #    2) Lemmas from all Italian root words
    for label, info in category_defs.items():
        eng_offset_str = info["english_offset"]
        ita_roots = info["italian_roots"]

        # Attempt to load the English synset by offset name
        try:
            eng_syn = wn.synset(eng_offset_str)
            eng_ita_lemmas = gather_ita_lemmas_from_english_offset(eng_syn)
        except:
            eng_ita_lemmas = set()

        # Gather from the Italian root words
        ita_lemmas = set()
        for root_word in ita_roots:
            ita_lemmas.update(gather_lemmas_from_italian_root(root_word))

        # Merge them
        all_lemmas = eng_ita_lemmas.union(ita_lemmas)

        # Now store them in lemma_dict
        for lemma in all_lemmas:
            lemma_dict[lemma] = label

        # Also store each root_word itself (in case it wasn’t captured by the function)
        for root_word in ita_roots:
            lemma_dict[root_word] = label

    print(lemma_dict)
    return lemma_dict

# -----------------------------------------------------------------------------
# 4. TEXT PROCESSING HELPERS
# -----------------------------------------------------------------------------
def standardize_category_name(category_name):
    """
    Remove any trailing digits and trailing spaces. E.g.:
    "Le città e la memoria 1" -> "Le città e la memoria"
    """
    new_name = re.sub(r"\s?\d+$", "", category_name)
    return new_name.strip()

def tokenize_and_lemmatize_italian(text):
    text = accent_correction(text)
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if not token.is_punct and not token.is_space:
            lemma = token.lemma_.lower().strip()
            if lemma:
                lemmas.append((lemma, token.pos_))  # Keep the POS info
    return lemmas

# -----------------------------------------------------------------------------
# 5. MAIN LOGIC: BUILD NODES & EDGES
# -----------------------------------------------------------------------------
def process_documents(records, lemma_dict):
    all_nodes = Counter()
    all_synset_labels = {}
    all_edges = Counter()

    category_nodes = defaultdict(Counter)
    category_synsets = defaultdict(dict)
    category_edges = defaultdict(Counter)

    MISC_ADJ_LABEL = "misc_adjective"  # Our label for leftover adjectives

    for record in records:
        cat_raw = record.get("category", "Unknown")
        cat = standardize_category_name(cat_raw)

        # Content
        if isinstance(record["content"], list):
            text_block = " ".join(record["content"])
        else:
            text_block = str(record["content"])

        # Tokenize & lemmatize in Italian (but we also get POS info)
        lemma_pos_list = tokenize_and_lemmatize_italian(text_block)

        # For storing the final “keywords” we want to track
        matched_keywords = []

        for (lemma, pos) in lemma_pos_list:
            # If this lemma is in our known categories (like color/building/etc.)
            if lemma in lemma_dict:
                matched_keywords.append(lemma)
            else:
                # If it's not in the dictionary, but it's an adjective, add to fallback
                if pos == "ADJ":
                    # We'll treat it as "misc_adjective"
                    matched_keywords.append(lemma)
                    # Also, we should store that in all_synset_labels if not known
                    if lemma not in all_synset_labels:
                        all_synset_labels[lemma] = MISC_ADJ_LABEL

        # Now we count how often each matched keyword appears in this doc
        kw_counts = Counter(matched_keywords)

        # Update global nodes
        for kw, cnt in kw_counts.items():
            all_nodes[kw] += cnt
            # If kw was already in lemma_dict, use that label, otherwise use MISC_ADJ_LABEL
            if kw not in all_synset_labels:
                all_synset_labels[kw] = lemma_dict.get(kw, MISC_ADJ_LABEL)

        # Build edges
        unique_kws = list(kw_counts.keys())
        for i in range(len(unique_kws)):
            for j in range(i + 1, len(unique_kws)):
                kw1 = unique_kws[i]
                kw2 = unique_kws[j]
                edge = tuple(sorted([kw1, kw2]))
                coocc = min(kw_counts[kw1], kw_counts[kw2])
                all_edges[edge] += coocc

        # Save category-based data
        for kw, cnt in kw_counts.items():
            category_nodes[cat][kw] += cnt
            if kw not in category_synsets[cat]:
                category_synsets[cat][kw] = all_synset_labels[kw]

        for i in range(len(unique_kws)):
            for j in range(i + 1, len(unique_kws)):
                kw1 = unique_kws[i]
                kw2 = unique_kws[j]
                edge = tuple(sorted([kw1, kw2]))
                coocc = min(kw_counts[kw1], kw_counts[kw2])
                category_edges[cat][edge] += coocc

    return (all_nodes, all_synset_labels, all_edges,
            category_nodes, category_synsets, category_edges)

# -----------------------------------------------------------------------------
# 6. SAVE RESULTS AS CSV FOR GEPHI
# -----------------------------------------------------------------------------
def save_gephi_csv(nodes_counter, synset_labels, edges_counter, nodes_file, edges_file):
    """
    Save two CSV files:
       nodes_file: with columns [Id,Label,Synset,Weight]
       edges_file: with columns [Source,Target,Weight]
    """
    # NODES
    with open(nodes_file, "w", encoding="utf-8") as f:
        f.write("Id,Label,Synset,Weight\n")
        for kw, cnt in nodes_counter.items():
            syn_label = synset_labels.get(kw, "unknown")
            f.write(f"{kw},{kw},{syn_label},{cnt}\n")

    # EDGES
    with open(edges_file, "w", encoding="utf-8") as f:
        f.write("Source,Target,Weight\n")
        for (kw1, kw2), cnt in edges_counter.items():
            f.write(f"{kw1},{kw2},{cnt}\n")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to JSON data file (e.g. calvino_data.json).")
    parser.add_argument("--output", required=True, help="Output directory for CSV files.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Read data
    with open(args.input, "r", encoding="utf-8") as fin:
        records = json.load(fin)

    # Build dictionary of Italian lemma -> synset_label
    lemma_dict = build_synsets_dict()

    # Process the documents
    (all_nodes, all_synset_labels, all_edges,
     category_nodes, category_synsets, category_edges) = process_documents(records, lemma_dict)

    # Save CSV for the entire data
    nodes_all_csv = os.path.join(args.output, "nodes_all.csv")
    edges_all_csv = os.path.join(args.output, "edges_all.csv")
    save_gephi_csv(all_nodes, all_synset_labels, all_edges, nodes_all_csv, edges_all_csv)

    # Save CSV for each category
    for cat, cat_nodes_counter in category_nodes.items():
        cat_name_safe = re.sub(r"\s+", "_", cat.lower())
        nodes_cat_csv = os.path.join(args.output, f"nodes_{cat_name_safe}.csv")
        edges_cat_csv = os.path.join(args.output, f"edges_{cat_name_safe}.csv")

        cat_synset_labels = category_synsets[cat]
        cat_edges_counter = category_edges[cat]

        save_gephi_csv(cat_nodes_counter, cat_synset_labels, cat_edges_counter,
                       nodes_cat_csv, edges_cat_csv)

    print("Done! CSV files have been generated.")

if __name__ == "__main__":
    main()
