import nltk
import pandas as pd
from preprocessing import essay_to_sentences

nltk.download("averaged_perceptron_tagger")

def count_pos_tags(essay: str):
    """
    Counts POS tags: nouns, verbs, adjectives, adverbs.
    """
    sentences = essay_to_sentences(essay)

    noun = verb = adj = adv = 0
    for sent in sentences:
        for _, tag in nltk.pos_tag(sent):
            if tag.startswith("N"):
                noun += 1
            elif tag.startswith("V"):
                verb += 1
            elif tag.startswith("J"):
                adj += 1
            elif tag.startswith("R"):
                adv += 1

    return noun, verb, adj, adv


def extract_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds linguistic features to the dataframe.
    """
    df = df.copy()

    df["word_count"] = df["essay"].apply(lambda x: len(x.split()))
    df["char_count"] = df["essay"].apply(len)
    df["sent_count"] = df["essay"].apply(lambda x: len(nltk.sent_tokenize(x)))

    pos_counts = df["essay"].apply(count_pos_tags)
    df["noun_count"], df["verb_count"], df["adj_count"], df["adv_count"] = zip(*pos_counts)

    return df
