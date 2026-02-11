import re
import nltk

nltk.download("punkt")

def clean_sentence(sentence: str) -> list[str]:
    """
    Cleans a sentence by removing non-alphabetic characters and lowercasing.
    """
    sentence = re.sub(r"[^A-Za-z]", " ", sentence)
    sentence = sentence.lower()
    return sentence.split()


def essay_to_sentences(essay: str) -> list[list[str]]:
    """
    Splits an essay into sentences and tokenizes each sentence.
    """
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    sentences = tokenizer.tokenize(essay.strip())
    return [clean_sentence(sent) for sent in sentences if sent.strip()]
