import re
import string
from functools import lru_cache

import numpy as np
import pandas as pd

# the order here is the column order of the feature matrix
FEATURE_NAMES = [
    "word_count",
    "sentence_count",
    "avg_sentence_len",
    "sentence_len_var",
    "avg_word_len",
    "vocab_richness",         # unique words / total words
    "stopword_ratio",
    "punctuation_ratio",
    "comma_ratio",
    "question_ratio",
    "exclamation_ratio",
    "digit_ratio",
    "uppercase_ratio",
    "repeated_word_ratio",
    "repeated_bigram_ratio",
]

# used only if nltk stopwords can't be loaded
FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for", "with", "about", "to", "from",
    "in", "on", "off", "over", "under", "up", "down", "out", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your",
    "his", "our", "their", "so", "not", "no", "than", "too", "very", "can", "will", "just", "there",
    "what", "which", "who", "whom", "when", "where", "why", "how", "all", "any", "both", "each",
    "more", "most", "other", "some", "such", "only", "own", "same", "then", "once", "into", "through",
}

WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


@lru_cache(maxsize=1)
def get_stopwords():
    try:
        from nltk.corpus import stopwords
        try:
            return frozenset(stopwords.words("english"))
        except LookupError:
            import nltk
            if nltk.download("stopwords", quiet=True):
                return frozenset(stopwords.words("english"))
    except Exception:
        pass
    print("nltk stopwords not found, using a small built-in list instead")
    return frozenset(FALLBACK_STOPWORDS)


def safe_divide(a, b):
    return a / b if b else 0.0


def extract_features(text):
    # empty or weird text just gives all zeros
    if not isinstance(text, str) or not text.strip():
        return [0.0] * len(FEATURE_NAMES)

    n_chars = len(text)
    words = WORD_RE.findall(text.lower())
    n_words = len(words)

    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s and re.search(r"\w", s)]
    sentence_lengths = [len(WORD_RE.findall(s)) for s in sentences] or [0]

    stop = get_stopwords()
    content_words = [w for w in words if w not in stop]
    bigrams = list(zip(words, words[1:]))
    n_letters = sum(c.isalpha() for c in text)

    features = {
        "word_count": float(n_words),
        "sentence_count": float(len(sentences)),
        "avg_sentence_len": float(np.mean(sentence_lengths)),
        "sentence_len_var": float(np.var(sentence_lengths)),
        "avg_word_len": safe_divide(sum(len(w) for w in words), n_words),
        "vocab_richness": safe_divide(len(set(words)), n_words),
        "stopword_ratio": safe_divide(n_words - len(content_words), n_words),
        "punctuation_ratio": safe_divide(sum(c in string.punctuation for c in text), n_chars),
        "comma_ratio": safe_divide(text.count(","), n_chars),
        "question_ratio": safe_divide(text.count("?"), n_chars),
        "exclamation_ratio": safe_divide(text.count("!"), n_chars),
        "digit_ratio": safe_divide(sum(c.isdigit() for c in text), n_chars),
        "uppercase_ratio": safe_divide(sum(c.isupper() for c in text), n_letters),
        "repeated_word_ratio": safe_divide(len(content_words) - len(set(content_words)), len(content_words)),
        "repeated_bigram_ratio": safe_divide(len(bigrams) - len(set(bigrams)), len(bigrams)),
    }
    return [features[name] for name in FEATURE_NAMES]


def extract_feature_matrix(texts):
    matrix = np.array([extract_features(t) for t in texts], dtype=float)
    matrix = matrix.reshape(-1, len(FEATURE_NAMES))
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def features_to_dataframe(texts):
    return pd.DataFrame(extract_feature_matrix(texts), columns=FEATURE_NAMES)
