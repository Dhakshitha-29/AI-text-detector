# loads the saved models and predicts new text without retraining
# the output is a probability from the model, it does not prove who wrote the text
import joblib
import numpy as np
from scipy import sparse

from src import config
from src.config import ProjectError
from src.data_preparation import clean_text
from src.feature_extraction import FEATURE_NAMES, extract_features

INDICATORS = [
    ("Vocabulary diversity", "vocab_richness"),
    ("Sentence variation", "sentence_len_var"),
    ("Repetition", "repeated_bigram_ratio"),
    ("Avg sentence length", "avg_sentence_len"),
    ("Punctuation use", "punctuation_ratio"),
]


def load_model(path, name):
    if not path.exists():
        raise ProjectError(f"{name} model not found ({path.name}). Train first:\n"
                           "  python main.py prepare\n  python main.py train")
    try:
        return joblib.load(path)
    except Exception as error:
        raise ProjectError(f"Could not load {path.name} ({error}). Retrain with: python main.py train")


def load_reference():
    if not config.FEATURE_REFERENCE_PATH.exists():
        raise ProjectError("feature_reference.pkl missing. Retrain with: python main.py train")
    return joblib.load(config.FEATURE_REFERENCE_PATH)


def _level(value, low, high):
    if value <= low:
        return "LOW"
    if value >= high:
        return "HIGH"
    return "MEDIUM"


def linguistic_indicators(text, reference):
    # compare the text features with the low/high limits saved during training
    values = dict(zip(FEATURE_NAMES, extract_features(text)))
    result = []
    for title, feature in INDICATORS:
        ref = reference[feature]
        result.append((title, _level(values[feature], ref["low"], ref["high"]), values[feature]))
    return result


def top_terms(model, text, k=5):
    # contribution of a word = its coefficient x its tfidf value
    # positive pushes towards AI, negative towards human
    try:
        union = model.named_steps["features"]
        names = union.transformer_list[0][1].get_feature_names_out()
        row = union.transform([text])
        row = row.toarray()[0] if sparse.issparse(row) else np.asarray(row)[0]
        contribution = row[:len(names)] * model.named_steps["clf"].coef_[0][:len(names)]
        order = np.argsort(contribution)
        toward_ai = [names[i] for i in order[::-1][:k] if contribution[i] > 0]
        toward_human = [names[i] for i in order[:k] if contribution[i] < 0]
        return toward_ai, toward_human
    except Exception:
        return [], []


def predict_text(text, proposed, baseline=None, reference=None):
    text = clean_text(text)
    n_words = len(text.split())
    if n_words == 0:
        raise ProjectError("Empty text. Please type or paste some text.")
    if n_words < config.MIN_WORDS_PREDICT:
        raise ProjectError(f"Text is too short ({n_words} words). Please enter at least {config.MIN_WORDS_PREDICT} words.")

    warnings = []
    if n_words < config.WARN_WORDS_PREDICT:
        warnings.append(f"Short text ({n_words} words): the prediction is unreliable.")

    p_ai = float(proposed.predict_proba([text])[0][1])
    label = "AI-GENERATED" if p_ai >= 0.5 else "HUMAN-WRITTEN"
    result = {
        "label": label, "confidence": max(p_ai, 1 - p_ai), "p_ai": p_ai, "warnings": warnings,
        "baseline_p_ai": float(baseline.predict_proba([text])[0][1]) if baseline is not None else None,
        "indicators": linguistic_indicators(text, reference) if reference else [],
    }
    result["terms_ai"], result["terms_human"] = top_terms(proposed, text)
    return result


def print_prediction(result):
    print("\n" + "=" * 44)
    print("PREDICTION (model estimate, not proof)")
    print(f"Class      : {result['label']}")
    print(f"Confidence : {result['confidence'] * 100:.1f}%")
    print(f"P(AI)      : {result['p_ai'] * 100:.1f}%")
    if result["baseline_p_ai"] is not None:
        print(f"Baseline model P(AI) for comparison: {result['baseline_p_ai'] * 100:.1f}%")
    for warning in result["warnings"]:
        print(f"[!] {warning}")
    if result["indicators"]:
        print("\nLinguistic Indicators (relative to training texts)")
        for title, level, value in result["indicators"]:
            print(f"  {title:<22}: {level:<6} (value {value:.3f})")
    if result["terms_ai"] or result["terms_human"]:
        print("\nWords that pushed the model")
        print(f"  towards AI    : {', '.join(result['terms_ai']) or '-'}")
        print(f"  towards human : {', '.join(result['terms_human']) or '-'}")
    print("=" * 44)


def detect(text):
    proposed = load_model(config.PROPOSED_MODEL_PATH, "Proposed")
    reference = load_reference()
    baseline = joblib.load(config.BASELINE_MODEL_PATH) if config.BASELINE_MODEL_PATH.exists() else None
    print_prediction(predict_text(text, proposed, baseline, reference))
