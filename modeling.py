import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from src import config
from src.data_preparation import select_training_rows, to_xy
from src.feature_extraction import extract_feature_matrix


def make_tfidf():
    # same tfidf settings for every model so the comparison is fair
    return TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, max_features=20000, strip_accents="unicode",
    )


def clip_features(matrix):
    # stops very large values from dominating
    return np.clip(matrix, -5.0, 5.0)


def build_baseline_lr(C=1.0):
    # text -> tfidf -> logistic regression
    return Pipeline([
        ("tfidf", make_tfidf()),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


def build_baseline_svm(C=1.0):
    # text -> tfidf -> linear svm
    return Pipeline([
        ("tfidf", make_tfidf()),
        ("clf", LinearSVC(C=C, max_iter=5000, class_weight="balanced")),
    ])


def build_proposed(C=1.0):
    # text -> (tfidf + style features) -> logistic regression
    style_features = Pipeline([
        ("extract", FunctionTransformer(extract_feature_matrix)),
        ("scale", StandardScaler()),
        ("clip", FunctionTransformer(clip_features)),
    ])
    return Pipeline([
        ("features", FeatureUnion([("tfidf", make_tfidf()), ("linguistic", style_features)])),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
    ])


MODEL_BUILDERS = {
    "TF-IDF + LogReg (baseline)": build_baseline_lr,
    "TF-IDF + LinearSVM": build_baseline_svm,
    "TF-IDF + Linguistic + LogReg (proposed)": build_proposed,
}


def train_model(df, builder, augmented):
    # train on the train split, pick the best C using the validation split
    # (the test split is not used here)
    train_x, train_y = to_xy(select_training_rows(df, "train", augmented))
    val_x, val_y = to_xy(select_training_rows(df, "val", augmented))

    best = None
    for C in config.C_GRID:
        model = builder(C)
        model.fit(train_x, train_y)
        score = f1_score(val_y, model.predict(val_x), zero_division=0)
        if best is None or score > best[0]:
            best = (score, C, model)
    return best[2], best[1], best[0]


def compute_metrics(model, texts, y_true):
    y_pred = model.predict(texts)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(texts)[:, 1]
    else:
        scores = model.decision_function(texts)   # svm has no probabilities
    auc = roc_auc_score(y_true, scores) if len(set(y_true)) == 2 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
