import joblib
import numpy as np

from src import config
from src.data_preparation import load_processed, select_training_rows
from src.feature_extraction import FEATURE_NAMES, extract_feature_matrix
from src.modeling import build_proposed, train_model


def save_feature_reference(df):
    # saves what "low" and "high" mean for each feature (33rd and 67th percentile of training texts)
    # predict.py uses this to show LOW / MEDIUM / HIGH
    texts = select_training_rows(df, "train", augmented=True)["text"]
    matrix = extract_feature_matrix(texts)
    reference = {}
    for i, name in enumerate(FEATURE_NAMES):
        low, high = np.percentile(matrix[:, i], [33.3, 66.7])
        reference[name] = {"low": float(low), "high": float(high)}
    joblib.dump(reference, config.FEATURE_REFERENCE_PATH)


def train_and_save_proposed(df):
    config.ensure_dirs()
    # proposed model is trained WITH the modified AI texts
    model, best_c, val_f1 = train_model(df, build_proposed, augmented=True)
    joblib.dump(model, config.PROPOSED_MODEL_PATH)
    save_feature_reference(df)
    print(f"Proposed model trained (TF-IDF + linguistic + LogReg, C={best_c}, validation F1={val_f1:.3f})")
    print(f"Saved to {config.PROPOSED_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save_proposed(load_processed())
