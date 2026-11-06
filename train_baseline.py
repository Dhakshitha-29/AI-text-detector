import joblib

from src import config
from src.data_preparation import load_processed
from src.modeling import build_baseline_lr, train_model


def train_and_save_baseline(df):
    config.ensure_dirs()
    # baseline is trained without any modified text
    model, best_c, val_f1 = train_model(df, build_baseline_lr, augmented=False)
    joblib.dump(model, config.BASELINE_MODEL_PATH)
    print(f"Baseline trained (TF-IDF + LogReg, C={best_c}, validation F1={val_f1:.3f})")
    print(f"Saved to {config.BASELINE_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_and_save_baseline(load_processed())
