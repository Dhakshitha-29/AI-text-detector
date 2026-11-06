# robustness experiment: train every model two ways (clean / augmented) and test on original and modified AI text
# clean = human + original AI, augmented = clean + modified AI. all numbers come from the fitted models
import matplotlib
matplotlib.use("Agg")   # save plots to files, no window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.config import ProjectError
from src.data_preparation import to_xy
from src.feature_extraction import FEATURE_NAMES, features_to_dataframe
from src.modeling import MODEL_BUILDERS, compute_metrics, train_model

TRAININGS = {"clean": False, "augmented": True}
MODIFIED_ALL = "modified (all types)"


def build_test_sets(df):
    # all test sets come only from the test split
    test = df[df["split"] == "test"]
    human = test[test["label"] == "human"]
    ai_original = test[(test["label"] == "ai") & (test["variant"] == "original")]
    ai_modified = test[(test["label"] == "ai") & (test["variant"] != "original")]
    if ai_modified.empty or human.empty or ai_original.empty:
        raise ProjectError("The test split has no human / AI / modified-AI texts. Use a larger dataset.")

    sets = {"original": pd.concat([human, ai_original])}
    # one random modified version per AI text so human and AI stay balanced
    one_each = ai_modified.groupby("group_id", group_keys=False).sample(n=1, random_state=config.RANDOM_STATE)
    sets[MODIFIED_ALL] = pd.concat([human, one_each])
    for kind in sorted(ai_modified["variant"].unique()):
        sets[f"modified: {kind}"] = pd.concat([human, ai_modified[ai_modified["variant"] == kind]])
    return sets


def run_experiments(df):
    # 3 models x 2 training types, each tested on every test set
    config.ensure_dirs()
    test_sets = build_test_sets(df)
    rows, matrices = [], {}

    for model_name, builder in MODEL_BUILDERS.items():
        for training_name, augmented in TRAININGS.items():
            print(f"  training: {model_name} | {training_name} ...")
            model, best_c, val_f1 = train_model(df, builder, augmented)
            for set_name, part in test_sets.items():
                x, y = to_xy(part)
                m = compute_metrics(model, x, y)
                matrices[(model_name, training_name, set_name)] = m.pop("confusion_matrix")
                rows.append({"model": model_name, "training": training_name, "test_set": set_name,
                             "n_test": len(part), "best_C": best_c, **m})

    results = pd.DataFrame(rows)
    results.to_csv(config.ALL_RESULTS_CSV, index=False)
    plot_confusion_matrices(matrices)
    plot_model_comparison(results)
    write_metrics_report(df, results, test_sets)
    print(f"Results saved in {config.RESULTS_DIR}")
    return results


def robustness_table(results):
    def pick(test_set, column):
        part = results[results["test_set"] == test_set].set_index(["model", "training"])
        return part[column]

    table = pd.DataFrame({
        "Original F1": pick("original", "f1"),
        "Modified F1": pick(MODIFIED_ALL, "f1"),
        "Modified AI recall": pick(MODIFIED_ALL, "recall"),
    })
    table["F1 drop"] = table["Original F1"] - table["Modified F1"]
    return table.reset_index().rename(columns={"model": "Model", "training": "Training"})


def comparison_table(results, test_set="original"):
    part = results[results["test_set"] == test_set]
    cols = ["model", "training", "accuracy", "precision", "recall", "f1", "roc_auc"]
    return part[cols].rename(columns={"model": "Model", "training": "Training", "accuracy": "Accuracy",
                                      "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"})


def per_type_table(results):
    part = results[results["test_set"].str.startswith("modified: ")]
    table = part.pivot_table(index=["model", "training"], columns="test_set", values="f1")
    return table.reset_index()


def print_robustness(results):
    print("\nROBUSTNESS EXPERIMENT (AI = positive class; all values measured on unseen test texts)")
    print(robustness_table(results).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nF1 per modification type:")
    print(per_type_table(results).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def print_model_comparison(results):
    print("\nMODEL COMPARISON on the ORIGINAL (unmodified) test set")
    print(comparison_table(results, "original").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nMODEL COMPARISON on the MODIFIED test set")
    print(comparison_table(results, MODIFIED_ALL).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def plot_confusion_matrices(matrices):
    # baseline (clean) vs proposed (augmented) on original and modified test sets
    chosen = [("TF-IDF + LogReg (baseline)", "clean"),
              ("TF-IDF + Linguistic + LogReg (proposed)", "augmented")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for r, (model_name, training) in enumerate(chosen):
        for c, set_name in enumerate(["original", MODIFIED_ALL]):
            sns.heatmap(matrices[(model_name, training, set_name)], annot=True, fmt="d", cmap="Blues",
                        cbar=False, ax=axes[r][c], xticklabels=["Human", "AI"], yticklabels=["Human", "AI"])
            axes[r][c].set_title(f"{model_name.split(' (')[-1].rstrip(')')} / {training}\n{set_name}", fontsize=9)
            axes[r][c].set_xlabel("Predicted")
            axes[r][c].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(config.CONFUSION_PNG, dpi=150)
    plt.close(fig)


def plot_model_comparison(results):
    table = robustness_table(results)
    short = {"TF-IDF + LogReg (baseline)": "Baseline LR", "TF-IDF + LinearSVM": "LinearSVM",
             "TF-IDF + Linguistic + LogReg (proposed)": "Proposed"}
    labels = [f"{short[m]}\n({t})" for m, t in zip(table["Model"], table["Training"])]
    x = range(len(table))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], table["Original F1"], width, label="Original AI text")
    ax.bar([i + width / 2 for i in x], table["Modified F1"], width, label="Modified AI text")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("F1 score (AI = positive class)")
    ax.set_ylim(0, 1.2)
    ax.set_title("Robustness: F1 on original vs modified AI text")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.COMPARISON_PNG, dpi=150)
    plt.close(fig)


def write_metrics_report(df, results, test_sets):
    fmt = lambda v: f"{v:.3f}"
    originals = df[df["variant"] == "original"]
    train_features = features_to_dataframe(originals[originals["split"] == "train"]["text"])
    train_features["label"] = originals[originals["split"] == "train"]["label"].values
    means = train_features.groupby("label")[FEATURE_NAMES].mean().T

    lines = [
        "AI TEXT DETECTION - EXPERIMENT REPORT",
        "=" * 60,
        "All numbers were computed by the program on unseen test texts (AI = positive class).",
        "Modified AI texts come from data/modified_ai_text.csv if it exists, otherwise from src/robustness.py.",
        "",
        "Documents per split (original texts):",
        pd.crosstab(originals["split"], originals["label"]).to_string(),
        "",
        "Modified AI texts per split and type:",
        pd.crosstab(df[df["variant"] != "original"]["variant"], df[df["variant"] != "original"]["split"]).to_string(),
        "",
        "Test set sizes: " + ", ".join(f"{k}={len(v)}" for k, v in test_sets.items()),
        "",
        "ROBUSTNESS TABLE", "-" * 60,
        robustness_table(results).to_string(index=False, float_format=fmt),
        "",
        "F1 PER MODIFICATION TYPE", "-" * 60,
        per_type_table(results).to_string(index=False, float_format=fmt),
        "",
        "MODEL COMPARISON - ORIGINAL TEST SET", "-" * 60,
        comparison_table(results, "original").to_string(index=False, float_format=fmt),
        "",
        "MODEL COMPARISON - MODIFIED TEST SET", "-" * 60,
        comparison_table(results, MODIFIED_ALL).to_string(index=False, float_format=fmt),
        "",
        "MEAN LINGUISTIC FEATURES (training texts, original only)", "-" * 60,
        means.to_string(float_format=lambda v: f"{v:.4f}"),
        "",
        "NOTES",
        "- The models give probabilities, not proof of who wrote a text.",
        "- Modified texts from robustness.py are simple rule based edits, not a real paraphrasing model.",
        "- On a small or fake dataset these numbers only show that the code runs.",
    ]
    config.METRICS_TXT.write_text("\n".join(lines), encoding="utf-8")


def show_saved_metrics():
    if not config.METRICS_TXT.exists():
        raise ProjectError("No metrics yet. Run:  python main.py evaluate")
    print(config.METRICS_TXT.read_text(encoding="utf-8"))


def load_or_run(df, force=False):
    # reuse saved results if they exist
    if not force and config.ALL_RESULTS_CSV.exists():
        return pd.read_csv(config.ALL_RESULTS_CSV)
    print("Running experiments (this trains 6 small models) ...")
    return run_experiments(df)


if __name__ == "__main__":
    from src.data_preparation import load_processed
    print_robustness(run_experiments(load_processed()))
